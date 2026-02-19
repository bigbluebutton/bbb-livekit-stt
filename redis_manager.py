import asyncio
import json
import logging
import time
import redis.asyncio as redis
from livekit.agents import stt
from config import RedisConfig
from livekit.agents.stt import SpeechData

class RedisManager:
    UPDATE_TRANSCRIPT_PUB_MSG = "UpdateTranscriptPubMsg"
    USER_SPEECH_LOCALE_CHANGED_EVT_MSG = "UserSpeechLocaleChangedEvtMsg"
    USER_SPEECH_OPTIONS_CHANGED_EVT_MSG = "UserSpeechOptionsChangedEvtMsg"
    TO_AKKA_APPS_CHANNEL = "to-akka-apps-redis-channel"
    FROM_AKKA_APPS_CHANNEL = "from-akka-apps-redis-channel"

    def __init__(self, config: RedisConfig):
        self.host = config.host
        self.port = config.port
        self.password = config.password
        self.pub_client = None
        self.sub_client = None

    async def connect(self):
        try:
            logging.debug(f"Connecting to Redis at {self.host}:{self.port}")
            self.pub_client = redis.Redis(host=self.host, port=self.port, password=self.password)
            self.sub_client = redis.Redis(host=self.host, port=self.port, password=self.password)
            await self.pub_client.ping()
            await self.sub_client.ping()
            logging.info("Connected to Redis")
        except Exception as e:
            logging.error(f"Failed to connect to Redis: {e}")
            self.pub_client = None
            self.sub_client = None

    async def publish_update_transcript_pub_msg(
        self,
        meeting_id: str,
        user_id: str,
        transcript_data: SpeechData,
        locale: str,
        result: bool = True,
    ):
        if not self.pub_client:
            logging.warning("Redis not connected, skipping transcription publish")
            return

        message = self._generate_update_transcript_pub_msg(
            meeting_id,
            user_id,
            locale,
            transcript_data.text,
            result,
            int(transcript_data.start_time * 1000),
            int(transcript_data.end_time * 1000),
        )

        try:
            await self.pub_client.publish(self.TO_AKKA_APPS_CHANNEL, json.dumps(message))
            logging.debug(f"Published to Redis: {message}")
        except Exception as e:
            logging.error(f"Failed to publish to Redis: {e}")

    async def listen(self, callback):
        if not self.sub_client:
            logging.error("Redis subscriber not connected, cannot listen for commands.")
            return

        async with self.sub_client.pubsub() as pubsub:
            await pubsub.subscribe(self.FROM_AKKA_APPS_CHANNEL)
            logging.info(f"Subscribed to Redis channel: {self.FROM_AKKA_APPS_CHANNEL}")

            while True:
                try:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)

                    if message and message['type'] == 'message':
                        await callback(message['data'].decode('utf-8'))
                except asyncio.CancelledError:
                    logging.info("Redis command listener cancelled.")
                    break
                except Exception as e:
                    logging.error(f"Error in Redis command listener: {e}")
                    await asyncio.sleep(1)

    async def aclose(self):
        if self.pub_client:
            await self.pub_client.aclose()

        if self.sub_client:
            await self.sub_client.aclose()

        logging.info("Redis connection closed")

    def _generate_update_transcript_pub_msg(
        self,
        meeting_id: str,
        user_id: str,
        locale: str,
        transcript: str,
        result: bool,
        start: int = 0,
        end: int = 0,
    ):
        return {
            "envelope": {
                "name": self.UPDATE_TRANSCRIPT_PUB_MSG,
                "routing": {
                    "meetingId": meeting_id,
                    "userId": user_id,
                },
                "timestamp": int(time.time() * 1000),
            },
            "core": {
                "header": {
                    "name": self.UPDATE_TRANSCRIPT_PUB_MSG,
                    "meetingId": meeting_id,
                    "userId": user_id,
                },
                "body": {
                    "transcriptId": f"{user_id}-{locale}-{start}",
                    "start": str(start),
                    "end": str(end),
                    "text": '',
                    "transcript": transcript,
                    "locale": locale,
                    "result": result,
                },
            }
        }
