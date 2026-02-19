import asyncio
import json
import logging

import nest_asyncio
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, stt
from livekit import rtc

from redis_manager import RedisManager
from gladia_stt_agent import GladiaSttAgent
from config import gladia_config, redis_config

load_dotenv()

async def entrypoint(ctx: JobContext):
    nest_asyncio.apply()

    redis_manager = RedisManager(redis_config)
    agent = GladiaSttAgent(gladia_config)

    async def on_redis_message(message_data: str):
        try:
            msg = json.loads(message_data)
            envelope = msg.get("envelope", {})
            core = msg.get("core", {})

            event_name = envelope.get("name")
            if event_name not in (
                RedisManager.USER_SPEECH_LOCALE_CHANGED_EVT_MSG,
                RedisManager.USER_SPEECH_OPTIONS_CHANGED_EVT_MSG,
            ):
                return

            routing = envelope.get("routing", {})
            body = core.get("body", {})
            meeting_id = routing.get("meetingId")
            user_id = routing.get("userId")

            if meeting_id != agent.room.name:
                return

            if event_name == RedisManager.USER_SPEECH_LOCALE_CHANGED_EVT_MSG:
                locale = body.get("locale")
                provider = body.get("provider")

                if not (provider and locale):
                    agent.stop_transcription_for_user(user_id)
                else:
                    current_locale = agent.participant_settings.get(user_id, {}).get('locale')
                    if current_locale and current_locale != locale:
                        agent.update_locale_for_user(user_id, locale)
                    elif not current_locale:
                        agent.start_transcription_for_user(user_id, locale, provider)

            elif event_name == RedisManager.USER_SPEECH_OPTIONS_CHANGED_EVT_MSG:
                partial_utterances = body.get("partialUtterances", False)
                min_utterance_length = body.get("minUtteranceLength", 0)
                settings = agent.participant_settings.setdefault(user_id, {})
                settings["partial_utterances"] = partial_utterances
                settings["min_utterance_length"] = min_utterance_length
                logging.info(f"User speech options changed for {user_id}: {settings}")

        except json.JSONDecodeError:
            logging.warning(f"Could not decode Redis message: {message_data}")
        except Exception as e:
            logging.error(f"Error processing Redis message: {e}")

    @agent.on("final_transcript")
    async def on_final_transcript(participant: rtc.RemoteParticipant, event: stt.SpeechEvent):
        p_settings = agent.participant_settings.get(participant.identity, {})
        original_locale = p_settings.get("locale")

        if not original_locale:
            logging.warning(f"Could not find original locale for participant {participant.identity}, cannot process transcripts.")
            return

        original_lang = original_locale.split('-')[0]

        for alternative in event.alternatives:
            transcript_lang = alternative.language
            text = alternative.text
            bbb_locale = None

            logging.info(f"Transcript for {participant.identity} ({transcript_lang}): {text}")

            if transcript_lang == original_lang:
                # This is the original transcript, use the original BBB locale
                bbb_locale = original_locale
            else:
                # This is a translated transcript, look it up in the map
                bbb_locale = gladia_config.translation_lang_map.get(transcript_lang)

            if not bbb_locale:
                logging.warning(f"Could not find a BBB locale mapping for language '{transcript_lang}'. "
                                f"Falling back to the language code itself. ")
                bbb_locale = transcript_lang

            await redis_manager.publish_update_transcript_pub_msg(
                agent.room.name, participant.identity, alternative, bbb_locale
            )

    @agent.on("interim_transcript")
    async def on_interim_transcript(participant: rtc.RemoteParticipant, event: stt.SpeechEvent):
        p_settings = agent.participant_settings.get(participant.identity, {})

        if not p_settings.get("partial_utterances", False):
            return

        original_locale = p_settings.get("locale")

        if not original_locale:
            logging.warning(f"Could not find original locale for participant {participant.identity}, cannot process interim transcripts.")
            return

        original_lang = original_locale.split('-')[0]
        min_utterance_length = p_settings.get("min_utterance_length", 0)

        for alternative in event.alternatives:
            transcript_lang = alternative.language
            text = alternative.text

            if min_utterance_length and len(text.split()) < min_utterance_length:
                logging.debug(f"Discarding interim transcript for {participant.identity}: too short ({len(text.split())} < {min_utterance_length} words).")
                continue

            bbb_locale = None

            logging.debug(f"Interim transcript for {participant.identity} ({transcript_lang}): {text}")

            if transcript_lang == original_lang:
                bbb_locale = original_locale
            else:
                bbb_locale = gladia_config.translation_lang_map.get(transcript_lang)

            if not bbb_locale:
                logging.warning(f"Could not find a BBB locale mapping for language '{transcript_lang}'. "
                                f"Falling back to the language code itself. ")
                bbb_locale = transcript_lang

            await redis_manager.publish_update_transcript_pub_msg(
                agent.room.name, participant.identity, alternative, bbb_locale, result=False
            )

    redis_listen_task = asyncio.create_task(redis_manager.listen(on_redis_message))

    try:
        await redis_manager.connect()
        logging.info(f"Received job for room {ctx.room.name}")
        await agent.start(ctx)
    finally:
        redis_listen_task.cancel()
        try:
            await redis_listen_task
        except asyncio.CancelledError:
            pass
        await redis_manager.aclose()


if __name__ == "__main__":
    opts = WorkerOptions(entrypoint_fnc=entrypoint)
    cli.run_app(opts)
