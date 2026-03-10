import asyncio
import logging
import time

import numpy as np
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    stt,
)
from livekit.plugins import openai as openai_plugin

from config import OpenAiConfig
from events import EventEmitter

# Energy-based voice activity detection parameters.
# RMS threshold (int16 scale 0–32768): frames below this are considered silence.
_SILENCE_THRESHOLD_RMS = 500
# Seconds of continuous silence after speech before the segment is flushed.
_SILENCE_DURATION_S = 0.8
# Maximum segment duration before a forced flush (prevents unbounded buffering).
_MAX_BUFFER_DURATION_S = 30.0


class OpenAiSttAgent(EventEmitter):
    def __init__(self, config: OpenAiConfig):
        super().__init__()
        self.config = config
        self.stt = openai_plugin.STT(**config.to_dict())
        self.ctx: JobContext | None = None
        self.room: rtc.Room | None = None
        self.processing_info = {}
        self.participant_settings = {}
        self.open_time = time.time()
        self._shutdown = asyncio.Event()

    async def start(self, ctx: JobContext):
        self.ctx = ctx
        await self.ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        self.room = self.ctx.room

        self.room.on("participant_disconnected", self._on_participant_disconnected)
        self.room.on("disconnected", self._on_disconnected)
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)

        try:
            await self._shutdown.wait()
        finally:
            await self._cleanup()

    async def _cleanup(self):
        for user_id in list(self.processing_info.keys()):
            self.stop_transcription_for_user(user_id)

        await asyncio.sleep(0.1)

    def start_transcription_for_user(self, user_id: str, locale: str, provider: str):
        settings = self.participant_settings.setdefault(user_id, {})
        settings["locale"] = locale
        settings["provider"] = provider

        participant = self._find_participant(user_id)

        if not participant:
            logging.error(
                f"Cannot start transcription, participant {user_id} not found."
            )
            return

        track = self._find_audio_track(participant)

        if not track:
            logging.warning(
                f"Won't start transcription yet, no audio track found for {user_id}."
            )
            return

        if participant.identity in self.processing_info:
            logging.debug(
                f"Transcription task already running for {participant.identity}, ignoring start command."
            )
            return

        language = self._sanitize_locale(locale)
        task = asyncio.create_task(
            self._run_transcription_pipeline(participant, track, language)
        )
        self.processing_info[participant.identity] = {"task": task}
        logging.info(
            f"Started transcription for {participant.identity} with locale {locale}."
        )

    def stop_transcription_for_user(self, user_id: str):
        logging.debug(f"Stopping transcription for {user_id}.")

        if user_id in self.processing_info:
            info = self.processing_info.pop(user_id)
            info["task"].cancel()
            logging.info(f"Stopped transcription for user {user_id}.")

    def update_locale_for_user(self, user_id: str, locale: str):
        if user_id in self.participant_settings:
            self.participant_settings[user_id]["locale"] = locale

        if user_id in self.processing_info:
            logging.info(f"Updating locale to '{locale}' for user {user_id}.")
            provider = self.participant_settings.get(user_id, {}).get(
                "provider", "openai"
            )
            # Restart the pipeline with the new locale.
            self.stop_transcription_for_user(user_id)
            self.start_transcription_for_user(user_id, locale, provider)
        else:
            logging.warning(
                f"Won't update locale, no active transcription for user {user_id}."
            )

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
            logging.debug(
                f"Skipping transcription for {participant.identity}'s track {track.sid} because it's not a microphone."
            )
            return

        settings = self.participant_settings.get(participant.identity)

        locale = settings.get("locale") if settings else None
        provider = settings.get("provider") if settings else None

        if locale and provider:
            logging.debug(
                f"Participant {participant.identity} subscribed with active settings, starting transcription.",
                extra={"settings": settings},
            )
            self.start_transcription_for_user(participant.identity, locale, provider)
        else:
            logging.debug(
                f"Participant {participant.identity} subscribed with no active settings, skipping transcription."
            )

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        self.stop_transcription_for_user(participant.identity)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant, *_):
        logging.debug(
            f"Participant {participant.identity} disconnected, stopping transcription."
        )
        self.stop_transcription_for_user(participant.identity)
        self.participant_settings.pop(participant.identity, None)

    def _on_disconnected(self):
        self._shutdown.set()

    def _find_participant(self, identity: str) -> rtc.RemoteParticipant | None:
        for p in self.room.remote_participants.values():
            if p.identity == identity:
                return p
        return None

    def _find_audio_track(self, participant: rtc.RemoteParticipant) -> rtc.Track | None:
        for pub in participant.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                return pub.track
        return None

    def _sanitize_locale(self, locale: str) -> str:
        # OpenAI STT accepts ISO 639-1 language codes (e.g. "en")
        # BBB uses <ISO 639-1>-<ISO 3166-1> format (e.g. "en-US")
        return locale.split("-")[0].lower()

    async def _run_transcription_pipeline(
        self,
        participant: rtc.RemoteParticipant,
        track: rtc.Track,
        language: str,
    ):
        """Collect audio, segment by silence, and transcribe via REST API.

        The OpenAI STT plugin's stream() method requires the Realtime WebSocket
        API which not all backends support.  Using recognize() hits the standard
        REST /audio/transcriptions endpoint and works with any Whisper-compatible
        backend.

        TODO: Support the /realtime WebSocket endpoint as an opt-in mode (e.g.
        via an OpenAiConfig flag like `use_realtime: bool`).  When enabled,
        delegate to openai_plugin.STT.stream() directly instead of this
        energy-based segmentation loop.  This would unlock interim results and
        lower latency for backends that implement the OpenAI Realtime API
        (e.g. gpt-4o-transcribe).
        """
        audio_stream = rtc.AudioStream(track)
        self.open_time = time.time()

        speech_buffer: list[rtc.AudioFrame] = []
        buffer_duration = 0.0
        silence_duration = 0.0
        was_speaking = False

        async def flush_segment(frames: list[rtc.AudioFrame]) -> None:
            if not frames:
                return
            try:
                event = await self.stt.recognize(buffer=frames, language=language)
                if event.alternatives and event.alternatives[0].text:
                    self.emit(
                        "final_transcript",
                        participant=participant,
                        event=event,
                        open_time=self.open_time,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logging.error(
                    f"Error transcribing segment for {participant.identity}: {e}"
                )

        try:
            async for audio_event in audio_stream:
                frame = audio_event.frame
                samples = np.frombuffer(frame.data, dtype=np.int16)
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                is_speaking = rms > _SILENCE_THRESHOLD_RMS
                frame_duration = frame.samples_per_channel / frame.sample_rate

                if is_speaking:
                    speech_buffer.append(frame)
                    buffer_duration += frame_duration
                    silence_duration = 0.0
                    was_speaking = True
                elif was_speaking:
                    # Carry silence frames so the segment has natural trailing audio.
                    speech_buffer.append(frame)
                    buffer_duration += frame_duration
                    silence_duration += frame_duration

                    if (
                        silence_duration >= _SILENCE_DURATION_S
                        or buffer_duration >= _MAX_BUFFER_DURATION_S
                    ):
                        await flush_segment(speech_buffer[:])
                        speech_buffer.clear()
                        buffer_duration = 0.0
                        silence_duration = 0.0
                        was_speaking = False
                elif buffer_duration >= _MAX_BUFFER_DURATION_S:
                    # Safety flush even without trailing silence.
                    await flush_segment(speech_buffer[:])
                    speech_buffer.clear()
                    buffer_duration = 0.0
                    silence_duration = 0.0
                    was_speaking = False

            # Flush any remaining buffered speech at end of stream.
            await flush_segment(speech_buffer[:])

        except asyncio.CancelledError:
            logging.info(f"Transcription for {participant.identity} was cancelled.")
        except Exception as e:
            logging.error(f"Error during transcription for track {track.sid}: {e}")
        finally:
            self.processing_info.pop(participant.identity, None)
