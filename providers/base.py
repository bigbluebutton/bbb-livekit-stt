import asyncio
import logging
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    stt,
)

from events import EventEmitter


@dataclass
class BaseSttConfig:
    interim_results: bool | None = None

    def to_stt_kwargs(self) -> dict:
        """Provider-specific kwargs for the LiveKit STT plugin. Subclasses implement."""
        raise NotImplementedError


class BaseSttAgent(EventEmitter, ABC):
    def __init__(self, config: BaseSttConfig):
        super().__init__()
        self.config = config
        self.ctx: JobContext | None = None
        self.room: rtc.Room | None = None
        self.processing_info = {}
        self.participant_settings = {}
        self.open_time = time.time()
        self._shutdown = asyncio.Event()

    @property
    def translation_lang_map(self) -> Dict[str, str]:
        """Map provider language codes to BBB locales. Override for translation support."""
        return {}

    @abstractmethod
    def _create_stt_stream(self, locale: str | None) -> stt.SpeechStream:
        """Create an STT stream for the given locale. Provider subclasses implement.

        A None locale means the BBB user asked for auto-detection: providers
        must omit the language so the backend detects it server-side.
        """
        ...

    @abstractmethod
    def _update_stream_locale(self, user_id: str, locale: str):
        """Apply a locale change to an active stream. Provider subclasses implement."""
        ...

    def _should_emit(self, event: stt.SpeechEvent) -> bool:
        """Filter events before emission. Override for provider-specific filtering."""
        return True

    async def start(self, ctx: JobContext):
        self.ctx = ctx
        # TODO: disable auto_subscribe. Should be on demand based on the participant's
        # transcription settings
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

        sanitized_locale = self._sanitize_locale(locale)
        stt_stream = self._create_stt_stream(sanitized_locale)
        task = asyncio.create_task(
            self._run_transcription_pipeline(participant, track, stt_stream)
        )
        self.processing_info[participant.identity] = {
            "stream": stt_stream,
            "task": task,
        }
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
            self._update_stream_locale(user_id, locale)
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

    def _sanitize_locale(self, locale: str) -> str | None:
        # STT providers typically accept ISO 639-1 locales (e.g. "en")
        # BBB uses <ISO 639-1>-<ISO 3166-1> format (e.g. "en-US")
        # Sanitization here is to ensure we use the provider's format.
        # "auto" is not a valid ISO language code — returning None lets the
        # provider fall back to server-side auto-detection.
        sanitized = locale.split("-")[0].lower()
        if sanitized == "auto":
            return None

        return sanitized

    async def _run_transcription_pipeline(
        self,
        participant: rtc.RemoteParticipant,
        track: rtc.Track,
        stt_stream: stt.SpeechStream,
    ):
        audio_stream = rtc.AudioStream(track)
        self.open_time = time.time()

        async def forward_audio_task():
            try:
                async for audio_event in audio_stream:
                    stt_stream.push_frame(audio_event.frame)
            finally:
                stt_stream.flush()

        async def process_stt_task():
            async for event in stt_stream:
                if not self._should_emit(event):
                    continue
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    self.emit(
                        "final_transcript",
                        participant=participant,
                        event=event,
                        open_time=self.open_time,
                    )
                elif (
                    event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT
                    and self.config.interim_results
                ):
                    self.emit(
                        "interim_transcript",
                        participant=participant,
                        event=event,
                        open_time=self.open_time,
                    )

        try:
            await asyncio.gather(forward_audio_task(), process_stt_task())
        except asyncio.CancelledError:
            logging.info(f"Transcription for {participant.identity} was cancelled.")
        except Exception as e:
            logging.error(f"Error during transcription for track {track.sid}: {e}")
        finally:
            self.processing_info.pop(participant.identity, None)
