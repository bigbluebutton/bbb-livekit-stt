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
from metrics import (
    AGENT_FAILURE_ROOM_CONNECT,
    SESSION_FAILURE_NO_AUDIO_TRACK,
    SESSION_FAILURE_PARTICIPANT_NOT_FOUND,
    SESSION_FAILURE_STREAM_ERROR,
    SttMetrics,
    stt_metrics,
)


@dataclass
class BaseSttConfig:
    interim_results: bool | None = None

    def to_stt_kwargs(self) -> dict:
        """Provider-specific kwargs for the LiveKit STT plugin. Subclasses implement."""
        raise NotImplementedError


class BaseSttAgent(EventEmitter, ABC):
    # Overridden by each provider; used as the `provider` metric label.
    provider_name: str = "unknown"

    def __init__(self, config: BaseSttConfig, metrics: SttMetrics | None = None):
        super().__init__()
        self.config = config
        self.metrics = stt_metrics if metrics is None else metrics
        self.ctx: JobContext | None = None
        self.room: rtc.Room | None = None
        self.processing_info = {}
        self.participant_settings = {}
        self.open_time = time.time()
        self._shutdown = asyncio.Event()

    @property
    def reports_confidence(self) -> bool:
        """Whether the provider populates SpeechData.confidence.

        Providers that do not leave it at its 0.0 default, which must not be
        observed into the confidence histogram.
        """
        return False

    @property
    def translation_enabled(self) -> bool:
        """Whether the provider emits transcripts in a translated language.

        When true, a transcript's language legitimately differs from the one the
        participant requested, so language mismatches must not be counted.
        """
        return False

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
        try:
            await self.ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        except Exception:
            self.metrics.agent_start_failed(
                self.provider_name, AGENT_FAILURE_ROOM_CONNECT
            )
            raise

        self.room = self.ctx.room

        self.room.on("participant_disconnected", self._on_participant_disconnected)
        self.room.on("disconnected", self._on_disconnected)
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)

        self.metrics.agent_started(self.provider_name)

        try:
            await self._shutdown.wait()
        finally:
            await self._cleanup()
            self.metrics.agent_stopped(self.provider_name)

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
            self.metrics.session_start_failed(
                self.provider_name, SESSION_FAILURE_PARTICIPANT_NOT_FOUND
            )
            return

        track = self._find_audio_track(participant)

        if not track:
            logging.warning(
                f"Won't start transcription yet, no audio track found for {user_id}."
            )
            # Usually transient: BBB sends the locale change before the track is
            # published, and _on_track_subscribed starts the session shortly after.
            self.metrics.session_start_failed(
                self.provider_name, SESSION_FAILURE_NO_AUDIO_TRACK
            )
            return

        if participant.identity in self.processing_info:
            logging.debug(
                f"Transcription task already running for {participant.identity}, ignoring start command."
            )
            # A no-op, not a failure: nothing to record.
            return

        sanitized_locale = self._sanitize_locale(locale)

        try:
            stt_stream = self._create_stt_stream(sanitized_locale)
        except Exception:
            self.metrics.session_start_failed(
                self.provider_name, SESSION_FAILURE_STREAM_ERROR
            )
            raise

        task = asyncio.create_task(
            self._run_transcription_pipeline(participant, track, stt_stream)
        )
        self.processing_info[participant.identity] = {
            "stream": stt_stream,
            "task": task,
            # The locale the gauge was incremented under. Kept separate from
            # participant_settings so a later mutation cannot make the decrement
            # land on a different label and strand a phantom session.
            "metrics_locale": locale,
        }
        self.metrics.session_started(self.provider_name, locale)
        logging.info(
            f"Started transcription for {participant.identity} with locale {locale}."
        )

    def handle_speech_locale_change(
        self, user_id: str, locale: str | None, provider: str | None
    ):
        """Apply a BBB speech-locale change to a participant's session.

        Only a complete pair means "transcribe": the html5 client clears both
        fields when a user unassigns their language, while other producers
        (a plugin pausing transcription) clear the locale and leave the provider
        filled, and akka passes either through verbatim. So anything short of
        both means "turn it off"; everything else is a request for transcription
        in `locale` — adjust the session in place when one is running, start one
        otherwise.
        """
        if not (locale and provider):
            self.disable_transcription_for_user(user_id)
            return

        if user_id not in self.processing_info:
            # Nothing to adjust: transcription is off, or its audio track went
            # away. start_transcription_for_user() records the settings either
            # way, so the session starts now or when the track is published.
            self.start_transcription_for_user(user_id, locale, provider)
            return

        if self.participant_settings.get(user_id, {}).get("locale") != locale:
            self.update_locale_for_user(user_id, locale)

    def disable_transcription_for_user(self, user_id: str):
        """Stop transcription and forget the locale BBB assigned to the user.

        Dropping the locale is what separates this from stopping a session:
        the settings are what _on_track_subscribed restarts from, so leaving
        them behind resurrects transcription the user turned off as soon as the
        microphone is republished, and makes re-enabling it look like a no-op.
        Speech options are left alone — they arrive on their own event and are
        not resent when the locale is reassigned.
        """
        self.stop_transcription_for_user(user_id)

        settings = self.participant_settings.get(user_id)
        if settings is not None:
            settings.pop("locale", None)
            settings.pop("provider", None)

    def stop_transcription_for_user(self, user_id: str):
        logging.debug(f"Stopping transcription for {user_id}.")

        if user_id in self.processing_info:
            info = self.processing_info.pop(user_id)
            info["task"].cancel()
            self.metrics.session_stopped(self.provider_name, info["metrics_locale"])
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
                elif event.type == stt.SpeechEventType.RECOGNITION_USAGE:
                    # Metrics only: this event carries the provider's own count of
                    # audio seconds processed and must not reach the transcript
                    # handlers.
                    if event.recognition_usage:
                        self.metrics.provider_audio_observed(
                            self.provider_name,
                            event.recognition_usage.audio_duration,
                        )

        try:
            await asyncio.gather(forward_audio_task(), process_stt_task())
        except asyncio.CancelledError:
            logging.info(f"Transcription for {participant.identity} was cancelled.")
        except Exception as e:
            logging.error(f"Error during transcription for track {track.sid}: {e}")
        finally:
            self.processing_info.pop(participant.identity, None)
