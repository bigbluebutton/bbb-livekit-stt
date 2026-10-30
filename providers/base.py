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
    SESSION_END_AUDIO_ENDED,
    SESSION_END_ERROR,
    SESSION_END_PROVIDER_CLOSED,
    SESSION_END_STOPPED,
    SESSION_FAILURE_NO_AUDIO_TRACK,
    SESSION_FAILURE_PARTICIPANT_NOT_FOUND,
    SESSION_FAILURE_STREAM_ERROR,
    SttMetrics,
    stt_metrics,
)


# How long a job's teardown waits for transcripts still being published.
TRANSCRIPT_DRAIN_TIMEOUT_S = 5.0


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
        self._shutdown = asyncio.Event()
        # Set once start() has finished, or failed; shutdown() waits on it.
        self._stopped = asyncio.Event()
        self._started = False
        # Every pipeline still running, including those the stop path already
        # took out of processing_info, so shutdown can wait for them.
        self._pipelines: set[asyncio.Task] = set()
        # Stream closes started by the stop path, so shutdown can wait for them.
        self._closing: set[asyncio.Future] = set()

    @property
    def reports_confidence(self) -> bool:
        """Whether the provider populates SpeechData.confidence.

        A provider that does not populate it leaves it at its 0.0 default,
        which must not be observed into the confidence histogram.
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

    def _should_emit(self, event: stt.SpeechEvent) -> bool:
        """Filter events before emission. Override for provider-specific filtering."""
        return True

    async def start(self, ctx: JobContext):
        if self._shutdown.is_set():
            # The job was shut down before the agent got here: do not join.
            return

        self._started = True
        try:
            await self._run(ctx)
        finally:
            self._stopped.set()

    async def shutdown(self):
        """Stop the agent and wait until start() has torn it down.

        LiveKit awaits a job's shutdown callbacks but not its entrypoint: the job
        process exits as soon as they return, with start() wherever it got to.
        """
        self._shutdown.set()
        if self._started:
            await self._stopped.wait()

    async def _run(self, ctx: JobContext):
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

        # Wait for every pipeline and stream close to finish. Otherwise the job
        # can exit before the streams are released and before the agent gauge
        # comes back down.
        pending = [*self._pipelines, *self._closing]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        # Handlers publish to Redis, which closes once this returns, and Voxtral
        # emits a final as it unwinds. Bounded so a hung publish cannot hold the job.
        await self.drain(timeout=TRANSCRIPT_DRAIN_TIMEOUT_S)

    def start_transcription_for_user(self, user_id: str, locale: str, provider: str):
        if self._shutdown.is_set():
            # A locale message handled during teardown would start a session that
            # outlives the agent.
            return

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

        task = self._spawn_pipeline(
            self._run_transcription_pipeline(participant, track, stt_stream, locale)
        )
        self.processing_info[participant.identity] = {
            "stream": stt_stream,
            "task": task,
            "track_sid": track.sid,
            # The locale the gauge was incremented under. Kept apart from
            # participant_settings: a later change there must not move the
            # decrement to another label and leave this one stuck above zero.
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

        A locale and a provider together mean "transcribe in `locale`": move a
        running session to it, or start one. Anything less means "turn it off".
        The html5 client sends both incomplete shapes — the transcription toggle
        clears the locale, the Audio Settings select clears both — and akka
        passes them through verbatim.
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

        Unlike stopping a session, this drops the locale: _on_track_subscribed
        restarts from the settings, so a locale left behind resurrects
        transcription as soon as the microphone is republished. Speech options
        stay: they arrive on their own event and are not resent when the locale
        is reassigned or the user reconnects.
        """
        self.stop_transcription_for_user(user_id)

        settings = self.participant_settings.get(user_id)
        if settings is not None:
            settings.pop("locale", None)
            settings.pop("provider", None)

    def stop_transcription_for_user(self, user_id: str):
        logging.debug(f"Stopping transcription for {user_id}.")

        if user_id not in self.processing_info:
            return

        info = self.processing_info.pop(user_id)
        info["task"].cancel()
        self.metrics.session_stopped(self.provider_name, info["metrics_locale"])
        self.metrics.session_ended(self.provider_name, SESSION_END_STOPPED)

        # A pipeline cancelled before its first step never runs its own teardown,
        # and the stream it was handed is already connected to the provider. Close
        # it here, or it stays open — and billed — for the life of the room job.
        if not info.get("started") and info.get("stream") is not None:
            closing = asyncio.ensure_future(self._close_stream(info["stream"]))
            self._closing.add(closing)
            closing.add_done_callback(self._closing.discard)

        logging.info(f"Stopped transcription for user {user_id}.")

    def _publishes_interims(self, user_id: str) -> bool:
        """Whether this participant's interims reach BBB: main.py publishes them
        only to a participant who asked for partial utterances."""
        settings = self.participant_settings.get(user_id, {})
        return bool(self.config.interim_results and settings.get("partial_utterances"))

    def _spawn_pipeline(self, pipeline) -> asyncio.Task:
        task = asyncio.create_task(pipeline)
        self._pipelines.add(task)
        task.add_done_callback(self._pipelines.discard)
        return task

    def _release_session_slot(self, user_id: str, end_reason: str):
        """Close out a finished pipeline's own session.

        Called from the pipeline task itself. A session that replaced it while
        it was being cancelled owns the slot by then. Closing that one would
        strand a running pipeline the agent can no longer stop or account for.

        Only a pipeline that ended by itself (the provider dropped the
        connection, or the audio ran out) still owns its slot here, so this is
        where its gauge is returned. The stop path removes the entry first and
        returns the gauge itself.
        """
        info = self.processing_info.get(user_id)
        if info is None or info["task"] is not asyncio.current_task():
            return

        del self.processing_info[user_id]
        self.metrics.session_stopped(self.provider_name, info["metrics_locale"])
        self.metrics.session_ended(self.provider_name, end_reason)

    def update_locale_for_user(self, user_id: str, locale: str):
        """Move a participant to a new locale by restarting their session.

        Providers restart their clock when the language changes: Gladia opens a
        fresh live session behind `update_options()`. An in-place update would
        stamp transcripts against the old stream, dating them back by the
        elapsed session and re-issuing transcript ids BBB already stored.
        """
        settings = self.participant_settings.get(user_id)

        if user_id not in self.processing_info:
            logging.warning(
                f"Won't update locale, no active transcription for user {user_id}."
            )
            # Still worth recording: it is what the next track subscription
            # starts transcription from.
            if settings is not None:
                settings["locale"] = locale
            return

        provider = (settings or {}).get("provider", self.provider_name)
        logging.info(f"Updating locale to '{locale}' for user {user_id}.")
        self.stop_transcription_for_user(user_id)
        self.start_transcription_for_user(user_id, locale, provider)

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
        info = self.processing_info.get(participant.identity)

        # AUDIO_ONLY subscribes screenshare audio too. Ending the session on any
        # other track would stop a live microphone's transcription, and nothing
        # would restart it.
        if info is None or info["track_sid"] != track.sid:
            return

        self.stop_transcription_for_user(participant.identity)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant, *_):
        logging.debug(
            f"Participant {participant.identity} disconnected, stopping transcription."
        )
        self.stop_transcription_for_user(participant.identity)
        # Settings stay until BBB removes the user: BBB announces a speech locale only
        # when it is set, not when the user reconnects.

    def _on_disconnected(self):
        self._shutdown.set()

    def _find_participant(self, identity: str) -> rtc.RemoteParticipant | None:
        for p in self.room.remote_participants.values():
            if p.identity == identity:
                return p
        return None

    def _find_audio_track(self, participant: rtc.RemoteParticipant) -> rtc.Track | None:
        # Skip screenshare audio: only the microphone carries the participant's
        # speech. _on_track_subscribed filters on the same source.
        for pub in participant.track_publications.values():
            if (
                pub.track
                and pub.track.kind == rtc.TrackKind.KIND_AUDIO
                and pub.source == rtc.TrackSource.SOURCE_MICROPHONE
            ):
                return pub.track
        return None

    async def _close_stream(self, stream) -> None:
        """Release a session's stream.

        Ending the task is not enough: the provider stream owns a websocket and
        the audio stream a native handle, and neither goes away with the
        coroutine. Nothing can be done about a close that fails — the session is
        over either way — so it must not take the rest of the teardown with it.
        """
        if stream is None:
            return

        try:
            await stream.aclose()
        except Exception as e:
            logging.debug(f"Error closing {type(stream).__name__}: {e}")

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
        locale: str,
    ):
        audio_stream = None
        # From here on this pipeline closes its own streams; see
        # stop_transcription_for_user().
        info = self.processing_info.get(participant.identity)
        if info is not None and info["task"] is asyncio.current_task():
            info["started"] = True

        # The provider reports offsets within this stream, so its open time turns
        # them into wall clock. It is per session: a shared one would be
        # overwritten by the next session to start.
        open_time = time.time()

        input_rejected = False

        async def forward_audio_task():
            nonlocal input_rejected
            try:
                async for audio_event in audio_stream:
                    try:
                        stt_stream.push_frame(audio_event.frame)
                    except RuntimeError:
                        # The provider closed its side a few turns before the
                        # pipeline sees it: the stream rejects further audio.
                        input_rejected = True
                        return
            finally:
                if process is not None and process.done():
                    # The provider ended before the audio did.
                    input_rejected = True
                elif not input_rejected:
                    # Ends the input, not just flushes it: a recognizer never
                    # told the audio is over keeps the pipeline waiting.
                    try:
                        stt_stream.end_input()
                    except RuntimeError:
                        # The same race as push_frame() above.
                        input_rejected = True

        async def process_stt_task():
            async for event in stt_stream:
                is_interim = event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT
                if is_interim and not self._publishes_interims(participant.identity):
                    # Dropped before the confidence filter: an interim the
                    # agent never publishes is not a discard.
                    continue
                if not self._should_emit(event):
                    continue
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    self.emit(
                        "final_transcript",
                        participant=participant,
                        event=event,
                        open_time=open_time,
                        locale=locale,
                    )
                elif is_interim:
                    self.emit(
                        "interim_transcript",
                        participant=participant,
                        event=event,
                        open_time=open_time,
                        locale=locale,
                    )
                elif event.type == stt.SpeechEventType.RECOGNITION_USAGE:
                    # Metrics only: it must not reach the transcript handlers.
                    if event.recognition_usage:
                        self.metrics.provider_audio_observed(
                            self.provider_name,
                            event.recognition_usage.audio_duration,
                        )

        forward: asyncio.Task | None = None
        process: asyncio.Task | None = None
        end_reason = SESSION_END_ERROR

        try:
            audio_stream = rtc.AudioStream(track)
            forward = asyncio.create_task(forward_audio_task())
            process = asyncio.create_task(process_stt_task())

            done, _ = await asyncio.wait(
                {forward, process}, return_when=asyncio.FIRST_COMPLETED
            )

            # A provider can close its side on its own, e.g. on an idle timeout
            # while the participant is muted. Then stop forwarding: that leg can
            # wait on a silent microphone forever. When the audio ends first, keep
            # reading: the provider still owes the last transcripts.
            # Left open: end_input() can succeed in the turn before livekit
            # closes the stream, which counts as audio_ended.
            provider_ended_first = forward not in done or input_rejected
            if provider_ended_first:
                forward.cancel()

            # The provider leg first: when both fail, its error is the cause.
            for result in await asyncio.gather(
                process, forward, return_exceptions=True
            ):
                if isinstance(result, BaseException) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    raise result
            end_reason = (
                SESSION_END_PROVIDER_CLOSED
                if provider_ended_first
                else SESSION_END_AUDIO_ENDED
            )
        except asyncio.CancelledError:
            logging.info(f"Transcription for {participant.identity} was cancelled.")
        except Exception as e:
            logging.error(f"Error during transcription for track {track.sid}: {e}")
        finally:
            # Before the first await: a stop that arrives while the legs unwind
            # would otherwise cancel this teardown and leave both streams open.
            self._release_session_slot(participant.identity, end_reason)

            legs = [leg for leg in (forward, process) if leg is not None]
            for leg in legs:
                leg.cancel()
            if legs:
                await asyncio.gather(*legs, return_exceptions=True)

            await self._close_stream(audio_stream)
            await self._close_stream(stt_stream)
