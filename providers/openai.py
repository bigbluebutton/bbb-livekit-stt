import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import stt

from metrics import (
    SESSION_FAILURE_NO_AUDIO_TRACK,
    SESSION_FAILURE_PARTICIPANT_NOT_FOUND,
    SttMetrics,
)
from providers.base import BaseSttAgent, BaseSttConfig

# Energy-based voice activity detection parameters.
# RMS threshold (int16 scale 0–32768): frames below this are considered silence.
_SILENCE_THRESHOLD_RMS = 500
# Seconds of continuous silence after speech before the segment is flushed.
_SILENCE_DURATION_S = 0.8
# Maximum segment duration before a forced flush (prevents unbounded buffering).
_MAX_BUFFER_DURATION_S = 12.0


@dataclass
class OpenAiConfig(BaseSttConfig):
    api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    model: str = field(
        default_factory=lambda: os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe")
    )
    base_url: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", None)
    )


openai_config = OpenAiConfig()


class OpenAiSttAgent(BaseSttAgent):
    provider_name = "openai"

    def __init__(self, config: OpenAiConfig, metrics: SttMetrics | None = None):
        super().__init__(config, metrics)
        self._http_session: aiohttp.ClientSession | None = None

    # --- HTTP session management ---

    def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._http_session

    async def _transcribe_wav(self, wav_bytes: bytes, language: str | None) -> str:
        """Call the OpenAI-compatible REST endpoint directly.

        Constructs the URL as ``{base_url}/v1/audio/transcriptions`` so that
        custom backends (e.g. ``http://my-server/api/``) work correctly
        regardless of how the OpenAI SDK would handle the ``/v1`` path segment.
        """
        base_url = (self.config.base_url or "https://api.openai.com").rstrip("/")
        url = f"{base_url}/v1/audio/transcriptions"

        form = aiohttp.FormData()
        form.add_field(
            "file", wav_bytes, filename="audio.wav", content_type="audio/wav"
        )
        form.add_field("model", self.config.model)
        form.add_field("response_format", "json")
        if language:
            form.add_field("language", language)

        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        session = self._get_http_session()
        async with session.post(url, data=form, headers=headers) as resp:
            resp.raise_for_status()
            result = await resp.json()
        return result.get("text", "").strip()

    # --- BaseSttAgent abstract method implementations ---

    def _create_stt_stream(self, locale: str | None) -> stt.SpeechStream:  # type: ignore[override]
        """Not used: REST mode overrides start_transcription_for_user directly."""
        raise NotImplementedError("OpenAI REST mode does not use STT streams")

    def _update_stream_locale(self, user_id: str, locale: str):
        """Restart the pipeline with the new locale.

        No explicit gauge transition: stop and start already move the session
        gauge between locale labels.
        """
        provider = self.participant_settings.get(user_id, {}).get("provider", "openai")
        self.stop_transcription_for_user(user_id)
        self.start_transcription_for_user(user_id, locale, provider)

    # --- Override start_transcription_for_user to pass language, not a stream ---

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

        language = self._sanitize_locale(locale)
        task = asyncio.create_task(
            self._run_transcription_pipeline(participant, track, language)
        )
        self.processing_info[participant.identity] = {
            "task": task,
            # See BaseSttAgent.start_transcription_for_user for why these are kept
            # separate from participant_settings.
            "track_sid": track.sid,
            "metrics_locale": locale,
        }
        self.metrics.session_started(self.provider_name, locale)
        logging.info(
            f"Started transcription for {participant.identity} with locale {locale}."
        )

    # --- Override _cleanup to close HTTP session ---

    async def _cleanup(self):
        await super()._cleanup()
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    # --- REST-based transcription pipeline ---

    async def _run_transcription_pipeline(  # type: ignore[override]
        self,
        participant: rtc.RemoteParticipant,
        track: rtc.Track,
        language: str | None,
    ):
        """Collect audio, segment by silence, and transcribe via REST API.

        The OpenAI plugin's stream() uses the Realtime WebSocket API which is
        not implemented by all OpenAI-compatible backends. This implementation
        uses energy-based silence detection (RMS threshold) to segment audio
        into utterances, then calls the standard REST /audio/transcriptions
        endpoint for each segment.

        TODO: Support the Realtime WebSocket endpoint as an opt-in mode (e.g.
        via an ``OpenAiConfig`` flag).  When enabled, delegate to the livekit
        openai plugin's stream() directly.  This would unlock lower latency for
        backends that implement the OpenAI Realtime API.
        """
        audio_stream = None
        open_time = time.time()
        self.open_time = open_time

        speech_buffer: list[rtc.AudioFrame] = []
        buffer_duration = 0.0
        silence_duration = 0.0
        was_speaking = False
        speech_start_time = 0.0

        async def flush_segment(frames: list[rtc.AudioFrame], seg_start: float) -> None:
            if not frames:
                return
            try:
                wav_bytes = rtc.combine_audio_frames(frames).to_wav_bytes()
                text = await self._transcribe_wav(wav_bytes, language)
                if text:
                    seg_end = time.time() - open_time
                    event = stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[
                            stt.SpeechData(
                                text=text,
                                language=language,
                                start_time=seg_start,
                                end_time=seg_end,
                            )
                        ],
                    )
                    self.emit(
                        "final_transcript",
                        participant=participant,
                        event=event,
                        open_time=open_time,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logging.error(
                    f"Error transcribing segment for {participant.identity}: {e}"
                )

        try:
            audio_stream = rtc.AudioStream(track)

            async for audio_event in audio_stream:
                frame = audio_event.frame
                samples = np.frombuffer(frame.data, dtype=np.int16)
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                is_speaking = rms > _SILENCE_THRESHOLD_RMS
                frame_duration = frame.samples_per_channel / frame.sample_rate

                if is_speaking:
                    if not was_speaking:
                        speech_start_time = time.time() - open_time
                    speech_buffer.append(frame)
                    buffer_duration += frame_duration
                    silence_duration = 0.0
                    was_speaking = True

                    if buffer_duration >= _MAX_BUFFER_DURATION_S:
                        # Safety flush: prevent unbounded buffer growth during
                        # continuous speech or sustained noise above the RMS threshold.
                        await flush_segment(speech_buffer[:], speech_start_time)
                        speech_buffer.clear()
                        buffer_duration = 0.0
                        speech_start_time = time.time() - open_time
                elif was_speaking:
                    # Carry silence frames so the segment has natural trailing audio.
                    speech_buffer.append(frame)
                    buffer_duration += frame_duration
                    silence_duration += frame_duration

                    if (
                        silence_duration >= _SILENCE_DURATION_S
                        or buffer_duration >= _MAX_BUFFER_DURATION_S
                    ):
                        await flush_segment(speech_buffer[:], speech_start_time)
                        speech_buffer.clear()
                        buffer_duration = 0.0
                        silence_duration = 0.0
                        was_speaking = False

            # Flush any remaining buffered speech at end of stream.
            await flush_segment(speech_buffer[:], speech_start_time)

        except asyncio.CancelledError:
            logging.info(f"Transcription for {participant.identity} was cancelled.")
        except Exception as e:
            logging.error(f"Error during transcription for track {track.sid}: {e}")
        finally:
            self._release_session_slot(participant.identity)
            await self._close_stream(audio_stream)
