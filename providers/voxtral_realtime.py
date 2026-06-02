"""STT provider for vLLM's Voxtral Realtime WebSocket API.

vLLM's protocol differs from the OpenAI Realtime Transcription API in three ways:
- session.update: model is at the top level, not nested inside session.audio
- No server-side VAD: client must send input_audio_buffer.commit to trigger generation
- Response events: transcription.delta / transcription.done (not conversation.item.*)

Audio must be PCM16, 16 kHz, mono, base64-encoded.
"""

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import stt

from providers.base import BaseSttAgent, BaseSttConfig

_SILENCE_THRESHOLD_RMS = float(os.getenv("VOXTRAL_SILENCE_THRESHOLD_RMS", "500"))
_SILENCE_DURATION_S = float(os.getenv("VOXTRAL_SILENCE_DURATION_S", "0.6"))
_MAX_BUFFER_DURATION_S = float(os.getenv("VOXTRAL_MAX_BUFFER_DURATION_S", "8.0"))
_TARGET_SAMPLE_RATE = int(os.getenv("VOXTRAL_TARGET_SAMPLE_RATE", "16000"))


@dataclass
class VoxtralRealtimeConfig(BaseSttConfig):
    api_key: str | None = field(default_factory=lambda: os.getenv("VOXTRAL_API_KEY"))
    model: str = field(
        default_factory=lambda: os.getenv(
            "VOXTRAL_MODEL", "mistralai/Voxtral-Mini-4B-Realtime-2602"
        )
    )
    base_url: str | None = field(
        default_factory=lambda: os.getenv("VOXTRAL_BASE_URL", None)
    )
    interim_results: bool = field(
        default_factory=lambda: (
            os.getenv("VOXTRAL_INTERIM_RESULTS", "true").lower() != "false"
        )
    )


voxtral_realtime_config = VoxtralRealtimeConfig()


class VoxtralRealtimeSttAgent(BaseSttAgent):
    def __init__(self, config: VoxtralRealtimeConfig):
        super().__init__(config)
        self._http_session: aiohttp.ClientSession | None = None

    def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=15)
            )
        return self._http_session

    def _build_ws_url(self) -> str:
        base = (self.config.base_url or "https://api.openai.com/v1").rstrip("/")
        base = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return f"{base}/realtime?intent=transcription"

    def _create_stt_stream(self, locale: str) -> stt.SpeechStream:
        raise NotImplementedError("VoxtralRealtime uses a custom pipeline")

    def _update_stream_locale(self, user_id: str, locale: str):
        provider = self.participant_settings.get(user_id, {}).get(
            "provider", "voxtral-realtime"
        )
        self.stop_transcription_for_user(user_id)
        self.start_transcription_for_user(user_id, locale, provider)

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
                f"Transcription already running for {participant.identity}, ignoring."
            )
            return

        language = self._sanitize_locale(locale)
        task = asyncio.create_task(
            self._run_transcription_pipeline(participant, track, language)
        )
        self.processing_info[participant.identity] = {"task": task}
        logging.info(
            f"Started Voxtral Realtime transcription for {participant.identity} ({locale})."
        )

    async def _cleanup(self):
        await super()._cleanup()
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    async def _run_transcription_pipeline(
        self,
        participant: rtc.RemoteParticipant,
        track: rtc.Track,
        language: str,
    ):
        ws_url = self._build_ws_url()
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        open_time = time.time()
        retry_delay = 1.0

        try:
            while True:
                audio_stream = rtc.AudioStream(track)
                try:
                    async with self._get_http_session().ws_connect(
                        ws_url, headers=headers
                    ) as ws:
                        msg = await asyncio.wait_for(ws.receive(), timeout=10.0)
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            logging.error(
                                "Voxtral WS: expected text for session.created"
                            )
                            return
                        data = json.loads(msg.data)
                        if data.get("type") != "session.created":
                            logging.error(
                                f"Voxtral WS: unexpected first message: {data}"
                            )
                            return
                        logging.info(
                            f"Voxtral WS session created for {participant.identity}"
                        )
                        # Connection is healthy again; reset reconnect backoff.
                        retry_delay = 1.0

                        # vLLM expects model at top level of session.update
                        await ws.send_json(
                            {"type": "session.update", "model": self.config.model}
                        )

                        await self._vad_loop(
                            ws, audio_stream, participant, language, open_time
                        )
                        return  # clean exit — audio stream finished normally

                except asyncio.CancelledError:
                    raise
                except aiohttp.ClientError as e:
                    logging.warning(
                        f"Voxtral WS connection lost for {participant.identity} "
                        f"({type(e).__name__}: {e}), reconnecting in {retry_delay:.0f}s"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30.0)
                except Exception as e:
                    logging.error(
                        f"Voxtral Realtime error for {participant.identity}: {e}",
                        exc_info=True,
                    )
                    return
                finally:
                    await audio_stream.aclose()

        except asyncio.CancelledError:
            logging.info(
                f"Voxtral Realtime transcription for {participant.identity} cancelled."
            )
        finally:
            self.processing_info.pop(participant.identity, None)

    async def _vad_loop(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        audio_stream: rtc.AudioStream,
        participant: rtc.RemoteParticipant,
        language: str,
        open_time: float,
    ):
        chunk_size = (
            _TARGET_SAMPLE_RATE // 20 * 2
        )  # 50 ms of int16 (matches official plugin)

        # Shared between _writer and _reader.  asyncio is single-threaded so no
        # lock is needed; the writer sets this synchronously before any await.
        speech_start_time = 0.0  # set by writer when speech starts

        # ── Reader ────────────────────────────────────────────────────────────

        async def _reader() -> None:
            """Read transcription events for all utterances on this session.

            Runs concurrently with _writer so that transcription.delta events
            emitted by the server while audio is still streaming are consumed
            in real time rather than buffered and replayed after each commit.
            """
            text = ""
            # Snapshot speech_start_time at the first delta of each utterance.
            # The writer may update speech_start_time for the next utterance
            # before this utterance's transcription.done arrives; snapshotting
            # here keeps all events for this utterance on the same transcriptId.
            utterance_start: float | None = None

            while True:
                try:
                    msg = await ws.receive()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logging.error(
                        f"Voxtral: reader error for {participant.identity}: {e}"
                    )
                    break

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    logging.debug(f"Voxtral: WS closed for {participant.identity}")
                    break

                if msg.type == aiohttp.WSMsgType.ERROR:
                    logging.warning(
                        f"Voxtral: WS error frame for {participant.identity}: "
                        f"{ws.exception()}"
                    )
                    break

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = json.loads(msg.data)
                except (ValueError, TypeError) as e:
                    logging.warning(
                        f"Voxtral: ignoring malformed WS message for "
                        f"{participant.identity}: {e}"
                    )
                    continue
                msg_type = data.get("type")

                if msg_type == "transcription.delta":
                    if utterance_start is None:
                        utterance_start = speech_start_time
                        logging.debug(
                            f"Voxtral: first delta for {participant.identity} "
                            f"at t={time.time() - open_time:.3f}s "
                            f"(utterance_start={utterance_start:.3f}s)"
                        )
                    text += data.get("delta", "")
                    if text and self.config.interim_results:
                        self.emit(
                            "interim_transcript",
                            participant=participant,
                            event=stt.SpeechEvent(
                                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                alternatives=[
                                    stt.SpeechData(
                                        text=text,
                                        language=language,
                                        start_time=utterance_start,
                                        end_time=time.time() - open_time,
                                    )
                                ],
                            ),
                            open_time=open_time,
                        )

                elif msg_type == "transcription.done":
                    # Prefer accumulated delta text over done.text: the realtime
                    # API sends content via deltas; done.text may be empty or absent.
                    server_text = data.get("text", "").strip()
                    logging.debug(
                        f"Voxtral: transcription.done for {participant.identity} "
                        f"at t={time.time() - open_time:.3f}s — "
                        f"delta_text='{text[:60]}', server_text='{server_text[:60]}'"
                    )
                    if server_text:
                        text = server_text

                    if text:
                        self.emit(
                            "final_transcript",
                            participant=participant,
                            event=stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                alternatives=[
                                    stt.SpeechData(
                                        text=text,
                                        language=language,
                                        start_time=utterance_start or speech_start_time,
                                        end_time=time.time() - open_time,
                                    )
                                ],
                            ),
                            open_time=open_time,
                        )

                    # Reset for next utterance
                    text = ""
                    utterance_start = None

                elif msg_type == "error":
                    logging.error(f"Voxtral WS error event: {data}")
                    break

        # ── Writer (RMS speech detection, server-side streaming) ─────────────
        #
        # Protocol (verified empirically against vLLM Voxtral, see
        # notes/progressive-transcription-investigation.md):
        #   1. On speech start: send an OPENING commit to start a streaming
        #      transcription request. Without this the server only buffers and
        #      batch-transcribes on the closing commit (no live deltas).
        #   2. During speech: append audio in real time. The server streams
        #      transcription.delta events back as audio arrives (first delta
        #      ~0.6s after speech start), which _reader emits as INTERIM.
        #   3. On silence (or max-buffer): send closing commit + commit(final)
        #      to end the request. The server emits transcription.done, which
        #      _reader emits as FINAL.
        # Each utterance is its own streaming request with its own
        # speech_start_time, so consecutive utterances never collide on BBB's
        # second-granularity transcriptId.

        async def _writer() -> None:
            nonlocal speech_start_time
            send_buffer_bytes = b""
            buffer_duration = 0.0
            silence_duration = 0.0
            was_speaking = False

            async def _append(data: bytes) -> None:
                await ws.send_json(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(data).decode(),
                    }
                )

            async def _open_stream() -> None:
                """Send the opening commit that starts a streaming request."""
                await ws.send_json({"type": "input_audio_buffer.commit"})

            async def _close_stream(tail_bytes: bytes) -> None:
                """Flush remaining audio and close the utterance's request."""
                if tail_bytes:
                    await _append(tail_bytes)
                await ws.send_json({"type": "input_audio_buffer.commit"})
                await ws.send_json({"type": "input_audio_buffer.commit", "final": True})

            async for audio_event in audio_stream:
                frame = audio_event.frame
                samples = np.frombuffer(frame.data, dtype=np.int16)
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                is_speaking = rms > _SILENCE_THRESHOLD_RMS
                frame_duration = frame.samples_per_channel / frame.sample_rate

                if is_speaking:
                    if not was_speaking:
                        # New utterance: open a streaming request before any audio.
                        speech_start_time = time.time() - open_time
                        await _open_stream()
                    was_speaking = True
                    silence_duration = 0.0

                    send_buffer_bytes += _to_pcm16_16k(frame)
                    buffer_duration += frame_duration

                    while len(send_buffer_bytes) >= chunk_size:
                        await _append(send_buffer_bytes[:chunk_size])
                        send_buffer_bytes = send_buffer_bytes[chunk_size:]

                    if buffer_duration >= _MAX_BUFFER_DURATION_S:
                        # Safety cap: close this utterance; the next speech frame
                        # reopens a fresh request with a new speech_start_time.
                        await _close_stream(send_buffer_bytes)
                        send_buffer_bytes = b""
                        buffer_duration = 0.0
                        silence_duration = 0.0
                        was_speaking = False

                elif was_speaking:
                    # Trailing audio after speech, before silence threshold met.
                    send_buffer_bytes += _to_pcm16_16k(frame)
                    buffer_duration += frame_duration
                    silence_duration += frame_duration

                    while len(send_buffer_bytes) >= chunk_size:
                        await _append(send_buffer_bytes[:chunk_size])
                        send_buffer_bytes = send_buffer_bytes[chunk_size:]

                    if silence_duration >= _SILENCE_DURATION_S:
                        await _close_stream(send_buffer_bytes)
                        send_buffer_bytes = b""
                        buffer_duration = 0.0
                        silence_duration = 0.0
                        was_speaking = False

            if was_speaking:
                await _close_stream(send_buffer_bytes)

        # ── Run reader and writer concurrently ────────────────────────────────

        reader_task = asyncio.create_task(_reader())
        try:
            await _writer()
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logging.error(
                    f"Voxtral: reader task crashed for {participant.identity}: {e}",
                    exc_info=True,
                )


def _to_pcm16_16k(frame: rtc.AudioFrame) -> bytes:
    """Resample an AudioFrame to 16 kHz mono PCM16."""
    samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)

    if frame.num_channels > 1:
        samples = samples.reshape(-1, frame.num_channels).mean(axis=1)

    if frame.sample_rate != _TARGET_SAMPLE_RATE:
        n_orig = len(samples)
        n_target = int(round(n_orig * _TARGET_SAMPLE_RATE / frame.sample_rate))
        samples = np.interp(
            np.linspace(0, n_orig - 1, n_target),
            np.arange(n_orig),
            samples,
        )

    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()
