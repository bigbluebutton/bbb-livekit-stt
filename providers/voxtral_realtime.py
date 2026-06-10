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
from collections import deque
from dataclasses import dataclass, field

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import stt
from livekit.agents import vad as agents_vad
from livekit.plugins import silero

from providers.base import BaseSttAgent, BaseSttConfig

_MAX_BUFFER_DURATION_S = float(os.getenv("VOXTRAL_MAX_BUFFER_DURATION_S", "8.0"))
_TARGET_SAMPLE_RATE = int(os.getenv("VOXTRAL_TARGET_SAMPLE_RATE", "16000"))
# Silero VAD parameters — replace the old RMS threshold and silence duration
_VAD_MIN_SILENCE_S = float(os.getenv("VOXTRAL_VAD_MIN_SILENCE_S", "0.6"))
_VAD_ACTIVATION_THRESHOLD = float(os.getenv("VOXTRAL_VAD_ACTIVATION_THRESHOLD", "0.5"))
# Rolling pre-roll kept while idle so the word onset preceding Silero's
# START_OF_SPEECH (its prefix padding) is sent once a streaming request opens.
_VAD_PREROLL_S = float(os.getenv("VOXTRAL_VAD_PREROLL_S", "0.5"))


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
    def __init__(
        self,
        config: VoxtralRealtimeConfig,
        vad: agents_vad.VAD | None = None,
    ):
        super().__init__(config)
        self._http_session: aiohttp.ClientSession | None = None
        self._vad: agents_vad.VAD = vad or silero.VAD.load(
            min_silence_duration=_VAD_MIN_SILENCE_S,
            activation_threshold=_VAD_ACTIVATION_THRESHOLD,
        )

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

        # Per-segment start times, pushed by _writer._open() and popped by
        # _reader at each segment's first delta. The server processes segments
        # strictly in order on one connection, so FIFO pairing is exact. This
        # gives every segment a distinct start_time — and therefore a distinct
        # BBB transcriptId — including max-buffer splits of one long utterance,
        # where Silero fires no new START_OF_SPEECH (a shared "speech start"
        # variable would collide and make segment 2 overwrite segment 1).
        segment_starts: deque[float] = deque()

        # ── Reader ────────────────────────────────────────────────────────────

        async def _reader() -> None:
            """Read transcription events for all utterances on this session.

            Runs concurrently with _writer so that transcription.delta events
            emitted by the server while audio is still streaming are consumed
            in real time rather than buffered and replayed after each commit.
            """
            text = ""
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
                        # Pair this segment with the start time its opener pushed.
                        utterance_start = (
                            segment_starts.popleft()
                            if segment_starts
                            else time.time() - open_time
                        )
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

                    if utterance_start is None:
                        # Zero-delta segment: consume its queued start anyway so
                        # later segments stay paired with their own openers.
                        utterance_start = (
                            segment_starts.popleft()
                            if segment_starts
                            else time.time() - open_time
                        )

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
                                        start_time=utterance_start,
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

        # ── VAD task + Writer (Silero VAD, server-side streaming) ────────────
        #
        # Protocol (verified empirically against vLLM Voxtral, see
        # notes/progressive-transcription-investigation.md):
        #   1. On speech start (Silero START_OF_SPEECH): send an OPENING commit
        #      to start a streaming request. Without this the server only buffers
        #      and batch-transcribes on the closing commit (no live deltas).
        #   2. While a request is open, audio is appended in real time. The
        #      server streams transcription.delta events back as audio arrives
        #      (first delta ~0.6 s after speech start), which _reader emits as
        #      INTERIM. While idle, only a bounded pre-roll is kept locally.
        #   3. On speech end (Silero END_OF_SPEECH or max-buffer): send closing
        #      commit + commit(final). The server emits transcription.done, which
        #      _reader emits as FINAL.
        # Each segment is its own streaming request with its own entry in
        # segment_starts, so consecutive segments — including max-buffer splits
        # of one long utterance — never collide on BBB's second-granularity
        # transcriptId.
        #
        # Silero VAD only controls WHEN commits are sent; audio filtering is not
        # needed and was the root cause of our earlier failed attempt.

        # Shared between _vad_task and _writer (asyncio single-threaded, no locks)
        is_in_speech = False
        commit_event = asyncio.Event()  # set by _vad_task on END_OF_SPEECH

        vad_stream = self._vad.stream()

        async def _vad_task() -> None:
            nonlocal is_in_speech
            try:
                async for ev in vad_stream:
                    if ev.type == agents_vad.VADEventType.START_OF_SPEECH:
                        is_in_speech = True
                        logging.debug(
                            f"Voxtral: speech start for {participant.identity}"
                        )
                    elif ev.type == agents_vad.VADEventType.END_OF_SPEECH:
                        is_in_speech = False
                        commit_event.set()
                        logging.debug(
                            f"Voxtral: speech end for {participant.identity}"
                        )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # A dead VAD task would silently freeze is_in_speech and stop all
                # commits; surface it rather than degrade to max-buffer-only.
                logging.error(
                    f"Voxtral: VAD task failed for {participant.identity}: {e}",
                    exc_info=True,
                )

        async def _writer() -> None:
            # `stream_open` is owned exclusively by _writer; `is_in_speech` is
            # owned exclusively by _vad_task. Keeping them separate is what lets a
            # max-buffer split mid-utterance reopen immediately on the next frame
            # (we never clobber the VAD's view of whether speech is ongoing).
            preroll_max = int(_VAD_PREROLL_S * _TARGET_SAMPLE_RATE) * 2  # int16 bytes
            preroll = bytearray()  # recent audio captured while no request is open
            pending = b""  # audio buffered for chunked append while open
            open_secs = 0.0  # duration of the current open segment
            stream_open = False

            async def _append(data: bytes) -> None:
                await ws.send_json(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(data).decode(),
                    }
                )

            async def _flush_pending() -> None:
                nonlocal pending
                while len(pending) >= chunk_size:
                    await _append(pending[:chunk_size])
                    pending = pending[chunk_size:]

            async def _open() -> None:
                """Open a streaming request and replay the captured lead-in."""
                nonlocal stream_open, pending, open_secs
                commit_event.clear()  # discard any stale END from a prior segment
                # Record this segment's start (backdated by the replayed pre-roll)
                # for the reader to pair with the segment's transcription events.
                preroll_secs = len(preroll) / (_TARGET_SAMPLE_RATE * 2)
                segment_starts.append(time.time() - open_time - preroll_secs)
                await ws.send_json({"type": "input_audio_buffer.commit"})
                stream_open = True
                open_secs = 0.0
                if preroll:
                    pending += bytes(preroll)
                    preroll.clear()
                    await _flush_pending()

            async def _close() -> None:
                """Flush remaining audio and close the utterance's request."""
                nonlocal stream_open, pending
                if pending:
                    await _append(pending)
                    pending = b""
                await ws.send_json({"type": "input_audio_buffer.commit"})
                await ws.send_json({"type": "input_audio_buffer.commit", "final": True})
                stream_open = False
                preroll.clear()  # committed; pre-roll only seeds the next onset

            async for audio_event in audio_stream:
                frame = audio_event.frame
                frame_duration = frame.samples_per_channel / frame.sample_rate

                # Feed the VAD, then yield once so _vad_task is scheduled. In
                # production the network awaits below also yield; this keeps the
                # interleave deterministic and lets tests drive a sync frame
                # iterator without starving the VAD task.
                vad_stream.push_frame(frame)
                await asyncio.sleep(0)

                resampled = _to_pcm16_16k(frame)

                # Open a request as soon as the VAD reports speech.
                if is_in_speech and not stream_open:
                    await _open()

                if stream_open:
                    pending += resampled
                    open_secs += frame_duration
                    await _flush_pending()
                else:
                    # Idle: keep only a bounded rolling pre-roll; send nothing.
                    preroll += resampled
                    if len(preroll) > preroll_max:
                        del preroll[:-preroll_max]

                # Close on end-of-speech (Silero) or the max-buffer safety cap.
                # is_in_speech is deliberately left untouched: if the speaker is
                # still talking past the cap, the next frame reopens immediately.
                if stream_open and (
                    commit_event.is_set() or open_secs >= _MAX_BUFFER_DURATION_S
                ):
                    commit_event.clear()
                    await _close()

            # End of stream: flush any open request.
            if stream_open:
                await _close()
            vad_stream.end_input()

        # ── Run all three tasks concurrently ──────────────────────────────────

        reader_task = asyncio.create_task(_reader())
        vad_task = asyncio.create_task(_vad_task())
        try:
            await _writer()
        finally:
            reader_task.cancel()
            vad_task.cancel()
            await asyncio.gather(reader_task, vad_task, return_exceptions=True)
            await vad_stream.aclose()


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
