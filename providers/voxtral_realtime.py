"""STT provider for vLLM's Voxtral Realtime WebSocket API.

vLLM's protocol differs from the OpenAI Realtime Transcription API in four ways:
- session.update: model is at the top level, not nested inside session.audio.
  It is also the ONLY field vLLM reads — its handler does event.get("model")
  and ignores the rest of the payload
  (vllm/entrypoints/speech_to_text/realtime/connection.py, handle_event).
- No server-side VAD: the client segments speech itself — a bare
  input_audio_buffer.commit opens a streaming request, commit(final: true)
  closes it (verified in notes/progressive-transcription-investigation.md)
- Response events: transcription.delta / transcription.done (not conversation.item.*)
- No language, in either direction. There is no field to request one and none
  is reported back; see _vad_loop's `language` parameter for why.

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

# Safety cap on a single streaming request's length. Only hit during pure
# monologue (no VAD pause). It serves two purposes: BBB only commits a caption
# on transcription.done, so the cap bounds FINAL latency and caption size when
# a speaker never pauses; and vLLM resets its position counter per commit
# cycle (not per connection), so the cap also bounds per-request context —
# exceeding --max-model-len crashes the whole engine. At 80 ms/token, 30 s is
# ~375 tokens, far from any realistic limit; splits replay _SPLIT_OVERLAP_S so
# lowering the cap mainly trades boundary-word duplication for faster FINALs.
_MAX_BUFFER_DURATION_S = float(os.getenv("VOXTRAL_MAX_BUFFER_DURATION_S", "30.0"))
# The model requires 16 kHz mono PCM16; deliberately not configurable.
_TARGET_SAMPLE_RATE = 16000
# Silero VAD parameters — replace the old RMS threshold and silence duration
_VAD_MIN_SILENCE_S = float(os.getenv("VOXTRAL_VAD_MIN_SILENCE_S", "0.6"))
_VAD_ACTIVATION_THRESHOLD = float(os.getenv("VOXTRAL_VAD_ACTIVATION_THRESHOLD", "0.5"))
# Rolling pre-roll kept while idle so the word onset preceding Silero's
# START_OF_SPEECH (its prefix padding) is sent once a streaming request opens.
_VAD_PREROLL_S = float(os.getenv("VOXTRAL_VAD_PREROLL_S", "0.5"))
# Audio replayed when a max-buffer split reopens mid-speech. The reopened
# request starts mid-utterance with no context; the Voxtral paper recommends
# ~1.28 s of lead-in (16 frames, "similar to attention sinks"), so short
# overlap loses or garbles the words right after a split. The overlap is
# transcribed twice — some duplication at split boundaries is the trade-off.
_SPLIT_OVERLAP_S = float(os.getenv("VOXTRAL_SPLIT_OVERLAP_S", "1.5"))
# Reconnect backoff bounds for a dropped WebSocket connection.
_RETRY_DELAY_INITIAL_S = 1.0
_RETRY_DELAY_MAX_S = 30.0
# WebSocket ping interval. A half-open connection (server killed without a
# FIN, idle NAT/load-balancer drop) leaves the reader blocked in receive()
# indefinitely, so aiohttp has to detect the loss for us.
_WS_HEARTBEAT_S = 20.0
# How long to keep retrying a handshake that connects but answers with
# something other than session.created. The budget has to outlast vLLM's
# 2–5 min of CUDA-graph warmup, during which the server may reject or error on
# the upgrade; past it, a handshake that still fails is a misconfiguration —
# wrong URL, or a server that does not speak vLLM's realtime protocol — and no
# amount of further retrying will fix it.
_HANDSHAKE_GIVE_UP_S = 600.0
# After the audio stream ends, wait up to this long for the server's
# transcription.done of the final segment before tearing the reader down;
# cancelling it immediately would drop the tail utterance's FINAL.
_FINAL_DRAIN_TIMEOUT_S = 3.0
# vLLM's realtime handler runs one generation per connection and SILENTLY
# drops any commit that arrives while the previous segment's generation is
# still running ("Generation already in progress, ignoring commit"). A dropped
# opener means the segment never streams (no interims, late batched FINAL); a
# dropped closer loses the segment's transcription.done entirely and desyncs
# the FIFO pairing. Openers are therefore gated until the previous segment's
# done has been read, buffering audio locally meanwhile. The timeout bounds
# the wait when a done never arrives (already-desynced session): give up,
# resync the counter, and open ungated as before.
_OPEN_GATE_TIMEOUT_S = float(os.getenv("VOXTRAL_OPEN_GATE_TIMEOUT_S", "10.0"))


# Key under which the prewarmed VAD is stashed in the worker process's
# userdata. Namespaced because the dict is shared with anything else the
# process prewarms.
VAD_USERDATA_KEY = "voxtral_realtime_vad"


class _HandshakeError(Exception):
    """The upgrade succeeded but the server did not answer with session.created."""


def _load_vad() -> agents_vad.VAD:
    return silero.VAD.load(
        min_silence_duration=_VAD_MIN_SILENCE_S,
        activation_threshold=_VAD_ACTIVATION_THRESHOLD,
    )


def prewarm(userdata: dict) -> None:
    """Load the Silero model once per worker process, before any job arrives.

    Called from the LiveKit worker's prewarm hook. Loading it lazily in the
    agent constructor instead would run once per room, synchronously on the
    job's event loop, and keep one copy of the model per concurrent job.
    """
    userdata[VAD_USERDATA_KEY] = _load_vad()


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
        if not config.base_url:
            # Fail at startup rather than with confusing auth/protocol errors
            # mid-meeting: Voxtral Realtime is served by a self-hosted vLLM
            # instance, never by api.openai.com.
            raise ValueError(
                "VOXTRAL_BASE_URL is required for the voxtral-realtime provider "
                "(the URL of your vLLM server, e.g. https://your-server:8000/v1)."
            )
        self._http_session: aiohttp.ClientSession | None = None
        if vad is None:
            # Normally supplied by prewarm(); loading here is the fallback for
            # a worker started without the hook, and costs the job the model
            # load on its own event loop.
            logging.debug("Voxtral: no prewarmed VAD supplied, loading inline.")
            vad = _load_vad()
        self._vad: agents_vad.VAD = vad

    def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=15)
            )
        return self._http_session

    def _build_ws_url(self) -> str:
        base = self.config.base_url.rstrip("/")
        base = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return f"{base}/realtime?intent=transcription"

    def _create_stt_stream(self, locale: str | None) -> stt.SpeechStream:
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

    async def _run_transcription_pipeline(  # type: ignore[override]
        self,
        participant: rtc.RemoteParticipant,
        track: rtc.Track,
        language: str | None,
    ):
        ws_url = self._build_ws_url()
        # A vLLM server started without VLLM_API_KEY takes anonymous requests,
        # so an unset key means "send no credentials" — not "send the string
        # None", which any auth proxy in front would reject outright.
        headers = (
            {"Authorization": f"Bearer {self.config.api_key}"}
            if self.config.api_key
            else {}
        )
        open_time = time.time()
        self.open_time = open_time
        retry_delay = _RETRY_DELAY_INITIAL_S
        # Monotonic timestamp of the first handshake failure in the current
        # streak; cleared whenever a handshake succeeds, so a long meeting with
        # occasional reconnects never accumulates its way to the give-up bound.
        handshake_failing_since: float | None = None

        try:
            while True:
                audio_stream = rtc.AudioStream(track)
                try:
                    async with self._get_http_session().ws_connect(
                        ws_url, headers=headers, heartbeat=_WS_HEARTBEAT_S
                    ) as ws:
                        msg = await asyncio.wait_for(ws.receive(), timeout=10.0)
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            raise _HandshakeError(
                                f"expected TEXT for session.created, got {msg.type}"
                            )
                        try:
                            data = json.loads(msg.data)
                        except (ValueError, TypeError) as e:
                            raise _HandshakeError(
                                f"malformed first message: {e}"
                            ) from e
                        if data.get("type") != "session.created":
                            raise _HandshakeError(f"unexpected first message: {data}")
                        logging.info(
                            f"Voxtral WS session created for {participant.identity}"
                        )
                        # Connection is healthy again; reset reconnect backoff
                        # and the handshake give-up streak.
                        retry_delay = _RETRY_DELAY_INITIAL_S
                        handshake_failing_since = None

                        # vLLM expects a FLAT session.update: its handler
                        # reads event.get("model") off the top level, so
                        # nesting under "session" fails validation with
                        # "Missing required field: model" (probe test 6).
                        # model is also the only field it reads — the
                        # SessionUpdate model declares nothing else, and the
                        # server hardcodes temperature=0.0 for every realtime
                        # generation, which is the greedy decoding the model
                        # card requires. temperature is sent anyway: it is
                        # ignored today and correct if the field is ever
                        # honoured.
                        await ws.send_json(
                            {
                                "type": "session.update",
                                "model": self.config.model,
                                "temperature": 0.0,
                            }
                        )

                        await self._vad_loop(
                            ws, audio_stream, participant, language, open_time
                        )
                        return  # clean exit — audio stream finished normally

                except asyncio.CancelledError:
                    raise
                except _HandshakeError as e:
                    # Retrying is what makes a server that is still warming up
                    # survivable. Returning here instead would end this
                    # participant's transcription for the rest of the meeting:
                    # nothing re-arms the pipeline, because _on_track_subscribed
                    # only fires for a track that is not already subscribed.
                    now = time.monotonic()
                    if handshake_failing_since is None:
                        handshake_failing_since = now
                    if now - handshake_failing_since >= _HANDSHAKE_GIVE_UP_S:
                        logging.error(
                            f"Voxtral WS handshake still failing for "
                            f"{participant.identity} after "
                            f"{_HANDSHAKE_GIVE_UP_S:.0f}s ({e}) — giving up. "
                            f"Check that VOXTRAL_BASE_URL points at a vLLM "
                            f"server serving the realtime API: {ws_url}"
                        )
                        return
                    logging.warning(
                        f"Voxtral WS handshake failed for {participant.identity} "
                        f"({e}), retrying in {retry_delay:.0f}s"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, _RETRY_DELAY_MAX_S)
                except (TimeoutError, aiohttp.ClientError, ConnectionResetError) as e:
                    # TimeoutError covers a slow session.created handshake —
                    # vLLM takes 2–5 min of CUDA-graph warmup after startup,
                    # during which giving up permanently would cost the
                    # participant the whole meeting. Retry with backoff.
                    # Deliberately unbounded, unlike _HandshakeError: a
                    # transport failure is as likely to be the server
                    # restarting mid-meeting as a permanent misconfiguration,
                    # and a reachable-but-silent endpoint costs one connect
                    # attempt every _RETRY_DELAY_MAX_S.
                    logging.warning(
                        f"Voxtral WS connection lost for {participant.identity} "
                        f"({type(e).__name__}: {e}), reconnecting in {retry_delay:.0f}s"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, _RETRY_DELAY_MAX_S)
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
            # Deregister only if this task still owns the entry: a locale
            # change (stop→start) registers the replacement task before this
            # cancelled task runs its finally block, so an unconditional pop
            # would deregister the replacement — leaving it running but
            # untracked (unstoppable, and a later start would spawn a
            # duplicate pipeline on the same track).
            info = self.processing_info.get(participant.identity)
            if info and info.get("task") is asyncio.current_task():
                self.processing_info.pop(participant.identity, None)

    async def _vad_loop(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        audio_stream: rtc.AudioStream,
        participant: rtc.RemoteParticipant,
        language: str | None,
        open_time: float,
    ):
        # `language` only labels the transcripts this pipeline emits; it is
        # never sent to the server, because there is nowhere to send it. vLLM's
        # SessionUpdate carries only `type` and `model`, and its realtime prompt
        # is tokenizer.instruct.start() + audio_encoder.encode_streaming_tokens()
        # — mistral_common encodes a "lang:<code>" prefix only in
        # _encode_instruct_transcription, the offline format; the streaming one
        # never reads request.language. Nothing comes back either:
        # transcription.delta carries `delta` and transcription.done carries
        # `text` and `usage`, neither a detected language. So Voxtral Realtime
        # always auto-detects, this label is the locale the participant asked
        # for, and "auto" (language None) means main.py has nothing to resolve a
        # BBB locale from and drops the transcript. See the README's caveats.
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

        # In-flight segment state. Owned by _reader, but hoisted to _vad_loop
        # scope so teardown can emit a best-effort FINAL from whatever delta
        # text was received when the segment's transcription.done never comes
        # (cancellation, reconnect, reader death). Without that, BBB keeps
        # showing the already-emitted interim as pending forever.
        seg_text = ""
        seg_start: float | None = None
        # Text of the last INTERIM emitted for the in-flight segment. vLLM
        # streams one transcription.delta per decoded frame and the frames that
        # decode to nothing carry an empty delta, so a segment ends with a run
        # of deltas that leave seg_text untouched — emitting on each of them
        # republishes identical text under the same BBB transcriptId dozens of
        # times per utterance.
        last_interim = ""

        # Segments opened whose transcription.done has not been read yet.
        # Incremented by _writer._open(), decremented by _reader on done.
        # _writer defers opening commits while this is non-zero — vLLM ignores
        # commits sent during an in-flight generation (see _OPEN_GATE_TIMEOUT_S).
        outstanding = 0

        def _pop_segment_start(context: str) -> float:
            if segment_starts:
                return segment_starts.popleft()
            # One transcription.done per opener commit is the pairing
            # invariant. An empty queue here means the server emitted more
            # transcription events than segments were opened, so this and
            # every later segment gets a mis-stamped start_time (and thus a
            # wrong BBB transcriptId that can overwrite a neighbor's caption).
            fallback = time.time() - open_time
            logging.warning(
                f"Voxtral: {context} with no queued segment start for "
                f"{participant.identity} — FIFO pairing desync; using "
                f"wall-clock fallback t={fallback:.3f}s"
            )
            return fallback

        def _emit_transcript(final: bool, text: str, start_time: float) -> None:
            self.emit(
                "final_transcript" if final else "interim_transcript",
                participant=participant,
                event=stt.SpeechEvent(
                    type=(
                        stt.SpeechEventType.FINAL_TRANSCRIPT
                        if final
                        else stt.SpeechEventType.INTERIM_TRANSCRIPT
                    ),
                    alternatives=[
                        stt.SpeechData(
                            text=text,
                            language=language,
                            start_time=start_time,
                            end_time=time.time() - open_time,
                        )
                    ],
                ),
                open_time=open_time,
            )

        # ── Reader ────────────────────────────────────────────────────────────

        async def _reader() -> None:
            """Read transcription events for all utterances on this session.

            Runs concurrently with _writer so that transcription.delta events
            emitted by the server while audio is still streaming are consumed
            in real time rather than buffered and replayed after each commit.
            """
            nonlocal seg_text, seg_start, outstanding, last_interim

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
                    if seg_start is None:
                        # Pair this segment with the start time its opener pushed.
                        seg_start = _pop_segment_start("transcription.delta")
                        logging.debug(
                            f"Voxtral: first delta for {participant.identity} "
                            f"at t={time.time() - open_time:.3f}s "
                            f"(seg_start={seg_start:.3f}s)"
                        )
                    seg_text += data.get("delta", "")
                    if (
                        seg_text
                        and seg_text != last_interim
                        and self.config.interim_results
                    ):
                        last_interim = seg_text
                        _emit_transcript(False, seg_text, seg_start)

                elif msg_type == "transcription.done":
                    # The server finished this segment's generation; the writer
                    # may open the next segment's request now.
                    outstanding = max(0, outstanding - 1)
                    # Prefer accumulated delta text over done.text: the realtime
                    # API sends content via deltas; done.text may be empty or absent.
                    server_text = data.get("text", "").strip()
                    logging.debug(
                        f"Voxtral: transcription.done for {participant.identity} "
                        f"at t={time.time() - open_time:.3f}s — "
                        f"delta_text='{seg_text[:60]}', server_text='{server_text[:60]}'"
                    )
                    if server_text:
                        seg_text = server_text

                    if seg_start is None:
                        # Zero-delta segment: consume its queued start anyway so
                        # later segments stay paired with their own openers.
                        seg_start = _pop_segment_start("transcription.done")

                    if seg_text:
                        _emit_transcript(True, seg_text, seg_start)

                    # Reset for next utterance
                    seg_text = ""
                    seg_start = None
                    last_interim = ""

                elif msg_type == "error":
                    logging.error(f"Voxtral WS error event: {data}")
                    break

            # Reaching here means the reader stopped while the writer may still
            # be streaming (error event, receive failure, server close). Close
            # the socket so the writer's next send fails and the pipeline's
            # reconnect logic takes over — otherwise audio keeps flowing into a
            # session nobody reads and transcription dies silently.
            await ws.close()

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
        #   3. On speech end (Silero END_OF_SPEECH or max-buffer): send
        #      commit(final). The server emits transcription.done, which
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
                        logging.debug(f"Voxtral: speech end for {participant.identity}")
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
            #
            # One rolling buffer of recent audio, two replay lengths at _open():
            # - Fresh onset: at most the VAD pre-roll, capped by min-silence —
            #   otherwise a normal onset could replay the previous utterance's
            #   tail (it would not have rolled out of the buffer during the
            #   gap) and duplicate it.
            # - Max-buffer split reopen: the full split overlap. Mid-speech the
            #   buffer holds only the current utterance, so the longer replay
            #   is safe and gives the reopened request the left-context the
            #   model needs (see _SPLIT_OVERLAP_S).
            nonlocal outstanding
            onset_max = (
                int(min(_VAD_PREROLL_S, _VAD_MIN_SILENCE_S) * _TARGET_SAMPLE_RATE) * 2
            )
            split_max = int(_SPLIT_OVERLAP_S * _TARGET_SAMPLE_RATE) * 2  # int16 bytes
            preroll_max = max(onset_max, split_max)
            preroll = bytearray()  # rolling window of the most recent audio
            split_reopen = False  # next _open() continues a cap-split utterance
            pending = b""  # audio buffered for chunked append while open
            open_secs = 0.0  # duration of the current open segment
            stream_open = False
            normalizer = _AudioNormalizer()  # stateful: one per audio stream
            # Commit gate (see _OPEN_GATE_TIMEOUT_S): while the previous
            # segment's transcription.done is outstanding, an opening commit
            # would be silently ignored by vLLM, so it is deferred. gate_buf
            # captures the lead-in plus all audio arriving during the wait and
            # becomes the eventual _open() replay — the segment starts a
            # little later but loses nothing.
            gated = False
            gate_buf: bytearray | None = None
            gate_deadline = 0.0
            pending_end = False  # END_OF_SPEECH arrived while gated
            loop = asyncio.get_running_loop()

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

            def _take_lead_in() -> bytes:
                """Consume the rolling buffer's tail as a new segment's lead-in."""
                nonlocal split_reopen
                limit = split_max if split_reopen else onset_max
                split_reopen = False
                lead = (
                    bytes(preroll[max(0, len(preroll) - limit) :]) if limit > 0 else b""
                )
                preroll.clear()
                return lead

            async def _open(replay: bytes) -> None:
                """Open a streaming request and replay the captured lead-in."""
                nonlocal stream_open, pending, open_secs, outstanding
                commit_event.clear()  # discard any stale END from a prior segment
                # Record this segment's start (backdated by the replayed lead-in)
                # for the reader to pair with the segment's transcription events.
                replay_secs = len(replay) / (_TARGET_SAMPLE_RATE * 2)
                segment_starts.append(time.time() - open_time - replay_secs)
                await ws.send_json({"type": "input_audio_buffer.commit"})
                outstanding += 1
                stream_open = True
                # The replay does not count toward the max-buffer cap: it is
                # bounded on its own (lead-in caps, gate timeout), and counting
                # it would make a long gated replay close the segment the
                # moment it opens.
                open_secs = 0.0
                if replay:
                    pending += replay
                    await _flush_pending()

            async def _close() -> None:
                """Flush remaining audio and close the utterance's request.

                Only commit(final) is sent, matching the vLLM reference client.
                Probe test 5 confirmed a preceding bare commit is redundant —
                same text, one done either way — and it delays the done by
                ~0.35 s (the server processes an extra commit boundary first).
                """
                nonlocal stream_open, pending
                if pending:
                    await _append(pending)
                    pending = b""
                await ws.send_json({"type": "input_audio_buffer.commit", "final": True})
                stream_open = False
                # The rolling buffer is intentionally NOT cleared: if this was a
                # max-buffer split, the next frame reopens immediately and
                # replays the split overlap so the boundary words carry fully
                # into the next segment. _open() clears it after replaying.

            try:
                async for audio_event in audio_stream:
                    frame = audio_event.frame
                    frame_duration = frame.samples_per_channel / frame.sample_rate

                    # Feed the VAD, then yield once so _vad_task is scheduled. In
                    # production the network awaits below also yield; this keeps the
                    # interleave deterministic and lets tests drive a sync frame
                    # iterator without starving the VAD task.
                    vad_stream.push_frame(frame)
                    await asyncio.sleep(0)

                    resampled = normalizer.process(frame)

                    # An END arriving while no request is open: if a gated
                    # segment is waiting, the utterance finished before the
                    # server freed up — remember to close it right after it
                    # opens. Otherwise it is stale (cap-split utterance ended
                    # before the reopen): whatever opens next is a fresh
                    # onset, not a split continuation.
                    if not stream_open and commit_event.is_set():
                        commit_event.clear()
                        if gated:
                            pending_end = True
                        else:
                            split_reopen = False

                    # Open a request as soon as the VAD reports speech — but
                    # only once the server has finished the previous segment's
                    # generation; a commit sent earlier would be silently
                    # ignored. While deferred, audio accumulates in gate_buf.
                    if not stream_open and (is_in_speech or gated):
                        if outstanding == 0:
                            replay = (
                                bytes(gate_buf)
                                if gate_buf is not None
                                else _take_lead_in()
                            )
                            gate_buf = None
                            gated = False
                            await _open(replay)
                            if pending_end:
                                pending_end = False
                                # The gated utterance already ended; close it
                                # now — unless the speaker has resumed, in
                                # which case keep streaming and let the next
                                # END close the merged segment.
                                if not is_in_speech:
                                    await _close()
                        elif not gated:
                            gated = True
                            gate_deadline = loop.time() + _OPEN_GATE_TIMEOUT_S
                            gate_buf = bytearray(_take_lead_in())
                        elif loop.time() >= gate_deadline:
                            # The done never arrived (lost closing commit or
                            # event desync); waiting longer only buffers more
                            # audio. Resync and open ungated on the next frame.
                            #
                            # Drop the starts still queued along with the
                            # counter. A segment that streamed has already had
                            # its start popped at its first delta, so what is
                            # left belongs to the segments this timeout is
                            # giving up on. Keeping them would pair every later
                            # segment with its predecessor's start — a wrong
                            # BBB transcriptId, for the life of the connection,
                            # and silently: _pop_segment_start warns about an
                            # empty queue, never an over-full one.
                            #
                            # The trade: a server merely slow past the timeout
                            # still owes deltas for a start just discarded, so
                            # that one segment falls back to a wall-clock stamp
                            # — loudly, from _pop_segment_start. A first delta
                            # arrives ~0.6 s after speech starts, so reaching
                            # here means the segment is wedged, not slow.
                            orphans = len(segment_starts)
                            segment_starts.clear()
                            logging.warning(
                                f"Voxtral: open gate timed out for "
                                f"{participant.identity} with {outstanding} "
                                f"transcription(s) outstanding — resyncing "
                                f"(discarding {orphans} unpaired segment start(s))"
                            )
                            outstanding = 0

                    if stream_open:
                        pending += resampled
                        open_secs += frame_duration
                        await _flush_pending()
                    elif gated:
                        gate_buf += resampled

                    # Keep a bounded rolling buffer of the most recent audio,
                    # whether idle or open. _open() replays its tail as the
                    # segment's lead-in: on a fresh onset the onset-capped
                    # pre-roll (the audio before the VAD fired); on a
                    # max-buffer split the full overlap that carries the
                    # boundary words into the next segment instead of cutting
                    # them in half.
                    preroll += resampled
                    if len(preroll) > preroll_max:
                        del preroll[: len(preroll) - preroll_max]

                    # Close on end-of-speech (Silero) or the max-buffer safety cap.
                    # is_in_speech is deliberately left untouched: if the speaker is
                    # still talking past the cap, the next frame reopens immediately.
                    if stream_open and (
                        commit_event.is_set() or open_secs >= _MAX_BUFFER_DURATION_S
                    ):
                        # A cap-triggered close (no END) splits mid-speech; the
                        # reopen replays the full overlap instead of the onset
                        # pre-roll.
                        split_reopen = not commit_event.is_set()
                        commit_event.clear()
                        await _close()
            except asyncio.CancelledError:
                # Frames stop before Silero can fire END_OF_SPEECH when a
                # speaker mutes or unpublishes right after talking, so an open
                # request would be dropped without its closing commit — losing
                # the utterance's FINAL and leaving the server request dangling
                # (abrupt drops are a known vLLM realtime crash trigger).
                # Best-effort close before propagating the cancellation. A
                # gated segment never sent its opener, so open-and-close it —
                # the server may still be mid-generation and drop the opener,
                # but that is no worse than losing the utterance outright.
                if stream_open or (gated and gate_buf):
                    try:

                        async def _cancel_flush() -> None:
                            if not stream_open:
                                await _open(bytes(gate_buf))
                            await _close()

                        await asyncio.wait_for(_cancel_flush(), timeout=1.0)
                    except Exception:
                        logging.debug(
                            f"Voxtral: cancel-time flush failed for "
                            f"{participant.identity}"
                        )
                raise

            # End of stream: drain the resampler tail and flush any open request.
            # A gated segment's opener was deferred the whole time; send it now,
            # best-effort, so the buffered utterance still gets transcribed.
            if stream_open:
                pending += normalizer.flush()
                await _close()
            elif gated and gate_buf:
                gate_buf += normalizer.flush()
                await _open(bytes(gate_buf))
                await _close()
            vad_stream.end_input()

        # ── Run all three tasks concurrently ──────────────────────────────────

        reader_task = asyncio.create_task(_reader())
        vad_task = asyncio.create_task(_vad_task())

        async def _drain_reader() -> None:
            """Wait (bounded) for the reader to consume the final segment's
            transcription.done after the writer finishes cleanly."""
            loop = asyncio.get_running_loop()
            deadline = loop.time() + _FINAL_DRAIN_TIMEOUT_S
            while (
                (segment_starts or seg_text)
                and not reader_task.done()
                and loop.time() < deadline
            ):
                await asyncio.sleep(0.05)

        try:
            await _writer()
            await _drain_reader()
        finally:
            reader_task.cancel()
            vad_task.cancel()
            await asyncio.gather(reader_task, vad_task, return_exceptions=True)
            await vad_stream.aclose()
            if segment_starts:
                # Expected when teardown interrupts an open segment; anything
                # beyond that means dones went missing (see _pop_segment_start).
                logging.info(
                    f"Voxtral: {len(segment_starts)} opened segment(s) without a "
                    f"transcription.done at teardown for {participant.identity}"
                )
            if seg_text:
                # Teardown caught a segment mid-flight: its transcription.done
                # will never be read, and BBB would show the already-emitted
                # interim as pending forever. Commit the best text we have.
                logging.info(
                    f"Voxtral: emitting best-effort FINAL for "
                    f"{participant.identity} on teardown: '{seg_text[:60]}'"
                )
                _emit_transcript(
                    True,
                    seg_text,
                    seg_start if seg_start is not None else time.time() - open_time,
                )


class _AudioNormalizer:
    """Downmix to mono and resample to _TARGET_SAMPLE_RATE PCM16.

    Wraps rtc.AudioResampler (SoX), which low-pass filters before decimating —
    naive linear interpolation aliases everything above the target Nyquist
    (8 kHz) into the speech band and measurably hurts transcription accuracy.
    The resampler is streaming: its filter state must persist across frames,
    so use one instance per audio stream, and call flush() at end of stream
    to drain the tail samples held back by the filter.
    """

    def __init__(self):
        self._resampler: rtc.AudioResampler | None = None
        self._input_rate: int | None = None

    def process(self, frame: rtc.AudioFrame) -> bytes:
        samples = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            samples = (
                samples.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
            )
        if frame.sample_rate == _TARGET_SAMPLE_RATE:
            return samples.tobytes()
        if self._input_rate != frame.sample_rate:
            self._resampler = rtc.AudioResampler(frame.sample_rate, _TARGET_SAMPLE_RATE)
            self._input_rate = frame.sample_rate
        frames = self._resampler.push(bytearray(samples.tobytes()))
        return b"".join(bytes(f.data) for f in frames)

    def flush(self) -> bytes:
        if self._resampler is None:
            return b""
        return b"".join(bytes(f.data) for f in self._resampler.flush())
