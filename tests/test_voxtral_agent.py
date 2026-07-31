import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import numpy as np
import pytest
from livekit import rtc
from livekit.agents import stt

from livekit.agents import vad as agents_vad

from providers.voxtral_realtime import (
    VoxtralRealtimeConfig,
    VoxtralRealtimeSttAgent,
    _AudioNormalizer,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_mock_vad(vad_events=None):
    """Return a mock VAD that emits the given VADEvent sequence."""
    mock_vad = MagicMock()
    # AsyncMock with __aiter__.return_value is the correct pattern for async-for
    # (same as AudioStream mocking in other test files).
    mock_vad_stream = AsyncMock()
    mock_vad_stream.push_frame = MagicMock()
    mock_vad_stream.end_input = MagicMock()
    mock_vad_stream.aclose = AsyncMock()
    events = vad_events or []
    mock_vad_stream.__aiter__.return_value = iter(events)
    mock_vad.stream.return_value = mock_vad_stream
    return mock_vad


def _make_config(**kwargs):
    kwargs.setdefault("base_url", "https://test-server.example.com/v1")
    return VoxtralRealtimeConfig(api_key="test-key", **kwargs)


def _make_agent(vad_events=None, **kwargs):
    return VoxtralRealtimeSttAgent(
        _make_config(**kwargs), vad=_make_mock_vad(vad_events)
    )


def _make_participant(identity, has_audio_track=True):
    participant = MagicMock(spec=rtc.RemoteParticipant)
    participant.identity = identity
    pubs = {}
    if has_audio_track:
        mock_track = MagicMock()
        mock_track.kind = rtc.TrackKind.KIND_AUDIO
        pub = MagicMock()
        pub.track = mock_track
        pubs["audio"] = pub
    participant.track_publications = pubs
    return participant


def _make_agent_with_room(participants=None, **kwargs):
    agent = _make_agent(**kwargs)
    mock_room = MagicMock()
    mock_room.remote_participants = participants or {}
    agent.room = mock_room
    return agent


def _text_ws_msg(data: dict) -> MagicMock:
    msg = MagicMock()
    msg.type = aiohttp.WSMsgType.TEXT
    msg.data = json.dumps(data)
    return msg


def _make_audio_frame(
    amplitude: int = 0,
    sample_rate: int = 16000,
    num_channels: int = 1,
    samples_per_channel: int = 160,
) -> MagicMock:
    total = samples_per_channel * num_channels
    samples = np.full(total, amplitude, dtype=np.int16)
    frame = MagicMock()
    frame.data = samples.tobytes()
    frame.sample_rate = sample_rate
    frame.samples_per_channel = samples_per_channel
    frame.num_channels = num_channels
    return frame


def _make_loud_frame():
    # Amplitude value is irrelevant; speech detection is now Silero-based (mocked in tests)
    return _make_audio_frame(amplitude=1000)


# ── Config ─────────────────────────────────────────────────────────────────────


class TestVoxtralRealtimeConfig:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for key in [
            "VOXTRAL_API_KEY",
            "VOXTRAL_MODEL",
            "VOXTRAL_BASE_URL",
            "VOXTRAL_INTERIM_RESULTS",
        ]:
            monkeypatch.delenv(key, raising=False)

    def test_default_model(self):
        assert (
            VoxtralRealtimeConfig().model == "mistralai/Voxtral-Mini-4B-Realtime-2602"
        )

    def test_default_api_key_is_none(self):
        assert VoxtralRealtimeConfig().api_key is None

    def test_default_base_url_is_none(self):
        assert VoxtralRealtimeConfig().base_url is None

    def test_default_interim_results_is_true(self):
        assert VoxtralRealtimeConfig().interim_results is True

    def test_interim_results_false_via_env(self, monkeypatch):
        monkeypatch.setenv("VOXTRAL_INTERIM_RESULTS", "false")
        assert VoxtralRealtimeConfig().interim_results is False

    def test_interim_results_true_via_env(self, monkeypatch):
        monkeypatch.setenv("VOXTRAL_INTERIM_RESULTS", "true")
        assert VoxtralRealtimeConfig().interim_results is True

    def test_custom_model_via_env(self, monkeypatch):
        monkeypatch.setenv("VOXTRAL_MODEL", "my-custom-model")
        assert VoxtralRealtimeConfig().model == "my-custom-model"

    def test_custom_api_key_via_env(self, monkeypatch):
        monkeypatch.setenv("VOXTRAL_API_KEY", "sk-test-key")
        assert VoxtralRealtimeConfig().api_key == "sk-test-key"

    def test_custom_base_url_via_env(self, monkeypatch):
        monkeypatch.setenv("VOXTRAL_BASE_URL", "http://localhost:8000")
        assert VoxtralRealtimeConfig().base_url == "http://localhost:8000"


# ── URL builder ────────────────────────────────────────────────────────────────


class TestBuildWsUrl:
    def test_missing_base_url_raises_at_agent_creation(self):
        """Voxtral is self-hosted; defaulting to api.openai.com can only
        produce confusing auth/protocol errors mid-meeting. Fail at startup."""
        with pytest.raises(ValueError, match="VOXTRAL_BASE_URL"):
            VoxtralRealtimeSttAgent(
                VoxtralRealtimeConfig(api_key="test-key", base_url=None),
                vad=_make_mock_vad(),
            )

    def test_custom_https_url_becomes_wss(self):
        agent = _make_agent(base_url="https://my-server.example.com/v1")
        assert agent._build_ws_url().startswith("wss://")

    def test_custom_http_url_becomes_ws(self):
        agent = _make_agent(base_url="http://localhost:8000/v1")
        assert agent._build_ws_url().startswith("ws://")

    def test_trailing_slash_in_base_url_is_stripped(self):
        agent = _make_agent(base_url="https://my-server.example.com/v1/")
        url = agent._build_ws_url()
        assert "//realtime" not in url

    def test_custom_host_is_preserved(self):
        agent = _make_agent(base_url="https://my-server.example.com/v1")
        assert "my-server.example.com" in agent._build_ws_url()


# ── PCM conversion ─────────────────────────────────────────────────────────────


class TestAudioNormalizer:
    def test_returns_bytes(self):
        frame = _make_audio_frame()
        assert isinstance(_AudioNormalizer().process(frame), bytes)

    def test_mono_16k_passthrough_preserves_values(self):
        frame = _make_audio_frame(amplitude=1000, sample_rate=16000, num_channels=1)
        result = _AudioNormalizer().process(frame)
        samples = np.frombuffer(result, dtype=np.int16)
        assert len(samples) == frame.samples_per_channel
        assert all(s == 1000 for s in samples)

    def test_stereo_downmix_to_mono(self):
        """Stereo frame with equal channels averages to same amplitude."""
        frame = _make_audio_frame(amplitude=1000, sample_rate=16000, num_channels=2)
        result = _AudioNormalizer().process(frame)
        samples = np.frombuffer(result, dtype=np.int16)
        assert len(samples) == frame.samples_per_channel
        assert all(s == 1000 for s in samples)

    def test_resampling_from_48k_produces_correct_total_length(self):
        """process() over many frames + flush() yields ~1/3 the samples.

        The resampler is streaming, so individual frames may return fewer
        samples (filter latency); only the drained total is deterministic.
        """
        normalizer = _AudioNormalizer()
        n_frames = 20  # 200 ms at 48 kHz
        out = b"".join(
            normalizer.process(_make_audio_frame(amplitude=500, sample_rate=48000))
            for _ in range(n_frames)
        )
        out += normalizer.flush()
        total_in = n_frames * 160
        expected = total_in * 16000 // 48000
        got = len(out) // 2
        assert abs(got - expected) <= 32, f"expected ~{expected} samples, got {got}"

    def test_downsampling_attenuates_above_target_nyquist(self):
        """Anti-aliasing regression: a 10 kHz tone at 48 kHz lies above the
        16 kHz target's Nyquist (8 kHz) and must be strongly attenuated.
        Naive linear interpolation instead folds it into the speech band at
        nearly full energy — the defect this normalizer replaces.
        """
        normalizer = _AudioNormalizer()
        rate_in, tone_hz, duration_s = 48000, 10000, 0.2
        t = np.arange(int(rate_in * duration_s)) / rate_in
        tone = (0.5 * 32767 * np.sin(2 * np.pi * tone_hz * t)).astype(np.int16)

        out = bytearray()
        frame_samples = 480  # 10 ms frames, as LiveKit delivers
        for i in range(0, len(tone), frame_samples):
            chunk = tone[i : i + frame_samples]
            frame = _make_audio_frame(sample_rate=rate_in)
            frame.data = chunk.tobytes()
            frame.samples_per_channel = len(chunk)
            out += normalizer.process(frame)
        out += normalizer.flush()

        in_rms = np.sqrt(np.mean(tone.astype(np.float64) ** 2))
        out_samples = np.frombuffer(bytes(out), dtype=np.int16).astype(np.float64)
        out_rms = np.sqrt(np.mean(out_samples**2)) if len(out_samples) else 0.0
        assert out_rms < 0.1 * in_rms, (
            f"10 kHz tone must be attenuated by the anti-aliasing filter "
            f"(in_rms={in_rms:.0f}, out_rms={out_rms:.0f})"
        )


# ── start_transcription_for_user ───────────────────────────────────────────────


class TestStartTranscriptionForUser:
    def test_participant_not_found_logs_error(self, caplog):
        agent = _make_agent_with_room(participants={})
        with caplog.at_level("ERROR"):
            agent.start_transcription_for_user(
                "ghost_user", "en-US", "voxtral-realtime"
            )
        assert "ghost_user" in caplog.text
        assert "ghost_user" not in agent.processing_info

    def test_no_audio_track_logs_warning(self, caplog):
        participant = _make_participant("user_1", has_audio_track=False)
        agent = _make_agent_with_room(participants={"p1": participant})
        with caplog.at_level("WARNING"):
            agent.start_transcription_for_user("user_1", "en-US", "voxtral-realtime")
        assert "user_1" in caplog.text
        assert "user_1" not in agent.processing_info

    def test_already_running_is_ignored(self):
        participant = _make_participant("user_1")
        agent = _make_agent_with_room(participants={"p1": participant})
        existing_task = MagicMock()
        agent.processing_info["user_1"] = {"task": existing_task}

        agent.start_transcription_for_user("user_1", "en-US", "voxtral-realtime")

        assert agent.processing_info["user_1"]["task"] is existing_task

    async def test_success_adds_task_to_processing_info(self):
        participant = _make_participant("user_1")
        agent = _make_agent_with_room(participants={"p1": participant})

        with patch.object(agent, "_run_transcription_pipeline", new_callable=AsyncMock):
            agent.start_transcription_for_user("user_1", "en-US", "voxtral-realtime")

        assert "user_1" in agent.processing_info
        assert "task" in agent.processing_info["user_1"]
        agent.processing_info.pop("user_1", None)

    async def test_locale_is_sanitized_to_language_code(self):
        participant = _make_participant("user_1")
        agent = _make_agent_with_room(participants={"p1": participant})

        with patch.object(
            agent, "_run_transcription_pipeline", new_callable=AsyncMock
        ) as mock_pipeline:
            agent.start_transcription_for_user("user_1", "pt-BR", "voxtral-realtime")
            await asyncio.sleep(0)

        # _run_transcription_pipeline(participant, track, language) — language must be "pt"
        assert mock_pipeline.call_args[0][2] == "pt"
        agent.processing_info.pop("user_1", None)

    async def test_settings_stored_on_start(self):
        participant = _make_participant("user_1")
        agent = _make_agent_with_room(participants={"p1": participant})

        with patch.object(agent, "_run_transcription_pipeline", new_callable=AsyncMock):
            agent.start_transcription_for_user("user_1", "de-DE", "voxtral-realtime")

        settings = agent.participant_settings.get("user_1", {})
        assert settings["locale"] == "de-DE"
        assert settings["provider"] == "voxtral-realtime"
        agent.processing_info.pop("user_1", None)


# ── _cleanup ───────────────────────────────────────────────────────────────────


class TestCleanup:
    async def test_closes_http_session_and_sets_none(self):
        agent = _make_agent()
        mock_session = AsyncMock()
        agent._http_session = mock_session

        await agent._cleanup()

        mock_session.close.assert_called_once()
        assert agent._http_session is None

    async def test_no_op_when_no_session(self):
        agent = _make_agent()
        assert agent._http_session is None
        await agent._cleanup()  # must not raise


# ── _run_transcription_pipeline — early exit paths ────────────────────────────


class TestRunTranscriptionPipeline:
    def _ws_context(self, first_message):
        """Build an async context manager that yields a mock WS with one receive."""
        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(return_value=first_message)
        mock_ws.send_json = AsyncMock()
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_ws)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    def _mock_session(self, first_message):
        session = MagicMock()
        session.ws_connect = MagicMock(return_value=self._ws_context(first_message))
        return session

    async def test_exits_cleanly_on_non_text_first_message(self, caplog):
        agent = _make_agent()
        participant = MagicMock(spec=rtc.RemoteParticipant)
        participant.identity = "user_1"

        binary_msg = MagicMock()
        binary_msg.type = aiohttp.WSMsgType.BINARY

        agent._http_session = self._mock_session(binary_msg)

        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter([])
        mock_stream.aclose = AsyncMock()

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream
        ):
            await agent._run_transcription_pipeline(participant, MagicMock(), "en")

        assert "user_1" not in agent.processing_info

    async def test_exits_cleanly_on_wrong_first_message_type(self, caplog):
        agent = _make_agent()
        participant = MagicMock(spec=rtc.RemoteParticipant)
        participant.identity = "user_1"

        agent._http_session = self._mock_session(
            _text_ws_msg({"type": "session.error"})  # not "session.created"
        )

        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter([])
        mock_stream.aclose = AsyncMock()

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream
        ):
            await agent._run_transcription_pipeline(participant, MagicMock(), "en")

        assert "user_1" not in agent.processing_info


# ── _vad_loop — speech detection and final flush ──────────────────────────────


def _make_vad_event(event_type: agents_vad.VADEventType) -> MagicMock:
    ev = MagicMock()
    ev.type = event_type
    return ev


class TestVadLoop:
    def _full_pipeline_setup(self, audio_frames, ws_messages, vad_events=None):
        """
        Return (agent, participant, mock_stream) wired up so that
        _run_transcription_pipeline can run end-to-end through the VAD loop.
        ws_messages is a list appended after the mandatory session.created message.
        vad_events: Silero VAD events emitted by the mock; defaults to
          [START_OF_SPEECH, END_OF_SPEECH] so a commit fires.
        """
        if vad_events is None:
            vad_events = [
                _make_vad_event(agents_vad.VADEventType.START_OF_SPEECH),
                _make_vad_event(agents_vad.VADEventType.END_OF_SPEECH),
            ]
        agent = _make_agent(interim_results=True, vad_events=vad_events)
        participant = MagicMock(spec=rtc.RemoteParticipant)
        participant.identity = "user_vad"

        all_ws = [_text_ws_msg({"type": "session.created"})] + ws_messages

        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(side_effect=all_ws)

        # Use a real async function so that awaiting send_json actually yields
        # to the event loop — this gives the concurrent _reader() task a chance
        # to process incoming WS messages while the writer is still sending.
        async def _send_json(_data):
            await asyncio.sleep(0)

        mock_ws.send_json = _send_json
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_ws)
        cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.ws_connect = MagicMock(return_value=cm)
        agent._http_session = mock_session

        audio_events = [MagicMock(frame=f) for f in audio_frames]
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter(audio_events)
        mock_stream.aclose = AsyncMock()

        return agent, participant, mock_stream

    async def test_silent_frames_do_not_trigger_flush(self):
        """No VAD events — no commit fires, no transcript event."""
        silence = _make_audio_frame(amplitude=0)
        agent, participant, mock_stream = self._full_pipeline_setup(
            audio_frames=[silence],
            ws_messages=[],
            vad_events=[],  # Silero never fires → no commit → no transcript
        )

        emitted = []
        agent.on("final_transcript", lambda **kw: emitted.append(kw))

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream
        ):
            await agent._run_transcription_pipeline(participant, MagicMock(), "en")

        assert emitted == []

    async def test_loud_frame_followed_by_end_of_stream_emits_final(self):
        """
        One loud frame followed by end-of-stream triggers the end-of-stream flush,
        which should produce a final_transcript event.
        """
        loud = _make_loud_frame()
        agent, participant, mock_stream = self._full_pipeline_setup(
            audio_frames=[loud],
            ws_messages=[
                _text_ws_msg({"type": "transcription.delta", "delta": "hello"}),
                _text_ws_msg({"type": "transcription.done", "text": "hello"}),
            ],
        )

        emitted = []
        agent.on("final_transcript", lambda **kw: emitted.append(kw))

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream
        ):
            await agent._run_transcription_pipeline(participant, MagicMock(), "en")
        await asyncio.sleep(0)

        assert len(emitted) == 1
        event = emitted[0]["event"]
        assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        assert event.alternatives[0].text == "hello"

    async def test_interim_deltas_emitted_during_flush(self):
        """Delta messages during flush are emitted as interim_transcript events."""
        loud = _make_loud_frame()
        agent, participant, mock_stream = self._full_pipeline_setup(
            audio_frames=[loud],
            ws_messages=[
                _text_ws_msg({"type": "transcription.delta", "delta": "hi"}),
                _text_ws_msg({"type": "transcription.done", "text": "hi"}),
            ],
        )

        interim = []
        final = []
        agent.on("interim_transcript", lambda **kw: interim.append(kw))
        agent.on("final_transcript", lambda **kw: final.append(kw))

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream
        ):
            await agent._run_transcription_pipeline(participant, MagicMock(), "en")
        await asyncio.sleep(0)

        assert len(interim) >= 1
        assert interim[0]["event"].type == stt.SpeechEventType.INTERIM_TRANSCRIPT
        assert len(final) == 1

    async def test_two_utterances_emit_two_finals_with_independent_text(self):
        """
        Two speech→silence cycles produce two independent FINAL transcripts.

        Guards the per-utterance reset: the delta accumulator and utterance_start
        are cleared on transcription.done so the second utterance does not inherit
        the first utterance's text (no "helloworld" bleed).
        """
        loud = _make_loud_frame()
        # A single silence frame long enough to cross _SILENCE_DURATION_S (0.6 s),
        # flushing the utterance: 9600 samples / 16000 Hz = 0.6 s.
        silence = _make_audio_frame(amplitude=0, samples_per_channel=9600)
        agent, participant, mock_stream = self._full_pipeline_setup(
            audio_frames=[loud, silence, loud, silence],
            ws_messages=[
                _text_ws_msg({"type": "transcription.delta", "delta": "hello"}),
                _text_ws_msg({"type": "transcription.done", "text": "hello"}),
                _text_ws_msg({"type": "transcription.delta", "delta": "world"}),
                _text_ws_msg({"type": "transcription.done", "text": "world"}),
            ],
        )

        interim = []
        final = []
        agent.on("interim_transcript", lambda **kw: interim.append(kw))
        agent.on("final_transcript", lambda **kw: final.append(kw))

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream
        ):
            await agent._run_transcription_pipeline(participant, MagicMock(), "en")
        await asyncio.sleep(0)

        final_texts = [kw["event"].alternatives[0].text for kw in final]
        assert final_texts == ["hello", "world"]

        # The second utterance's interim must not carry the first's text.
        second_interim = [
            kw["event"].alternatives[0].text
            for kw in interim
            if kw["event"].alternatives[0].text.startswith("world")
        ]
        assert second_interim, "expected an interim for the second utterance"
        assert not any(t.startswith("hello") for t in second_interim)

    async def test_all_events_of_an_utterance_share_start_time(self):
        """
        The interim and final transcripts for one utterance carry the same
        start_time — the snapshot taken at the first delta, so every event lands
        on the same BBB transcriptId rather than drifting with later deltas.
        """
        loud = _make_loud_frame()
        agent, participant, mock_stream = self._full_pipeline_setup(
            audio_frames=[loud],
            ws_messages=[
                _text_ws_msg({"type": "transcription.delta", "delta": "one "}),
                _text_ws_msg({"type": "transcription.delta", "delta": "two"}),
                _text_ws_msg({"type": "transcription.done", "text": "one two"}),
            ],
        )

        interim = []
        final = []
        agent.on("interim_transcript", lambda **kw: interim.append(kw))
        agent.on("final_transcript", lambda **kw: final.append(kw))

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream
        ):
            await agent._run_transcription_pipeline(participant, MagicMock(), "en")
        await asyncio.sleep(0)

        assert len(final) == 1
        start_times = {kw["event"].alternatives[0].start_time for kw in interim}
        start_times.add(final[0]["event"].alternatives[0].start_time)
        assert len(start_times) == 1, (
            f"all events for one utterance must share start_time, got {start_times}"
        )

    async def test_max_buffer_split_reopens_stream(self, monkeypatch):
        """
        Regression for the is_in_speech / stream_open dual-ownership bug: a
        continuous utterance that exceeds the max-buffer cap (speaker never
        pauses, so Silero emits START but no END) must REOPEN a fresh streaming
        request after each forced close — otherwise the rest of the utterance is
        appended with no opening commit and the server silently stops streaming.
        """
        import providers.voxtral_realtime as vr

        # Trip the safety cap after ~2 frames (each _make_audio_frame is 0.01 s).
        monkeypatch.setattr(vr, "_MAX_BUFFER_DURATION_S", 0.015)

        agent = _make_agent(
            interim_results=True,
            # START only, no END → speaker stays "in speech" the whole time.
            vad_events=[_make_vad_event(agents_vad.VADEventType.START_OF_SPEECH)],
        )
        participant = MagicMock(spec=rtc.RemoteParticipant)
        participant.identity = "user_split"

        sent: list[dict] = []

        async def _send_json(data):
            sent.append(data)
            await asyncio.sleep(0)

        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(
            side_effect=[_text_ws_msg({"type": "session.created"})]
        )
        mock_ws.send_json = _send_json
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_ws)
        cm.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.ws_connect = MagicMock(return_value=cm)
        agent._http_session = mock_session

        frames = [_make_loud_frame() for _ in range(6)]
        audio_events = [MagicMock(frame=f) for f in frames]
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter(audio_events)
        mock_stream.aclose = AsyncMock()

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream
        ):
            await agent._run_transcription_pipeline(participant, MagicMock(), "en")

        commits = [m for m in sent if m.get("type") == "input_audio_buffer.commit"]
        # Each open sends one bare commit; each close sends one final commit.
        openers = [m for m in commits if "final" not in m]
        assert len(openers) >= 2, (
            f"expected the stream to reopen after a max-buffer split "
            f"(>=2 opener commits), got {len(openers)}"
        )

    async def test_max_buffer_split_segments_get_distinct_start_times(
        self, monkeypatch
    ):
        """
        Regression for the transcript-overwrite bug: when one long utterance is
        split by the max-buffer cap, Silero fires no new START_OF_SPEECH, so a
        shared "speech start" would give both segments the same start_time —
        the same BBB transcriptId — and segment 2's text would REPLACE segment
        1's in the transcript. Each opener must record its own start time and
        the reader must pair segments with them in FIFO order.
        """
        import providers.voxtral_realtime as vr

        monkeypatch.setattr(vr, "_MAX_BUFFER_DURATION_S", 0.015)
        # Deterministic, strictly-increasing clock so the two openers cannot
        # land on the same wall-clock value (real splits are ~8 s apart).
        fake_now = [1000.0]

        def _fake_time():
            fake_now[0] += 1.0
            return fake_now[0]

        monkeypatch.setattr(vr.time, "time", _fake_time)

        agent = _make_agent(
            interim_results=True,
            # START only, no END → continuous speech across the split.
            vad_events=[_make_vad_event(agents_vad.VADEventType.START_OF_SPEECH)],
        )
        participant = MagicMock(spec=rtc.RemoteParticipant)
        participant.identity = "user_split_ts"

        sent: list[dict] = []

        async def _send_json(data):
            sent.append(data)
            await asyncio.sleep(0)

        def _bare_commits():
            return [
                m
                for m in sent
                if m.get("type") == "input_audio_buffer.commit" and "final" not in m
            ]

        def _closers():
            return [
                m
                for m in sent
                if m.get("type") == "input_audio_buffer.commit"
                and m.get("final") is True
            ]

        # Deliver each segment's transcription only after the writer has sent
        # the corresponding commits — mirroring the real server's causality.
        # Segment 1 events after its close; segment 2 events after reopen
        # (bare commits are openers only: open1 + open2 = 2 bare commits).
        closed_msg = MagicMock()
        closed_msg.type = aiohttp.WSMsgType.CLOSED
        script = [
            (lambda: True, _text_ws_msg({"type": "session.created"})),
            (
                lambda: len(_closers()) >= 1,
                _text_ws_msg({"type": "transcription.delta", "delta": "hello"}),
            ),
            (
                lambda: len(_closers()) >= 1,
                _text_ws_msg({"type": "transcription.done", "text": "hello"}),
            ),
            (
                lambda: len(_bare_commits()) >= 2,
                _text_ws_msg({"type": "transcription.delta", "delta": "world"}),
            ),
            (
                lambda: len(_bare_commits()) >= 2,
                _text_ws_msg({"type": "transcription.done", "text": "world"}),
            ),
            (lambda: True, closed_msg),
        ]
        script_iter = iter(script)

        async def _receive():
            try:
                cond, msg = next(script_iter)
            except StopIteration:
                return closed_msg
            while not cond():
                await asyncio.sleep(0)
            return msg

        mock_ws = AsyncMock()
        mock_ws.receive = _receive
        mock_ws.send_json = _send_json
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_ws)
        cm.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.ws_connect = MagicMock(return_value=cm)
        agent._http_session = mock_session

        frames = [_make_loud_frame() for _ in range(6)]
        audio_events = [MagicMock(frame=f) for f in frames]
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter(audio_events)
        mock_stream.aclose = AsyncMock()

        final = []
        agent.on("final_transcript", lambda **kw: final.append(kw))

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream
        ):
            await agent._run_transcription_pipeline(participant, MagicMock(), "en")
        await asyncio.sleep(0)

        texts = [kw["event"].alternatives[0].text for kw in final]
        assert texts == ["hello", "world"]

        starts = [kw["event"].alternatives[0].start_time for kw in final]
        assert starts[0] < starts[1], (
            f"split segments must have distinct, increasing start_times "
            f"(distinct BBB transcriptIds) — got {starts}; equal values mean "
            f"segment 2 overwrites segment 1 in the transcript"
        )


# ── Split overlap and onset pre-roll ───────────────────────────────────────────


def _segments_from_sent(sent: list[dict]) -> list[bytes]:
    """Group appended audio bytes into segments delimited by opener commits."""
    segments: list[bytearray] = []
    for m in sent:
        t = m.get("type")
        if t == "input_audio_buffer.commit" and "final" not in m:
            segments.append(bytearray())
        elif t == "input_audio_buffer.append" and segments:
            segments[-1] += base64.b64decode(m["audio"])
    return [bytes(s) for s in segments]


class _DeferredStartVadStream:
    """VAD stream double that fires START_OF_SPEECH after N pushed frames."""

    def __init__(self, after_frames: int):
        self._after = after_frames
        self.pushed = 0
        self._fired = False

    def push_frame(self, _frame):
        self.pushed += 1

    def end_input(self):
        pass

    async def aclose(self):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._fired:
            await asyncio.Event().wait()  # block until cancelled
        while self.pushed < self._after:
            await asyncio.sleep(0)
        self._fired = True
        return _make_vad_event(agents_vad.VADEventType.START_OF_SPEECH)


class TestSplitOverlap:
    def _run_pipeline(self, agent, frames):
        participant = MagicMock(spec=rtc.RemoteParticipant)
        participant.identity = "user_overlap"

        sent: list[dict] = []

        async def _send_json(data):
            sent.append(data)
            await asyncio.sleep(0)

        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(
            side_effect=[_text_ws_msg({"type": "session.created"})]
        )
        mock_ws.send_json = _send_json
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_ws)
        cm.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.ws_connect = MagicMock(return_value=cm)
        agent._http_session = mock_session

        audio_events = [MagicMock(frame=f) for f in frames]
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter(audio_events)
        mock_stream.aclose = AsyncMock()

        return participant, mock_stream, sent

    async def test_split_reopen_replays_overlap(self, monkeypatch):
        """
        Regression for word loss at max-buffer splits: the reopened segment
        must begin with the overlap — the tail of the audio already sent in
        the previous segment — so boundary words carry fully into the new
        request instead of being cut at the commit boundary.
        """
        import providers.voxtral_realtime as vr

        monkeypatch.setattr(vr, "_MAX_BUFFER_DURATION_S", 0.015)
        # Shrink the onset pre-roll far below one frame so a full-frame overlap
        # can ONLY come from the split path: if the reopen were mistaken for a
        # fresh onset, it would replay a sliver and the assertion would fail.
        monkeypatch.setattr(vr, "_VAD_PREROLL_S", 0.005)

        agent = _make_agent(
            # START only, no END → continuous speech across the split.
            vad_events=[_make_vad_event(agents_vad.VADEventType.START_OF_SPEECH)],
        )
        # Distinct per-frame amplitudes so overlapping byte ranges are
        # distinguishable (constant amplitude would make any comparison pass).
        frames = [_make_audio_frame(amplitude=100 + i) for i in range(6)]
        participant, mock_stream, sent = self._run_pipeline(agent, frames)

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream
        ):
            await agent._run_transcription_pipeline(participant, MagicMock(), "en")

        segments = _segments_from_sent(sent)
        assert len(segments) >= 2, f"expected a split, got {len(segments)} segment(s)"
        seg1, seg2 = segments[0], segments[1]

        frame_bytes = 320  # 160 samples of int16
        overlap = min(len(seg1), len(seg2))
        assert overlap >= frame_bytes and seg1.endswith(seg2[:overlap]), (
            "the reopened segment must start with the tail of the previous "
            "segment's audio (the split overlap)"
        )

    async def test_fresh_onset_replay_is_capped(self, monkeypatch):
        """
        The rolling buffer is now sized for the split overlap (long), but a
        FRESH onset must still replay only the onset pre-roll — otherwise it
        would replay the previous utterance's tail from before the silence
        gap and duplicate it.
        """
        import providers.voxtral_realtime as vr

        onset_secs = 0.02  # 2 frames
        monkeypatch.setattr(vr, "_VAD_PREROLL_S", onset_secs)
        monkeypatch.setattr(vr, "_SPLIT_OVERLAP_S", 10.0)  # buffer far larger

        idle_frames = 100  # 1 s of audio buffered before speech starts
        vad_stream = _DeferredStartVadStream(after_frames=idle_frames)
        mock_vad = MagicMock()
        mock_vad.stream.return_value = vad_stream
        agent = VoxtralRealtimeSttAgent(_make_config(), vad=mock_vad)

        frames = [_make_audio_frame(amplitude=100 + i) for i in range(idle_frames + 4)]
        participant, mock_stream, sent = self._run_pipeline(agent, frames)

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream
        ):
            await asyncio.wait_for(
                agent._run_transcription_pipeline(participant, MagicMock(), "en"),
                timeout=5.0,
            )

        segments = _segments_from_sent(sent)
        assert len(segments) == 1, f"expected one segment, got {len(segments)}"
        seg = segments[0]

        frame_bytes = 320
        onset_bytes = int(onset_secs * 16000) * 2
        full_audio = b"".join(f.data for f in frames)

        # The segment must be a contiguous suffix of the input audio (replay
        # directly precedes the live frames, no gap) …
        assert full_audio.endswith(seg), "segment audio must be a contiguous suffix"
        # … and bounded: onset replay + the few frames after the VAD fired —
        # NOT the ~32 kB of idle audio sitting in the oversized buffer.
        max_live_frames = 6
        assert len(seg) <= onset_bytes + max_live_frames * frame_bytes, (
            f"fresh onset replayed {len(seg)} bytes; the onset cap is "
            f"{onset_bytes} bytes — the long split-overlap buffer must not be "
            f"replayed on a fresh onset"
        )


# ── Commit gate ────────────────────────────────────────────────────────────────


class _ScheduledVadStream:
    """VAD double firing scripted events after N pushed frames."""

    def __init__(self, schedule):
        # schedule: list of (after_pushed_frames, VADEventType)
        self._schedule = list(schedule)
        self.pushed = 0

    def push_frame(self, _frame):
        self.pushed += 1

    def end_input(self):
        pass

    async def aclose(self):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            if not self._schedule:
                await asyncio.Event().wait()  # block until cancelled
            after, ev_type = self._schedule[0]
            if self.pushed >= after:
                self._schedule.pop(0)
                return _make_vad_event(ev_type)
            await asyncio.sleep(0)


class TestCommitGate:
    """vLLM silently drops a commit sent while the previous segment's
    generation is still running ("Generation already in progress, ignoring
    commit") — the segment then never streams, or loses its transcription.done
    entirely. Opening commits must wait for the previous segment's done."""

    _RELEASE_MARKER = {"type": "_test_done_released"}

    def _wire(self, agent, frames, script):
        """Wire agent to a scripted WS; returns (participant, stream, sent).

        script: list of (condition, message, mark) — receive() serves each
        message once its condition (over `sent`) holds; mark=True appends
        _RELEASE_MARKER to `sent` first, recording the release moment in the
        send/receive order.
        """
        participant = MagicMock(spec=rtc.RemoteParticipant)
        participant.identity = "user_gate"

        sent: list[dict] = []

        async def _send_json(data):
            sent.append(data)
            await asyncio.sleep(0)

        closed_msg = MagicMock()
        closed_msg.type = aiohttp.WSMsgType.CLOSED
        script_iter = iter(script)

        async def _receive(*args, **kwargs):
            try:
                cond, msg, mark = next(script_iter)
            except StopIteration:
                return closed_msg
            while not cond():
                await asyncio.sleep(0)
            if mark:
                sent.append(dict(self._RELEASE_MARKER))
            return msg

        mock_ws = AsyncMock()
        mock_ws.receive = _receive
        mock_ws.send_json = _send_json
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_ws)
        cm.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.ws_connect = MagicMock(return_value=cm)
        agent._http_session = mock_session

        audio_events = [MagicMock(frame=f) for f in frames]
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter(audio_events)
        mock_stream.aclose = AsyncMock()

        return participant, mock_stream, sent

    @staticmethod
    def _openers(sent):
        return [
            i
            for i, m in enumerate(sent)
            if m.get("type") == "input_audio_buffer.commit" and "final" not in m
        ]

    @staticmethod
    def _closers(sent):
        return [
            i
            for i, m in enumerate(sent)
            if m.get("type") == "input_audio_buffer.commit" and m.get("final") is True
        ]

    def _marker_index(self, sent):
        return next(i for i, m in enumerate(sent) if m == self._RELEASE_MARKER)

    async def test_opener_waits_for_previous_done_and_loses_no_audio(self, monkeypatch):
        """
        Regression for the dropped-commit degradation: after a max-buffer
        split, the reopen's opening commit must NOT be sent until the previous
        segment's transcription.done has been read — vLLM ignores commits
        during an in-flight generation, which cost the reopened segment its
        streaming (and, on a swallowed closer, its FINAL). The audio arriving
        during the wait must be buffered and replayed, not dropped.
        """
        import providers.voxtral_realtime as vr

        monkeypatch.setattr(vr, "_MAX_BUFFER_DURATION_S", 0.015)

        vad_stream = _ScheduledVadStream([(1, agents_vad.VADEventType.START_OF_SPEECH)])
        mock_vad = MagicMock()
        mock_vad.stream.return_value = vad_stream
        agent = VoxtralRealtimeSttAgent(_make_config(), vad=mock_vad)

        frames = [_make_audio_frame(amplitude=100 + i) for i in range(8)]

        # Hold segment 1's transcription until well after the split close
        # (6 frames pushed), so an ungated reopen would fire first.
        sent_holder: dict = {}

        def _cond_done1():
            sent = sent_holder["sent"]
            return vad_stream.pushed >= 6 and self._closers(sent)

        def _cond_seg2():
            return len(self._openers(sent_holder["sent"])) >= 2

        script = [
            (lambda: True, _text_ws_msg({"type": "session.created"}), False),
            (
                _cond_done1,
                _text_ws_msg({"type": "transcription.delta", "delta": "hello"}),
                True,
            ),
            (
                lambda: True,
                _text_ws_msg({"type": "transcription.done", "text": "hello"}),
                False,
            ),
            (
                _cond_seg2,
                _text_ws_msg({"type": "transcription.delta", "delta": "world"}),
                False,
            ),
            (
                lambda: True,
                _text_ws_msg({"type": "transcription.done", "text": "world"}),
                False,
            ),
        ]

        participant, mock_stream, sent2 = self._wire(agent, frames, script)
        sent_holder["sent"] = sent2

        final = []
        agent.on("final_transcript", lambda **kw: final.append(kw))

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream
        ):
            await asyncio.wait_for(
                agent._run_transcription_pipeline(participant, MagicMock(), "en"),
                timeout=5.0,
            )
        await asyncio.sleep(0)

        openers = self._openers(sent2)
        assert len(openers) >= 2, f"expected a reopen, got {len(openers)} opener(s)"
        marker = self._marker_index(sent2)
        assert openers[1] > marker, (
            "the reopen's opening commit must be deferred until the previous "
            "segment's transcription.done has been received — an earlier "
            "commit is silently dropped by vLLM"
        )

        # No audio lost while gated: the frames that arrived while the opener
        # was deferred (the gate window right after the split close) must be
        # replayed contiguously into the reopened segment.
        segments = _segments_from_sent(sent2)
        assert len(segments) >= 2
        gated_audio = b"".join(f.data for f in frames[2:5])
        assert gated_audio in segments[1], (
            "audio arriving while the opener was gated must be buffered and "
            "replayed into the reopened segment, not dropped"
        )

        texts = [kw["event"].alternatives[0].text for kw in final]
        assert texts == ["hello", "world"]

    async def test_utterance_ending_while_gated_is_still_sent(self):
        """
        A short utterance that starts AND ends while the previous segment's
        done is outstanding must still be sent (open + close) once the server
        frees up — not stay buffered forever or be dropped.
        """
        vad_stream = _ScheduledVadStream(
            [
                (1, agents_vad.VADEventType.START_OF_SPEECH),
                (3, agents_vad.VADEventType.END_OF_SPEECH),
                (6, agents_vad.VADEventType.START_OF_SPEECH),
                (9, agents_vad.VADEventType.END_OF_SPEECH),
            ]
        )
        mock_vad = MagicMock()
        mock_vad.stream.return_value = vad_stream
        agent = VoxtralRealtimeSttAgent(_make_config(), vad=mock_vad)

        frames = [_make_audio_frame(amplitude=100 + i) for i in range(16)]

        sent_holder: dict = {}

        def _cond_done1():
            sent = sent_holder["sent"]
            return vad_stream.pushed >= 12 and self._closers(sent)

        def _cond_seg2():
            return len(self._openers(sent_holder["sent"])) >= 2

        script = [
            (lambda: True, _text_ws_msg({"type": "session.created"}), False),
            (
                _cond_done1,
                _text_ws_msg({"type": "transcription.delta", "delta": "hello"}),
                True,
            ),
            (
                lambda: True,
                _text_ws_msg({"type": "transcription.done", "text": "hello"}),
                False,
            ),
            (
                _cond_seg2,
                _text_ws_msg({"type": "transcription.delta", "delta": "world"}),
                False,
            ),
            (
                lambda: True,
                _text_ws_msg({"type": "transcription.done", "text": "world"}),
                False,
            ),
        ]

        participant, mock_stream, sent = self._wire(agent, frames, script)
        sent_holder["sent"] = sent

        final = []
        agent.on("final_transcript", lambda **kw: final.append(kw))

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream
        ):
            await asyncio.wait_for(
                agent._run_transcription_pipeline(participant, MagicMock(), "en"),
                timeout=5.0,
            )
        await asyncio.sleep(0)

        openers = self._openers(sent)
        closers = self._closers(sent)
        assert len(openers) == 2 and len(closers) == 2, (
            f"gated utterance must be opened and closed once the server frees "
            f"up — got {len(openers)} openers / {len(closers)} closers"
        )
        marker = self._marker_index(sent)
        assert openers[1] > marker, (
            "utterance 2's opener must wait for utterance 1's done"
        )
        texts = [kw["event"].alternatives[0].text for kw in final]
        assert texts == ["hello", "world"]

    async def test_gate_timeout_resyncs_and_opens(self, monkeypatch, caplog):
        """
        When a transcription.done never arrives (swallowed closing commit,
        already-desynced session), the gate must not wedge the pipeline: after
        the timeout it resyncs the outstanding counter and opens ungated —
        degrading to the pre-gate behavior instead of buffering forever.
        """
        import providers.voxtral_realtime as vr

        monkeypatch.setattr(vr, "_MAX_BUFFER_DURATION_S", 0.015)
        monkeypatch.setattr(vr, "_OPEN_GATE_TIMEOUT_S", 0.0)

        vad_stream = _ScheduledVadStream([(1, agents_vad.VADEventType.START_OF_SPEECH)])
        mock_vad = MagicMock()
        mock_vad.stream.return_value = vad_stream
        agent = VoxtralRealtimeSttAgent(_make_config(), vad=mock_vad)

        frames = [_make_audio_frame(amplitude=100 + i) for i in range(8)]
        script = [
            (lambda: True, _text_ws_msg({"type": "session.created"}), False),
            # No transcription events ever — the done is lost.
        ]
        participant, mock_stream, sent = self._wire(agent, frames, script)

        with (
            patch(
                "providers.voxtral_realtime.rtc.AudioStream",
                return_value=mock_stream,
            ),
            caplog.at_level("WARNING"),
        ):
            await asyncio.wait_for(
                agent._run_transcription_pipeline(participant, MagicMock(), "en"),
                timeout=5.0,
            )

        assert "open gate timed out" in caplog.text
        assert len(self._openers(sent)) >= 2, (
            "after the gate timeout the next segment must still open"
        )


# ── Failure recovery and teardown flush ────────────────────────────────────────


class _ScriptedWs:
    """Minimal WS double: serves a fixed message list, then blocks until closed.

    send_json raises ClientError once the socket is closed — mirroring aiohttp,
    so the writer's send failure is what surfaces a reader-initiated close.
    """

    def __init__(self, messages):
        self._messages = list(messages)
        self.closed = False
        self.sent: list[dict] = []

    async def receive(self, *args, **kwargs):
        await asyncio.sleep(0)
        if self._messages:
            return self._messages.pop(0)
        while not self.closed:
            await asyncio.sleep(0)
        msg = MagicMock()
        msg.type = aiohttp.WSMsgType.CLOSED
        return msg

    async def send_json(self, data):
        if self.closed:
            raise aiohttp.ClientError("socket closed")
        self.sent.append(data)
        await asyncio.sleep(0)

    async def close(self):
        self.closed = True


class _EndlessAudioStream:
    """Async audio stream that yields loud frames until abandoned."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        return MagicMock(frame=_make_loud_frame())

    async def aclose(self):
        pass


def _ws_context(ws):
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


class TestReaderFailureRecovery:
    async def test_server_error_event_closes_ws_and_reconnects(self, monkeypatch):
        """
        Regression for silent transcription death: a server `error` event makes
        the reader exit while the connection stays alive. The reader must close
        the socket so the writer's sends fail and the pipeline reconnects —
        otherwise audio keeps streaming into a session nobody reads and every
        subsequent utterance is lost without a trace.
        """
        import providers.voxtral_realtime as vr

        monkeypatch.setattr(vr, "_RETRY_DELAY_INITIAL_S", 0.01)

        agent = _make_agent(
            vad_events=[_make_vad_event(agents_vad.VADEventType.START_OF_SPEECH)],
        )
        participant = MagicMock(spec=rtc.RemoteParticipant)
        participant.identity = "user_err"

        ws1 = _ScriptedWs(
            [
                _text_ws_msg({"type": "session.created"}),
                _text_ws_msg({"type": "error", "error": "boom"}),
            ]
        )
        ws2 = _ScriptedWs([_text_ws_msg({"type": "session.created"})])

        mock_session = MagicMock()
        mock_session.ws_connect = MagicMock(
            side_effect=[_ws_context(ws1), _ws_context(ws2)]
        )
        agent._http_session = mock_session

        # First connection: endless frames so the writer keeps sending until
        # the reader-initiated close makes a send fail. Second connection:
        # empty stream so the pipeline exits cleanly.
        empty_stream = AsyncMock()
        empty_stream.__aiter__.return_value = iter([])
        empty_stream.aclose = AsyncMock()

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream",
            side_effect=[_EndlessAudioStream(), empty_stream],
        ):
            await asyncio.wait_for(
                agent._run_transcription_pipeline(participant, MagicMock(), "en"),
                timeout=5.0,
            )

        assert ws1.closed, "reader must close the WS after a server error event"
        assert mock_session.ws_connect.call_count == 2, (
            "pipeline must reconnect after the reader-initiated close"
        )


class TestSessionUpdate:
    async def test_sends_flat_model_and_greedy_temperature(self):
        """vLLM requires a FLAT session.update (nesting under "session" is
        rejected with "Missing required field: model" — probe test 6), and
        the model card mandates temperature 0.0 for stable transcription."""
        agent = _make_agent(vad_events=[])
        participant = MagicMock(spec=rtc.RemoteParticipant)
        participant.identity = "user_cfg"

        ws = _ScriptedWs([_text_ws_msg({"type": "session.created"})])
        mock_session = MagicMock()
        mock_session.ws_connect = MagicMock(return_value=_ws_context(ws))
        agent._http_session = mock_session

        empty = AsyncMock()
        empty.__aiter__.return_value = iter([])
        empty.aclose = AsyncMock()

        with patch("providers.voxtral_realtime.rtc.AudioStream", return_value=empty):
            await asyncio.wait_for(
                agent._run_transcription_pipeline(participant, MagicMock(), "en"),
                timeout=5.0,
            )

        updates = [m for m in ws.sent if m.get("type") == "session.update"]
        assert updates == [
            {
                "type": "session.update",
                "model": agent.config.model,
                "temperature": 0.0,
            }
        ]


class TestHandshakeTimeout:
    async def test_slow_session_created_retries_instead_of_giving_up(self, monkeypatch):
        """
        Regression for permanent give-up during server warmup: vLLM takes
        minutes of CUDA-graph warmup after startup, during which the WS may
        connect but session.created arrives late. A handshake timeout must
        retry with backoff like a connection error — not end transcription
        for the participant's whole meeting.
        """
        import providers.voxtral_realtime as vr

        monkeypatch.setattr(vr, "_RETRY_DELAY_INITIAL_S", 0.01)

        agent = _make_agent(vad_events=[])
        participant = MagicMock(spec=rtc.RemoteParticipant)
        participant.identity = "user_warmup"

        # First connection: session.created never arrives (handshake timeout).
        slow_ws = AsyncMock()
        slow_ws.receive = AsyncMock(side_effect=asyncio.TimeoutError)
        ws2 = _ScriptedWs([_text_ws_msg({"type": "session.created"})])

        mock_session = MagicMock()
        mock_session.ws_connect = MagicMock(
            side_effect=[_ws_context(slow_ws), _ws_context(ws2)]
        )
        agent._http_session = mock_session

        empty = AsyncMock()
        empty.__aiter__.return_value = iter([])
        empty.aclose = AsyncMock()
        empty2 = AsyncMock()
        empty2.__aiter__.return_value = iter([])
        empty2.aclose = AsyncMock()

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream", side_effect=[empty, empty2]
        ):
            await asyncio.wait_for(
                agent._run_transcription_pipeline(participant, MagicMock(), "en"),
                timeout=5.0,
            )

        assert mock_session.ws_connect.call_count == 2, (
            "a handshake timeout must reconnect with backoff, not give up"
        )


class TestLocaleUpdateRace:
    async def test_locale_change_does_not_orphan_replacement_pipeline(self):
        """
        Regression for the stop→start restart race: _update_stream_locale
        cancels the old pipeline task and synchronously registers a new one
        under the same identity. task.cancel() only schedules the
        cancellation, so the old task's cleanup runs AFTER the new entry
        exists — an unconditional pop there deregisters the replacement,
        leaving it running but untracked (unstoppable, and a later start
        would spawn a duplicate pipeline on the same track).
        """
        participant = _make_participant("user_1")
        agent = _make_agent_with_room(participants={"p1": participant})

        ws1 = _ScriptedWs([_text_ws_msg({"type": "session.created"})])
        ws2 = _ScriptedWs([_text_ws_msg({"type": "session.created"})])
        mock_session = MagicMock()
        mock_session.ws_connect = MagicMock(
            side_effect=[_ws_context(ws1), _ws_context(ws2)]
        )
        agent._http_session = mock_session

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream",
            side_effect=[_EndlessAudioStream(), _EndlessAudioStream()],
        ):
            agent.start_transcription_for_user("user_1", "en-US", "voxtral-realtime")
            task1 = agent.processing_info["user_1"]["task"]
            await asyncio.sleep(0.05)  # let pipeline 1 get going

            agent._update_stream_locale("user_1", "de-DE")
            task2 = agent.processing_info["user_1"]["task"]
            assert task2 is not task1

            # Let the cancelled task run its cleanup to completion.
            await asyncio.wait_for(task1, timeout=5.0)
            await asyncio.sleep(0)

            assert agent.processing_info.get("user_1", {}).get("task") is task2, (
                "the cancelled pipeline's cleanup must not deregister its replacement"
            )

            agent.stop_transcription_for_user("user_1")
            await asyncio.wait_for(task2, timeout=5.0)
            assert "user_1" not in agent.processing_info


class TestTeardownFlush:
    async def test_cancel_mid_utterance_emits_synthetic_final_and_closing_commit(
        self,
    ):
        """
        Regression for the speak-then-mute loss: when frames stop before Silero
        fires END_OF_SPEECH and the task is cancelled (mute / track
        unsubscribed), the open segment must still (a) send its closing
        commit(final) so the server request is not left dangling, and (b) emit
        a FINAL from the delta text already received — otherwise BBB keeps the
        interim caption pending forever and the utterance is lost.
        """
        agent = _make_agent(
            interim_results=True,
            vad_events=[_make_vad_event(agents_vad.VADEventType.START_OF_SPEECH)],
        )
        participant = MagicMock(spec=rtc.RemoteParticipant)
        participant.identity = "user_mute"

        # session.created + one delta; transcription.done never arrives.
        ws = _ScriptedWs(
            [
                _text_ws_msg({"type": "session.created"}),
                _text_ws_msg({"type": "transcription.delta", "delta": "hello"}),
            ]
        )
        mock_session = MagicMock()
        mock_session.ws_connect = MagicMock(return_value=_ws_context(ws))
        agent._http_session = mock_session

        interim = []
        final = []
        agent.on("interim_transcript", lambda **kw: interim.append(kw))
        agent.on("final_transcript", lambda **kw: final.append(kw))

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream",
            return_value=_EndlessAudioStream(),
        ):
            task = asyncio.create_task(
                agent._run_transcription_pipeline(participant, MagicMock(), "en")
            )
            # Wait until the delta has been received and emitted as interim.
            for _ in range(500):
                if interim:
                    break
                await asyncio.sleep(0.01)
            assert interim, "expected an interim before cancelling"

            task.cancel()
            await asyncio.wait_for(task, timeout=5.0)

        # Let the emit task scheduled during teardown run.
        for _ in range(5):
            await asyncio.sleep(0)

        texts = [kw["event"].alternatives[0].text for kw in final]
        assert texts == ["hello"], (
            f"cancellation mid-utterance must emit a best-effort FINAL from the "
            f"accumulated delta text, got {texts}"
        )
        closers = [m for m in ws.sent if m.get("final") is True]
        assert closers, "cancellation with an open segment must send commit(final=True)"

    async def test_end_of_stream_waits_for_late_transcription_done(self):
        """
        Regression for the tail-utterance drop at clean stream end: the server's
        transcription.done for the flushed segment arrives after the writer has
        finished. The reader must be drained, not cancelled immediately, so the
        tail utterance still gets its real FINAL.
        """
        agent = _make_agent(
            interim_results=True,
            vad_events=[_make_vad_event(agents_vad.VADEventType.START_OF_SPEECH)],
        )
        participant = MagicMock(spec=rtc.RemoteParticipant)
        participant.identity = "user_tail"

        ws = _ScriptedWs([_text_ws_msg({"type": "session.created"})])

        # Release the transcription only after the writer has sent the closing
        # commit(final) — mirroring real server causality at end of stream.
        real_receive = ws.receive
        released = False

        async def _receive(*args, **kwargs):
            nonlocal released
            if not ws._messages and not ws.closed and not released:
                while not any(m.get("final") is True for m in ws.sent):
                    await asyncio.sleep(0)
                released = True
                ws._messages = [
                    _text_ws_msg({"type": "transcription.delta", "delta": "tail"}),
                    _text_ws_msg({"type": "transcription.done", "text": "tail words"}),
                ]
            return await real_receive()

        ws.receive = _receive

        mock_session = MagicMock()
        mock_session.ws_connect = MagicMock(return_value=_ws_context(ws))
        agent._http_session = mock_session

        audio_events = [MagicMock(frame=_make_loud_frame()) for _ in range(3)]
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter(audio_events)
        mock_stream.aclose = AsyncMock()

        final = []
        agent.on("final_transcript", lambda **kw: final.append(kw))

        with patch(
            "providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream
        ):
            await asyncio.wait_for(
                agent._run_transcription_pipeline(participant, MagicMock(), "en"),
                timeout=5.0,
            )
        await asyncio.sleep(0)

        texts = [kw["event"].alternatives[0].text for kw in final]
        assert texts == ["tail words"], (
            f"the tail segment's late transcription.done must still be read "
            f"before teardown, got {texts}"
        )
