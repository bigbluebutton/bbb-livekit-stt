import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import numpy as np
import pytest
from livekit import rtc
from livekit.agents import stt

from providers.voxtral_realtime import (
    VoxtralRealtimeConfig,
    VoxtralRealtimeSttAgent,
    _SILENCE_THRESHOLD_RMS,
    _to_pcm16_16k,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_config(**kwargs):
    return VoxtralRealtimeConfig(api_key="test-key", **kwargs)


def _make_agent(**kwargs):
    return VoxtralRealtimeSttAgent(_make_config(**kwargs))


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
    return _make_audio_frame(amplitude=int(_SILENCE_THRESHOLD_RMS * 2))


# ── Config ─────────────────────────────────────────────────────────────────────


class TestVoxtralRealtimeConfig:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for key in ["VOXTRAL_API_KEY", "VOXTRAL_MODEL", "VOXTRAL_BASE_URL", "VOXTRAL_INTERIM_RESULTS"]:
            monkeypatch.delenv(key, raising=False)

    def test_default_model(self):
        assert VoxtralRealtimeConfig().model == "mistralai/Voxtral-Mini-4B-Realtime-2602"

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
    def test_default_url(self):
        agent = _make_agent()
        assert agent._build_ws_url() == "wss://api.openai.com/v1/realtime?intent=transcription"

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


class TestToPcm16_16k:
    def test_returns_bytes(self):
        frame = _make_audio_frame()
        assert isinstance(_to_pcm16_16k(frame), bytes)

    def test_mono_16k_passthrough_preserves_values(self):
        frame = _make_audio_frame(amplitude=1000, sample_rate=16000, num_channels=1)
        result = _to_pcm16_16k(frame)
        samples = np.frombuffer(result, dtype=np.int16)
        assert len(samples) == frame.samples_per_channel
        assert all(s == 1000 for s in samples)

    def test_stereo_downmix_to_mono(self):
        """Stereo frame with equal channels averages to same amplitude."""
        frame = _make_audio_frame(amplitude=1000, sample_rate=16000, num_channels=2)
        result = _to_pcm16_16k(frame)
        samples = np.frombuffer(result, dtype=np.int16)
        assert len(samples) == frame.samples_per_channel
        assert all(s == 1000 for s in samples)

    def test_resampling_from_48k_produces_correct_length(self):
        frame = _make_audio_frame(amplitude=500, sample_rate=48000, num_channels=1)
        result = _to_pcm16_16k(frame)
        samples = np.frombuffer(result, dtype=np.int16)
        expected = round(frame.samples_per_channel * 16000 / 48000)
        assert len(samples) == expected

    def test_values_are_clipped_to_int16_range(self):
        """numpy clip must keep all output values within ±32767."""
        frame = _make_audio_frame(amplitude=0, sample_rate=16000, num_channels=1)
        # Override with float32 extremes stored as int16 (will saturate on cast)
        raw = np.array([32767, -32768, 0], dtype=np.float32)
        frame.data = raw.astype(np.int16).tobytes()
        frame.samples_per_channel = 3
        result = _to_pcm16_16k(frame)
        out = np.frombuffer(result, dtype=np.int16)
        assert all(-32768 <= s <= 32767 for s in out)


# ── start_transcription_for_user ───────────────────────────────────────────────


class TestStartTranscriptionForUser:
    def test_participant_not_found_logs_error(self, caplog):
        agent = _make_agent_with_room(participants={})
        with caplog.at_level("ERROR"):
            agent.start_transcription_for_user("ghost_user", "en-US", "voxtral-realtime")
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

        with patch.object(agent, "_run_transcription_pipeline", new_callable=AsyncMock) as mock_pipeline:
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

        with patch("providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream):
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

        with patch("providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream):
            await agent._run_transcription_pipeline(participant, MagicMock(), "en")

        assert "user_1" not in agent.processing_info


# ── _vad_loop — speech detection and final flush ──────────────────────────────


class TestVadLoop:
    def _full_pipeline_setup(self, audio_frames, ws_messages):
        """
        Return (agent, participant, mock_stream, mock_session) wired up so that
        _run_transcription_pipeline can run end-to-end through the VAD loop.
        ws_messages is a list appended after the mandatory session.created message.
        """
        agent = _make_agent(interim_results=True)
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
        """Only silence frames — no flush, no transcript event."""
        silence = _make_audio_frame(amplitude=0)
        agent, participant, mock_stream = self._full_pipeline_setup(
            audio_frames=[silence],
            ws_messages=[],
        )

        emitted = []
        agent.on("final_transcript", lambda **kw: emitted.append(kw))

        with patch("providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream):
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

        with patch("providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream):
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

        with patch("providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream):
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

        with patch("providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream):
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

        with patch("providers.voxtral_realtime.rtc.AudioStream", return_value=mock_stream):
            await agent._run_transcription_pipeline(participant, MagicMock(), "en")
        await asyncio.sleep(0)

        assert len(final) == 1
        start_times = {kw["event"].alternatives[0].start_time for kw in interim}
        start_times.add(final[0]["event"].alternatives[0].start_time)
        assert len(start_times) == 1, (
            f"all events for one utterance must share start_time, got {start_times}"
        )
