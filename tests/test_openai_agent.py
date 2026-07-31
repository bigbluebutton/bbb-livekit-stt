import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from livekit import rtc
from livekit.agents import stt

from providers.openai import (
    OpenAiConfig,
    OpenAiSttAgent,
    _SILENCE_THRESHOLD_RMS,
)


def _make_agent(**kwargs):
    config = OpenAiConfig(api_key="fake-key", **kwargs)
    return OpenAiSttAgent(config)


def _make_agent_with_room(participants=None, **kwargs):
    agent = _make_agent(**kwargs)
    mock_room = MagicMock()
    mock_room.remote_participants = participants or {}
    agent.room = mock_room
    return agent


def _make_participant(identity, audio_track=None):
    participant = MagicMock(spec=rtc.RemoteParticipant)
    participant.identity = identity
    pubs = {}
    if audio_track:
        pub = MagicMock()
        pub.track = audio_track
        pub.track.kind = rtc.TrackKind.KIND_AUDIO
        pubs["audio"] = pub
    participant.track_publications = pubs
    return participant


def _make_audio_event(amplitude: int = 0) -> MagicMock:
    """Create a mock audio event with PCM bytes at the given amplitude."""
    samples = np.full(160, amplitude, dtype=np.int16)
    event = MagicMock()
    event.frame.data = samples.tobytes()
    event.frame.sample_rate = 16000
    event.frame.samples_per_channel = 160
    return event


def _make_loud_event() -> MagicMock:
    """Audio event with RMS energy above the speech threshold."""
    return _make_audio_event(amplitude=int(_SILENCE_THRESHOLD_RMS * 2))


class TestOpenAiConfigDefaults:
    @pytest.fixture(autouse=True)
    def _clean_openai_env(self, monkeypatch):
        for key in list(__import__("os").environ):
            if key.startswith("OPENAI_"):
                monkeypatch.delenv(key, raising=False)

    def test_model_defaults_to_gpt4o_transcribe(self):
        assert OpenAiConfig().model == "gpt-4o-transcribe"

    def test_api_key_defaults_to_none(self):
        assert OpenAiConfig().api_key is None

    def test_base_url_defaults_to_none(self):
        assert OpenAiConfig().base_url is None


class TestUpdateLocaleForUser:
    def test_updates_locale_in_participant_settings(self):
        agent = _make_agent_with_room()
        agent.participant_settings["user_1"] = {"locale": "en", "provider": "openai"}

        agent.update_locale_for_user("user_1", "fr")

        assert agent.participant_settings["user_1"]["locale"] == "fr"

    def test_restarts_transcription_when_active(self):
        """OpenAI REST requires stop+restart to change locale."""
        mock_track = MagicMock()
        mock_track.kind = rtc.TrackKind.KIND_AUDIO
        participant = _make_participant("user_1", audio_track=mock_track)
        agent = _make_agent_with_room(participants={"pid": participant})
        agent.participant_settings["user_1"] = {"locale": "en", "provider": "openai"}
        agent.processing_info["user_1"] = {"task": MagicMock()}

        with (
            patch.object(agent, "stop_transcription_for_user") as mock_stop,
            patch.object(agent, "start_transcription_for_user") as mock_start,
        ):
            agent.update_locale_for_user("user_1", "de")

        mock_stop.assert_called_once_with("user_1")
        mock_start.assert_called_once_with("user_1", "de", "openai")

    def test_does_not_restart_when_no_active_transcription(self):
        agent = _make_agent_with_room()
        agent.participant_settings["user_1"] = {"locale": "en", "provider": "openai"}

        with (
            patch.object(agent, "stop_transcription_for_user") as mock_stop,
            patch.object(agent, "start_transcription_for_user") as mock_start,
        ):
            agent.update_locale_for_user("user_1", "fr")

        mock_stop.assert_not_called()
        mock_start.assert_not_called()
        assert agent.participant_settings["user_1"]["locale"] == "fr"


class TestStartTranscriptionForUser:
    async def test_passes_sanitized_locale_to_pipeline(self):
        """Locale 'pt-BR' should be sanitized to 'pt' when starting the pipeline."""
        mock_track = MagicMock()
        mock_track.kind = rtc.TrackKind.KIND_AUDIO
        participant = _make_participant("user_1", audio_track=mock_track)
        agent = _make_agent_with_room(participants={"pid": participant})

        with patch.object(
            agent, "_run_transcription_pipeline", new_callable=AsyncMock
        ) as mock_pipeline:
            agent.start_transcription_for_user("user_1", "pt-BR", "openai")
            await asyncio.sleep(0)

        mock_pipeline.assert_called_once_with(participant, mock_track, "pt")
        agent.processing_info.pop("user_1", None)

    async def test_processing_info_has_no_stream_key(self):
        """REST mode stores only 'task' in processing_info."""
        mock_track = MagicMock()
        mock_track.kind = rtc.TrackKind.KIND_AUDIO
        participant = _make_participant("user_1", audio_track=mock_track)
        agent = _make_agent_with_room(participants={"pid": participant})

        with patch.object(agent, "_run_transcription_pipeline", new_callable=AsyncMock):
            agent.start_transcription_for_user("user_1", "en", "openai")

        assert "task" in agent.processing_info["user_1"]
        assert "stream" not in agent.processing_info["user_1"]
        agent.processing_info.pop("user_1", None)


class TestRunTranscriptionPipeline:
    async def test_cancellation_cleans_up_processing_info(self):
        """CancelledError should be caught and processing_info entry removed."""
        agent = _make_agent()
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.side_effect = asyncio.CancelledError

        agent.processing_info["user_1"] = {"task": MagicMock()}

        with patch("providers.openai.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(mock_participant, mock_track, "en")

        assert "user_1" not in agent.processing_info

    async def test_emits_final_transcript_for_speech_frames(self):
        """Speech frames trigger a final_transcript event via REST API."""
        agent = _make_agent()
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        # One loud audio frame followed by end-of-stream triggers end-of-stream flush
        loud_event = _make_loud_event()
        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([loud_event])

        emitted = []
        agent.on("final_transcript", lambda **kw: emitted.append(kw))
        agent._transcribe_wav = AsyncMock(return_value="hello world")

        with patch("providers.openai.rtc.AudioStream", return_value=mock_audio_stream):
            with patch("providers.openai.rtc.combine_audio_frames"):
                await agent._run_transcription_pipeline(
                    mock_participant, mock_track, "en"
                )
            await asyncio.sleep(0)

        assert len(emitted) == 1
        assert emitted[0]["participant"] is mock_participant
        event = emitted[0]["event"]
        assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        assert event.alternatives[0].text == "hello world"

    async def test_does_not_emit_for_empty_transcript(self):
        """No event emitted when REST returns empty text."""
        agent = _make_agent()
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        loud_event = _make_loud_event()
        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([loud_event])

        emitted = []
        agent.on("final_transcript", lambda **kw: emitted.append(kw))
        agent._transcribe_wav = AsyncMock(return_value="")

        with patch("providers.openai.rtc.AudioStream", return_value=mock_audio_stream):
            with patch("providers.openai.rtc.combine_audio_frames"):
                await agent._run_transcription_pipeline(
                    mock_participant, mock_track, "en"
                )

        assert len(emitted) == 0

    async def test_does_not_call_transcribe_for_silent_audio(self):
        """Silent frames (below energy threshold) should not trigger REST calls."""
        agent = _make_agent()
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        silent_event = _make_audio_event(amplitude=0)
        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([silent_event])

        agent._transcribe_wav = AsyncMock()

        with patch("providers.openai.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(mock_participant, mock_track, "en")

        agent._transcribe_wav.assert_not_called()

    async def test_generic_exception_cleans_up_processing_info(self):
        """Unexpected exceptions should be caught and processing_info cleaned up."""
        agent = _make_agent()
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.side_effect = RuntimeError("boom")

        agent.processing_info["user_1"] = {"task": MagicMock()}

        with patch("providers.openai.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(mock_participant, mock_track, "en")

        assert "user_1" not in agent.processing_info

    async def test_segment_has_start_and_end_times(self):
        """Emitted events must have non-zero start_time and end_time on SpeechData."""
        agent = _make_agent()
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        loud_event = _make_loud_event()
        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([loud_event])

        emitted = []
        agent.on("final_transcript", lambda **kw: emitted.append(kw))
        agent._transcribe_wav = AsyncMock(return_value="hello")

        with patch("providers.openai.rtc.AudioStream", return_value=mock_audio_stream):
            with patch("providers.openai.rtc.combine_audio_frames"):
                await agent._run_transcription_pipeline(
                    mock_participant, mock_track, "en"
                )
            await asyncio.sleep(0)

        assert len(emitted) == 1
        alt = emitted[0]["event"].alternatives[0]
        assert alt.start_time >= 0.0
        assert alt.end_time >= alt.start_time


class TestCleanup:
    async def test_closes_http_session_on_cleanup(self):
        """_cleanup() should close the aiohttp session."""
        agent = _make_agent()
        mock_session = AsyncMock()
        agent._http_session = mock_session

        await agent._cleanup()

        mock_session.close.assert_called_once()
        assert agent._http_session is None


def _make_http_session(payload=None):
    """Mock aiohttp session whose post() works as an async context manager."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = AsyncMock(return_value=payload or {"text": "hello"})

    post_cm = MagicMock()
    post_cm.__aenter__ = AsyncMock(return_value=response)
    post_cm.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.post = MagicMock(return_value=post_cm)
    return session


async def _transcribe_with_mocked_form(agent, language):
    """Run _transcribe_wav against a mocked session, returning the form fields."""
    mock_form = MagicMock()
    agent._get_http_session = MagicMock(return_value=_make_http_session())

    with patch("providers.openai.aiohttp.FormData", return_value=mock_form):
        await agent._transcribe_wav(b"fake-wav-bytes", language)

    return {call.args[0]: call.args[1] for call in mock_form.add_field.call_args_list}


class TestAutoLocale:
    """The 'auto' locale sanitizes to None; it must never reach the API."""

    async def test_start_transcription_passes_none_language_to_pipeline(self):
        mock_track = MagicMock()
        mock_track.kind = rtc.TrackKind.KIND_AUDIO
        participant = _make_participant("user_1", audio_track=mock_track)
        agent = _make_agent_with_room(participants={"pid": participant})

        with patch.object(
            agent, "_run_transcription_pipeline", new_callable=AsyncMock
        ) as mock_pipeline:
            agent.start_transcription_for_user("user_1", "auto", "openai")
            await asyncio.sleep(0)

        mock_pipeline.assert_called_once_with(participant, mock_track, None)
        agent.processing_info.pop("user_1", None)

    async def test_transcribe_wav_omits_the_language_field(self):
        """Omitting the field is what makes OpenAI detect the language itself."""
        fields = await _transcribe_with_mocked_form(_make_agent(), None)

        assert "language" not in fields
        assert fields["model"] == "gpt-4o-transcribe"
        assert fields["response_format"] == "json"

    async def test_transcribe_wav_sends_the_language_field_when_known(self):
        fields = await _transcribe_with_mocked_form(_make_agent(), "pt")

        assert fields["language"] == "pt"

    async def test_emitted_transcript_carries_no_language(self):
        """Under 'auto' the provider cannot label the transcript, so main.py
        has to resolve the BBB locale without one."""
        agent = _make_agent()
        participant = MagicMock(spec=rtc.RemoteParticipant)
        participant.identity = "user_1"

        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([_make_loud_event()])

        emitted = []
        agent.on("final_transcript", lambda **kw: emitted.append(kw))
        agent._transcribe_wav = AsyncMock(return_value="olá mundo")

        with patch("providers.openai.rtc.AudioStream", return_value=mock_audio_stream):
            with patch("providers.openai.rtc.combine_audio_frames"):
                await agent._run_transcription_pipeline(participant, MagicMock(), None)
            await asyncio.sleep(0)

        assert len(emitted) == 1
        assert emitted[0]["event"].alternatives[0].language is None
