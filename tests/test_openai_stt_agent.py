import asyncio
import contextlib
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from livekit import rtc
from livekit.agents import stt

from config import OpenAiConfig
from openai_stt_agent import OpenAiSttAgent, _SILENCE_THRESHOLD_RMS


def _make_agent(interim_results=None, **kwargs):
    config = OpenAiConfig(api_key="fake-key", interim_results=interim_results, **kwargs)
    with patch("openai_stt_agent.openai_plugin") as mock_plugin:
        mock_plugin.STT.return_value = MagicMock()
        agent = OpenAiSttAgent(config)
    return agent


def _make_track_subscribed_args(source=rtc.TrackSource.SOURCE_MICROPHONE):
    mock_track = MagicMock()
    mock_publication = MagicMock()
    mock_publication.source = source
    mock_participant = MagicMock()
    return mock_track, mock_publication, mock_participant


def _make_agent_with_room(interim_results=None, participants=None, **kwargs):
    """Create an agent with a mocked room containing the given participants."""
    agent = _make_agent(interim_results=interim_results, **kwargs)
    mock_room = MagicMock()
    participants = participants or {}
    mock_room.remote_participants = participants
    agent.room = mock_room
    return agent


def _make_participant(identity, audio_track=None):
    """Create a mock RemoteParticipant with an optional audio track."""
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


class TestSanitizeLocale:
    def test_strips_region_from_bcp47_locale(self):
        agent = _make_agent()
        assert agent._sanitize_locale("en-US") == "en"
        assert agent._sanitize_locale("pt-BR") == "pt"
        assert agent._sanitize_locale("zh-CN") == "zh"
        assert agent._sanitize_locale("fr-FR") == "fr"

    def test_returns_language_code_unchanged_when_no_region(self):
        agent = _make_agent()
        assert agent._sanitize_locale("en") == "en"
        assert agent._sanitize_locale("de") == "de"

    def test_lowercases_language_code(self):
        agent = _make_agent()
        assert agent._sanitize_locale("EN-US") == "en"
        assert agent._sanitize_locale("PT") == "pt"


class TestStopTranscriptionForUser:
    def test_cancels_task_and_removes_from_processing_info(self):
        agent = _make_agent()
        mock_task = MagicMock()
        agent.processing_info["user_123"] = {"task": mock_task}

        agent.stop_transcription_for_user("user_123")

        mock_task.cancel.assert_called_once()
        assert "user_123" not in agent.processing_info

    def test_no_op_when_user_not_in_processing_info(self):
        agent = _make_agent()
        # Should not raise even if user_id is unknown
        agent.stop_transcription_for_user("unknown_user")


class TestUpdateLocaleForUser:
    def test_updates_locale_in_participant_settings(self):
        agent = _make_agent_with_room()
        agent.participant_settings["user_1"] = {"locale": "en", "provider": "openai"}

        agent.update_locale_for_user("user_1", "fr")

        assert agent.participant_settings["user_1"]["locale"] == "fr"

    def test_restarts_transcription_when_active(self):
        """OpenAI STT requires stop+restart to change locale (no update_options)."""
        mock_track = MagicMock()
        mock_track.kind = rtc.TrackKind.KIND_AUDIO
        participant = _make_participant("user_1", audio_track=mock_track)
        agent = _make_agent_with_room(participants={"pid": participant})
        agent.participant_settings["user_1"] = {"locale": "en", "provider": "openai"}
        mock_task = MagicMock()
        agent.processing_info["user_1"] = {"task": mock_task}

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

    def test_no_op_in_settings_when_user_not_in_participant_settings(self):
        agent = _make_agent_with_room()
        # Should not raise or create settings entry
        agent.update_locale_for_user("unknown_user", "en")
        assert "unknown_user" not in agent.participant_settings


class TestOnParticipantDisconnected:
    def test_stops_transcription_and_clears_settings(self):
        agent = _make_agent()
        agent.participant_settings["user_1"] = {"locale": "en", "provider": "openai"}
        mock_task = MagicMock()
        agent.processing_info["user_1"] = {"task": mock_task}

        mock_participant = MagicMock()
        mock_participant.identity = "user_1"

        agent._on_participant_disconnected(mock_participant)

        mock_task.cancel.assert_called_once()
        assert "user_1" not in agent.processing_info
        assert "user_1" not in agent.participant_settings

    def test_no_op_for_unknown_participant(self):
        agent = _make_agent()
        mock_participant = MagicMock()
        mock_participant.identity = "ghost_user"

        agent._on_participant_disconnected(mock_participant)

        assert "ghost_user" not in agent.participant_settings


class TestOnTrackSubscribed:
    def test_skips_non_microphone_tracks(self):
        agent = _make_agent()
        agent.participant_settings["user_1"] = {"locale": "en-US", "provider": "openai"}
        mock_track, mock_publication, mock_participant = _make_track_subscribed_args(
            source=rtc.TrackSource.SOURCE_CAMERA
        )
        mock_participant.identity = "user_1"

        with patch.object(agent, "start_transcription_for_user") as mock_start:
            agent._on_track_subscribed(mock_track, mock_publication, mock_participant)
            mock_start.assert_not_called()

    def test_skips_transcription_when_no_settings(self):
        agent = _make_agent()
        mock_track, mock_publication, mock_participant = _make_track_subscribed_args()
        mock_participant.identity = "user_no_settings"

        with patch.object(agent, "start_transcription_for_user") as mock_start:
            agent._on_track_subscribed(mock_track, mock_publication, mock_participant)
            mock_start.assert_not_called()

    def test_starts_transcription_when_locale_and_provider_present(self):
        agent = _make_agent()
        agent.participant_settings["user_1"] = {"locale": "en-US", "provider": "openai"}
        mock_track, mock_publication, mock_participant = _make_track_subscribed_args()
        mock_participant.identity = "user_1"

        with patch.object(agent, "start_transcription_for_user") as mock_start:
            agent._on_track_subscribed(mock_track, mock_publication, mock_participant)
            mock_start.assert_called_once_with("user_1", "en-US", "openai")


class TestStartTranscriptionForUser:
    def test_logs_error_when_participant_not_found(self, caplog):
        agent = _make_agent_with_room(participants={})

        with caplog.at_level(logging.ERROR):
            agent.start_transcription_for_user("missing_user", "en-US", "openai")

        assert "missing_user" not in agent.processing_info
        assert any("not found" in r.message for r in caplog.records)

    def test_logs_warning_when_no_audio_track(self, caplog):
        participant = _make_participant("user_1", audio_track=None)
        agent = _make_agent_with_room(participants={"pid": participant})

        with caplog.at_level(logging.WARNING):
            agent.start_transcription_for_user("user_1", "en-US", "openai")

        assert "user_1" not in agent.processing_info
        assert any("no audio track" in r.message for r in caplog.records)

    def test_skips_when_already_processing(self, caplog):
        mock_track = MagicMock()
        mock_track.kind = rtc.TrackKind.KIND_AUDIO
        participant = _make_participant("user_1", audio_track=mock_track)
        agent = _make_agent_with_room(participants={"pid": participant})

        existing_task = MagicMock()
        agent.processing_info["user_1"] = {"task": existing_task}

        with caplog.at_level(logging.DEBUG):
            agent.start_transcription_for_user("user_1", "en-US", "openai")

        assert agent.processing_info["user_1"]["task"] is existing_task

    async def test_creates_task_on_success(self):
        """Happy path: participant with audio track → task created."""
        mock_track = MagicMock()
        mock_track.kind = rtc.TrackKind.KIND_AUDIO
        participant = _make_participant("user_1", audio_track=mock_track)
        agent = _make_agent_with_room(participants={"pid": participant})

        with patch("openai_stt_agent.rtc.AudioStream"):
            agent.start_transcription_for_user("user_1", "en-US", "openai")

            assert "user_1" in agent.processing_info
            info = agent.processing_info["user_1"]
            assert "task" in info
            assert agent.participant_settings["user_1"]["locale"] == "en-US"
            assert agent.participant_settings["user_1"]["provider"] == "openai"

            info["task"].cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await info["task"]

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
            await asyncio.sleep(0)  # let the task start

        mock_pipeline.assert_called_once_with(participant, mock_track, "pt")

        # Clean up the task
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

        with patch("openai_stt_agent.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(mock_participant, mock_track, "en")

        assert "user_1" not in agent.processing_info

    async def test_emits_final_transcript_event(self):
        """Speech frames trigger a final_transcript event via recognize()."""
        agent = _make_agent()
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        # One loud audio frame followed by end-of-stream → triggers end-of-stream flush
        loud_event = _make_loud_event()
        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([loud_event])

        mock_stt_event = MagicMock()
        mock_stt_event.alternatives = [MagicMock(text="hello world")]
        agent.stt.recognize = AsyncMock(return_value=mock_stt_event)

        emitted = []
        agent.on("final_transcript", lambda **kw: emitted.append(kw))

        with patch("openai_stt_agent.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(mock_participant, mock_track, "en")
            await asyncio.sleep(0)

        assert len(emitted) == 1
        assert emitted[0]["participant"] is mock_participant
        assert emitted[0]["event"] is mock_stt_event

    async def test_does_not_emit_when_recognize_returns_empty_text(self):
        """No event should be emitted if the transcription result is empty."""
        agent = _make_agent()
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        loud_event = _make_loud_event()
        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([loud_event])

        mock_stt_event = MagicMock()
        mock_stt_event.alternatives = [MagicMock(text="")]
        agent.stt.recognize = AsyncMock(return_value=mock_stt_event)

        emitted = []
        agent.on("final_transcript", lambda **kw: emitted.append(kw))

        with patch("openai_stt_agent.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(mock_participant, mock_track, "en")

        assert len(emitted) == 0

    async def test_does_not_emit_for_silent_audio(self):
        """Silent frames (below energy threshold) should not trigger recognition."""
        agent = _make_agent()
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        silent_event = _make_audio_event(amplitude=0)
        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([silent_event])

        agent.stt.recognize = AsyncMock()

        emitted = []
        agent.on("final_transcript", lambda **kw: emitted.append(kw))

        with patch("openai_stt_agent.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(mock_participant, mock_track, "en")

        agent.stt.recognize.assert_not_called()
        assert len(emitted) == 0

    async def test_generic_exception_cleans_up_processing_info(self):
        """Unexpected exceptions should be caught and processing_info cleaned up."""
        agent = _make_agent()
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.side_effect = RuntimeError("boom")

        agent.processing_info["user_1"] = {"task": MagicMock()}

        with patch("openai_stt_agent.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(mock_participant, mock_track, "en")

        assert "user_1" not in agent.processing_info


class TestCleanup:
    async def test_cleanup_stops_all_active_transcriptions(self):
        """_cleanup() should cancel all active tasks."""
        agent = _make_agent()
        tasks = {}
        for uid in ("user_1", "user_2", "user_3"):
            mock_task = MagicMock()
            agent.processing_info[uid] = {"task": mock_task}
            tasks[uid] = mock_task

        await agent._cleanup()

        for uid, mock_task in tasks.items():
            mock_task.cancel.assert_called_once()
            assert uid not in agent.processing_info
