import asyncio
import contextlib
import logging
from unittest.mock import AsyncMock, MagicMock, patch

from livekit import rtc
from livekit.agents import stt
from livekit.plugins.gladia import STT as GladiaSTT

from config import GladiaConfig
from gladia_stt_agent import GladiaSttAgent


def _make_agent(interim_results=None, **kwargs):
    config = GladiaConfig(api_key="fake-key", interim_results=interim_results, **kwargs)
    with patch("gladia_stt_agent.GladiaSTT", spec=GladiaSTT):
        agent = GladiaSttAgent(config)
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
        agent.processing_info["user_123"] = {"task": mock_task, "stream": MagicMock()}

        agent.stop_transcription_for_user("user_123")

        mock_task.cancel.assert_called_once()
        assert "user_123" not in agent.processing_info

    def test_no_op_when_user_not_in_processing_info(self):
        agent = _make_agent()
        # Should not raise even if user_id is unknown
        agent.stop_transcription_for_user("unknown_user")


class TestUpdateLocaleForUser:
    def test_updates_locale_in_participant_settings(self):
        agent = _make_agent()
        agent.participant_settings["user_1"] = {"locale": "en", "provider": "gladia"}

        agent.update_locale_for_user("user_1", "fr")

        assert agent.participant_settings["user_1"]["locale"] == "fr"

    def test_calls_stream_update_options_when_transcription_active(self):
        agent = _make_agent()
        agent.participant_settings["user_1"] = {"locale": "en", "provider": "gladia"}
        mock_stream = MagicMock()
        agent.processing_info["user_1"] = {"stream": mock_stream, "task": MagicMock()}

        agent.update_locale_for_user("user_1", "de")

        mock_stream.update_options.assert_called_once_with(languages=["de"])

    def test_sanitizes_bcp47_locale_for_stream_update(self):
        """update_locale_for_user should sanitize 'de-DE' → 'de' for the stream."""
        agent = _make_agent()
        agent.participant_settings["user_1"] = {"locale": "en", "provider": "gladia"}
        mock_stream = MagicMock()
        agent.processing_info["user_1"] = {"stream": mock_stream, "task": MagicMock()}

        agent.update_locale_for_user("user_1", "de-DE")

        mock_stream.update_options.assert_called_once_with(languages=["de"])

    def test_does_not_call_update_options_when_no_active_transcription(self):
        agent = _make_agent()
        agent.participant_settings["user_1"] = {"locale": "en", "provider": "gladia"}

        # update_locale_for_user should not raise when no active stream
        agent.update_locale_for_user("user_1", "fr")
        assert agent.participant_settings["user_1"]["locale"] == "fr"

    def test_no_op_in_settings_when_user_not_in_participant_settings(self):
        agent = _make_agent()
        # Should not raise or create settings entry
        agent.update_locale_for_user("unknown_user", "en")
        assert "unknown_user" not in agent.participant_settings


class TestOnParticipantDisconnected:
    def test_stops_transcription_and_clears_settings(self):
        agent = _make_agent()
        agent.participant_settings["user_1"] = {"locale": "en", "provider": "gladia"}
        mock_task = MagicMock()
        agent.processing_info["user_1"] = {"task": mock_task, "stream": MagicMock()}

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
        agent.participant_settings["user_1"] = {"locale": "en-US", "provider": "gladia"}
        mock_track, mock_publication, mock_participant = _make_track_subscribed_args(
            source=rtc.TrackSource.SOURCE_CAMERA
        )
        mock_participant.identity = "user_1"

        with patch.object(agent, "start_transcription_for_user") as mock_start:
            agent._on_track_subscribed(mock_track, mock_publication, mock_participant)
            mock_start.assert_not_called()

    def test_skips_transcription_when_no_settings(self):
        """Regression: must not raise when no settings exist for the participant."""
        agent = _make_agent()
        mock_track, mock_publication, mock_participant = _make_track_subscribed_args()
        mock_participant.identity = "user_no_settings"

        with patch.object(agent, "start_transcription_for_user") as mock_start:
            agent._on_track_subscribed(mock_track, mock_publication, mock_participant)
            mock_start.assert_not_called()

    def test_skips_transcription_when_locale_missing(self):
        """Regression: must not raise when provider is set but locale is absent."""
        agent = _make_agent()
        agent.participant_settings["user_1"] = {"provider": "gladia"}  # no locale
        mock_track, mock_publication, mock_participant = _make_track_subscribed_args()
        mock_participant.identity = "user_1"

        with patch.object(agent, "start_transcription_for_user") as mock_start:
            agent._on_track_subscribed(mock_track, mock_publication, mock_participant)
            mock_start.assert_not_called()

    def test_skips_transcription_when_provider_missing(self):
        """Regression: must not raise when locale is set but provider is absent."""
        agent = _make_agent()
        agent.participant_settings["user_1"] = {"locale": "en-US"}  # no provider
        mock_track, mock_publication, mock_participant = _make_track_subscribed_args()
        mock_participant.identity = "user_1"

        with patch.object(agent, "start_transcription_for_user") as mock_start:
            agent._on_track_subscribed(mock_track, mock_publication, mock_participant)
            mock_start.assert_not_called()

    def test_starts_transcription_when_locale_and_provider_present(self):
        agent = _make_agent()
        agent.participant_settings["user_1"] = {"locale": "en-US", "provider": "gladia"}
        mock_track, mock_publication, mock_participant = _make_track_subscribed_args()
        mock_participant.identity = "user_1"

        with patch.object(agent, "start_transcription_for_user") as mock_start:
            agent._on_track_subscribed(mock_track, mock_publication, mock_participant)
            mock_start.assert_called_once_with("user_1", "en-US", "gladia")


class TestStartTranscriptionForUser:
    def test_logs_error_when_participant_not_found(self, caplog):
        """Participant not in room → error log, no stream created."""
        agent = _make_agent_with_room(participants={})

        with caplog.at_level(logging.ERROR):
            agent.start_transcription_for_user("missing_user", "en-US", "gladia")

        assert "missing_user" not in agent.processing_info
        assert any("not found" in r.message for r in caplog.records)

    def test_logs_warning_when_no_audio_track(self, caplog):
        """Participant exists but has no audio track → warning, no stream."""
        participant = _make_participant("user_1", audio_track=None)
        agent = _make_agent_with_room(participants={"pid": participant})

        with caplog.at_level(logging.WARNING):
            agent.start_transcription_for_user("user_1", "en-US", "gladia")

        assert "user_1" not in agent.processing_info
        assert any("no audio track" in r.message for r in caplog.records)

    def test_skips_when_already_processing(self, caplog):
        """Already running → skip, existing task untouched."""
        mock_track = MagicMock()
        mock_track.kind = rtc.TrackKind.KIND_AUDIO
        participant = _make_participant("user_1", audio_track=mock_track)
        agent = _make_agent_with_room(participants={"pid": participant})

        existing_task = MagicMock()
        agent.processing_info["user_1"] = {"task": existing_task, "stream": MagicMock()}

        with caplog.at_level(logging.DEBUG):
            agent.start_transcription_for_user("user_1", "en-US", "gladia")

        # Existing task should not have been replaced
        assert agent.processing_info["user_1"]["task"] is existing_task

    async def test_creates_stream_and_task_on_success(self):
        """Happy path: participant with audio track → stream + task created."""
        mock_track = MagicMock()
        mock_track.kind = rtc.TrackKind.KIND_AUDIO
        participant = _make_participant("user_1", audio_track=mock_track)
        agent = _make_agent_with_room(participants={"pid": participant})

        with patch("gladia_stt_agent.rtc.AudioStream"):
            agent.start_transcription_for_user("user_1", "en-US", "gladia")

            assert "user_1" in agent.processing_info
            info = agent.processing_info["user_1"]
            assert "stream" in info
            assert "task" in info
            # Settings should be persisted
            assert agent.participant_settings["user_1"]["locale"] == "en-US"
            assert agent.participant_settings["user_1"]["provider"] == "gladia"

            # Cancel the background task so it doesn't leak after the patch exits
            info["task"].cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await info["task"]

    async def test_sanitizes_locale_before_creating_stream(self):
        """Locale 'pt-BR' should be sanitized to 'pt' for Gladia."""
        mock_track = MagicMock()
        mock_track.kind = rtc.TrackKind.KIND_AUDIO
        participant = _make_participant("user_1", audio_track=mock_track)
        agent = _make_agent_with_room(participants={"pid": participant})

        with patch("gladia_stt_agent.rtc.AudioStream"):
            agent.start_transcription_for_user("user_1", "pt-BR", "gladia")
            agent.stt.stream.assert_called_once_with(language="pt")

            # Cancel the background task to avoid leaking past the patch
            agent.processing_info["user_1"]["task"].cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await agent.processing_info["user_1"]["task"]


class TestRunTranscriptionPipeline:
    async def test_cancellation_cleans_up_processing_info(self):
        """CancelledError should be caught and processing_info entry removed."""
        agent = _make_agent()
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        # Simulate an audio stream that raises CancelledError immediately
        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.side_effect = asyncio.CancelledError

        mock_stt_stream = AsyncMock()
        mock_stt_stream.__aiter__.return_value = iter([])

        agent.processing_info["user_1"] = {
            "stream": mock_stt_stream,
            "task": MagicMock(),
        }

        with patch("gladia_stt_agent.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(
                mock_participant, mock_track, mock_stt_stream
            )

        assert "user_1" not in agent.processing_info

    async def test_emits_final_transcript_event(self):
        """FINAL_TRANSCRIPT events from STT should be emitted."""
        agent = _make_agent()
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        # Empty audio stream (finishes immediately)
        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([])

        # STT stream that yields one FINAL_TRANSCRIPT event
        mock_event = MagicMock()
        mock_event.type = stt.SpeechEventType.FINAL_TRANSCRIPT
        mock_stt_stream = AsyncMock()
        mock_stt_stream.__aiter__.return_value = iter([mock_event])

        emitted = []
        agent.on("final_transcript", lambda **kw: emitted.append(kw))

        with patch("gladia_stt_agent.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(
                mock_participant, mock_track, mock_stt_stream
            )
            await asyncio.sleep(0)

        assert len(emitted) == 1
        assert emitted[0]["participant"] is mock_participant
        assert emitted[0]["event"] is mock_event

    async def test_emits_interim_transcript_when_enabled(self):
        """INTERIM_TRANSCRIPT events should be emitted only when interim_results is set."""
        agent = _make_agent(interim_results=True)
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([])

        mock_event = MagicMock()
        mock_event.type = stt.SpeechEventType.INTERIM_TRANSCRIPT
        mock_stt_stream = AsyncMock()
        mock_stt_stream.__aiter__.return_value = iter([mock_event])

        emitted = []
        agent.on("interim_transcript", lambda **kw: emitted.append(kw))

        with patch("gladia_stt_agent.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(
                mock_participant, mock_track, mock_stt_stream
            )
            await asyncio.sleep(0)

        assert len(emitted) == 1

    async def test_suppresses_interim_transcript_when_disabled(self):
        """INTERIM_TRANSCRIPT should NOT be emitted when interim_results is off."""
        agent = _make_agent(interim_results=False)
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([])

        mock_event = MagicMock()
        mock_event.type = stt.SpeechEventType.INTERIM_TRANSCRIPT
        mock_stt_stream = AsyncMock()
        mock_stt_stream.__aiter__.return_value = iter([mock_event])

        emitted = []
        agent.on("interim_transcript", lambda **kw: emitted.append(kw))

        with patch("gladia_stt_agent.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(
                mock_participant, mock_track, mock_stt_stream
            )
            await asyncio.sleep(0)

        assert len(emitted) == 0

    async def test_generic_exception_cleans_up_processing_info(self):
        """Unexpected exceptions should be caught and processing_info cleaned up."""
        agent = _make_agent()
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.side_effect = RuntimeError("boom")

        mock_stt_stream = AsyncMock()
        mock_stt_stream.__aiter__.return_value = iter([])

        agent.processing_info["user_1"] = {
            "stream": mock_stt_stream,
            "task": MagicMock(),
        }

        with patch("gladia_stt_agent.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(
                mock_participant, mock_track, mock_stt_stream
            )

        assert "user_1" not in agent.processing_info


class TestCleanup:
    async def test_cleanup_stops_all_active_transcriptions(self):
        """_cleanup() should cancel all active tasks."""
        agent = _make_agent()
        tasks = {}
        for uid in ("user_1", "user_2", "user_3"):
            mock_task = MagicMock()
            agent.processing_info[uid] = {"task": mock_task, "stream": MagicMock()}
            tasks[uid] = mock_task

        await agent._cleanup()

        for uid, mock_task in tasks.items():
            mock_task.cancel.assert_called_once()
            assert uid not in agent.processing_info
