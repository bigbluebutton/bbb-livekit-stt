import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from livekit import rtc
from livekit.agents import stt

from metrics import (
    AGENT_FAILURE_ROOM_CONNECT,
    SESSION_FAILURE_NO_AUDIO_TRACK,
    SESSION_FAILURE_PARTICIPANT_NOT_FOUND,
    SESSION_FAILURE_STREAM_ERROR,
    SttMetrics,
)
from prometheus_client import CollectorRegistry

from providers.base import BaseSttAgent, BaseSttConfig


class StubSttAgent(BaseSttAgent):
    """Concrete test subclass for BaseSttAgent."""

    def _create_stt_stream(self, locale):
        return MagicMock()

    def _update_stream_locale(self, user_id, locale):
        stream = self.processing_info[user_id]["stream"]
        sanitized = self._sanitize_locale(locale)
        stream.update_options(languages=[sanitized])


def _make_stub_agent(interim_results=None):
    config = BaseSttConfig(interim_results=interim_results)
    return StubSttAgent(config)


class TestShouldEmitDefault:
    def test_returns_true_for_any_event(self):
        agent = _make_stub_agent()
        mock_event = MagicMock()
        assert agent._should_emit(mock_event) is True


class TestTranslationLangMapDefault:
    def test_returns_empty_dict(self):
        agent = _make_stub_agent()
        assert agent.translation_lang_map == {}


class TestShouldEmitIntegration:
    async def test_pipeline_skips_event_when_should_emit_returns_false(self):
        """Events filtered by _should_emit should not be emitted."""

        class FilteringAgent(StubSttAgent):
            def _should_emit(self, event):
                return False

        config = BaseSttConfig(interim_results=True)
        agent = FilteringAgent(config)
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([])

        mock_event = MagicMock()
        mock_event.type = stt.SpeechEventType.FINAL_TRANSCRIPT
        mock_stt_stream = AsyncMock()
        mock_stt_stream.__aiter__.return_value = iter([mock_event])

        emitted = []
        agent.on("final_transcript", lambda **kw: emitted.append(kw))

        with patch("providers.base.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(
                mock_participant, mock_track, mock_stt_stream
            )
            await asyncio.sleep(0)

        assert len(emitted) == 0

    async def test_pipeline_emits_event_when_should_emit_returns_true(self):
        agent = _make_stub_agent()
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([])

        mock_event = MagicMock()
        mock_event.type = stt.SpeechEventType.FINAL_TRANSCRIPT
        mock_stt_stream = AsyncMock()
        mock_stt_stream.__aiter__.return_value = iter([mock_event])

        emitted = []
        agent.on("final_transcript", lambda **kw: emitted.append(kw))

        with patch("providers.base.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(
                mock_participant, mock_track, mock_stt_stream
            )
            await asyncio.sleep(0)

        assert len(emitted) == 1


def _make_instrumented_agent(interim_results=None):
    """Returns (agent, registry) so tests can read real sample values."""
    registry = CollectorRegistry()
    agent = StubSttAgent(
        BaseSttConfig(interim_results=interim_results), metrics=SttMetrics(registry)
    )
    return agent, registry


def _stub_participant(identity="user_1"):
    participant = MagicMock(spec=rtc.RemoteParticipant)
    participant.identity = identity
    return participant


def _with_audio_track(participant):
    track = MagicMock()
    track.kind = rtc.TrackKind.KIND_AUDIO
    publication = MagicMock()
    publication.track = track
    participant.track_publications = {"t": publication}
    return participant


class TestProviderCapabilityDefaults:
    def test_reports_confidence_is_false(self):
        agent, _ = _make_instrumented_agent()
        assert agent.reports_confidence is False

    def test_translation_enabled_is_false(self):
        agent, _ = _make_instrumented_agent()
        assert agent.translation_enabled is False


class TestSessionStartFailures:
    def test_records_participant_not_found(self):
        agent, registry = _make_instrumented_agent()
        agent.room = MagicMock()
        agent.room.remote_participants = {}

        agent.start_transcription_for_user("ghost", "en-US", "gladia")

        assert (
            registry.get_sample_value(
                "bbb_stt_session_start_failures_total",
                {
                    "provider": agent.provider_name,
                    "reason": SESSION_FAILURE_PARTICIPANT_NOT_FOUND,
                },
            )
            == 1.0
        )

    def test_records_no_audio_track(self):
        agent, registry = _make_instrumented_agent()
        participant = _stub_participant()
        participant.track_publications = {}
        agent.room = MagicMock()
        agent.room.remote_participants = {"p": participant}

        agent.start_transcription_for_user("user_1", "en-US", "gladia")

        assert (
            registry.get_sample_value(
                "bbb_stt_session_start_failures_total",
                {
                    "provider": agent.provider_name,
                    "reason": SESSION_FAILURE_NO_AUDIO_TRACK,
                },
            )
            == 1.0
        )

    def test_records_stream_error_and_reraises(self):
        registry = CollectorRegistry()

        class BrokenStreamAgent(StubSttAgent):
            def _create_stt_stream(self, locale):
                raise RuntimeError("provider unreachable")

        agent = BrokenStreamAgent(BaseSttConfig(), metrics=SttMetrics(registry))
        participant = _with_audio_track(_stub_participant())
        agent.room = MagicMock()
        agent.room.remote_participants = {"p": participant}

        with pytest.raises(RuntimeError):
            agent.start_transcription_for_user("user_1", "en-US", "gladia")

        assert (
            registry.get_sample_value(
                "bbb_stt_session_start_failures_total",
                {
                    "provider": agent.provider_name,
                    "reason": SESSION_FAILURE_STREAM_ERROR,
                },
            )
            == 1.0
        )


class TestSessionGauge:
    async def test_start_then_stop_returns_the_gauge_to_zero(self):
        agent, registry = _make_instrumented_agent()
        participant = _with_audio_track(_stub_participant())
        agent.room = MagicMock()
        agent.room.remote_participants = {"p": participant}

        with patch("providers.base.rtc.AudioStream"):
            agent.start_transcription_for_user("user_1", "en-US", "gladia")

            assert (
                registry.get_sample_value(
                    "bbb_stt_active_sessions",
                    {"provider": agent.provider_name, "locale": "en-US"},
                )
                == 1.0
            )

            agent.stop_transcription_for_user("user_1")

        assert (
            registry.get_sample_value(
                "bbb_stt_active_sessions",
                {"provider": agent.provider_name, "locale": "en-US"},
            )
            == 0.0
        )

    async def test_stop_uses_the_locale_the_gauge_was_incremented_under(self):
        """processing_info carries its own copy so a settings mutation cannot
        make the decrement land on the wrong label."""
        agent, registry = _make_instrumented_agent()
        participant = _with_audio_track(_stub_participant())
        agent.room = MagicMock()
        agent.room.remote_participants = {"p": participant}

        with patch("providers.base.rtc.AudioStream"):
            agent.start_transcription_for_user("user_1", "en-US", "gladia")
            agent.participant_settings["user_1"]["locale"] = "pt-BR"
            agent.stop_transcription_for_user("user_1")

        assert (
            registry.get_sample_value(
                "bbb_stt_active_sessions",
                {"provider": agent.provider_name, "locale": "en-US"},
            )
            == 0.0
        )
        assert (
            registry.get_sample_value(
                "bbb_stt_active_sessions",
                {"provider": agent.provider_name, "locale": "pt-BR"},
            )
            is None
        )


class TestAgentStartFailure:
    async def test_records_room_connect_error_and_reraises(self):
        agent, registry = _make_instrumented_agent()
        ctx = MagicMock()
        ctx.connect = AsyncMock(side_effect=RuntimeError("no route to host"))

        with pytest.raises(RuntimeError):
            await agent.start(ctx)

        assert (
            registry.get_sample_value(
                "bbb_stt_agent_start_failures_total",
                {"provider": agent.provider_name, "reason": AGENT_FAILURE_ROOM_CONNECT},
            )
            == 1.0
        )
        assert (
            registry.get_sample_value(
                "bbb_stt_active_agents", {"provider": agent.provider_name}
            )
            is None
        )


class TestRecognitionUsage:
    async def test_records_audio_seconds_and_emits_no_transcript(self):
        """livekit-agents drops RECOGNITION_USAGE today; it carries the provider's
        own count of audio seconds processed."""
        agent, registry = _make_instrumented_agent(interim_results=True)
        participant = _stub_participant()
        track = MagicMock()

        usage_event = stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            alternatives=[],
            recognition_usage=stt.RecognitionUsage(audio_duration=5.0),
        )

        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([])
        mock_stt_stream = AsyncMock()
        mock_stt_stream.__aiter__.return_value = iter([usage_event])

        # EventEmitter awaits its callbacks, so these must be coroutines.
        emitted = []

        async def collect(**kwargs):
            emitted.append(kwargs)

        agent.on("final_transcript", collect)
        agent.on("interim_transcript", collect)

        with patch("providers.base.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(participant, track, mock_stt_stream)
        await asyncio.sleep(0)

        assert (
            registry.get_sample_value(
                "bbb_stt_provider_audio_seconds_total",
                {"provider": agent.provider_name},
            )
            == 5.0
        )
        assert emitted == []
