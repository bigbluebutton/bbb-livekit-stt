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


class _BlockingStream:
    """Async iterator that blocks forever, keeping a pipeline alive until cancelled."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Event().wait()
        raise StopAsyncIteration

    def push_frame(self, frame):
        pass

    def flush(self):
        pass

    async def aclose(self):
        pass


def _agent_with_participant(identity="user_1"):
    """Instrumented agent whose room holds one participant with a live mic track."""
    agent, registry = _make_instrumented_agent()
    participant = _with_audio_track(_stub_participant(identity))
    agent.room = MagicMock()
    agent.room.remote_participants = {"p": participant}
    return agent, registry, participant


async def _drain(*tasks):
    """Cancel the pipeline tasks a test started so none outlive the test."""
    live = [task for task in tasks if task is not None]
    for task in live:
        task.cancel()
    await asyncio.gather(*live, return_exceptions=True)


class TestSpeechLocaleChangeDispatch:
    """BBB's UserSpeechLocaleChangedEvtMsg drives the whole session lifecycle:
    a locale plus a provider enables transcription, an empty pair disables it.
    """

    async def test_enabling_starts_a_session(self):
        agent, registry, _ = _agent_with_participant()

        with patch("providers.base.rtc.AudioStream"):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]

        assert agent.participant_settings["user_1"]["locale"] == "pt-BR"
        assert (
            registry.get_sample_value(
                "bbb_stt_active_sessions",
                {"provider": agent.provider_name, "locale": "pt-BR"},
            )
            == 1.0
        )
        await _drain(task)

    async def test_disabling_stops_the_session_and_forgets_the_locale(self):
        """BBB sends an empty locale and provider when the user unassigns the
        transcription language. The stored locale must go with the session, or
        the agent still believes transcription is on."""
        agent, _, _ = _agent_with_participant()

        with patch("providers.base.rtc.AudioStream"):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]
            agent.handle_speech_locale_change("user_1", "", "")

        assert "user_1" not in agent.processing_info
        assert not agent.participant_settings["user_1"].get("locale")
        assert not agent.participant_settings["user_1"].get("provider")
        await _drain(task)

    async def test_disabling_keeps_the_speech_options(self):
        """Speech options arrive on their own event and are not resent when the
        locale is reassigned, so a disable must not drop them."""
        agent, _, _ = _agent_with_participant()
        agent.participant_settings["user_1"] = {
            "partial_utterances": True,
            "min_utterance_length": 1.0,
        }

        with patch("providers.base.rtc.AudioStream"):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]
            agent.handle_speech_locale_change("user_1", "", "")

        assert agent.participant_settings["user_1"]["partial_utterances"] is True
        assert agent.participant_settings["user_1"]["min_utterance_length"] == 1.0
        await _drain(task)

    async def test_reenabling_the_same_locale_starts_a_new_session(self):
        """Regression: disabling and re-enabling the same locale left the agent
        with a stale locale that made the re-enable a silent no-op, so
        transcription never came back."""
        agent, registry, _ = _agent_with_participant()

        with patch("providers.base.rtc.AudioStream"):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            first = agent.processing_info["user_1"]["task"]
            agent.handle_speech_locale_change("user_1", "", "")
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")

            assert "user_1" in agent.processing_info
            second = agent.processing_info["user_1"]["task"]

        assert second is not first
        assert agent.participant_settings["user_1"]["locale"] == "pt-BR"
        assert (
            registry.get_sample_value(
                "bbb_stt_active_sessions",
                {"provider": agent.provider_name, "locale": "pt-BR"},
            )
            == 1.0
        )
        await _drain(first, second)

    async def test_reenabling_a_different_locale_starts_a_new_session(self):
        agent, _, _ = _agent_with_participant()

        with patch("providers.base.rtc.AudioStream"):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            first = agent.processing_info["user_1"]["task"]
            agent.handle_speech_locale_change("user_1", "", "")
            agent.handle_speech_locale_change("user_1", "en-US", "gladia")

            assert "user_1" in agent.processing_info
            second = agent.processing_info["user_1"]["task"]

        assert agent.participant_settings["user_1"]["locale"] == "en-US"
        await _drain(first, second)

    async def test_changing_locale_while_running_updates_in_place(self):
        agent, _, _ = _agent_with_participant()

        with patch("providers.base.rtc.AudioStream"):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]
            stream = agent.processing_info["user_1"]["stream"]

            agent.handle_speech_locale_change("user_1", "en-US", "gladia")

            assert agent.processing_info["user_1"]["task"] is task
            stream.update_options.assert_called_once_with(languages=["en"])

        assert agent.participant_settings["user_1"]["locale"] == "en-US"
        await _drain(task)

    async def test_reasserting_the_running_locale_is_a_no_op(self):
        agent, _, _ = _agent_with_participant()

        with patch("providers.base.rtc.AudioStream"):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]
            stream = agent.processing_info["user_1"]["stream"]

            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")

            assert agent.processing_info["user_1"]["task"] is task
            stream.update_options.assert_not_called()

        await _drain(task)

    async def test_enabling_without_a_track_starts_on_track_subscribed(self):
        """BBB routinely sends the locale before the mic track is published."""
        agent, _, participant = _agent_with_participant()
        participant.track_publications = {}

        agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
        assert "user_1" not in agent.processing_info

        _with_audio_track(participant)
        publication = MagicMock()
        publication.source = rtc.TrackSource.SOURCE_MICROPHONE

        with patch("providers.base.rtc.AudioStream"):
            agent._on_track_subscribed(MagicMock(), publication, participant)
            task = agent.processing_info["user_1"]["task"]

        await _drain(task)

    async def test_track_resubscribing_after_a_disable_does_not_restart(self):
        """Republishing the mic must not resurrect transcription the user turned
        off — after a disable there is no locale left to start with."""
        agent, _, participant = _agent_with_participant()

        with patch("providers.base.rtc.AudioStream"):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]
            agent.handle_speech_locale_change("user_1", "", "")

            publication = MagicMock()
            publication.source = rtc.TrackSource.SOURCE_MICROPHONE
            agent._on_track_subscribed(MagicMock(), publication, participant)

        assert "user_1" not in agent.processing_info
        await _drain(task)

    async def test_reenabling_after_the_track_dropped_starts_a_new_session(self):
        """A muted participant's track is unpublished, which stops the session
        but leaves the locale in place. Re-asserting it must start again."""
        agent, _, participant = _agent_with_participant()

        with patch("providers.base.rtc.AudioStream"):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            first = agent.processing_info["user_1"]["task"]

            agent._on_track_unsubscribed(MagicMock(), MagicMock(), participant)
            assert "user_1" not in agent.processing_info

            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            second = agent.processing_info["user_1"]["task"]

        await _drain(first, second)

    async def test_restarting_within_one_loop_turn_keeps_the_new_session(self):
        """A disable and a re-enable can both be handled before the cancelled
        pipeline gets to run its cleanup. That cleanup must not evict the
        session that took the slot over."""

        class BlockingAgent(StubSttAgent):
            def _create_stt_stream(self, locale):
                return _BlockingStream()

        agent = BlockingAgent(BaseSttConfig(), metrics=SttMetrics(CollectorRegistry()))
        agent.room = MagicMock()
        agent.room.remote_participants = {"p": _with_audio_track(_stub_participant())}

        with patch("providers.base.rtc.AudioStream", return_value=_BlockingStream()):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            first = agent.processing_info["user_1"]["task"]

            # Let the pipeline actually start, so its cancellation later has a
            # try/finally to unwind.
            for _ in range(3):
                await asyncio.sleep(0)

            agent.handle_speech_locale_change("user_1", "", "")
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            second = agent.processing_info["user_1"]["task"]

            # Let the cancelled pipeline reach its cleanup.
            for _ in range(5):
                await asyncio.sleep(0)

            assert agent.processing_info.get("user_1", {}).get("task") is second
            await _drain(first, second)

    async def test_an_empty_locale_disables_even_with_the_provider_still_set(self):
        """Seen on the wire when transcription is paused/deactivated:
        `{"locale": "", "provider": "gladia"}`. Only the pair being complete
        means "enabled", so a cleared locale disables regardless of provider."""
        agent, registry, _ = _agent_with_participant()

        with patch("providers.base.rtc.AudioStream"):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]

            agent.handle_speech_locale_change("user_1", "", "gladia")

        assert "user_1" not in agent.processing_info
        assert not agent.participant_settings["user_1"].get("locale")
        assert not agent.participant_settings["user_1"].get("provider")
        assert (
            registry.get_sample_value(
                "bbb_stt_active_sessions",
                {"provider": agent.provider_name, "locale": "pt-BR"},
            )
            == 0.0
        )
        await _drain(task)

    async def test_resuming_after_a_provider_only_pause_starts_a_new_session(self):
        """The pause/resume round trip: locale cleared with the provider left in
        place, then the same locale reasserted."""
        agent, _, _ = _agent_with_participant()

        with patch("providers.base.rtc.AudioStream"):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            first = agent.processing_info["user_1"]["task"]
            agent.handle_speech_locale_change("user_1", "", "gladia")
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")

            assert "user_1" in agent.processing_info
            second = agent.processing_info["user_1"]["task"]

        assert second is not first
        assert agent.participant_settings["user_1"]["locale"] == "pt-BR"
        await _drain(first, second)

    async def test_track_resubscribing_after_a_provider_only_pause_does_not_restart(
        self,
    ):
        agent, _, participant = _agent_with_participant()

        with patch("providers.base.rtc.AudioStream"):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]
            agent.handle_speech_locale_change("user_1", "", "gladia")

            publication = MagicMock()
            publication.source = rtc.TrackSource.SOURCE_MICROPHONE
            agent._on_track_subscribed(MagicMock(), publication, participant)

        assert "user_1" not in agent.processing_info
        await _drain(task)

    async def test_absent_locale_and_provider_fields_disable(self):
        """`body.get()` yields None when akka omits a field; that is not an
        enable either."""
        agent, _, _ = _agent_with_participant()

        with patch("providers.base.rtc.AudioStream"):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]

            agent.handle_speech_locale_change("user_1", None, None)

        assert "user_1" not in agent.processing_info
        assert not agent.participant_settings["user_1"].get("locale")
        await _drain(task)

    def test_disabling_an_unknown_user_is_a_no_op(self):
        agent, _, _ = _agent_with_participant()
        agent.handle_speech_locale_change("ghost", "", "")
        assert "ghost" not in agent.processing_info
