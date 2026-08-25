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


def _audio_publication(source=rtc.TrackSource.SOURCE_MICROPHONE, sid="TR_mic"):
    track = MagicMock()
    track.kind = rtc.TrackKind.KIND_AUDIO
    track.sid = sid
    publication = MagicMock()
    publication.source = source
    publication.track = track
    return publication


def _with_audio_track(participant):
    participant.track_publications = {"t": _audio_publication()}
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
    """Async iterator that blocks until its input is ended or it is closed.

    Models the provider stream contract: a recognizer keeps yielding until the
    pipeline tells it no more audio is coming.
    """

    def __init__(self):
        self.closed = False
        self._ended = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self._ended.wait()
        raise StopAsyncIteration

    def push_frame(self, frame):
        pass

    def flush(self):
        pass

    def end_input(self):
        self._ended.set()

    async def aclose(self):
        self.closed = True
        self._ended.set()


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

            mic = participant.track_publications["t"]
            agent._on_track_unsubscribed(mic.track, mic, participant)
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

        with patch(
            "providers.base.rtc.AudioStream",
            side_effect=lambda *_, **__: _BlockingStream(),
        ):
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


class _DyingStream:
    """Audio stream that fails once the pipeline is already running."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        raise RuntimeError("provider connection dropped")


class _EndingStream:
    """Audio stream that ends, as it does when the track closes."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class TestSessionAccountingWhenAPipelineDies:
    """Only stop_transcription_for_user used to decrement the session gauge, so a
    pipeline that ended on its own left its label stranded for the process's life."""

    async def _run_until_death(self, audio_stream):
        agent, registry = _make_instrumented_agent()
        agent.room = MagicMock()
        agent.room.remote_participants = {"p": _with_audio_track(_stub_participant())}

        with patch("providers.base.rtc.AudioStream", return_value=audio_stream):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]
            await asyncio.gather(task, return_exceptions=True)

        return agent, registry

    def _sessions(self, registry, agent, locale="pt-BR"):
        return registry.get_sample_value(
            "bbb_stt_active_sessions",
            {"provider": agent.provider_name, "locale": locale},
        )

    async def test_gauge_returns_to_zero_when_the_pipeline_raises(self):
        agent, registry = await self._run_until_death(_DyingStream())

        assert "user_1" not in agent.processing_info
        assert self._sessions(registry, agent) == 0.0

    async def test_gauge_returns_to_zero_when_the_audio_stream_ends(self):
        agent, registry = await self._run_until_death(_EndingStream())

        assert "user_1" not in agent.processing_info
        assert self._sessions(registry, agent) == 0.0

    async def test_stopping_a_session_decrements_exactly_once(self):
        """The stop path decrements before cancelling; the cancelled pipeline must
        not decrement the same session again on its way out."""
        agent, registry = _make_instrumented_agent()
        agent.room = MagicMock()
        agent.room.remote_participants = {"p": _with_audio_track(_stub_participant())}

        with patch(
            "providers.base.rtc.AudioStream",
            side_effect=lambda *_, **__: _BlockingStream(),
        ):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]

            for _ in range(3):
                await asyncio.sleep(0)

            agent.handle_speech_locale_change("user_1", "", "")
            await asyncio.gather(task, return_exceptions=True)

        assert self._sessions(registry, agent) == 0.0

    async def test_a_replaced_session_keeps_its_gauge(self):
        """The cancelled pipeline no longer owns the slot, so its accounting must
        not follow the replacement."""

        class BlockingAgent(StubSttAgent):
            def _create_stt_stream(self, locale):
                return _BlockingStream()

        registry = CollectorRegistry()
        agent = BlockingAgent(BaseSttConfig(), metrics=SttMetrics(registry))
        agent.room = MagicMock()
        agent.room.remote_participants = {"p": _with_audio_track(_stub_participant())}

        with patch(
            "providers.base.rtc.AudioStream",
            side_effect=lambda *_, **__: _BlockingStream(),
        ):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            first = agent.processing_info["user_1"]["task"]

            for _ in range(3):
                await asyncio.sleep(0)

            agent.handle_speech_locale_change("user_1", "", "")
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            second = agent.processing_info["user_1"]["task"]

            for _ in range(5):
                await asyncio.sleep(0)

            assert self._sessions(registry, agent) == 1.0
            await _drain(first, second)

    async def test_an_unopenable_audio_stream_leaves_no_session_behind(self):
        """Opening the audio stream is part of the pipeline: when it fails there is
        no session, and the participant can start one again."""
        agent, registry = _make_instrumented_agent()
        agent.room = MagicMock()
        agent.room.remote_participants = {"p": _with_audio_track(_stub_participant())}

        with patch(
            "providers.base.rtc.AudioStream", side_effect=RuntimeError("no such track")
        ):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]
            await asyncio.gather(task, return_exceptions=True)

            assert "user_1" not in agent.processing_info
            assert self._sessions(registry, agent) == 0.0

        with patch(
            "providers.base.rtc.AudioStream",
            side_effect=lambda *_, **__: _BlockingStream(),
        ):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            assert "user_1" in agent.processing_info
            await _drain(agent.processing_info["user_1"]["task"])


class TestMicrophoneTrackSelection:
    """The agent transcribes microphones. Screenshare audio is subscribed too,
    because the room is joined with AutoSubscribe.AUDIO_ONLY."""

    def _participant_with_screenshare_first(self):
        participant = _stub_participant()
        participant.track_publications = {
            "screen": _audio_publication(
                source=rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO, sid="TR_screen"
            ),
            "mic": _audio_publication(sid="TR_mic"),
        }
        return participant

    def test_ignores_screenshare_audio_when_choosing_a_track(self):
        agent, _ = _make_instrumented_agent()
        participant = self._participant_with_screenshare_first()

        track = agent._find_audio_track(participant)

        assert track is participant.track_publications["mic"].track

    async def test_unsubscribing_screenshare_audio_leaves_the_session_running(self):
        agent, registry = _make_instrumented_agent()
        participant = self._participant_with_screenshare_first()
        agent.room = MagicMock()
        agent.room.remote_participants = {"p": participant}

        with patch(
            "providers.base.rtc.AudioStream",
            side_effect=lambda *_, **__: _BlockingStream(),
        ):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]

            screen = participant.track_publications["screen"]
            agent._on_track_unsubscribed(screen.track, screen, participant)

            assert "user_1" in agent.processing_info
            assert (
                registry.get_sample_value(
                    "bbb_stt_active_sessions",
                    {"provider": agent.provider_name, "locale": "pt-BR"},
                )
                == 1.0
            )
            await _drain(task)

    async def test_unsubscribing_the_session_track_stops_it(self):
        agent, _ = _make_instrumented_agent()
        participant = self._participant_with_screenshare_first()
        agent.room = MagicMock()
        agent.room.remote_participants = {"p": participant}

        with patch(
            "providers.base.rtc.AudioStream",
            side_effect=lambda *_, **__: _BlockingStream(),
        ):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]

            mic = participant.track_publications["mic"]
            agent._on_track_unsubscribed(mic.track, mic, participant)

            assert "user_1" not in agent.processing_info
            await _drain(task)


class TestStreamTeardown:
    """Cancelling the pipeline task ends the coroutine; it does not release the
    provider websocket or the audio stream's native handle."""

    def _closing_agent(self):
        class ClosingAgent(StubSttAgent):
            def _create_stt_stream(self, locale):
                return _BlockingStream()

        agent = ClosingAgent(BaseSttConfig(), metrics=SttMetrics(CollectorRegistry()))
        agent.room = MagicMock()
        agent.room.remote_participants = {"p": _with_audio_track(_stub_participant())}
        return agent

    async def test_closes_both_streams_when_a_session_is_stopped(self):
        agent = self._closing_agent()
        audio_stream = _BlockingStream()

        with patch("providers.base.rtc.AudioStream", return_value=audio_stream):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]
            stt_stream = agent.processing_info["user_1"]["stream"]

            for _ in range(3):
                await asyncio.sleep(0)

            agent.handle_speech_locale_change("user_1", "", "")
            await asyncio.gather(task, return_exceptions=True)

        assert audio_stream.closed
        assert stt_stream.closed

    async def test_an_exhausted_audio_track_ends_the_session(self):
        """When the audio runs out the provider has to be told, or the pipeline
        waits on a recognizer that will never speak again."""
        agent = self._closing_agent()

        with patch("providers.base.rtc.AudioStream", return_value=_EndingStream()):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]
            stt_stream = agent.processing_info["user_1"]["stream"]
            await asyncio.wait_for(asyncio.shield(task), timeout=2)

        assert stt_stream.closed
        assert "user_1" not in agent.processing_info

    async def test_a_failing_close_does_not_break_the_teardown(self):
        """The session is over either way; a close that raises must not take the
        gauge decrement or the slot release down with it."""

        class UnclosableStream(_BlockingStream):
            async def aclose(self):
                raise RuntimeError("already gone")

        class UnclosableAgent(StubSttAgent):
            def _create_stt_stream(self, locale):
                return UnclosableStream()

        registry = CollectorRegistry()
        agent = UnclosableAgent(BaseSttConfig(), metrics=SttMetrics(registry))
        agent.room = MagicMock()
        agent.room.remote_participants = {"p": _with_audio_track(_stub_participant())}

        with patch("providers.base.rtc.AudioStream", return_value=UnclosableStream()):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]

            for _ in range(3):
                await asyncio.sleep(0)

            agent.handle_speech_locale_change("user_1", "", "")
            await asyncio.gather(task, return_exceptions=True)

        assert "user_1" not in agent.processing_info
        assert (
            registry.get_sample_value(
                "bbb_stt_active_sessions",
                {"provider": agent.provider_name, "locale": "pt-BR"},
            )
            == 0.0
        )


class TestLocaleUpdateFailure:
    """A locale change is only real once the provider has taken it."""

    def _running_agent(self):
        agent, registry = _make_instrumented_agent()
        agent.room = MagicMock()
        agent.room.remote_participants = {"p": _with_audio_track(_stub_participant())}
        return agent, registry

    async def test_keeps_the_stored_locale_when_the_stream_rejects_the_change(self):
        agent, registry = self._running_agent()

        with patch(
            "providers.base.rtc.AudioStream",
            side_effect=lambda *_, **__: _BlockingStream(),
        ):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]
            stream = agent.processing_info["user_1"]["stream"]
            stream.update_options.side_effect = RuntimeError("stream is closed")

            agent.handle_speech_locale_change("user_1", "en-US", "gladia")

            assert agent.participant_settings["user_1"]["locale"] == "pt-BR"
            assert (
                registry.get_sample_value(
                    "bbb_stt_locale_update_failures_total",
                    {"provider": agent.provider_name},
                )
                == 1.0
            )
            await _drain(task)

    async def test_the_participant_can_retry_the_same_locale(self):
        """Committing the locale before the provider took it made the retry look
        like a no-op, leaving the participant on the old language for good."""
        agent, _ = self._running_agent()

        with patch(
            "providers.base.rtc.AudioStream",
            side_effect=lambda *_, **__: _BlockingStream(),
        ):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]
            stream = agent.processing_info["user_1"]["stream"]

            stream.update_options.side_effect = RuntimeError("stream is closed")
            agent.handle_speech_locale_change("user_1", "en-US", "gladia")

            stream.update_options.side_effect = None
            agent.handle_speech_locale_change("user_1", "en-US", "gladia")

            assert stream.update_options.call_count == 2
            assert agent.participant_settings["user_1"]["locale"] == "en-US"
            await _drain(task)

    async def test_a_successful_change_is_committed(self):
        agent, registry = self._running_agent()

        with patch(
            "providers.base.rtc.AudioStream",
            side_effect=lambda *_, **__: _BlockingStream(),
        ):
            agent.handle_speech_locale_change("user_1", "pt-BR", "gladia")
            task = agent.processing_info["user_1"]["task"]

            agent.handle_speech_locale_change("user_1", "en-US", "gladia")

            assert agent.participant_settings["user_1"]["locale"] == "en-US"
            assert (
                registry.get_sample_value(
                    "bbb_stt_locale_update_failures_total",
                    {"provider": agent.provider_name},
                )
                is None
            )
            await _drain(task)

    def test_records_the_locale_without_a_session_to_update(self):
        """The locale still has to be remembered: it is what the next track
        subscription starts from."""
        agent, _ = self._running_agent()
        agent.participant_settings["user_1"] = {"locale": "pt-BR", "provider": "gladia"}

        agent.update_locale_for_user("user_1", "en-US")

        assert agent.participant_settings["user_1"]["locale"] == "en-US"


class _ScriptedSttStream:
    """Provider stream that yields the transcripts a test hands it."""

    def __init__(self):
        self.queue = asyncio.Queue()

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.queue.get()

    def push_frame(self, frame):
        pass

    def flush(self):
        pass

    def end_input(self):
        pass

    async def aclose(self):
        pass

    def speak(self, start_time):
        """Queue a final transcript at an offset within this stream's session."""
        self.queue.put_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text="teste",
                        language="pt",
                        start_time=start_time,
                        end_time=start_time + 1,
                    )
                ],
            )
        )


class TestTranscriptEpoch:
    """A provider reports offsets within its own stream. Turning them into wall
    clock needs the moment that stream opened — which belongs to the session, not
    to the agent that serves the whole room."""

    def _scripted_agent(self, identities):
        class ScriptedAgent(StubSttAgent):
            def _create_stt_stream(self, locale):
                return _ScriptedSttStream()

        agent = ScriptedAgent(BaseSttConfig(), metrics=SttMetrics(CollectorRegistry()))
        participants = {}
        for index, identity in enumerate(identities):
            participant = _stub_participant(identity)
            participant.track_publications = {
                "mic": _audio_publication(sid=f"TR_{index}")
            }
            participants[identity] = participant
        agent.room = MagicMock()
        agent.room.remote_participants = participants
        return agent

    async def test_a_new_session_leaves_another_session_epoch_alone(self):
        """Regression: every session read the same open_time, so a participant
        joining late pushed everyone else's transcripts into the future."""
        clock = [1_000_000.0]
        agent = self._scripted_agent(["alice", "bob"])
        published = []

        async def record(participant, event, open_time):
            published.append(
                (participant.identity, open_time + event.alternatives[0].start_time)
            )

        agent.on("final_transcript", record)

        with (
            patch(
                "providers.base.rtc.AudioStream",
                side_effect=lambda *_, **__: _BlockingStream(),
            ),
            patch("providers.base.time.time", lambda: clock[0]),
        ):
            agent.handle_speech_locale_change("alice", "pt-BR", "gladia")
            alice_stream = agent.processing_info["alice"]["stream"]
            for _ in range(3):
                await asyncio.sleep(0)

            # Alice speaks five seconds into her own session.
            alice_stream.speak(5.0)
            for _ in range(3):
                await asyncio.sleep(0)

            # Bob joins four hundred seconds later.
            clock[0] += 400
            agent.handle_speech_locale_change("bob", "pt-BR", "gladia")
            for _ in range(3):
                await asyncio.sleep(0)

            # Alice speaks again, still measured from her own session's start.
            alice_stream.speak(410.0)
            for _ in range(4):
                await asyncio.sleep(0)

            tasks = [info["task"] for info in agent.processing_info.values()]

        assert published == [
            ("alice", 1_000_005.0),
            ("alice", 1_000_410.0),
        ]
        await _drain(*tasks)

    async def test_each_session_stamps_from_its_own_start(self):
        clock = [1_000_000.0]
        agent = self._scripted_agent(["alice", "bob"])
        published = []

        async def record(participant, event, open_time):
            published.append(
                (participant.identity, open_time + event.alternatives[0].start_time)
            )

        agent.on("final_transcript", record)

        with (
            patch(
                "providers.base.rtc.AudioStream",
                side_effect=lambda *_, **__: _BlockingStream(),
            ),
            patch("providers.base.time.time", lambda: clock[0]),
        ):
            agent.handle_speech_locale_change("alice", "pt-BR", "gladia")
            for _ in range(3):
                await asyncio.sleep(0)

            clock[0] += 400
            agent.handle_speech_locale_change("bob", "pt-BR", "gladia")
            for _ in range(3):
                await asyncio.sleep(0)

            agent.processing_info["bob"]["stream"].speak(2.0)
            for _ in range(4):
                await asyncio.sleep(0)

            tasks = [info["task"] for info in agent.processing_info.values()]

        assert published == [("bob", 1_000_402.0)]
        await _drain(*tasks)
