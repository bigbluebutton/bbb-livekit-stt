from prometheus_client import CollectorRegistry, Gauge

from metrics import (
    AGENT_FAILURE_ROOM_CONNECT,
    DISCARD_LOW_CONFIDENCE,
    EVENT_TYPE_FINAL,
    EVENT_TYPE_INTERIM,
    SESSION_FAILURE_NO_AUDIO_TRACK,
    SESSION_FAILURE_PARTICIPANT_NOT_FOUND,
    SttMetrics,
)


def _make_metrics():
    """Each test gets an isolated registry so collectors never collide."""
    return SttMetrics(CollectorRegistry())


class TestCollectorRegistration:
    def test_registers_all_collectors_on_the_injected_registry(self):
        registry = CollectorRegistry()
        SttMetrics(registry)

        names = {
            "bbb_stt_active_sessions",
            "bbb_stt_active_agents",
            "bbb_stt_session_starts_total",
            "bbb_stt_session_start_failures_total",
            "bbb_stt_agent_start_failures_total",
            "bbb_stt_transcripts_total",
            "bbb_stt_transcripts_discarded_total",
            "bbb_stt_transcript_confidence",
            "bbb_stt_utterance_duration_seconds",
            "bbb_stt_provider_audio_seconds_total",
            "bbb_stt_language_detection_mismatch_total",
            "bbb_stt_transcript_publish_failures_total",
        }
        collected = {m.name for m in registry.collect()}
        # Counters are collected without their _total suffix.
        expected = {n[: -len("_total")] if n.endswith("_total") else n for n in names}
        assert expected <= collected

    def test_two_instances_do_not_collide(self):
        SttMetrics(CollectorRegistry())
        SttMetrics(CollectorRegistry())


class TestMultiprocessSafety:
    def test_every_gauge_declares_a_multiprocess_mode(self):
        """Jobs run in child processes; a gauge without a mode raises at
        collection time in multiprocess mode, which unit tests never exercise."""
        metrics = _make_metrics()
        gauges = [value for value in vars(metrics).values() if isinstance(value, Gauge)]
        assert gauges, "expected SttMetrics to expose at least one Gauge"
        for gauge in gauges:
            assert gauge._multiprocess_mode == "livesum", (
                f"{gauge._name} must use livesum so a dead job's contribution drops out"
            )


class TestLabelCardinality:
    def test_no_collector_carries_an_unbounded_label(self):
        metrics = _make_metrics()
        forbidden = {"meeting_id", "user_id", "room", "room_name", "participant"}
        for value in vars(metrics).values():
            labelnames = getattr(value, "_labelnames", ())
            assert not forbidden & set(labelnames), (
                f"{value._name} carries an unbounded label"
            )


class TestAgentLifecycle:
    def test_started_then_stopped_returns_to_zero(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.agent_started("gladia")
        assert (
            registry.get_sample_value("bbb_stt_active_agents", {"provider": "gladia"})
            == 1.0
        )

        metrics.agent_stopped("gladia")
        assert (
            registry.get_sample_value("bbb_stt_active_agents", {"provider": "gladia"})
            == 0.0
        )

    def test_start_failure_increments_only_its_own_reason(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.agent_start_failed("gladia", AGENT_FAILURE_ROOM_CONNECT)

        assert (
            registry.get_sample_value(
                "bbb_stt_agent_start_failures_total",
                {"provider": "gladia", "reason": AGENT_FAILURE_ROOM_CONNECT},
            )
            == 1.0
        )


class TestSessionLifecycle:
    def test_session_started_increments_gauge_and_counter(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.session_started("gladia", "en-US")

        assert (
            registry.get_sample_value(
                "bbb_stt_active_sessions", {"provider": "gladia", "locale": "en-US"}
            )
            == 1.0
        )
        assert (
            registry.get_sample_value(
                "bbb_stt_session_starts_total", {"provider": "gladia"}
            )
            == 1.0
        )

    def test_session_stopped_decrements_gauge_but_not_the_counter(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.session_started("gladia", "en-US")
        metrics.session_stopped("gladia", "en-US")

        assert (
            registry.get_sample_value(
                "bbb_stt_active_sessions", {"provider": "gladia", "locale": "en-US"}
            )
            == 0.0
        )
        assert (
            registry.get_sample_value(
                "bbb_stt_session_starts_total", {"provider": "gladia"}
            )
            == 1.0
        )

    def test_session_start_failure_increments_only_its_own_reason(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.session_start_failed("gladia", SESSION_FAILURE_PARTICIPANT_NOT_FOUND)

        assert (
            registry.get_sample_value(
                "bbb_stt_session_start_failures_total",
                {"provider": "gladia", "reason": SESSION_FAILURE_PARTICIPANT_NOT_FOUND},
            )
            == 1.0
        )
        assert (
            registry.get_sample_value(
                "bbb_stt_session_start_failures_total",
                {"provider": "gladia", "reason": SESSION_FAILURE_NO_AUDIO_TRACK},
            )
            is None
        )


class TestSessionLocaleChange:
    def test_moves_the_gauge_from_the_old_locale_to_the_new_one(self):
        """A stale label would leave a phantom session pinned to the old locale."""
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.session_started("gladia", "en-US")
        metrics.session_locale_changed("gladia", "en-US", "pt-BR")

        assert (
            registry.get_sample_value(
                "bbb_stt_active_sessions", {"provider": "gladia", "locale": "en-US"}
            )
            == 0.0
        )
        assert (
            registry.get_sample_value(
                "bbb_stt_active_sessions", {"provider": "gladia", "locale": "pt-BR"}
            )
            == 1.0
        )

    def test_does_not_count_a_new_session_start(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.session_started("gladia", "en-US")
        metrics.session_locale_changed("gladia", "en-US", "pt-BR")

        assert (
            registry.get_sample_value(
                "bbb_stt_session_starts_total", {"provider": "gladia"}
            )
            == 1.0
        )

    def test_is_a_noop_when_the_locale_is_unchanged(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.session_started("gladia", "en-US")
        metrics.session_locale_changed("gladia", "en-US", "en-US")

        assert (
            registry.get_sample_value(
                "bbb_stt_active_sessions", {"provider": "gladia", "locale": "en-US"}
            )
            == 1.0
        )


class TestTranscriptEmitted:
    def test_counts_the_transcript(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.transcript_emitted("gladia", EVENT_TYPE_FINAL)

        assert (
            registry.get_sample_value(
                "bbb_stt_transcripts_total",
                {"provider": "gladia", "event_type": EVENT_TYPE_FINAL},
            )
            == 1.0
        )

    def test_observes_confidence_when_supplied(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.transcript_emitted("gladia", EVENT_TYPE_FINAL, confidence=0.85)

        assert (
            registry.get_sample_value(
                "bbb_stt_transcript_confidence_count",
                {"provider": "gladia", "event_type": EVENT_TYPE_FINAL},
            )
            == 1.0
        )

    def test_skips_confidence_when_the_provider_does_not_report_it(self):
        """OpenAI leaves SpeechData.confidence at its 0.0 default; observing it
        would pile a fabricated spike into the bottom bucket."""
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.transcript_emitted("openai", EVENT_TYPE_FINAL, confidence=None)

        assert (
            registry.get_sample_value(
                "bbb_stt_transcript_confidence_count",
                {"provider": "openai", "event_type": EVENT_TYPE_FINAL},
            )
            is None
        )
        assert (
            registry.get_sample_value(
                "bbb_stt_transcripts_total",
                {"provider": "openai", "event_type": EVENT_TYPE_FINAL},
            )
            == 1.0
        )

    def test_observes_utterance_duration_when_supplied(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.transcript_emitted("gladia", EVENT_TYPE_INTERIM, duration_seconds=2.5)

        assert (
            registry.get_sample_value(
                "bbb_stt_utterance_duration_seconds_sum",
                {"provider": "gladia", "event_type": EVENT_TYPE_INTERIM},
            )
            == 2.5
        )


class TestTranscriptDiscarded:
    def test_increments_the_given_reason(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.transcript_discarded("gladia", DISCARD_LOW_CONFIDENCE)

        assert (
            registry.get_sample_value(
                "bbb_stt_transcripts_discarded_total",
                {"provider": "gladia", "reason": DISCARD_LOW_CONFIDENCE},
            )
            == 1.0
        )


class TestProviderAudioObserved:
    def test_accumulates_reported_seconds(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.provider_audio_observed("gladia", 5.0)
        metrics.provider_audio_observed("gladia", 5.0)

        assert (
            registry.get_sample_value(
                "bbb_stt_provider_audio_seconds_total", {"provider": "gladia"}
            )
            == 10.0
        )

    def test_ignores_non_positive_durations(self):
        """prometheus_client raises on a negative Counter increment."""
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.provider_audio_observed("gladia", 5.0)
        metrics.provider_audio_observed("gladia", 0.0)
        metrics.provider_audio_observed("gladia", -1.0)

        assert (
            registry.get_sample_value(
                "bbb_stt_provider_audio_seconds_total", {"provider": "gladia"}
            )
            == 5.0
        )


class TestLanguageChecked:
    def test_counts_a_mismatch(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.language_checked(
            "gladia",
            requested_lang="en",
            reported_lang="pt",
            translation_enabled=False,
        )

        assert (
            registry.get_sample_value(
                "bbb_stt_language_detection_mismatch_total", {"provider": "gladia"}
            )
            == 1.0
        )

    def test_ignores_a_match(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.language_checked(
            "gladia",
            requested_lang="en",
            reported_lang="en",
            translation_enabled=False,
        )

        assert (
            registry.get_sample_value(
                "bbb_stt_language_detection_mismatch_total", {"provider": "gladia"}
            )
            is None
        )

    def test_ignores_everything_when_translation_is_enabled(self):
        """Gladia suppresses the original-language final transcript and emits one
        carrying the target language, so every translated utterance would
        otherwise register as a mismatch."""
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.language_checked(
            "gladia",
            requested_lang="en",
            reported_lang="pt",
            translation_enabled=True,
        )

        assert (
            registry.get_sample_value(
                "bbb_stt_language_detection_mismatch_total", {"provider": "gladia"}
            )
            is None
        )

    def test_ignores_auto_detection(self):
        """The participant requested detection, so there is nothing to differ from.
        _sanitize_locale maps "auto" to None."""
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.language_checked(
            "gladia",
            requested_lang=None,
            reported_lang="pt",
            translation_enabled=False,
        )

        assert (
            registry.get_sample_value(
                "bbb_stt_language_detection_mismatch_total", {"provider": "gladia"}
            )
            is None
        )

    def test_ignores_a_transcript_with_no_reported_language(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.language_checked(
            "openai",
            requested_lang="en",
            reported_lang=None,
            translation_enabled=False,
        )

        assert (
            registry.get_sample_value(
                "bbb_stt_language_detection_mismatch_total", {"provider": "openai"}
            )
            is None
        )


class TestTranscriptPublishFailed:
    def test_increments_the_counter(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.transcript_publish_failed("gladia")

        assert (
            registry.get_sample_value(
                "bbb_stt_transcript_publish_failures_total", {"provider": "gladia"}
            )
            == 1.0
        )


class TestLocaleUpdateFailed:
    def test_increments_the_counter(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        metrics.locale_update_failed("gladia")

        assert (
            registry.get_sample_value(
                "bbb_stt_locale_update_failures_total", {"provider": "gladia"}
            )
            == 1.0
        )

    def test_leaves_the_session_gauge_alone(self):
        """The session keeps running in the locale it already had."""
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)
        metrics.session_started("gladia", "pt-BR")

        metrics.locale_update_failed("gladia")

        assert (
            registry.get_sample_value(
                "bbb_stt_active_sessions", {"provider": "gladia", "locale": "pt-BR"}
            )
            == 1.0
        )
