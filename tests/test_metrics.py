from prometheus_client import CollectorRegistry, Gauge

from metrics import (
    AGENT_FAILURE_ROOM_CONNECT,
    DISCARD_LOW_CONFIDENCE,
    EVENT_TYPE_FINAL,
    EVENT_TYPE_INTERIM,
    SESSION_FAILURE_NO_AUDIO_TRACK,
    SESSION_FAILURE_PARTICIPANT_NOT_FOUND,
    SttMetrics,
    language_label,
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
                "bbb_stt_active_sessions", {"provider": "gladia", "language": "en"}
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
                "bbb_stt_active_sessions", {"provider": "gladia", "language": "en"}
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

        metrics.transcript_discarded("gladia", "interim", DISCARD_LOW_CONFIDENCE)

        assert (
            registry.get_sample_value(
                "bbb_stt_transcripts_discarded_total",
                {
                    "provider": "gladia",
                    "event_type": "interim",
                    "reason": DISCARD_LOW_CONFIDENCE,
                },
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
        """_sanitize_locale maps "auto" to None."""
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


class TestLanguageLabel:
    def test_a_bbb_locale_is_labelled_with_its_language(self):
        assert language_label("pt-BR") == "pt"
        assert language_label("EN-us") == "en"

    def test_auto_detection_keeps_its_own_label(self):
        assert language_label("auto") == "auto"

    def test_anything_outside_iso_639_1_is_other(self):
        assert language_label("xx-YY") == "other"
        assert language_label("en<script>") == "other"
        assert language_label("") == "other"

    def test_locale_changes_cannot_grow_the_series_count(self):
        registry = CollectorRegistry()
        metrics = SttMetrics(registry)

        for i in range(200):
            metrics.session_started("gladia", f"en-x{i:05d}")
            metrics.session_started("gladia", f"zz-{i:05d}")

        languages = {
            sample.labels["language"]
            for metric in registry.collect()
            if metric.name == "bbb_stt_active_sessions"
            for sample in metric.samples
        }
        assert languages == {"en", "other"}
