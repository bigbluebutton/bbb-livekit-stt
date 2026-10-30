"""Prometheus collectors for the BBB LiveKit STT agent.

Collectors are registered into the default ``prometheus_client`` registry, which
the LiveKit worker's built-in ``/metrics`` endpoint serves alongside its own
``lk_agents_*`` metrics. Set ``BBB_STT_PROMETHEUS_PORT`` to expose it.

Jobs run in child processes, so every ``Gauge`` declares an explicit
``multiprocess_mode``. ``livesum`` matches LiveKit's own choice for its active-job
gauge: values sum across processes and a dead process stops contributing.
``PROMETHEUS_MULTIPROC_DIR`` comes from the environment, not ``WorkerOptions``:
``prometheus_client`` reads it on first import, before ``WorkerOptions`` is built.

This module must not import from ``providers`` — providers import it.
"""

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

# --- Label values. Closed sets: never caller-supplied strings. ---

EVENT_TYPE_FINAL = "final"
EVENT_TYPE_INTERIM = "interim"

SESSION_FAILURE_PARTICIPANT_NOT_FOUND = "participant_not_found"
# Usually benign; the README's metric caveats say how to alert on it.
SESSION_FAILURE_NO_AUDIO_TRACK = "no_audio_track"
SESSION_FAILURE_STREAM_ERROR = "stream_error"

# Why a session that did start came to an end.
SESSION_END_STOPPED = "stopped"
SESSION_END_PROVIDER_CLOSED = "provider_closed"
SESSION_END_AUDIO_ENDED = "audio_ended"
# A provider can reject a session after accepting it. That failure is counted
# here, not in bbb_stt_session_start_failures_total.
SESSION_END_ERROR = "error"

AGENT_FAILURE_UNKNOWN_PROVIDER = "unknown_provider"
AGENT_FAILURE_ROOM_CONNECT = "room_connect_error"
AGENT_FAILURE_REDIS_CONNECT = "redis_connect_error"

DISCARD_LOW_CONFIDENCE = "low_confidence"
DISCARD_UNRESOLVABLE_LOCALE = "unresolvable_locale"
DISCARD_BELOW_MIN_UTTERANCE_LENGTH = "below_min_utterance_length"

# The session gauge's language label. The locale comes from BBB unvalidated, so
# the label is its ISO 639-1 language when it has one, and LANGUAGE_OTHER for
# anything else.
LANGUAGE_AUTO = "auto"
LANGUAGE_OTHER = "other"
ISO_639_1 = frozenset(
    """
    aa ab ae af ak am an ar as av ay az ba be bg bh bi bm bn bo br bs ca ce ch
    co cr cs cu cv cy da de dv dz ee el en eo es et eu fa ff fi fj fo fr fy ga
    gd gl gn gu gv ha he hi ho hr ht hu hy hz ia id ie ig ii ik io is it iu ja
    jv ka kg ki kj kk kl km kn ko kr ks ku kv kw ky la lb lg li ln lo lt lu lv
    mg mh mi mk ml mn mr ms mt my na nb nd ne ng nl nn no nr nv ny oc oj om or
    os pa pi pl ps pt qu rm rn ro ru rw sa sc sd se sg si sk sl sm sn so sq sr
    ss st su sv sw ta te tg th ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi
    vo wa wo xh yi yo za zh zu
    """.split()
)

# prometheus_client's default buckets are latency-oriented and useless for a
# [0, 1] score.
CONFIDENCE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
UTTERANCE_DURATION_BUCKETS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 60.0)


def language_label(locale: str) -> str:
    """The session gauge's label for a BBB locale such as "pt-BR"."""
    language = locale.split("-")[0].lower()
    if language == LANGUAGE_AUTO:
        return LANGUAGE_AUTO
    return language if language in ISO_639_1 else LANGUAGE_OTHER


class SttMetrics:
    """Owns every collector.

    The registry is injectable so tests can assert real sample values against an
    isolated registry instead of mocking. Production uses the module singleton
    ``stt_metrics``, bound to the default registry.
    """

    def __init__(self, registry: CollectorRegistry | None = None):
        registry = REGISTRY if registry is None else registry

        self.active_sessions = Gauge(
            "bbb_stt_active_sessions",
            "Transcription sessions currently active, one per subscribed microphone track",
            ["provider", "language"],
            multiprocess_mode="livesum",
            registry=registry,
        )
        self.active_agents = Gauge(
            "bbb_stt_active_agents",
            "STT agents currently running, one per LiveKit room job",
            ["provider"],
            multiprocess_mode="livesum",
            registry=registry,
        )
        self.session_starts = Counter(
            "bbb_stt_session_starts_total",
            "Transcription sessions successfully started",
            ["provider"],
            registry=registry,
        )
        self.session_start_failures = Counter(
            "bbb_stt_session_start_failures_total",
            "Transcription sessions that could not be established",
            ["provider", "reason"],
            registry=registry,
        )
        self.session_ends = Counter(
            "bbb_stt_session_ends_total",
            "Transcription sessions that ended, by why they ended",
            ["provider", "reason"],
            registry=registry,
        )
        self.agent_start_failures = Counter(
            "bbb_stt_agent_start_failures_total",
            "STT agents that could not be established",
            ["provider", "reason"],
            registry=registry,
        )
        self.transcripts = Counter(
            "bbb_stt_transcripts_total",
            # Counted whether or not Redis accepted the transcript, because
            # transcript_emitted() also feeds the confidence histogram. The
            # README's metric caveats say how to read it.
            "Transcripts that passed filtering and were sent to BBB",
            ["provider", "event_type"],
            registry=registry,
        )
        self.transcripts_discarded = Counter(
            "bbb_stt_transcripts_discarded_total",
            "Transcripts dropped before publication",
            ["provider", "event_type", "reason"],
            registry=registry,
        )
        self.transcript_confidence = Histogram(
            "bbb_stt_transcript_confidence",
            "Provider-reported confidence of published transcripts",
            ["provider", "event_type"],
            buckets=CONFIDENCE_BUCKETS,
            registry=registry,
        )
        self.utterance_duration = Histogram(
            "bbb_stt_utterance_duration_seconds",
            "Duration of transcribed utterances",
            ["provider", "event_type"],
            buckets=UTTERANCE_DURATION_BUCKETS,
            registry=registry,
        )
        self.provider_audio_seconds = Counter(
            "bbb_stt_provider_audio_seconds_total",
            "Audio seconds sent to the STT provider, as counted by its plugin (Gladia only)",
            ["provider"],
            registry=registry,
        )
        self.language_detection_mismatch = Counter(
            "bbb_stt_language_detection_mismatch_total",
            "Transcripts whose reported language differs from the requested one",
            ["provider"],
            registry=registry,
        )
        self.transcript_publish_failures = Counter(
            "bbb_stt_transcript_publish_failures_total",
            "Transcripts lost because publishing to Redis failed",
            ["provider"],
            registry=registry,
        )

    # --- Agent lifecycle ---

    def agent_started(self, provider: str) -> None:
        self.active_agents.labels(provider=provider).inc()

    def agent_stopped(self, provider: str) -> None:
        self.active_agents.labels(provider=provider).dec()

    def agent_start_failed(self, provider: str, reason: str) -> None:
        self.agent_start_failures.labels(provider=provider, reason=reason).inc()

    # --- Session lifecycle ---

    def session_started(self, provider: str, locale: str) -> None:
        self.active_sessions.labels(
            provider=provider, language=language_label(locale)
        ).inc()
        self.session_starts.labels(provider=provider).inc()

    def session_stopped(self, provider: str, locale: str) -> None:
        self.active_sessions.labels(
            provider=provider, language=language_label(locale)
        ).dec()

    def session_ended(self, provider: str, reason: str) -> None:
        self.session_ends.labels(provider=provider, reason=reason).inc()

    def session_start_failed(self, provider: str, reason: str) -> None:
        self.session_start_failures.labels(provider=provider, reason=reason).inc()

    # --- Transcripts ---

    def transcript_emitted(
        self,
        provider: str,
        event_type: str,
        *,
        confidence: float | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        """Record a transcript that passed filtering and was sent to BBB.

        ``confidence`` is None for providers that do not report one.
        """
        self.transcripts.labels(provider=provider, event_type=event_type).inc()

        if confidence is not None:
            self.transcript_confidence.labels(
                provider=provider, event_type=event_type
            ).observe(confidence)

        if duration_seconds is not None:
            self.utterance_duration.labels(
                provider=provider, event_type=event_type
            ).observe(duration_seconds)

    def transcript_discarded(self, provider: str, event_type: str, reason: str) -> None:
        self.transcripts_discarded.labels(
            provider=provider, event_type=event_type, reason=reason
        ).inc()

    def provider_audio_observed(self, provider: str, seconds: float) -> None:
        """Accumulate audio seconds sent to the provider, as its plugin reports them.

        The closest available proxy for provider billing. Non-positive values are
        ignored because prometheus_client raises on a negative increment.
        """
        if seconds <= 0:
            return

        self.provider_audio_seconds.labels(provider=provider).inc(seconds)

    def language_checked(
        self,
        provider: str,
        *,
        requested_lang: str | None,
        reported_lang: str | None,
        translation_enabled: bool,
    ) -> None:
        """Count transcripts whose language differs from the requested one.

        Surfaces both code-switching and misdetection. Two exclusions are
        mandatory or the metric becomes actively misleading:

        - Translation: Gladia suppresses the original-language final transcript
          and emits one carrying the *target* language, so every translated
          utterance would register as a mismatch.
        - Auto-detection: the participant asked the provider to choose, so there
          is no requested language to differ from.
        """
        if translation_enabled or requested_lang is None or not reported_lang:
            return

        if reported_lang != requested_lang:
            self.language_detection_mismatch.labels(provider=provider).inc()

    def transcript_publish_failed(self, provider: str) -> None:
        self.transcript_publish_failures.labels(provider=provider).inc()


stt_metrics = SttMetrics()
