from unittest.mock import AsyncMock, MagicMock, patch

from livekit import rtc
from livekit.agents import stt
from livekit.plugins.gladia import STT as GladiaSTT
from prometheus_client import CollectorRegistry

from metrics import DISCARD_LOW_CONFIDENCE, SttMetrics
from providers.gladia import GladiaSttAgent, GladiaConfig


def _make_gladia_agent(**kwargs):
    defaults = {"api_key": "fake-key"}
    defaults.update(kwargs)
    config = GladiaConfig(**defaults)
    with patch("providers.gladia.GladiaSTT", spec=GladiaSTT):
        agent = GladiaSttAgent(config)
    return agent


def _make_speech_event(event_type, confidence):
    event = MagicMock()
    event.type = event_type
    alt = MagicMock()
    alt.confidence = confidence
    event.alternatives = [alt]
    return event


class TestGladiaShouldEmit:
    def test_allows_final_above_threshold(self):
        agent = _make_gladia_agent(min_confidence_final=0.5)
        event = _make_speech_event(stt.SpeechEventType.FINAL_TRANSCRIPT, 0.8)
        assert agent._should_emit(event) is True

    def test_blocks_final_below_threshold(self):
        agent = _make_gladia_agent(min_confidence_final=0.5)
        event = _make_speech_event(stt.SpeechEventType.FINAL_TRANSCRIPT, 0.3)
        assert agent._should_emit(event) is False

    def test_allows_interim_above_threshold(self):
        agent = _make_gladia_agent(min_confidence_interim=0.2)
        event = _make_speech_event(stt.SpeechEventType.INTERIM_TRANSCRIPT, 0.5)
        assert agent._should_emit(event) is True

    def test_blocks_interim_below_threshold(self):
        agent = _make_gladia_agent(min_confidence_interim=0.5)
        event = _make_speech_event(stt.SpeechEventType.INTERIM_TRANSCRIPT, 0.1)
        assert agent._should_emit(event) is False

    def test_allows_unknown_event_types(self):
        agent = _make_gladia_agent()
        event = MagicMock()
        event.type = stt.SpeechEventType.END_OF_SPEECH
        assert agent._should_emit(event) is True


class TestGladiaTranslationLangMap:
    def test_returns_config_translation_lang_map(self):
        lang_map = {"en": "en-US", "fr": "fr-FR"}
        agent = _make_gladia_agent(translation_lang_map=lang_map)
        assert agent.translation_lang_map == lang_map

    def test_returns_empty_dict_when_map_is_empty(self):
        agent = _make_gladia_agent(translation_lang_map={})
        assert agent.translation_lang_map == {}


def _make_metered_gladia_agent(**config_kwargs):
    registry = CollectorRegistry()
    defaults = {"api_key": "fake-key"}
    defaults.update(config_kwargs)
    with patch("providers.gladia.GladiaSTT", spec=GladiaSTT):
        agent = GladiaSttAgent(GladiaConfig(**defaults), metrics=SttMetrics(registry))
    return agent, registry


class TestLowConfidenceIsRecorded:
    def test_discarding_a_low_confidence_transcript_is_counted(self):
        """A misconfigured confidence threshold rejects every transcript, which
        looks like an empty room. The counter tells the two apart."""
        agent, registry = _make_metered_gladia_agent(min_confidence_final=0.5)
        event = stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language="en", text="hello", confidence=0.1)],
        )

        assert agent._should_emit(event) is False
        assert (
            registry.get_sample_value(
                "bbb_stt_transcripts_discarded_total",
                {
                    "provider": "gladia",
                    "event_type": "final",
                    "reason": DISCARD_LOW_CONFIDENCE,
                },
            )
            == 1.0
        )

    def test_an_accepted_transcript_is_not_counted_as_discarded(self):
        agent, registry = _make_metered_gladia_agent(min_confidence_final=0.5)
        event = stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language="en", text="hello", confidence=0.9)],
        )

        assert agent._should_emit(event) is True
        assert (
            registry.get_sample_value(
                "bbb_stt_transcripts_discarded_total",
                {
                    "provider": "gladia",
                    "event_type": "final",
                    "reason": DISCARD_LOW_CONFIDENCE,
                },
            )
            is None
        )

    def test_a_low_confidence_interim_is_counted_as_an_interim(self):
        """Interims are discarded more often than finals, and not all are
        published. A separate label keeps them from hiding the final rate."""
        agent, registry = _make_metered_gladia_agent(min_confidence_interim=0.5)
        event = stt.SpeechEvent(
            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
            alternatives=[stt.SpeechData(language="en", text="hel", confidence=0.1)],
        )

        assert agent._should_emit(event) is False
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


class TestInterimsTheAgentDoesNotPublish:
    async def _discards_after_one_low_confidence_interim(
        self, interim_results, partial_utterances
    ):
        agent, registry = _make_metered_gladia_agent(min_confidence_interim=0.5)
        agent.config.interim_results = interim_results
        agent.participant_settings["user_1"] = {
            "locale": "en-US",
            "provider": "gladia",
            "partial_utterances": partial_utterances,
        }
        participant = MagicMock(spec=rtc.RemoteParticipant)
        participant.identity = "user_1"
        audio_stream = AsyncMock()
        audio_stream.__aiter__.return_value = iter([])
        stt_stream = AsyncMock()
        stt_stream.__aiter__.return_value = iter(
            [
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(language="en", text="hel", confidence=0.1)
                    ],
                )
            ]
        )
        stt_stream.end_input = MagicMock()

        with patch("providers.base.rtc.AudioStream", return_value=audio_stream):
            await agent._run_transcription_pipeline(
                participant, MagicMock(), stt_stream, "en-US"
            )

        return registry.get_sample_value(
            "bbb_stt_transcripts_discarded_total",
            {
                "provider": "gladia",
                "event_type": "interim",
                "reason": DISCARD_LOW_CONFIDENCE,
            },
        )

    async def test_with_interim_results_off_are_not_counted_as_discarded(self):
        """With interim_results unset the plugin still streams interims, but the
        agent publishes none."""
        assert await self._discards_after_one_low_confidence_interim(None, True) is None

    async def test_for_a_participant_without_partials_are_not_counted(self):
        """main.py publishes interims only to participants who asked for them."""
        assert (
            await self._discards_after_one_low_confidence_interim(True, False) is None
        )

    async def test_for_a_participant_with_partials_are_counted(self):
        assert await self._discards_after_one_low_confidence_interim(True, True) == 1.0
