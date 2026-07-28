from unittest.mock import MagicMock, patch

from livekit.agents import stt
from livekit.plugins.gladia import STT as GladiaSTT

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
