from unittest.mock import MagicMock, patch

import pytest
from livekit.plugins.gladia import STT as GladiaSTT

from providers import create_agent, prewarm_provider
from providers.gladia import GladiaSttAgent
from providers.openai import OpenAiSttAgent
from providers.voxtral_realtime import (
    VAD_USERDATA_KEY,
    VoxtralRealtimeSttAgent,
    voxtral_realtime_config,
)


@pytest.fixture
def _voxtral_base_url(monkeypatch):
    """The config singleton reads env at import; set the field directly."""
    monkeypatch.setattr(
        voxtral_realtime_config, "base_url", "https://vllm.example.com/v1"
    )


class TestCreateAgent:
    def test_returns_gladia_agent_for_gladia_provider(self):
        with patch("providers.gladia.GladiaSTT", spec=GladiaSTT):
            agent = create_agent("gladia")
        assert isinstance(agent, GladiaSttAgent)

    def test_returns_openai_agent_for_openai_provider(self):
        agent = create_agent("openai")
        assert isinstance(agent, OpenAiSttAgent)

    def test_raises_for_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown STT provider"):
            create_agent("nonexistent")

    def test_case_insensitive_provider_name(self):
        with patch("providers.gladia.GladiaSTT", spec=GladiaSTT):
            agent = create_agent("gladia")
        assert isinstance(agent, GladiaSttAgent)

    def test_voxtral_reuses_the_prewarmed_vad(self, _voxtral_base_url):
        prewarmed = MagicMock()
        userdata = {VAD_USERDATA_KEY: prewarmed}

        with patch("providers.voxtral_realtime._load_vad") as load:
            agent = create_agent("voxtral-realtime", userdata)

        assert isinstance(agent, VoxtralRealtimeSttAgent)
        assert agent._vad is prewarmed
        load.assert_not_called()

    def test_voxtral_loads_a_vad_when_the_process_was_not_prewarmed(
        self, _voxtral_base_url
    ):
        # Falling back keeps a worker started without the prewarm hook working,
        # at the cost of loading the model on the job's own event loop.
        with patch("providers.voxtral_realtime._load_vad") as load:
            agent = create_agent("voxtral-realtime")

        assert agent._vad is load.return_value
        load.assert_called_once()


class TestPrewarmProvider:
    def test_voxtral_stores_a_vad_in_userdata(self):
        userdata = {}
        with patch("providers.voxtral_realtime._load_vad") as load:
            prewarm_provider("voxtral-realtime", userdata)
        assert userdata[VAD_USERDATA_KEY] is load.return_value

    def test_providers_without_assets_are_a_no_op(self):
        userdata = {}
        prewarm_provider("gladia", userdata)
        prewarm_provider("openai", userdata)
        assert userdata == {}

    def test_unknown_provider_does_not_raise(self):
        # create_agent() is the single place that validates the name; failing
        # the prewarm would kill the worker before it could report the error.
        prewarm_provider("nonexistent", {})
