from unittest.mock import patch

import pytest
from livekit.plugins.gladia import STT as GladiaSTT

from providers import create_agent
from providers.gladia import GladiaSttAgent


class TestCreateAgent:
    def test_returns_gladia_agent_for_gladia_provider(self):
        with patch("providers.gladia.GladiaSTT", spec=GladiaSTT):
            agent = create_agent("gladia")
        assert isinstance(agent, GladiaSttAgent)

    def test_raises_for_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown STT provider"):
            create_agent("nonexistent")

    def test_case_insensitive_provider_name(self):
        with patch("providers.gladia.GladiaSTT", spec=GladiaSTT):
            agent = create_agent("gladia")
        assert isinstance(agent, GladiaSttAgent)
