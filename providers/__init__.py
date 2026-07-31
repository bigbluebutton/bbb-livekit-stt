from providers.base import BaseSttAgent, BaseSttConfig as BaseSttConfig


def create_agent(provider: str) -> BaseSttAgent:
    if provider == "gladia":
        from providers.gladia import GladiaSttAgent, gladia_config

        return GladiaSttAgent(gladia_config)
    if provider == "openai":
        from providers.openai import OpenAiSttAgent, openai_config

        return OpenAiSttAgent(openai_config)
    if provider == "voxtral-realtime":
        from providers.voxtral_realtime import (
            VoxtralRealtimeSttAgent,
            voxtral_realtime_config,
        )

        return VoxtralRealtimeSttAgent(voxtral_realtime_config)
    raise ValueError(f"Unknown STT provider: {provider}")
