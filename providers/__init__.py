from providers.base import BaseSttAgent, BaseSttConfig as BaseSttConfig


def prewarm_provider(provider: str, userdata: dict) -> None:
    """Load a provider's slow assets once per worker process.

    Called from the LiveKit worker's prewarm hook, before any job is assigned.
    Providers with nothing to prewarm are a no-op, and an unknown provider is
    ignored here rather than raising: create_agent() is the single place that
    validates the name, and failing the prewarm would take down the worker
    before it could report the real error.
    """
    if provider == "voxtral-realtime":
        from providers import voxtral_realtime

        voxtral_realtime.prewarm(userdata)


def create_agent(provider: str, userdata: dict | None = None) -> BaseSttAgent:
    if provider == "gladia":
        from providers.gladia import GladiaSttAgent, gladia_config

        return GladiaSttAgent(gladia_config)
    if provider == "openai":
        from providers.openai import OpenAiSttAgent, openai_config

        return OpenAiSttAgent(openai_config)
    if provider == "voxtral-realtime":
        from providers.voxtral_realtime import (
            VAD_USERDATA_KEY,
            VoxtralRealtimeSttAgent,
            voxtral_realtime_config,
        )

        return VoxtralRealtimeSttAgent(
            voxtral_realtime_config,
            vad=userdata.get(VAD_USERDATA_KEY) if userdata else None,
        )
    raise ValueError(f"Unknown STT provider: {provider}")
