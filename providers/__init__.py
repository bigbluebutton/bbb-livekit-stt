from providers.base import BaseSttAgent, BaseSttConfig as BaseSttConfig


def create_agent(provider: str) -> BaseSttAgent:
    if provider == "gladia":
        from providers.gladia import GladiaSttAgent, gladia_config

        return GladiaSttAgent(gladia_config)
    raise ValueError(f"Unknown STT provider: {provider}")
