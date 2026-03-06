import logging
from typing import Dict

from livekit.agents import stt
from livekit.plugins.gladia import STT as GladiaSTT

from config import GladiaConfig
from providers.base import BaseSttAgent

gladia_config = GladiaConfig()


class GladiaSttAgent(BaseSttAgent):
    def __init__(self, config: GladiaConfig):
        super().__init__(config)
        self.stt = GladiaSTT(**config.to_stt_kwargs())

    @property
    def translation_lang_map(self) -> Dict[str, str]:
        return self.config.translation_lang_map

    def _create_stt_stream(self, locale: str) -> stt.SpeechStream:
        return self.stt.stream(language=locale)

    def _update_stream_locale(self, user_id: str, locale: str):
        stream = self.processing_info[user_id]["stream"]
        sanitized_locale = self._sanitize_locale(locale)
        stream.update_options(languages=[sanitized_locale])

    def _should_emit(self, event: stt.SpeechEvent) -> bool:
        if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            min_confidence = self.config.min_confidence_final
        elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            min_confidence = self.config.min_confidence_interim
        else:
            return True

        for alt in event.alternatives:
            if alt.confidence < min_confidence:
                logging.debug(
                    f"Discarding transcript: low confidence "
                    f"({alt.confidence} < {min_confidence})."
                )
                return False

        return True
