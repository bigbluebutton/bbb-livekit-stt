import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from livekit.agents import stt
from livekit.plugins.gladia import STT as GladiaSTT

from config import (
    _get_bool_env,
    _get_float_env,
    _get_json_env,
    _get_list_env,
    _get_map_env,
)
from providers.base import BaseSttAgent, BaseSttConfig

DEFAULT_TRANSLATION_LANG_MAP = "de:de-DE,en:en-US,es:es-ES,fr:fr-FR,hi:hi-IN,it:it-IT,ja:ja-JP,pt:pt-BR,ru:ru-RU,zh:zh-CN"


# This is a mapping of the Gladia STT plugin configuration options to environment variables.
# The plugin uses the defaults if the environment variables are not set.
# See https://github.com/livekit/livekit-agents/blob/main/livekit/plugins/gladia/stt.py
@dataclass
class GladiaConfig(BaseSttConfig):
    api_key: str | None = field(default_factory=lambda: os.getenv("GLADIA_API_KEY"))
    min_confidence_interim: float = field(
        default_factory=lambda: _get_float_env(
            "GLADIA_MIN_CONFIDENCE_INTERIM",
            _get_float_env("GLADIA_MIN_CONFIDENCE", 0.1),
        )
    )
    min_confidence_final: float = field(
        default_factory=lambda: _get_float_env(
            "GLADIA_MIN_CONFIDENCE_FINAL",
            _get_float_env("GLADIA_MIN_CONFIDENCE", 0.1),
        )
    )
    model: str | None = field(default_factory=lambda: os.getenv("GLADIA_MODEL", None))
    base_url: str | None = field(
        default_factory=lambda: os.getenv("GLADIA_BASE_URL", None)
    )
    interim_results: bool | None = field(
        default_factory=lambda: _get_bool_env("GLADIA_INTERIM_RESULTS", None)
    )
    languages: List[str] | None = field(
        default_factory=lambda: _get_list_env("GLADIA_LANGUAGES", None)
    )
    code_switching: bool | None = field(
        default_factory=lambda: _get_bool_env("GLADIA_CODE_SWITCHING", False)
    )
    sample_rate: int = field(
        default_factory=lambda: int(os.getenv("GLADIA_SAMPLE_RATE", 16000))
    )
    bit_depth: int = field(
        default_factory=lambda: int(os.getenv("GLADIA_BIT_DEPTH", 16))
    )
    channels: int = field(default_factory=lambda: int(os.getenv("GLADIA_CHANNELS", 1)))
    encoding: Literal["wav/pcm", "wav/alaw", "wav/ulaw"] = field(
        default_factory=lambda: os.getenv("GLADIA_ENCODING", "wav/pcm")
    )
    endpointing: float | None = field(
        default_factory=lambda: _get_float_env("GLADIA_ENDPOINTING", 0.01)
    )
    maximum_duration_without_endpointing: float | None = field(
        default_factory=lambda: _get_float_env(
            "GLADIA_MAXIMUM_DURATION_WITHOUT_ENDPOINTING", None
        )
    )
    region: str | None = field(default_factory=lambda: os.getenv("GLADIA_REGION", None))
    energy_filter: bool | None = field(
        default_factory=lambda: _get_bool_env("GLADIA_ENERGY_FILTER", None)
    )

    translation_enabled: bool | None = field(
        default_factory=lambda: _get_bool_env("GLADIA_TRANSLATION_ENABLED", None)
    )
    translation_target_languages: List[str] | None = field(
        default_factory=lambda: _get_list_env(
            "GLADIA_TRANSLATION_TARGET_LANGUAGES", None
        )
    )
    translation_model: str | None = field(
        default_factory=lambda: os.getenv("GLADIA_TRANSLATION_MODEL", None)
    )
    translation_match_original_utterances: bool | None = field(
        default_factory=lambda: _get_bool_env(
            "GLADIA_TRANSLATION_MATCH_ORIGINAL_UTTERANCES", None
        )
    )
    translation_lipsync: bool | None = field(
        default_factory=lambda: _get_bool_env("GLADIA_TRANSLATION_LIPSYNC", None)
    )
    translation_context_adaptation: bool | None = field(
        default_factory=lambda: _get_bool_env(
            "GLADIA_TRANSLATION_CONTEXT_ADAPTATION", None
        )
    )
    translation_context: str | None = field(
        default_factory=lambda: os.getenv("GLADIA_TRANSLATION_CONTEXT", None)
    )
    translation_informal: bool | None = field(
        default_factory=lambda: _get_bool_env("GLADIA_TRANSLATION_INFORMAL", None)
    )

    translation_lang_map: Dict[str, str] = field(
        default_factory=lambda: _get_map_env(
            "GLADIA_TRANSLATION_LANG_MAP", DEFAULT_TRANSLATION_LANG_MAP
        )
    )

    custom_vocabulary: List[Any] | None = field(
        default_factory=lambda: _get_json_env("GLADIA_CUSTOM_VOCABULARY")
    )
    custom_spelling: Dict[str, List[str]] | None = field(
        default_factory=lambda: _get_json_env("GLADIA_CUSTOM_SPELLING")
    )

    pre_processing_audio_enhancer: bool = field(
        default_factory=lambda: _get_bool_env(
            "GLADIA_PRE_PROCESSING_AUDIO_ENHANCER", True
        )
    )
    pre_processing_speech_threshold: float | None = field(
        default_factory=lambda: _get_float_env(
            "GLADIA_PRE_PROCESSING_SPEECH_THRESHOLD", 0.7
        )
    )

    def to_stt_kwargs(self):
        # Exclude None values so defaults are used by the plugin
        data = {
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "interim_results": self.interim_results,
            "languages": self.languages,
            "code_switching": self.code_switching,
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "channels": self.channels,
            "encoding": self.encoding,
            "endpointing": self.endpointing,
            "maximum_duration_without_endpointing": self.maximum_duration_without_endpointing,
            "region": self.region,
            "energy_filter": self.energy_filter,
            "translation_enabled": self.translation_enabled,
            "translation_target_languages": self.translation_target_languages,
            "translation_model": self.translation_model,
            "translation_match_original_utterances": self.translation_match_original_utterances,
            "translation_lipsync": self.translation_lipsync,
            "translation_context_adaptation": self.translation_context_adaptation,
            "translation_context": self.translation_context,
            "translation_informal": self.translation_informal,
            "custom_vocabulary": self.custom_vocabulary,
            "custom_spelling": self.custom_spelling,
            "pre_processing_audio_enhancer": self.pre_processing_audio_enhancer,
            "pre_processing_speech_threshold": self.pre_processing_speech_threshold,
        }
        return {k: v for k, v in data.items() if v is not None}


gladia_config = GladiaConfig()


class GladiaSttAgent(BaseSttAgent):
    def __init__(self, config: GladiaConfig):
        super().__init__(config)
        self.stt = GladiaSTT(**config.to_stt_kwargs())

    @property
    def translation_lang_map(self) -> Dict[str, str]:
        return self.config.translation_lang_map

    def _create_stt_stream(self, locale: str | None) -> stt.SpeechStream:
        # A None locale ("auto") omits the language so Gladia auto-detects.
        if locale is None:
            return self.stt.stream()

        return self.stt.stream(language=locale)

    def _update_stream_locale(self, user_id: str, locale: str):
        stream = self.processing_info[user_id]["stream"]
        sanitized_locale = self._sanitize_locale(locale)
        # An empty languages list re-enables Gladia's auto-detection.
        stream.update_options(languages=[sanitized_locale] if sanitized_locale else [])

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
