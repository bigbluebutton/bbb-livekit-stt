import os
from dataclasses import dataclass, field
from typing import List, Literal, Dict

DEFAULT_TRANSLATION_LANG_MAP = "de:de-DE,en:en-US,es:es-ES,fr:fr-FR,hi:hi-IN,it:it-IT,ja:ja-JP,pt:pt-BR,ru:ru-RU,zh:zh-CN"

def _get_float_env(key: str, default: float) -> float:
    val = os.getenv(key)

    if val is None:
        return default

    return float(val)

def _get_bool_env(key: str, default: bool | None) -> bool | None:
    val = os.getenv(key)

    if val is None:
        return default

    return val.lower() in ('true', '1', 't')

def _get_list_env(key: str, default: List[str] | None) -> List[str] | None:
    val = os.getenv(key)

    if val is None:
        return default

    if not val:
        return []

    return [item.strip() for item in val.split(',')]

def _get_map_env(key: str, default_str: str = "") -> Dict[str, str]:
    val = os.getenv(key, default_str)

    if not val:
        return {}

    lang_map = {}

    try:
        pairs = val.split(',')
        for pair in pairs:
            if ':' in pair:
                lang, bbb_locale = pair.split(':', 1)
                lang_map[lang.strip()] = bbb_locale.strip()
    except Exception as e:
        print(f"Warning: Could not parse {key}: {e}")

    return lang_map

@dataclass
class RedisConfig:
    host: str = field(default_factory=lambda: os.getenv('REDIS_HOST', '127.0.0.1'))
    port: int = field(default_factory=lambda: int(os.getenv('REDIS_PORT', 6379)))
    password: str = field(default_factory=lambda: os.getenv('REDIS_PASSWORD', ''))

redis_config = RedisConfig()

# This is a mapping of the Gladia STT plugin configuration options to environment variables.
# The plugin uses the defaults if the environment variables are not set.
# See https://github.com/livekit/livekit-agents/blob/main/livekit/plugins/gladia/stt.py
@dataclass
class GladiaConfig:
    api_key: str | None = field(default_factory=lambda: os.getenv('GLADIA_API_KEY'))
    interim_results: bool | None = field(default_factory=lambda: _get_bool_env('GLADIA_INTERIM_RESULTS', None))
    languages: List[str] | None = field(default_factory=lambda: _get_list_env('GLADIA_LANGUAGES', None))
    code_switching: bool | None = field(default_factory=lambda: _get_bool_env('GLADIA_CODE_SWITCHING', None))
    sample_rate: int = field(default_factory=lambda: int(os.getenv('GLADIA_SAMPLE_RATE', 16000)))
    bit_depth: int = field(default_factory=lambda: int(os.getenv('GLADIA_BIT_DEPTH', 16)))
    channels: int = field(default_factory=lambda: int(os.getenv('GLADIA_CHANNELS', 1)))
    encoding: Literal["wav/pcm", "wav/alaw", "wav/ulaw"] = field(default_factory=lambda: os.getenv('GLADIA_ENCODING', "wav/pcm"))
    energy_filter: bool | None = field(default_factory=lambda: _get_bool_env('GLADIA_ENERGY_FILTER', None))

    translation_enabled: bool | None = field(default_factory=lambda: _get_bool_env('GLADIA_TRANSLATION_ENABLED', None))
    translation_target_languages: List[str] | None = field(default_factory=lambda: _get_list_env('GLADIA_TRANSLATION_TARGET_LANGUAGES', None))
    translation_model: str | None = field(default_factory=lambda: os.getenv('GLADIA_TRANSLATION_MODEL', None))
    translation_match_original_utterances: bool | None = field(default_factory=lambda: _get_bool_env('GLADIA_TRANSLATION_MATCH_ORIGINAL_UTTERANCES', None))
    translation_lipsync: bool | None = field(default_factory=lambda: _get_bool_env('GLADIA_TRANSLATION_LIPSYNC', None))
    translation_context_adaptation: bool | None = field(default_factory=lambda: _get_bool_env('GLADIA_TRANSLATION_CONTEXT_ADAPTATION', None))
    translation_context: str | None = field(default_factory=lambda: os.getenv('GLADIA_TRANSLATION_CONTEXT', None))
    translation_informal: bool | None = field(default_factory=lambda: _get_bool_env('GLADIA_TRANSLATION_INFORMAL', None))

    translation_lang_map: Dict[str, str] = field(default_factory=lambda: _get_map_env('GLADIA_TRANSLATION_LANG_MAP', DEFAULT_TRANSLATION_LANG_MAP))

    custom_vocabulary: List[str] | None = field(default_factory=lambda: _get_list_env('GLADIA_CUSTOM_VOCABULARY', None))

    pre_processing_audio_enhancer: bool = field(default_factory=lambda: _get_bool_env('GLADIA_PRE_PROCESSING_AUDIO_ENHANCER', False))
    pre_processing_speech_threshold: float | None = field(default_factory=lambda: _get_float_env('GLADIA_PRE_PROCESSING_SPEECH_THRESHOLD', None))

    def to_dict(self):
        # Exclude None values so defaults are used by the plugin
        data = {
            "api_key": self.api_key,
            "interim_results": self.interim_results,
            "languages": self.languages,
            "code_switching": self.code_switching,
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "channels": self.channels,
            "encoding": self.encoding,
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
            "pre_processing_audio_enhancer": self.pre_processing_audio_enhancer,
            "pre_processing_speech_threshold": self.pre_processing_speech_threshold,
        }
        return {k: v for k, v in data.items() if v is not None}

gladia_config = GladiaConfig()
