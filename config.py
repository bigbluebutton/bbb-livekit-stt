import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal

DEFAULT_TRANSLATION_LANG_MAP = "de:de-DE,en:en-US,es:es-ES,fr:fr-FR,hi:hi-IN,it:it-IT,ja:ja-JP,pt:pt-BR,ru:ru-RU,zh:zh-CN"
REDACTED_CONFIG_KEYS = {"api_key", "password", "secret", "token"}


def _get_float_env(key: str, default: float) -> float:
    val = os.getenv(key)

    if val is None:
        return default

    return float(val)


def _get_bool_env(key: str, default: bool | None) -> bool | None:
    val = os.getenv(key)

    if val is None:
        return default

    return val.lower() in ("true", "1", "t")


def _get_list_env(key: str, default: List[str] | None) -> List[str] | None:
    val = os.getenv(key)

    if val is None:
        return default

    if not val:
        return []

    return [item.strip() for item in val.split(",")]


def _get_json_env(key: str) -> Any | None:
    val = os.getenv(key)
    if not val:
        return None
    try:
        return json.loads(val)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse {key} as JSON: {e}")
        return None


def _get_map_env(key: str, default_str: str = "") -> Dict[str, str]:
    val = os.getenv(key, default_str)

    if not val:
        return {}

    lang_map = {}

    try:
        pairs = val.split(",")
        for pair in pairs:
            if ":" in pair:
                lang, bbb_locale = pair.split(":", 1)
                lang_map[lang.strip()] = bbb_locale.strip()
    except Exception as e:
        print(f"Warning: Could not parse {key}: {e}")

    return lang_map


@dataclass
class RedisConfig:
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", 6379)))
    password: str = field(default_factory=lambda: os.getenv("REDIS_PASSWORD", ""))


redis_config = RedisConfig()


# This is a mapping of the Gladia STT plugin configuration options to environment variables.
# The plugin uses the defaults if the environment variables are not set.
# See https://github.com/livekit/livekit-agents/blob/main/livekit/plugins/gladia/stt.py
@dataclass
class GladiaConfig:
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
        default_factory=lambda: _get_float_env("GLADIA_ENDPOINTING", None)
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

    def to_dict(self):
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


def redact_config_values(value: object, key: str | None = None) -> object:
    if key and key.lower() in REDACTED_CONFIG_KEYS:
        return "***REDACTED***" if value not in (None, "") else value

    if isinstance(value, dict):
        return {k: redact_config_values(v, k) for k, v in value.items()}

    if isinstance(value, list):
        return [redact_config_values(item) for item in value]

    return value


def get_redacted_app_config() -> Dict[str, Any]:
    config_payload = {
        "redis": asdict(redis_config),
        "gladia": asdict(gladia_config),
    }
    return redact_config_values(config_payload)
