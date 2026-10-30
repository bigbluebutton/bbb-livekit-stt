import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

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


def _get_int_env(key: str, default: int | None) -> int | None:
    val = os.getenv(key)

    if val is None:
        return default

    try:
        return int(val)
    except ValueError:
        print(f"Warning: Could not parse {key} as an integer, using {default}")
        return default


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


@dataclass
class MetricsConfig:
    # Opt-in: unset means the worker exposes no /metrics listener.
    prometheus_port: int | None = field(
        default_factory=lambda: _get_int_env("BBB_STT_PROMETHEUS_PORT", None)
    )


metrics_config = MetricsConfig()

stt_provider = os.getenv("STT_PROVIDER", "gladia").lower()


def redact_config_values(value: object, key: str | None = None) -> object:
    if key and key.lower() in REDACTED_CONFIG_KEYS:
        return "***REDACTED***" if value not in (None, "") else value

    if isinstance(value, dict):
        return {k: redact_config_values(v, k) for k, v in value.items()}

    if isinstance(value, list):
        return [redact_config_values(item) for item in value]

    return value


def get_redacted_app_config(stt_config) -> Dict[str, Any]:
    config_payload = {
        "redis": asdict(redis_config),
        "metrics": asdict(metrics_config),
        "stt": asdict(stt_config),
    }
    return redact_config_values(config_payload)
