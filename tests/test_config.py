import os

import pytest

from config import (
    GladiaConfig,
    _get_bool_env,
    _get_float_env,
    _get_json_env,
    _get_list_env,
    _get_map_env,
    redact_config_values,
)


class TestGetBoolEnv:
    def test_returns_default_when_not_set(self, monkeypatch):
        monkeypatch.delenv("TEST_BOOL", raising=False)
        assert _get_bool_env("TEST_BOOL", None) is None
        assert _get_bool_env("TEST_BOOL", True) is True
        assert _get_bool_env("TEST_BOOL", False) is False

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "t", "T"])
    def test_returns_true_for_truthy_strings(self, monkeypatch, value):
        monkeypatch.setenv("TEST_BOOL", value)
        assert _get_bool_env("TEST_BOOL", False) is True

    @pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "f", "no"])
    def test_returns_false_for_falsy_strings(self, monkeypatch, value):
        monkeypatch.setenv("TEST_BOOL", value)
        assert _get_bool_env("TEST_BOOL", True) is False

    def test_empty_string_is_falsy(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL", "")
        assert _get_bool_env("TEST_BOOL", True) is False


class TestGetFloatEnv:
    def test_returns_default_when_not_set(self, monkeypatch):
        monkeypatch.delenv("TEST_FLOAT", raising=False)
        assert _get_float_env("TEST_FLOAT", 0.5) == 0.5

    def test_parses_float_string(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "3.14")
        assert _get_float_env("TEST_FLOAT", 0.0) == pytest.approx(3.14)

    def test_raises_on_invalid_string(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "not-a-number")
        with pytest.raises(ValueError):
            _get_float_env("TEST_FLOAT", 0.0)


class TestGetListEnv:
    def test_returns_default_when_not_set(self, monkeypatch):
        monkeypatch.delenv("TEST_LIST", raising=False)
        assert _get_list_env("TEST_LIST", None) is None
        assert _get_list_env("TEST_LIST", ["a"]) == ["a"]

    def test_returns_empty_list_for_empty_string(self, monkeypatch):
        monkeypatch.setenv("TEST_LIST", "")
        assert _get_list_env("TEST_LIST", None) == []

    def test_parses_comma_separated_values(self, monkeypatch):
        monkeypatch.setenv("TEST_LIST", "en,fr,de")
        assert _get_list_env("TEST_LIST", None) == ["en", "fr", "de"]

    def test_strips_whitespace_from_items(self, monkeypatch):
        monkeypatch.setenv("TEST_LIST", "en, fr , de")
        assert _get_list_env("TEST_LIST", None) == ["en", "fr", "de"]

    def test_single_value(self, monkeypatch):
        monkeypatch.setenv("TEST_LIST", "en")
        assert _get_list_env("TEST_LIST", None) == ["en"]


class TestGetMapEnv:
    def test_parses_default_translation_map(self):
        result = _get_map_env(
            "NONEXISTENT_MAP_VAR",
            "de:de-DE,en:en-US,fr:fr-FR",
        )
        assert result == {"de": "de-DE", "en": "en-US", "fr": "fr-FR"}

    def test_returns_empty_dict_for_empty_string(self, monkeypatch):
        monkeypatch.setenv("TEST_MAP", "")
        assert _get_map_env("TEST_MAP") == {}

    def test_ignores_pairs_without_colon(self, monkeypatch):
        monkeypatch.setenv("TEST_MAP", "en:en-US,invalid,fr:fr-FR")
        result = _get_map_env("TEST_MAP")
        assert result == {"en": "en-US", "fr": "fr-FR"}

    def test_strips_whitespace_from_keys_and_values(self, monkeypatch):
        monkeypatch.setenv("TEST_MAP", " en : en-US , fr : fr-FR ")
        result = _get_map_env("TEST_MAP")
        assert result == {"en": "en-US", "fr": "fr-FR"}

    def test_overrides_via_env(self, monkeypatch):
        monkeypatch.setenv("TEST_MAP", "en:en-AU,pt:pt-PT")
        result = _get_map_env("TEST_MAP", "en:en-US")
        assert result == {"en": "en-AU", "pt": "pt-PT"}


class TestGladiaConfigToDict:
    def test_excludes_optional_none_fields(self):
        config = GladiaConfig(
            api_key="test-key",
            interim_results=None,
            languages=None,
            code_switching=None,
            energy_filter=None,
        )
        result = config.to_dict()
        assert "interim_results" not in result
        assert "languages" not in result
        assert "code_switching" not in result
        assert "energy_filter" not in result

    def test_includes_non_none_optional_fields(self):
        config = GladiaConfig(
            api_key="test-key",
            interim_results=True,
            languages=["en", "fr"],
            code_switching=False,
        )
        result = config.to_dict()
        assert result["api_key"] == "test-key"
        assert result["interim_results"] is True
        assert result["languages"] == ["en", "fr"]
        assert result["code_switching"] is False
        # Default non-None fields are always present
        assert "sample_rate" in result
        assert "bit_depth" in result
        assert "channels" in result
        assert "encoding" in result


class TestGetJsonEnv:
    def test_returns_none_when_not_set(self, monkeypatch):
        monkeypatch.delenv("TEST_JSON", raising=False)
        assert _get_json_env("TEST_JSON") is None

    def test_returns_none_for_empty_string(self, monkeypatch):
        monkeypatch.setenv("TEST_JSON", "")
        assert _get_json_env("TEST_JSON") is None

    def test_parses_json_list(self, monkeypatch):
        monkeypatch.setenv("TEST_JSON", '["hello", "world"]')
        assert _get_json_env("TEST_JSON") == ["hello", "world"]

    def test_parses_json_dict(self, monkeypatch):
        monkeypatch.setenv("TEST_JSON", '{"key": "value", "num": 42}')
        assert _get_json_env("TEST_JSON") == {"key": "value", "num": 42}

    def test_returns_none_for_invalid_json(self, monkeypatch):
        monkeypatch.setenv("TEST_JSON", "not valid json {")
        assert _get_json_env("TEST_JSON") is None


class TestRedactConfigValues:
    @pytest.mark.parametrize("key", ["api_key", "password", "secret", "token"])
    def test_redacts_sensitive_key_values(self, key):
        result = redact_config_values("some-secret-value", key=key)
        assert result == "***REDACTED***"

    def test_does_not_redact_non_sensitive_keys(self):
        result = redact_config_values("en-US", key="locale")
        assert result == "en-US"

    def test_preserves_none_value_for_sensitive_key(self):
        result = redact_config_values(None, key="api_key")
        assert result is None

    def test_preserves_empty_string_value_for_sensitive_key(self):
        result = redact_config_values("", key="api_key")
        assert result == ""

    def test_recursively_redacts_nested_dict(self):
        payload = {"api_key": "secret", "host": "localhost"}
        result = redact_config_values(payload)
        assert result["api_key"] == "***REDACTED***"
        assert result["host"] == "localhost"

    def test_recursively_redacts_deeply_nested_dict(self):
        payload = {"redis": {"password": "redis-pass", "host": "127.0.0.1"}}
        result = redact_config_values(payload)
        assert result["redis"]["password"] == "***REDACTED***"
        assert result["redis"]["host"] == "127.0.0.1"

    def test_passes_through_list_values_unchanged(self):
        result = redact_config_values(["en", "fr", "de"])
        assert result == ["en", "fr", "de"]


class TestGladiaConfigDefaults:
    @pytest.fixture(autouse=True)
    def _clean_gladia_env(self, monkeypatch):
        """Remove all GLADIA_* env vars so dataclass defaults are exercised."""
        for key in list(os.environ):
            if key.startswith("GLADIA_"):
                monkeypatch.delenv(key, raising=False)

    def test_code_switching_defaults_to_false(self):
        config = GladiaConfig()
        assert config.code_switching is False

    def test_pre_processing_audio_enhancer_defaults_to_true(self):
        config = GladiaConfig()
        assert config.pre_processing_audio_enhancer is True

    def test_pre_processing_speech_threshold_defaults_to_0_7(self):
        config = GladiaConfig()
        assert config.pre_processing_speech_threshold == pytest.approx(0.7)

    def test_min_confidence_defaults_to_0_1(self):
        config = GladiaConfig()
        assert config.min_confidence_interim == pytest.approx(0.1)
        assert config.min_confidence_final == pytest.approx(0.1)

    def test_min_confidence_interim_overrides_base(self, monkeypatch):
        monkeypatch.setenv("GLADIA_MIN_CONFIDENCE", "0.5")
        monkeypatch.setenv("GLADIA_MIN_CONFIDENCE_INTERIM", "0.2")
        config = GladiaConfig()
        assert config.min_confidence_interim == pytest.approx(0.2)
        assert config.min_confidence_final == pytest.approx(0.5)
