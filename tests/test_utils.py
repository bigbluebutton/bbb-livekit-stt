import logging

import pytest

from utils import (
    coerce_min_utterance_length_seconds,
    coerce_partial_utterances,
    resolve_bbb_locale,
)


class TestCoercePartialUtterances:
    def test_nonzero_int_returns_true(self):
        assert coerce_partial_utterances(1) is True
        assert coerce_partial_utterances(-1) is True

    def test_zero_int_returns_false(self):
        assert coerce_partial_utterances(0) is False

    def test_nonzero_float_returns_true(self):
        assert coerce_partial_utterances(0.5) is True

    def test_zero_float_returns_false(self):
        assert coerce_partial_utterances(0.0) is False

    @pytest.mark.parametrize(
        "value", [True, "true", "1", "t", "yes", "y", "TRUE", "YES"]
    )
    def test_truthy_values_return_true(self, value):
        assert coerce_partial_utterances(value) is True

    @pytest.mark.parametrize("value", [False, "false", "0", "f", "no", "n", "False"])
    def test_falsy_values_return_false(self, value):
        assert coerce_partial_utterances(value) is False

    def test_string_whitespace_is_stripped(self):
        assert coerce_partial_utterances("  true  ") is True
        assert coerce_partial_utterances("  false  ") is False

    def test_unknown_string_returns_default(self):
        assert coerce_partial_utterances("maybe", default=False) is False
        assert coerce_partial_utterances("maybe", default=True) is True

    def test_none_returns_default(self):
        assert coerce_partial_utterances(None, default=False) is False
        assert coerce_partial_utterances(None, default=True) is True


class TestCoerceMinUtteranceLengthSeconds:
    def test_none_returns_default(self):
        assert coerce_min_utterance_length_seconds(None) == pytest.approx(0.0)
        assert coerce_min_utterance_length_seconds(None, default=1.5) == pytest.approx(
            1.5
        )

    def test_empty_string_returns_default(self):
        assert coerce_min_utterance_length_seconds("") == pytest.approx(0.0)
        assert coerce_min_utterance_length_seconds("", default=2.0) == pytest.approx(
            2.0
        )

    def test_valid_float_string(self):
        assert coerce_min_utterance_length_seconds("3.5") == pytest.approx(3.5)

    def test_valid_int_string(self):
        result = coerce_min_utterance_length_seconds("2")
        assert result == pytest.approx(2.0)

    def test_zero_returns_zero(self):
        assert coerce_min_utterance_length_seconds(0) == pytest.approx(0.0)
        assert coerce_min_utterance_length_seconds("0") == pytest.approx(0.0)

    def test_invalid_string_returns_default_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = coerce_min_utterance_length_seconds("not_a_number", default=0.5)

        assert result == pytest.approx(0.5)
        assert any("Invalid" in r.message for r in caplog.records)

    def test_negative_value_is_clamped_to_default_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = coerce_min_utterance_length_seconds(-1.0, default=0.0)

        assert result == pytest.approx(0.0)
        assert any("Negative" in r.message for r in caplog.records)


LANG_MAP = {"en": "en-US", "pt": "pt-BR", "de": "de-DE"}


class TestResolveBbbLocale:
    def test_matching_language_returns_the_original_locale(self):
        """A transcript in the selected language keeps the participant's region."""
        assert resolve_bbb_locale("en", "en-GB", LANG_MAP) == "en-GB"

    def test_translated_language_is_mapped(self):
        assert resolve_bbb_locale("pt", "en-US", LANG_MAP) == "pt-BR"

    def test_unmapped_language_falls_back_to_the_language_code(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = resolve_bbb_locale("ja", "en-US", LANG_MAP)

        assert result == "ja"
        assert any("Could not find a BBB locale" in r.message for r in caplog.records)

    def test_auto_locale_resolves_from_the_detected_language(self):
        """With 'auto' there is no selected language, so the map always decides."""
        assert resolve_bbb_locale("pt", "auto", LANG_MAP) == "pt-BR"

    def test_returns_none_when_language_is_unknown(self):
        """Auto-detection plus a provider that reports no language leaves
        nothing to resolve - callers must drop the transcript."""
        assert resolve_bbb_locale(None, "auto", LANG_MAP) is None
        assert resolve_bbb_locale("", "auto", LANG_MAP) is None
