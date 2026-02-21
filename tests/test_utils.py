import logging

import pytest

from utils import coerce_min_utterance_length_seconds, coerce_partial_utterances


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
