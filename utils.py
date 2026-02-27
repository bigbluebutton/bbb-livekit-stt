import logging


def coerce_partial_utterances(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "t", "yes", "y"}:
            return True
        if normalized in {"false", "0", "f", "no", "n"}:
            return False

    return default


def coerce_min_utterance_length_seconds(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logging.warning(
            f"Invalid minUtteranceLength value '{value}', falling back to {default}."
        )
        return default

    if parsed < 0:
        logging.warning(
            f"Negative minUtteranceLength value '{value}', clamping to {default}."
        )
        return default

    return parsed
