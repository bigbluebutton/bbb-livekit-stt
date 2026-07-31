import logging
from typing import Dict


def resolve_bbb_locale(
    transcript_lang: str | None,
    original_locale: str,
    translation_lang_map: Dict[str, str],
) -> str | None:
    """Resolve the BBB locale (<ISO 639-1>-<ISO 3166-1>) a transcript belongs to.

    Returns None when the transcript's language is unknown, which happens when
    the participant asked for auto-detection and the provider reported no
    language on the transcript. There is no locale to fall back to in that
    case, so callers must drop the transcript instead of publishing a null
    locale to BBB.
    """
    if not transcript_lang:
        return None

    is_auto = original_locale.lower() == "auto"

    if not is_auto and transcript_lang == original_locale.split("-")[0]:
        # The transcript is in the language the participant selected, so it
        # keeps their region suffix (e.g. "en" under "en-GB" stays "en-GB").
        return original_locale

    # Either a translation or an auto-detected language: map it back to BBB's
    # locale format.
    bbb_locale = translation_lang_map.get(transcript_lang)

    if not bbb_locale:
        logging.warning(
            f"Could not find a BBB locale mapping for language '{transcript_lang}'. "
            f"Falling back to the language code itself."
        )
        bbb_locale = transcript_lang

    return bbb_locale


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
