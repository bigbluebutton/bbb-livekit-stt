"""Integration tests for the OpenAI STT pipeline.

These tests require a valid OPENAI_API_KEY environment variable and make real
requests to the OpenAI transcription service.  They are skipped automatically
when the key is absent.
"""

import os

import numpy as np
import pytest
from livekit import rtc

from providers.openai import OpenAiConfig, OpenAiSttAgent

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable is not set",
)


def _make_agent() -> OpenAiSttAgent:
    config = OpenAiConfig(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ.get("OPENAI_STT_MODEL", "gpt-4o-transcribe"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    return OpenAiSttAgent(config)


def _silent_wav_bytes(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Return WAV bytes containing silent PCM audio."""
    num_samples = int(duration_s * sample_rate)
    frame = rtc.AudioFrame(
        data=bytes(num_samples * 2),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=num_samples,
    )
    return frame.to_wav_bytes()


def _tone_wav_bytes(
    duration_s: float = 0.5, sample_rate: int = 16000, freq_hz: float = 440.0
) -> bytes:
    """Return WAV bytes containing a sine-wave tone (not speech)."""
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * freq_hz * t) * 16000).astype(np.int16)
    frame = rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=num_samples,
    )
    return frame.to_wav_bytes()


@pytest.mark.integration
async def test_transcribe_wav_silent_audio_returns_empty():
    """Silent audio should return an empty or near-empty transcript."""
    agent = _make_agent()
    try:
        result = await agent._transcribe_wav(_silent_wav_bytes(), language="en")
        assert isinstance(result, str)
        # Silent audio may return empty or a minimal artefact — just no long text.
        assert len(result) < 20, f"Unexpected transcript for silence: {result!r}"
    finally:
        await agent._cleanup()


@pytest.mark.integration
async def test_transcribe_wav_returns_string():
    """A tone clip should return a string (possibly empty) without raising."""
    agent = _make_agent()
    try:
        result = await agent._transcribe_wav(_tone_wav_bytes(), language="en")
        assert isinstance(result, str)
    finally:
        await agent._cleanup()


@pytest.mark.integration
async def test_cleanup_closes_http_session():
    """_cleanup() should close the aiohttp session opened during a request."""
    agent = _make_agent()
    # Trigger session creation via a real request
    await agent._transcribe_wav(_silent_wav_bytes(), language="en")
    assert agent._http_session is not None
    await agent._cleanup()
    assert agent._http_session is None
