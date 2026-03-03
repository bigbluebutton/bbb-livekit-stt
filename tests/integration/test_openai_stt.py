"""Integration tests for the OpenAI STT pipeline.

These tests require a valid OPENAI_API_KEY environment variable and make real
requests to the OpenAI transcription service.  They are skipped automatically
when the key is absent.
"""

import asyncio
import os

import pytest
from livekit import rtc
from livekit.agents import stt
from livekit.plugins import openai as openai_plugin

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable is not set",
)


@pytest.mark.integration
@pytest.mark.usefixtures("job_process")
async def test_openai_stt_stream_opens_and_closes():
    """Verify that an OpenAI STT stream can be created and closed without errors."""
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("OPENAI_STT_MODEL", "gpt-4o-transcribe")
    base_url = os.environ.get("OPENAI_BASE_URL")

    kwargs = {"api_key": api_key, "model": model}
    if base_url:
        kwargs["base_url"] = base_url

    async with openai_plugin.STT(**kwargs) as openai_stt:
        stream = openai_stt.stream(language="en")
        await stream.aclose()


@pytest.mark.integration
@pytest.mark.usefixtures("job_process")
async def test_openai_stt_stream_accepts_silent_audio():
    """Verify that the OpenAI STT stream processes silent PCM audio without errors.

    This tests end-to-end connectivity: frames are pushed through the STT
    stream, the stream is flushed, and no exceptions are raised.  Silent audio
    is expected to produce no transcript events.
    """
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("OPENAI_STT_MODEL", "gpt-4o-transcribe")
    base_url = os.environ.get("OPENAI_BASE_URL")

    kwargs = {"api_key": api_key, "model": model}
    if base_url:
        kwargs["base_url"] = base_url

    async with openai_plugin.STT(**kwargs) as openai_stt:
        stream = openai_stt.stream(language="en")

        # Build a 100 ms silent PCM frame (16-bit mono @ 16 kHz → 1600 samples)
        samples_per_frame = 1600
        silent_frame = rtc.AudioFrame(
            data=bytes(samples_per_frame * 2),  # 2 bytes per int16 sample
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=samples_per_frame,
        )

        events_received = []

        async def collect_events():
            async for event in stream:
                events_received.append(event)

        collector = asyncio.create_task(collect_events())

        # Push 500 ms of silence in five 100 ms chunks
        for _ in range(5):
            stream.push_frame(silent_frame)
        stream.flush()

        # Give the service a moment to respond, then close
        await asyncio.sleep(3)
        await stream.aclose()
        collector.cancel()
        try:
            await collector
        except asyncio.CancelledError:
            pass

        # Silent audio should not produce any FINAL_TRANSCRIPT events
        final_transcripts = [
            e for e in events_received if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert len(final_transcripts) == 0
