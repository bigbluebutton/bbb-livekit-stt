import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from livekit import rtc
from livekit.agents import stt

import main
from events import EventEmitter


class StubAgent(EventEmitter):
    """Stands in for a provider agent: main.py only needs its events and settings."""

    provider_name = "gladia"
    reports_confidence = True
    translation_enabled = False
    translation_lang_map = {"pt": "pt-BR"}

    def __init__(self):
        super().__init__()
        self.config = MagicMock()
        self.participant_settings = {}
        self.processing_info = {}
        self.room = MagicMock()
        self.room.name = "meeting-1"
        self.running = asyncio.Event()
        self.release = asyncio.Event()

    async def start(self, ctx):
        self.running.set()
        await self.release.wait()


def _participant(identity="user_1"):
    participant = MagicMock(spec=rtc.RemoteParticipant)
    participant.identity = identity
    return participant


def _final_transcript(start_time, end_time, language="pt"):
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                text="teste de transcrição",
                language=language,
                start_time=start_time,
                end_time=end_time,
                confidence=0.9,
            )
        ],
    )


@pytest.fixture
async def entrypoint():
    """Run main.entrypoint far enough to register its transcript handlers."""
    agent = StubAgent()
    redis = MagicMock()
    redis.connect = AsyncMock(return_value=True)
    redis.listen = AsyncMock()
    redis.aclose = AsyncMock()
    redis.publish_update_transcript_pub_msg = AsyncMock(return_value=True)

    ctx = MagicMock()
    ctx.room.name = "meeting-1"

    with (
        patch("main.create_agent", return_value=agent),
        patch("main.RedisManager", return_value=redis),
        patch("main.nest_asyncio.apply"),
        patch("main.get_redacted_app_config", return_value={}),
    ):
        task = asyncio.create_task(main.entrypoint(ctx))
        done, _ = await asyncio.wait(
            [task, asyncio.create_task(agent.running.wait())],
            timeout=2,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task in done:
            # Surface whatever made the entrypoint give up early.
            await task

        yield agent, redis

        agent.release.set()
        await asyncio.wait_for(task, timeout=2)


class TestTranscriptPublication:
    async def test_stamps_the_transcript_with_the_emitted_epoch(self, entrypoint):
        """The epoch travels with the transcript: it is the moment that
        participant's own provider stream opened."""
        agent, redis = entrypoint
        agent.participant_settings["user_1"] = {
            "locale": "pt-BR",
            "provider": "gladia",
        }

        agent.emit(
            "final_transcript",
            participant=_participant(),
            event=_final_transcript(5.0, 6.5),
            open_time=1_000_000.0,
        )
        for _ in range(3):
            await asyncio.sleep(0)

        redis.publish_update_transcript_pub_msg.assert_awaited_once()
        args = redis.publish_update_transcript_pub_msg.await_args
        assert args.args[0] == "meeting-1"
        assert args.args[1] == "user_1"
        assert args.args[3] == "pt-BR"
        assert args.args[4] == math.floor(1_000_000.0 + 5.0)
        assert args.args[5] == math.floor(1_000_000.0 + 6.5)
        assert args.kwargs["result"] is True

    async def test_two_sessions_publish_against_their_own_epochs(self, entrypoint):
        """Regression: the handlers used to fall back to an agent-wide epoch, so
        a late joiner moved everyone else's timestamps."""
        agent, redis = entrypoint
        for identity in ("alice", "bob"):
            agent.participant_settings[identity] = {
                "locale": "pt-BR",
                "provider": "gladia",
            }

        agent.emit(
            "final_transcript",
            participant=_participant("alice"),
            event=_final_transcript(410.0, 411.0),
            open_time=1_000_000.0,
        )
        agent.emit(
            "final_transcript",
            participant=_participant("bob"),
            event=_final_transcript(2.0, 3.0),
            open_time=1_000_400.0,
        )
        for _ in range(4):
            await asyncio.sleep(0)

        published = {
            call.args[1]: call.args[4]
            for call in redis.publish_update_transcript_pub_msg.await_args_list
        }
        assert published == {"alice": 1_000_410, "bob": 1_000_402}

    async def test_interim_transcripts_need_the_partial_utterances_option(
        self, entrypoint
    ):
        agent, redis = entrypoint
        agent.participant_settings["user_1"] = {
            "locale": "pt-BR",
            "provider": "gladia",
        }

        interim = _final_transcript(1.0, 4.0)
        interim.type = stt.SpeechEventType.INTERIM_TRANSCRIPT

        agent.emit(
            "interim_transcript",
            participant=_participant(),
            event=interim,
            open_time=1_000_000.0,
        )
        for _ in range(3):
            await asyncio.sleep(0)

        redis.publish_update_transcript_pub_msg.assert_not_awaited()

        agent.participant_settings["user_1"]["partial_utterances"] = True
        agent.emit(
            "interim_transcript",
            participant=_participant(),
            event=interim,
            open_time=1_000_000.0,
        )
        for _ in range(3):
            await asyncio.sleep(0)

        redis.publish_update_transcript_pub_msg.assert_awaited_once()
        assert (
            redis.publish_update_transcript_pub_msg.await_args.kwargs["result"] is False
        )
