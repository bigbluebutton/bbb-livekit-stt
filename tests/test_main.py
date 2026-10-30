import asyncio
import json
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from livekit import rtc
from livekit.agents import stt

import main
from events import EventEmitter
from redis_manager import RedisManager


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
        self.stopped = asyncio.Event()
        self.locale_changes = []

    def handle_speech_locale_change(self, user_id, locale, provider):
        self.locale_changes.append((user_id, locale, provider))

    async def start(self, ctx):
        self.running.set()
        try:
            await self.release.wait()
        finally:
            self.stopped.set()

    async def shutdown(self):
        # Like BaseSttAgent: returns once start() has finished, or at once if
        # it never ran.
        self.release.set()
        if self.running.is_set():
            await self.stopped.wait()


def _participant(identity="user_1"):
    participant = MagicMock(spec=rtc.RemoteParticipant)
    participant.identity = identity
    return participant


def _speech_locale_message(locale, provider, meeting_id="meeting-1", user_id="user_1"):
    return json.dumps(
        {
            "envelope": {
                "name": "UserSpeechLocaleChangedEvtMsg",
                "routing": {"meetingId": meeting_id, "userId": user_id},
            },
            "core": {
                "header": {"name": "UserSpeechLocaleChangedEvtMsg"},
                "body": {"locale": locale, "provider": provider},
            },
        }
    )


def _speech_options_message(partial_utterances, min_utterance_length):
    return json.dumps(
        {
            "envelope": {
                "name": "UserSpeechOptionsChangedEvtMsg",
                "routing": {"meetingId": "meeting-1", "userId": "user_1"},
            },
            "core": {
                "header": {"name": "UserSpeechOptionsChangedEvtMsg"},
                "body": {
                    "partialUtterances": partial_utterances,
                    "minUtteranceLength": min_utterance_length,
                },
            },
        }
    )


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
    """Run main.entrypoint far enough to register its handlers.

    RedisManager's methods are patched one by one, not the whole class. The
    router compares event names with RedisManager's class attributes, so a
    mocked class would route nothing.
    """
    agent = StubAgent()
    publish = AsyncMock(return_value=True)
    listen = AsyncMock()

    ctx = MagicMock()
    ctx.room.name = "meeting-1"

    with (
        patch("main.create_agent", return_value=agent),
        patch.object(RedisManager, "connect", AsyncMock(return_value=True)),
        patch.object(RedisManager, "listen", listen),
        patch.object(RedisManager, "aclose", AsyncMock()),
        patch.object(RedisManager, "publish_update_transcript_pub_msg", publish),
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

        # The router is reachable only as the callback passed to listen().
        route = listen.await_args.args[0]

        yield agent, publish, route

        agent.release.set()
        await asyncio.wait_for(task, timeout=2)


class TestTranscriptPublication:
    async def test_stamps_the_transcript_with_the_emitted_epoch(self, entrypoint):
        """open_time comes with each transcript: it is when that participant's
        own provider stream opened."""
        agent, publish, _route = entrypoint
        agent.participant_settings["user_1"] = {
            "locale": "pt-BR",
            "provider": "gladia",
        }

        agent.emit(
            "final_transcript",
            participant=_participant(),
            event=_final_transcript(5.0, 6.5),
            open_time=1_000_000.0,
            locale="pt-BR",
        )
        for _ in range(3):
            await asyncio.sleep(0)

        publish.assert_awaited_once()
        args = publish.await_args
        assert args.args[0] == "meeting-1"
        assert args.args[1] == "user_1"
        assert args.args[3] == "pt-BR"
        assert args.args[4] == math.floor(1_000_000.0 + 5.0)
        assert args.args[5] == math.floor(1_000_000.0 + 6.5)
        assert args.kwargs["result"] is True

    async def test_two_sessions_publish_against_their_own_epochs(self, entrypoint):
        """A late joiner must not move anyone else's timestamps."""
        agent, publish, _route = entrypoint
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
            locale="pt-BR",
        )
        agent.emit(
            "final_transcript",
            participant=_participant("bob"),
            event=_final_transcript(2.0, 3.0),
            open_time=1_000_400.0,
            locale="pt-BR",
        )
        for _ in range(4):
            await asyncio.sleep(0)

        published = {call.args[1]: call.args[4] for call in publish.await_args_list}
        assert published == {"alice": 1_000_410, "bob": 1_000_402}

    async def test_publishes_under_the_locale_the_transcript_carries(self, entrypoint):
        """A transcript in flight while the participant switches language is
        published under the locale it was transcribed in, not the new one."""
        agent, publish, _route = entrypoint
        agent.participant_settings["user_1"] = {
            "locale": "pt-BR",
            "provider": "gladia",
        }

        agent.emit(
            "final_transcript",
            participant=_participant(),
            event=_final_transcript(1.0, 2.0, language="en"),
            open_time=1_000_000.0,
            locale="en-US",
        )
        for _ in range(3):
            await asyncio.sleep(0)

        publish.assert_awaited_once()
        assert publish.await_args.args[3] == "en-US"

    async def test_publishes_a_transcript_from_a_session_just_turned_off(
        self, entrypoint
    ):
        """Turning transcription off clears the stored locale. The final
        transcript of the last words can arrive after that. It was transcribed
        while transcription was on, so it is published."""
        agent, publish, _route = entrypoint
        agent.participant_settings["user_1"] = {"partial_utterances": True}

        agent.emit(
            "final_transcript",
            participant=_participant(),
            event=_final_transcript(1.0, 2.0),
            open_time=1_000_000.0,
            locale="pt-BR",
        )
        for _ in range(3):
            await asyncio.sleep(0)

        publish.assert_awaited_once()
        assert publish.await_args.args[3] == "pt-BR"

    async def test_interim_transcripts_need_the_partial_utterances_option(
        self, entrypoint
    ):
        agent, publish, _route = entrypoint
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
            locale="pt-BR",
        )
        for _ in range(3):
            await asyncio.sleep(0)

        publish.assert_not_awaited()

        agent.participant_settings["user_1"]["partial_utterances"] = True
        agent.emit(
            "interim_transcript",
            participant=_participant(),
            event=interim,
            open_time=1_000_000.0,
            locale="pt-BR",
        )
        for _ in range(3):
            await asyncio.sleep(0)

        publish.assert_awaited_once()
        assert publish.await_args.kwargs["result"] is False


class TestRedisRouting:
    """The router is the only thing between BBB's events and the agent."""

    async def test_a_complete_pair_reaches_the_agent(self, entrypoint):
        agent, _publish, route = entrypoint

        await route(_speech_locale_message("pt-BR", "gladia"))

        assert agent.locale_changes == [("user_1", "pt-BR", "gladia")]

    async def test_an_empty_pair_reaches_the_agent_as_it_arrived(self, entrypoint):
        """The agent decides what an incomplete locale/provider pair means."""
        agent, _publish, route = entrypoint

        await route(_speech_locale_message("", "gladia"))

        assert agent.locale_changes == [("user_1", "", "gladia")]

    async def test_ignores_events_for_another_meeting(self, entrypoint):
        agent, _publish, route = entrypoint

        await route(_speech_locale_message("pt-BR", "gladia", meeting_id="other"))

        assert agent.locale_changes == []

    async def test_ignores_unrelated_events(self, entrypoint):
        agent, _publish, route = entrypoint
        message = json.loads(_speech_locale_message("pt-BR", "gladia"))
        message["envelope"]["name"] = "SomeOtherEvtMsg"

        await route(json.dumps(message))

        assert agent.locale_changes == []

    async def test_speech_options_land_in_the_participant_settings(self, entrypoint):
        agent, _publish, route = entrypoint

        await route(_speech_options_message(True, 1))

        assert agent.participant_settings["user_1"] == {
            "partial_utterances": True,
            "min_utterance_length": 1.0,
        }

    async def test_a_malformed_message_does_not_take_the_listener_down(
        self, entrypoint
    ):
        agent, _publish, route = entrypoint

        await route("{not json")
        await route(_speech_locale_message("pt-BR", "gladia"))

        assert agent.locale_changes == [("user_1", "pt-BR", "gladia")]


class TestJobShutdown:
    async def test_shutdown_callback_tears_down_the_agent_and_redis(self):
        """LiveKit awaits a job's shutdown callbacks, not its entrypoint: the
        callback must finish the teardown by itself."""
        agent = StubAgent()
        closed = asyncio.Event()

        async def slow_close(*_):
            for _ in range(5):
                await asyncio.sleep(0)
            closed.set()

        aclose = AsyncMock(side_effect=slow_close)
        ctx = MagicMock()
        ctx.room.name = "meeting-1"

        with (
            patch("main.create_agent", return_value=agent),
            patch.object(RedisManager, "connect", AsyncMock(return_value=True)),
            patch.object(RedisManager, "listen", AsyncMock()),
            patch.object(RedisManager, "aclose", aclose),
            patch("main.nest_asyncio.apply"),
            patch("main.get_redacted_app_config", return_value={}),
        ):
            task = asyncio.create_task(main.entrypoint(ctx))
            await asyncio.wait_for(agent.running.wait(), timeout=2)

            (callback,) = ctx.add_shutdown_callback.call_args.args
            await asyncio.wait_for(callback(), timeout=2)

            assert agent.release.is_set()
            assert closed.is_set(), "the callback returned before Redis closed"
            aclose.assert_awaited_once()

            await asyncio.wait_for(task, timeout=2)
            aclose.assert_awaited_once()

    @pytest.mark.parametrize("connect_succeeds", [True, False])
    async def test_a_shutdown_during_connect_starts_nothing(self, connect_succeeds):
        """Teardown can close the Redis clients while connect() is still pinging
        them. No listener may start afterwards. A connect that the teardown cut
        short is not recorded as a start failure."""
        agent = StubAgent()
        connecting = asyncio.Event()
        connected = asyncio.Event()

        async def slow_connect(*_):
            connecting.set()
            await connected.wait()
            return connect_succeeds

        listen = AsyncMock()
        metrics = MagicMock()
        ctx = MagicMock()
        ctx.room.name = "meeting-1"

        with (
            patch("main.create_agent", return_value=agent),
            patch("main.stt_metrics", metrics),
            patch.object(RedisManager, "connect", slow_connect),
            patch.object(RedisManager, "listen", listen),
            patch.object(RedisManager, "aclose", AsyncMock()),
            patch("main.nest_asyncio.apply"),
            patch("main.get_redacted_app_config", return_value={}),
        ):
            task = asyncio.create_task(main.entrypoint(ctx))
            await asyncio.wait_for(connecting.wait(), timeout=2)

            (callback,) = ctx.add_shutdown_callback.call_args.args
            await asyncio.wait_for(callback(), timeout=2)
            connected.set()
            await asyncio.wait_for(task, timeout=2)

        listen.assert_not_called()
        metrics.agent_start_failed.assert_not_called()


class TestMetricsWorkerOptions:
    def test_no_port_means_no_listener(self, monkeypatch):
        monkeypatch.setattr(main.metrics_config, "prometheus_port", None)
        assert main._metrics_worker_kwargs() == {}

    def test_port_with_a_multiprocess_dir(self, monkeypatch, caplog):
        monkeypatch.setattr(main.metrics_config, "prometheus_port", 8082)
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus-multiproc")

        assert main._metrics_worker_kwargs() == {"prometheus_port": 8082}
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_port_without_a_multiprocess_dir_warns(self, monkeypatch, caplog):
        """Every bbb_stt_* metric is recorded in a job process and reaches the
        worker's /metrics only through the multiprocess directory."""
        monkeypatch.setattr(main.metrics_config, "prometheus_port", 8082)
        monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)

        assert main._metrics_worker_kwargs() == {"prometheus_port": 8082}
        assert any(
            "PROMETHEUS_MULTIPROC_DIR" in r.getMessage()
            for r in caplog.records
            if r.levelname == "WARNING"
        )


class TestRedisDownAtJobStart:
    async def test_the_job_ends_cleanly_and_closes_redis(self, monkeypatch):
        """With nothing listening on the Redis port, the job must still end
        cleanly and close the Redis clients."""
        monkeypatch.setattr(main.redis_config, "host", "127.0.0.1")
        monkeypatch.setattr(main.redis_config, "port", 1)
        agent = StubAgent()
        aclose = AsyncMock()
        ctx = MagicMock()
        ctx.room.name = "meeting-1"

        with (
            patch("main.create_agent", return_value=agent),
            patch.object(RedisManager, "aclose", aclose),
            patch("main.nest_asyncio.apply"),
            patch("main.get_redacted_app_config", return_value={}),
        ):
            task = asyncio.create_task(main.entrypoint(ctx))
            await asyncio.wait_for(agent.running.wait(), timeout=5)
            for _ in range(10):
                await asyncio.sleep(0)
            agent.release.set()
            await asyncio.wait_for(task, timeout=5)

        aclose.assert_awaited_once()
