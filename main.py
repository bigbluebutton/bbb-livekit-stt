import load_env  # noqa: F401 - must stay first; see load_env.py

import asyncio
import json
import logging
import math
import os

from livekit.agents import JobContext, JobProcess, WorkerOptions, cli, stt
from livekit import rtc

from redis_manager import RedisManager
from providers import create_agent, import_provider, prewarm_provider
from metrics import (
    AGENT_FAILURE_REDIS_CONNECT,
    AGENT_FAILURE_UNKNOWN_PROVIDER,
    DISCARD_BELOW_MIN_UTTERANCE_LENGTH,
    DISCARD_UNRESOLVABLE_LOCALE,
    EVENT_TYPE_FINAL,
    EVENT_TYPE_INTERIM,
    start_dead_process_sweeper,
    stt_metrics,
)
from config import (
    get_redacted_app_config,
    metrics_config,
    redis_config,
    stt_provider,
)
from utils import (
    coerce_min_utterance_length_seconds,
    coerce_partial_utterances,
    resolve_bbb_locale,
)

# At module level rather than under __main__: `main.py dev` runs the worker in
# a child process that imports this file as __mp_main__.
import_provider(stt_provider)


def _log_startup_configuration(stt_config):
    logging.debug(
        "Application configuration: %s",
        json.dumps(get_redacted_app_config(stt_config), sort_keys=True),
    )


def prewarm(proc: JobProcess):
    # Runs once per worker process, before any job is assigned. Providers that
    # need to load models do it here rather than per room.
    prewarm_provider(stt_provider, proc.userdata)


async def entrypoint(ctx: JobContext):
    redis_manager = RedisManager(redis_config)

    try:
        agent = create_agent(stt_provider, ctx.proc.userdata)
    except ValueError:
        stt_metrics.agent_start_failed(stt_provider, AGENT_FAILURE_UNKNOWN_PROVIDER)
        raise

    _log_startup_configuration(agent.config)

    async def on_redis_message(message_data: str):
        try:
            msg = json.loads(message_data)
            envelope = msg.get("envelope", {})
            core = msg.get("core", {})

            event_name = envelope.get("name")
            if event_name not in (
                RedisManager.USER_SPEECH_LOCALE_CHANGED_EVT_MSG,
                RedisManager.USER_SPEECH_OPTIONS_CHANGED_EVT_MSG,
                RedisManager.USER_LEFT_MEETING_EVT_MSG,
            ):
                return

            routing = envelope.get("routing", {})
            body = core.get("body", {})
            meeting_id = routing.get("meetingId")
            user_id = routing.get("userId")

            if agent.room is None or meeting_id != agent.room.name:
                return

            if event_name == RedisManager.USER_SPEECH_LOCALE_CHANGED_EVT_MSG:
                agent.handle_speech_locale_change(
                    user_id, body.get("locale"), body.get("provider")
                )

            elif event_name == RedisManager.USER_SPEECH_OPTIONS_CHANGED_EVT_MSG:
                partial_utterances = coerce_partial_utterances(
                    body.get("partialUtterances", False)
                )
                min_utterance_length = coerce_min_utterance_length_seconds(
                    body.get("minUtteranceLength", 0)
                )
                settings = agent.participant_settings.setdefault(user_id, {})
                settings["partial_utterances"] = partial_utterances
                settings["min_utterance_length"] = min_utterance_length
                logging.info(f"User speech options changed for {user_id}: {settings}")

            elif event_name == RedisManager.USER_LEFT_MEETING_EVT_MSG:
                agent.disable_transcription_for_user(user_id)

        except json.JSONDecodeError:
            logging.warning(f"Could not decode Redis message: {message_data}")
        except Exception as e:
            logging.error(f"Error processing Redis message: {e}")

    @agent.on("final_transcript")
    async def on_final_transcript(
        participant: rtc.RemoteParticipant,
        event: stt.SpeechEvent,
        open_time: float,
        locale: str,
    ):
        # The locale comes with the transcript, as open_time does: the
        # participant may have switched locale since this audio was transcribed.
        original_locale = locale

        # "auto" means the provider detects the language, so there is no
        # selected language to fall back to when a transcript omits one.
        original_lang = (
            None if original_locale.lower() == "auto" else original_locale.split("-")[0]
        )

        for alternative in event.alternatives:
            # Some providers (e.g. OpenAI) may not report a language; fall back to original.
            transcript_lang = alternative.language or original_lang
            text = alternative.text
            start_time_adjusted = math.floor(open_time + alternative.start_time)
            end_time_adjusted = math.floor(open_time + alternative.end_time)
            utterance_duration_seconds = max(
                0.0, alternative.end_time - alternative.start_time
            )
            logging.debug(
                f"FINAL transcript for {participant.identity} = [{transcript_lang}] {text}",
                extra={
                    "utterance_duration_seconds": utterance_duration_seconds,
                    "open_time": open_time,
                    "start_time": alternative.start_time,
                    "end_time": alternative.end_time,
                    "start_time_adjusted": start_time_adjusted,
                    "end_time_adjusted": end_time_adjusted,
                    "confidence": alternative.confidence,
                    "original_lang": original_lang,
                    "alternative": alternative,
                },
            )
            bbb_locale = resolve_bbb_locale(
                transcript_lang, original_locale, agent.translation_lang_map
            )

            if not bbb_locale:
                logging.warning(
                    f"Discarding final transcript for {participant.identity}: "
                    f"the provider reported no language and the participant's "
                    f"locale is '{original_locale}', so there is no BBB locale "
                    f"to publish it under."
                )
                stt_metrics.transcript_discarded(
                    agent.provider_name, EVENT_TYPE_FINAL, DISCARD_UNRESOLVABLE_LOCALE
                )
                continue

            published = await redis_manager.publish_update_transcript_pub_msg(
                agent.room.name,
                participant.identity,
                alternative,
                bbb_locale,
                start_time_adjusted,
                end_time_adjusted,
                result=True,
            )

            # Recorded after the publish, so a metrics error cannot drop the
            # transcript.
            if not published:
                stt_metrics.transcript_publish_failed(agent.provider_name)

            stt_metrics.transcript_emitted(
                agent.provider_name,
                EVENT_TYPE_FINAL,
                confidence=(
                    alternative.confidence if agent.reports_confidence else None
                ),
                duration_seconds=utterance_duration_seconds,
            )
            stt_metrics.language_checked(
                agent.provider_name,
                requested_lang=original_lang,
                reported_lang=alternative.language,
                translation_enabled=agent.translation_enabled,
            )

    @agent.on("interim_transcript")
    async def on_interim_transcript(
        participant: rtc.RemoteParticipant,
        event: stt.SpeechEvent,
        open_time: float,
        locale: str,
    ):
        # Speech options are read live: the participant can change them at any
        # time.
        p_settings = agent.participant_settings.get(participant.identity, {})

        if not p_settings.get("partial_utterances", False):
            return

        original_locale = locale

        original_lang = (
            None if original_locale.lower() == "auto" else original_locale.split("-")[0]
        )
        min_utterance_length = p_settings.get("min_utterance_length", 0)

        for alternative in event.alternatives:
            # Some providers (e.g. OpenAI) may not report a language; fall back to original.
            transcript_lang = alternative.language or original_lang
            text = alternative.text
            start_time_adjusted = math.floor(open_time + alternative.start_time)
            end_time_adjusted = math.floor(open_time + alternative.end_time)
            utterance_duration_seconds = max(
                0.0, alternative.end_time - alternative.start_time
            )

            if (
                min_utterance_length
                and utterance_duration_seconds <= min_utterance_length
            ):
                logging.debug(
                    f"Discarding interim transcript for {participant.identity}: too short "
                    f"({utterance_duration_seconds:.3f}s <= {min_utterance_length}s).",
                    extra={
                        "utterance_duration_seconds": utterance_duration_seconds,
                        "min_utterance_length": min_utterance_length,
                        "open_time": open_time,
                        "start_time": alternative.start_time,
                        "end_time": alternative.end_time,
                        "start_time_adjusted": start_time_adjusted,
                        "end_time_adjusted": end_time_adjusted,
                    },
                )
                stt_metrics.transcript_discarded(
                    agent.provider_name,
                    EVENT_TYPE_INTERIM,
                    DISCARD_BELOW_MIN_UTTERANCE_LENGTH,
                )
                continue

            logging.debug(
                f"INTERIM transcript for {participant.identity} = [{transcript_lang}] {text}",
                extra={
                    "utterance_duration_seconds": utterance_duration_seconds,
                    "open_time": open_time,
                    "start_time": alternative.start_time,
                    "end_time": alternative.end_time,
                    "start_time_adjusted": start_time_adjusted,
                    "end_time_adjusted": end_time_adjusted,
                    "confidence": alternative.confidence,
                    "original_lang": original_lang,
                    "alternative": alternative,
                },
            )

            bbb_locale = resolve_bbb_locale(
                transcript_lang, original_locale, agent.translation_lang_map
            )

            if not bbb_locale:
                logging.warning(
                    f"Discarding interim transcript for {participant.identity}: "
                    f"the provider reported no language and the participant's "
                    f"locale is '{original_locale}', so there is no BBB locale "
                    f"to publish it under."
                )
                stt_metrics.transcript_discarded(
                    agent.provider_name, EVENT_TYPE_INTERIM, DISCARD_UNRESOLVABLE_LOCALE
                )
                continue

            published = await redis_manager.publish_update_transcript_pub_msg(
                agent.room.name,
                participant.identity,
                alternative,
                bbb_locale,
                start_time_adjusted,
                end_time_adjusted,
                result=False,
            )

            # After the publish, as in on_final_transcript.
            if not published:
                stt_metrics.transcript_publish_failed(agent.provider_name)

            stt_metrics.transcript_emitted(
                agent.provider_name,
                EVENT_TYPE_INTERIM,
                confidence=(
                    alternative.confidence if agent.reports_confidence else None
                ),
                duration_seconds=utterance_duration_seconds,
            )
            stt_metrics.language_checked(
                agent.provider_name,
                requested_lang=original_lang,
                reported_lang=alternative.language,
                translation_enabled=agent.translation_enabled,
            )

    redis_listen_task: asyncio.Task | None = None
    redis_closing: asyncio.Future | None = None

    async def close_redis():
        # Every caller awaits the same close: the entrypoint's finally can get
        # here first, and LiveKit's shutdown callback must still wait for it.
        nonlocal redis_closing
        if redis_closing is None:
            redis_closing = asyncio.ensure_future(_close_redis())
        await asyncio.shield(redis_closing)

    async def _close_redis():
        if redis_listen_task is not None:
            redis_listen_task.cancel()
            try:
                await redis_listen_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logging.error(f"Redis listener failed: {e}")
        await redis_manager.aclose()

    async def on_job_shutdown():
        # LiveKit waits for this teardown; see BaseSttAgent.shutdown().
        await agent.shutdown()
        await close_redis()

    ctx.add_shutdown_callback(on_job_shutdown)

    try:
        connected = await redis_manager.connect()
        # If a shutdown closed Redis during the connect, start no listener and
        # record no failure, since the shutdown ended the connect.
        if redis_closing is None:
            if connected:
                redis_listen_task = asyncio.create_task(
                    redis_manager.listen(on_redis_message)
                )
            else:
                # Not fatal: the agent still joins the room. But no locale change
                # arrives and no transcript is published, so record the failure.
                stt_metrics.agent_start_failed(
                    agent.provider_name, AGENT_FAILURE_REDIS_CONNECT
                )

        logging.info(f"Received job for room {ctx.room.name}")
        await agent.start(ctx)
    finally:
        await close_redis()


def _metrics_worker_kwargs() -> dict:
    if metrics_config.prometheus_port is None:
        return {}

    if not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        logging.warning(
            "BBB_STT_PROMETHEUS_PORT is set without PROMETHEUS_MULTIPROC_DIR: "
            "/metrics will serve no bbb_stt_* metric, since jobs record them "
            "in their own processes."
        )

    return {"prometheus_port": metrics_config.prometheus_port}


if __name__ == "__main__":
    metrics_kwargs = _metrics_worker_kwargs()
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if metrics_kwargs and multiproc_dir:
        start_dead_process_sweeper(multiproc_dir)

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            **metrics_kwargs,
        )
    )
