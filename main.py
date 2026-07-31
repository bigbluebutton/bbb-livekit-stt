import asyncio
import json
import logging
import math

import nest_asyncio
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, stt
from livekit import rtc

from redis_manager import RedisManager
from providers import create_agent
from config import get_redacted_app_config, redis_config, stt_provider
from utils import (
    coerce_min_utterance_length_seconds,
    coerce_partial_utterances,
    resolve_bbb_locale,
)

load_dotenv()


def _log_startup_configuration(stt_config):
    logging.debug(
        "Application configuration: %s",
        json.dumps(get_redacted_app_config(stt_config), sort_keys=True),
    )


async def entrypoint(ctx: JobContext):
    nest_asyncio.apply()

    redis_manager = RedisManager(redis_config)
    agent = create_agent(stt_provider)

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
            ):
                return

            routing = envelope.get("routing", {})
            body = core.get("body", {})
            meeting_id = routing.get("meetingId")
            user_id = routing.get("userId")

            if agent.room is None or meeting_id != agent.room.name:
                return

            if event_name == RedisManager.USER_SPEECH_LOCALE_CHANGED_EVT_MSG:
                locale = body.get("locale")
                provider = body.get("provider")

                if not (provider and locale):
                    agent.stop_transcription_for_user(user_id)
                else:
                    current_locale = agent.participant_settings.get(user_id, {}).get(
                        "locale"
                    )
                    if current_locale and current_locale != locale:
                        agent.update_locale_for_user(user_id, locale)
                    elif not current_locale:
                        agent.start_transcription_for_user(user_id, locale, provider)

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

        except json.JSONDecodeError:
            logging.warning(f"Could not decode Redis message: {message_data}")
        except Exception as e:
            logging.error(f"Error processing Redis message: {e}")

    @agent.on("final_transcript")
    async def on_final_transcript(
        participant: rtc.RemoteParticipant,
        event: stt.SpeechEvent,
        open_time: float = agent.open_time,
    ):
        p_settings = agent.participant_settings.get(participant.identity, {})
        original_locale = p_settings.get("locale")

        if not original_locale:
            logging.warning(
                f"Could not find original locale for participant {participant.identity}, cannot process transcripts."
            )
            return

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
                continue

            await redis_manager.publish_update_transcript_pub_msg(
                agent.room.name,
                participant.identity,
                alternative,
                bbb_locale,
                start_time_adjusted,
                end_time_adjusted,
                result=True,
            )

    @agent.on("interim_transcript")
    async def on_interim_transcript(
        participant: rtc.RemoteParticipant,
        event: stt.SpeechEvent,
        open_time: float = agent.open_time,
    ):
        p_settings = agent.participant_settings.get(participant.identity, {})

        if not p_settings.get("partial_utterances", False):
            return

        original_locale = p_settings.get("locale")

        if not original_locale:
            logging.warning(
                f"Could not find original locale for participant {participant.identity}, cannot process interim transcripts."
            )
            return

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
                continue

            await redis_manager.publish_update_transcript_pub_msg(
                agent.room.name,
                participant.identity,
                alternative,
                bbb_locale,
                start_time_adjusted,
                end_time_adjusted,
                result=False,
            )

    redis_listen_task = asyncio.create_task(redis_manager.listen(on_redis_message))

    try:
        await redis_manager.connect()
        logging.info(f"Received job for room {ctx.room.name}")
        await agent.start(ctx)
    finally:
        redis_listen_task.cancel()
        try:
            await redis_listen_task
        except asyncio.CancelledError:
            pass
        await redis_manager.aclose()


if __name__ == "__main__":
    opts = WorkerOptions(entrypoint_fnc=entrypoint)
    cli.run_app(opts)
