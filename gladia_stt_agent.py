import asyncio
import logging

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    stt,
)
from livekit.plugins.gladia import STT as GladiaSTT

from config import GladiaConfig
from events import EventEmitter


class GladiaSttAgent(EventEmitter):
    def __init__(self, config: GladiaConfig):
        super().__init__()
        self.stt = GladiaSTT(**config.to_dict())
        self.ctx: JobContext | None = None
        self.room: rtc.Room | None = None
        self.processing_info = {}
        self.participant_settings = {}
        self._shutdown = asyncio.Event()

    async def start(self, ctx: JobContext):
        self.ctx = ctx
        # TODO: disable auto_subscribe. Should be on demand based on the participant's
        # transcription settings
        await self.ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        self.room = self.ctx.room

        self.room.on("participant_disconnected", self._on_participant_disconnected)
        self.room.on("disconnected", self._on_disconnected)
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)

        try:
            await self._shutdown.wait()
        finally:
            await self._cleanup()

    async def _cleanup(self):
        for user_id in list(self.processing_info.keys()):
            self.stop_transcription_for_user(user_id)

        await asyncio.sleep(0.1)

    def start_transcription_for_user(self, user_id: str, locale: str, provider: str):
        settings = self.participant_settings.get(user_id, {})
        self.participant_settings[user_id] = {
            "locale": locale,
            "provider": provider,
            "partial_utterances": settings.get("partial_utterances", False),
            "min_utterance_length": settings.get("min_utterance_length", 0),
        }

        participant = self._find_participant(user_id)

        if not participant:
            logging.error(
                f"Cannot start transcription, participant {user_id} not found."
            )
            return

        track = self._find_audio_track(participant)

        if not track:
            logging.warning(
                f"Won't start transcription yet, no audio track found for {user_id}."
            )
            return

        if participant.identity in self.processing_info:
            logging.debug(
                f"Transcription task already running for {participant.identity}, ignoring start command."
            )
            return

        gladia_locale = self._sanitize_locale(locale)
        stt_stream = self.stt.stream(language=gladia_locale)
        task = asyncio.create_task(
            self._run_transcription_pipeline(participant, track, stt_stream)
        )
        self.processing_info[participant.identity] = {
            "stream": stt_stream,
            "task": task,
        }
        logging.info(
            f"Started transcription for {participant.identity} with locale {locale}."
        )

    def stop_transcription_for_user(self, user_id: str):
        logging.debug(f"Stopping transcription for {user_id}.")

        if user_id in self.processing_info:
            info = self.processing_info.pop(user_id)
            info["task"].cancel()
            logging.info(f"Stopped transcription for user {user_id}.")

    def update_locale_for_user(self, user_id: str, locale: str):
        if user_id in self.participant_settings:
            self.participant_settings[user_id]["locale"] = locale

        if user_id in self.processing_info:
            logging.info(f"Updating locale to '{locale}' for user {user_id}.")
            stream = self.processing_info[user_id]["stream"]
            gladia_locale = self._sanitize_locale(locale)
            stream.update_options(languages=[gladia_locale])
        else:
            logging.warning(
                f"Won't update locale, no active transcription for user {user_id}."
            )

    def update_speech_options_for_user(
        self, user_id: str, partial_utterances: bool, min_utterance_length: int
    ):
        if user_id not in self.participant_settings:
            self.participant_settings[user_id] = {}
        self.participant_settings[user_id]["partial_utterances"] = partial_utterances
        self.participant_settings[user_id]["min_utterance_length"] = (
            min_utterance_length
        )

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
            logging.debug(
                f"Skipping transcription for {participant.identity}'s track {track.sid} because it's not a microphone."
            )
            return

        settings = self.participant_settings.get(participant.identity)

        if settings:
            logging.debug(
                f"Participant {participant.identity} subscribed with active settings, starting transcription."
            )
            self.start_transcription_for_user(
                participant.identity, settings["locale"], settings["provider"]
            )
        else:
            logging.debug(
                f"Participant {participant.identity} subscribed with no active settings, skipping transcription."
            )

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        self.stop_transcription_for_user(participant.identity)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant, *_):
        logging.debug(
            f"Participant {participant.identity} disconnected, stopping transcription."
        )
        self.stop_transcription_for_user(participant.identity)
        self.participant_settings.pop(participant.identity, None)

    def _on_disconnected(self):
        self._shutdown.set()

    def _find_participant(self, identity: str) -> rtc.RemoteParticipant | None:
        for p in self.room.remote_participants.values():
            if p.identity == identity:
                return p
        return None

    def _find_audio_track(self, participant: rtc.RemoteParticipant) -> rtc.Track | None:
        for pub in participant.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                return pub.track
        return None

    def _sanitize_locale(self, locale: str) -> str:
        # Gladia only accepts ISO 639-1 locales (e.g. "en")
        # BBB uses <ISO 639-1>-<ISO 3166-1> format (e.g. "en-US")
        # Sanitization here is to ensure we use Gladia's format.
        gladia_locale = locale.split("-")[0].lower()

        return gladia_locale

    async def _run_transcription_pipeline(
        self,
        participant: rtc.RemoteParticipant,
        track: rtc.Track,
        stt_stream: stt.SpeechStream,
    ):
        audio_stream = rtc.AudioStream(track)

        async def forward_audio_task():
            try:
                async for audio_event in audio_stream:
                    stt_stream.push_frame(audio_event.frame)
            finally:
                stt_stream.flush()

        async def process_stt_task():
            async for event in stt_stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    self.emit("final_transcript", participant=participant, event=event)
                elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                    self.emit(
                        "interim_transcript", participant=participant, event=event
                    )

        try:
            await asyncio.gather(forward_audio_task(), process_stt_task())
        except asyncio.CancelledError:
            logging.info(f"Transcription for {participant.identity} was cancelled.")
        except Exception as e:
            logging.error(f"Error during transcription for track {track.sid}: {e}")
        finally:
            self.processing_info.pop(participant.identity, None)
