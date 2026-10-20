from unittest.mock import AsyncMock, MagicMock

from livekit import rtc

from transcript_stream import (
    ATTRIBUTE_FINAL,
    ATTRIBUTE_LOCALE,
    ATTRIBUTE_SEGMENT_ID,
    ATTRIBUTE_TRACK_ID,
    TOPIC_TRANSCRIPTION,
    publish_transcript,
)


def _participant(
    identity, kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD, mic=None
):
    participant = MagicMock()
    participant.identity = identity
    participant.kind = kind
    publications = {}
    if mic:
        publication = MagicMock()
        publication.sid = mic
        publication.source = rtc.TrackSource.SOURCE_MICROPHONE
        publications[mic] = publication
    participant.track_publications = publications
    return participant


def _room(*remote):
    room = MagicMock()
    room.remote_participants = {p.identity: p for p in remote}
    writer = MagicMock()
    writer.write = AsyncMock()
    writer.aclose = AsyncMock()
    room.local_participant.stream_text = AsyncMock(return_value=writer)
    return room, writer


class TestPublishTranscript:
    async def test_sends_to_agents_on_behalf_of_the_speaker(self):
        speaker = _participant("w_alice", mic="TR_mic")
        assistant = _participant(
            "assistant", kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        )
        room, writer = _room(speaker, assistant)

        sent = await publish_transcript(
            room, speaker, "Olá, turma", "pt-BR", "w_alice-pt-BR-100", final=True
        )

        assert sent is True
        room.local_participant.stream_text.assert_awaited_once_with(
            topic=TOPIC_TRANSCRIPTION,
            destination_identities=["assistant"],
            sender_identity="w_alice",
            attributes={
                ATTRIBUTE_SEGMENT_ID: "w_alice-pt-BR-100",
                ATTRIBUTE_FINAL: "true",
                ATTRIBUTE_LOCALE: "pt-BR",
                ATTRIBUTE_TRACK_ID: "TR_mic",
            },
        )
        writer.write.assert_awaited_once_with("Olá, turma")
        writer.aclose.assert_awaited_once()

    async def test_interim_is_marked_not_final(self):
        speaker = _participant("w_alice", mic="TR_mic")
        assistant = _participant(
            "assistant", kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        )
        room, _ = _room(speaker, assistant)

        await publish_transcript(room, speaker, "Olá", "pt-BR", "seg", final=False)

        attributes = room.local_participant.stream_text.await_args.kwargs["attributes"]
        assert attributes[ATTRIBUTE_FINAL] == "false"

    async def test_end_users_never_receive_it(self):
        speaker = _participant("w_alice", mic="TR_mic")
        room, _ = _room(speaker, _participant("w_bob"))

        sent = await publish_transcript(
            room, speaker, "Olá", "pt-BR", "seg", final=True
        )

        assert sent is False
        room.local_participant.stream_text.assert_not_awaited()

    async def test_no_track_attribute_without_a_microphone(self):
        speaker = _participant("w_alice")
        assistant = _participant(
            "assistant", kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        )
        room, _ = _room(speaker, assistant)

        await publish_transcript(room, speaker, "Olá", "pt-BR", "seg", final=True)

        attributes = room.local_participant.stream_text.await_args.kwargs["attributes"]
        assert ATTRIBUTE_TRACK_ID not in attributes

    async def test_empty_text_is_not_sent(self):
        speaker = _participant("w_alice", mic="TR_mic")
        assistant = _participant(
            "assistant", kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        )
        room, _ = _room(speaker, assistant)

        sent = await publish_transcript(room, speaker, "  ", "pt-BR", "seg", final=True)

        assert sent is False
        room.local_participant.stream_text.assert_not_awaited()

    async def test_a_failed_stream_does_not_raise(self):
        speaker = _participant("w_alice", mic="TR_mic")
        assistant = _participant(
            "assistant", kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        )
        room, _ = _room(speaker, assistant)
        room.local_participant.stream_text.side_effect = RuntimeError("room closed")

        sent = await publish_transcript(
            room, speaker, "Olá", "pt-BR", "seg", final=True
        )

        assert sent is False
