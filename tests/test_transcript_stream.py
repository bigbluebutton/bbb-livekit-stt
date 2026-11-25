from unittest.mock import AsyncMock, MagicMock

from livekit import rtc

from transcript_stream import (
    ATTRIBUTE_FINAL,
    ATTRIBUTE_LOCALE,
    ATTRIBUTE_SEGMENT_ID,
    ATTRIBUTE_TRACK_ID,
    REQUEST_ATTRIBUTE,
    TOPIC_TRANSCRIPTION,
    publish_transcript,
    requested_locale,
)

AGENT = rtc.ParticipantKind.PARTICIPANT_KIND_AGENT


def _participant(
    identity,
    kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
    mic=None,
    attributes=None,
):
    participant = MagicMock()
    participant.identity = identity
    participant.kind = kind
    participant.attributes = attributes or {}
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
            room,
            speaker,
            "Olá, turma",
            "pt-BR",
            "w_alice-pt-BR-100",
            final=True,
            destinations=["assistant"],
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

        await publish_transcript(
            room,
            speaker,
            "Olá",
            "pt-BR",
            "seg",
            final=False,
            destinations=["assistant"],
        )

        attributes = room.local_participant.stream_text.await_args.kwargs["attributes"]
        assert attributes[ATTRIBUTE_FINAL] == "false"

    async def test_nothing_is_sent_without_an_agent_that_asked(self):
        speaker = _participant("w_alice", mic="TR_mic")
        room, _ = _room(speaker, _participant("w_bob"))

        sent = await publish_transcript(
            room, speaker, "Olá", "pt-BR", "seg", final=True, destinations=[]
        )

        assert sent is False
        room.local_participant.stream_text.assert_not_awaited()

    async def test_no_track_attribute_without_a_microphone(self):
        speaker = _participant("w_alice")
        assistant = _participant(
            "assistant", kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        )
        room, _ = _room(speaker, assistant)

        await publish_transcript(
            room, speaker, "Olá", "pt-BR", "seg", final=True, destinations=["assistant"]
        )

        attributes = room.local_participant.stream_text.await_args.kwargs["attributes"]
        assert ATTRIBUTE_TRACK_ID not in attributes

    async def test_empty_text_is_not_sent(self):
        speaker = _participant("w_alice", mic="TR_mic")
        assistant = _participant(
            "assistant", kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        )
        room, _ = _room(speaker, assistant)

        sent = await publish_transcript(
            room, speaker, "  ", "pt-BR", "seg", final=True, destinations=["assistant"]
        )

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
            room, speaker, "Olá", "pt-BR", "seg", final=True, destinations=["assistant"]
        )

        assert sent is False


class TestRequestedLocale:
    def test_an_agent_asks_for_everyone_in_a_locale(self):
        agent = _participant(
            "assistant",
            kind=AGENT,
            attributes={REQUEST_ATTRIBUTE: '{"v": 1, "who": "all", "locale": "pt-BR"}'},
        )
        assert requested_locale(agent) == "pt-BR"

    def test_the_locale_is_auto_when_left_out(self):
        agent = _participant(
            "assistant",
            kind=AGENT,
            attributes={REQUEST_ATTRIBUTE: '{"v": 1, "who": "all"}'},
        )
        assert requested_locale(agent) == "auto"

    def test_a_standard_participant_never_asks(self):
        user = _participant(
            "w_alice", attributes={REQUEST_ATTRIBUTE: '{"v": 1, "who": "all"}'}
        )
        assert requested_locale(user) is None

    def test_an_agent_without_the_attribute_asks_nothing(self):
        assert requested_locale(_participant("assistant", kind=AGENT)) is None

    def test_a_request_it_cannot_read_is_ignored(self):
        for raw in (
            "not json",
            "[]",
            '{"v": 2, "who": "all"}',
            '{"v": 1, "who": ["w_alice"]}',
            '{"v": 1, "who": "all", "locale": 7}',
        ):
            agent = _participant(
                "assistant", kind=AGENT, attributes={REQUEST_ATTRIBUTE: raw}
            )
            assert requested_locale(agent) is None, raw
