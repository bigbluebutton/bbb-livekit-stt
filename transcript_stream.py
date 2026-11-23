"""Transcripts published into the LiveKit room, for agents in the same meeting.

Each utterance goes out as a text stream on LiveKit's standard transcription topic,
sent on behalf of the speaker, so another agent can follow the meeting without
reading BBB's Redis. Only agent participants receive it: end users' clients never
get the transcript through this path.
"""

import logging

from livekit import rtc

# The conventions LiveKit Agents uses for transcriptions (livekit.agents.types).
TOPIC_TRANSCRIPTION = "lk.transcription"
ATTRIBUTE_TRACK_ID = "lk.transcribed_track_id"
ATTRIBUTE_SEGMENT_ID = "lk.segment_id"
ATTRIBUTE_FINAL = "lk.transcription_final"
ATTRIBUTE_LOCALE = "bbb.locale"


def _agent_identities(room: rtc.Room) -> list[str]:
    return [
        p.identity
        for p in room.remote_participants.values()
        if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    ]


def _microphone_track_sid(participant: rtc.RemoteParticipant) -> str | None:
    for publication in participant.track_publications.values():
        if publication.source == rtc.TrackSource.SOURCE_MICROPHONE:
            return publication.sid
    return None


async def publish_transcript(
    room: rtc.Room,
    participant: rtc.RemoteParticipant,
    text: str,
    locale: str,
    segment_id: str,
    final: bool,
) -> bool:
    """Send one utterance to the agents in the room; return whether it was sent."""
    destinations = _agent_identities(room)
    if not destinations or not text.strip():
        return False

    attributes = {
        ATTRIBUTE_SEGMENT_ID: segment_id,
        ATTRIBUTE_FINAL: "true" if final else "false",
        ATTRIBUTE_LOCALE: locale,
    }
    track_sid = _microphone_track_sid(participant)
    if track_sid:
        attributes[ATTRIBUTE_TRACK_ID] = track_sid

    try:
        writer = await room.local_participant.stream_text(
            topic=TOPIC_TRANSCRIPTION,
            destination_identities=destinations,
            sender_identity=participant.identity,
            attributes=attributes,
        )
        await writer.write(text)
        await writer.aclose()
    except Exception as e:
        # Redis stays the source of truth for BBB; a failed stream only costs the agents.
        logging.warning(
            f"Could not stream transcript for {participant.identity} to agents: {e}"
        )
        return False
    return True
