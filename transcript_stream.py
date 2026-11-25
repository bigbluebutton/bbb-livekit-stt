"""Transcripts published into the LiveKit room, for the agents that ask for them.

An agent asks by declaring an attribute on its own participant (REQUEST_ATTRIBUTE):
while it is in the room, everyone else is transcribed for it. Each utterance goes out
as a text stream on LiveKit's standard transcription topic, sent on behalf of the
speaker, to the agents that asked. Only an agent participant can ask: BBB's clients
join as standard participants, so an end user never gets the transcript this way.
"""

import json
import logging

from livekit import rtc

# The conventions LiveKit Agents uses for transcriptions (livekit.agents.types).
TOPIC_TRANSCRIPTION = "lk.transcription"
ATTRIBUTE_TRACK_ID = "lk.transcribed_track_id"
ATTRIBUTE_SEGMENT_ID = "lk.segment_id"
ATTRIBUTE_FINAL = "lk.transcription_final"
ATTRIBUTE_LOCALE = "bbb.locale"
# What an agent declares to have the room transcribed for it, as JSON:
# {"v": 1, "who": "all", "locale": "auto"}. The locale is "auto" when left out.
REQUEST_ATTRIBUTE = "bbb.transcription"
REQUEST_VERSION = 1


def requested_locale(participant: rtc.RemoteParticipant) -> str | None:
    """The locale an agent asks the room to be transcribed in, or None if it asks
    nothing. A standard participant's attribute is never a request."""
    if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
        return None
    raw = participant.attributes.get(REQUEST_ATTRIBUTE)
    if not raw:
        return None
    try:
        request = json.loads(raw)
    except ValueError:
        request = None
    locale = request.get("locale", "auto") if isinstance(request, dict) else None
    if (
        not isinstance(request, dict)
        or request.get("v") != REQUEST_VERSION
        or request.get("who") != "all"
        or not isinstance(locale, str)
        or not locale
    ):
        logging.warning(
            f"Ignoring the transcription request of {participant.identity}: {raw!r}"
        )
        return None
    return locale


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
    destinations: list[str],
) -> bool:
    """Send one utterance to the agents that asked; return whether it was sent."""
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
