import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from livekit import rtc
from livekit.agents import stt

from providers.base import BaseSttAgent, BaseSttConfig


class StubSttAgent(BaseSttAgent):
    """Concrete test subclass for BaseSttAgent."""

    def _create_stt_stream(self, locale):
        return MagicMock()

    def _update_stream_locale(self, user_id, locale):
        stream = self.processing_info[user_id]["stream"]
        sanitized = self._sanitize_locale(locale)
        stream.update_options(languages=[sanitized])


def _make_stub_agent(interim_results=None):
    config = BaseSttConfig(interim_results=interim_results)
    return StubSttAgent(config)


class TestShouldEmitDefault:
    def test_returns_true_for_any_event(self):
        agent = _make_stub_agent()
        mock_event = MagicMock()
        assert agent._should_emit(mock_event) is True


class TestTranslationLangMapDefault:
    def test_returns_empty_dict(self):
        agent = _make_stub_agent()
        assert agent.translation_lang_map == {}


class TestShouldEmitIntegration:
    async def test_pipeline_skips_event_when_should_emit_returns_false(self):
        """Events filtered by _should_emit should not be emitted."""

        class FilteringAgent(StubSttAgent):
            def _should_emit(self, event):
                return False

        config = BaseSttConfig(interim_results=True)
        agent = FilteringAgent(config)
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([])

        mock_event = MagicMock()
        mock_event.type = stt.SpeechEventType.FINAL_TRANSCRIPT
        mock_stt_stream = AsyncMock()
        mock_stt_stream.__aiter__.return_value = iter([mock_event])

        emitted = []
        agent.on("final_transcript", lambda **kw: emitted.append(kw))

        with patch("providers.base.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(
                mock_participant, mock_track, mock_stt_stream
            )
            await asyncio.sleep(0)

        assert len(emitted) == 0

    async def test_pipeline_emits_event_when_should_emit_returns_true(self):
        agent = _make_stub_agent()
        mock_participant = MagicMock(spec=rtc.RemoteParticipant)
        mock_participant.identity = "user_1"
        mock_track = MagicMock()

        mock_audio_stream = AsyncMock()
        mock_audio_stream.__aiter__.return_value = iter([])

        mock_event = MagicMock()
        mock_event.type = stt.SpeechEventType.FINAL_TRANSCRIPT
        mock_stt_stream = AsyncMock()
        mock_stt_stream.__aiter__.return_value = iter([mock_event])

        emitted = []
        agent.on("final_transcript", lambda **kw: emitted.append(kw))

        with patch("providers.base.rtc.AudioStream", return_value=mock_audio_stream):
            await agent._run_transcription_pipeline(
                mock_participant, mock_track, mock_stt_stream
            )
            await asyncio.sleep(0)

        assert len(emitted) == 1


def _microphone(identity="user_1"):
    participant = MagicMock(spec=rtc.RemoteParticipant)
    participant.identity = identity
    publication = MagicMock()
    publication.source = rtc.TrackSource.SOURCE_MICROPHONE
    return participant, publication


class TestSettingsAcrossReconnects:
    """BBB announces a speech locale only when it is set, not when the user comes back."""

    def test_a_returning_participant_is_transcribed_again(self):
        agent = _make_stub_agent()
        agent.participant_settings["user_1"] = {"locale": "pt-BR", "provider": "gladia"}
        participant, publication = _microphone()

        agent._on_participant_disconnected(participant)
        with patch.object(agent, "start_transcription_for_user") as start:
            agent._on_track_subscribed(MagicMock(), publication, participant)

        start.assert_called_once_with("user_1", "pt-BR", "gladia")

    def test_transcription_turned_off_stays_off_when_the_participant_returns(self):
        agent = _make_stub_agent()
        agent.participant_settings["user_1"] = {
            "locale": "pt-BR",
            "provider": "gladia",
            "partial_utterances": True,
        }
        participant, publication = _microphone()

        agent.disable_transcription_for_user("user_1")
        agent._on_participant_disconnected(participant)
        with patch.object(agent, "start_transcription_for_user") as start:
            agent._on_track_subscribed(MagicMock(), publication, participant)

        start.assert_not_called()
        assert agent.participant_settings["user_1"] == {"partial_utterances": True}
