from unittest.mock import AsyncMock, MagicMock

from config import RedisConfig
from redis_manager import RedisManager


def _make_manager():
    config = RedisConfig(host="localhost", port=6379, password="")
    return RedisManager(config)


class TestGenerateUpdateTranscriptPubMsg:
    def test_message_contains_required_envelope_fields(self):
        manager = _make_manager()
        msg = manager._generate_update_transcript_pub_msg(
            meeting_id="meeting-1",
            user_id="user-1",
            locale="en-US",
            transcript="Hello world",
            result=True,
            start=0,
            end=1000,
        )

        envelope = msg["envelope"]
        assert envelope["name"] == RedisManager.UPDATE_TRANSCRIPT_PUB_MSG
        assert envelope["routing"]["meetingId"] == "meeting-1"
        assert envelope["routing"]["userId"] == "user-1"
        assert "timestamp" in envelope

    def test_message_contains_required_core_fields(self):
        manager = _make_manager()
        msg = manager._generate_update_transcript_pub_msg(
            meeting_id="meeting-2",
            user_id="user-2",
            locale="pt-BR",
            transcript="Olá mundo",
            result=False,
            start=500,
            end=2000,
        )

        core = msg["core"]
        assert core["header"]["name"] == RedisManager.UPDATE_TRANSCRIPT_PUB_MSG
        assert core["header"]["meetingId"] == "meeting-2"
        assert core["header"]["userId"] == "user-2"
        body = core["body"]
        assert body["transcript"] == "Olá mundo"
        assert body["locale"] == "pt-BR"
        assert body["result"] is False
        assert body["start"] == "500"
        assert body["end"] == "2000"

    def test_transcript_id_encodes_user_locale_and_start(self):
        manager = _make_manager()
        msg = manager._generate_update_transcript_pub_msg(
            meeting_id="m",
            user_id="u",
            locale="en-US",
            transcript="hi",
            result=True,
            start=1234,
            end=5678,
        )
        assert msg["core"]["body"]["transcriptId"] == "u-en-US-1234"

    def test_start_and_end_default_to_zero(self):
        manager = _make_manager()
        msg = manager._generate_update_transcript_pub_msg(
            meeting_id="m",
            user_id="u",
            locale="en",
            transcript="test",
            result=True,
        )
        body = msg["core"]["body"]
        assert body["start"] == "0"
        assert body["end"] == "0"


class TestPublishUpdateTranscriptPubMsg:
    async def test_skips_publish_when_not_connected(self, caplog):
        manager = _make_manager()
        manager.pub_client = None

        speech_data = MagicMock()
        speech_data.text = "Hello"
        speech_data.start_time = 0.0
        speech_data.end_time = 1.0

        # Should not raise even when pub_client is None
        await manager.publish_update_transcript_pub_msg(
            meeting_id="m", user_id="u", transcript_data=speech_data, locale="en-US"
        )

    async def test_publishes_to_correct_channel(self):
        manager = _make_manager()
        mock_client = AsyncMock()
        manager.pub_client = mock_client

        speech_data = MagicMock()
        speech_data.text = "Test transcript"
        speech_data.start_time = 0.5
        speech_data.end_time = 2.0

        await manager.publish_update_transcript_pub_msg(
            meeting_id="meeting-1",
            user_id="user-1",
            transcript_data=speech_data,
            locale="en-US",
        )

        mock_client.publish.assert_called_once()
        call_args = mock_client.publish.call_args
        assert call_args[0][0] == RedisManager.TO_AKKA_APPS_CHANNEL
