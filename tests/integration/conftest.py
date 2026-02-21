import os

import pytest

from livekit.agents import utils


@pytest.fixture
async def job_process():
    """Initialize the LiveKit HTTP context, simulating an agent job process.

    This is required for plugins (e.g. GladiaSTT) that use the managed
    http_context session internally, matching the production code path.
    """
    utils.http_context._new_session_ctx()
    yield
    await utils.http_context._close_http_ctx()


def pytest_collection_modifyitems(config, items):
    """Skip all integration tests when GLADIA_API_KEY is not set."""
    if os.environ.get("GLADIA_API_KEY"):
        return

    skip_marker = pytest.mark.skip(
        reason="GLADIA_API_KEY environment variable is not set"
    )
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip_marker)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test requiring external services",
    )
