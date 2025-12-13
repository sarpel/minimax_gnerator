from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Import the application
from wakegen.web.app import create_app
from wakegen.web.config import WebConfig


@pytest.fixture
def app():
    """Create a fresh FastAPI app instance for each test."""
    config = WebConfig(debug=True)
    return create_app(config)


@pytest.fixture
def client(app):
    """Create a TestClient instance."""
    return TestClient(app)


@pytest.fixture
def mock_provider_registry():
    """
    Mock the provider registry to avoid loading real providers/models.
    This is critical for CI/CD and fast testing.
    """
    # Patch where it is USED, not just where it is defined
    # Since multiple routers import get_provider, we might need multiple patches
    # or patch the underlying registry dictionary if possible.
    # For now, let's patch the most common usage points.
    with patch("wakegen.web.routers.providers.get_provider") as mock_get_1, \
         patch("wakegen.generation.orchestrator.get_provider") as mock_get_3, \
         patch("wakegen.providers.registry.get_provider") as mock_get_orig:

        mock_get = mock_get_orig # Use the original one as the primary mock configuration source

        # Configure all mocks to behave the same
        for m in [mock_get_1, mock_get_3, mock_get_orig]:
             m.side_effect = mock_get.side_effect

        # Create a mock provider instance
        mock_provider = MagicMock()
        mock_provider.id = "mock_provider"
        mock_provider.name = "Mock Provider"

        # Mock list_voices
        mock_voice = MagicMock()
        mock_voice.id = "voice_1"
        mock_voice.name = "Test Voice"
        mock_voice.language = "en-US"

        # Async mock for list_voices
        async def async_list_voices():
            return [mock_voice]

        mock_provider.list_voices.side_effect = async_list_voices

        # Async mock for generate
        async def async_generate(*args, **kwargs):
            return "output/path.wav"

        mock_provider.generate.side_effect = async_generate

        # Return this mock when get_provider() is called
        mock_get.return_value = mock_provider
        yield mock_get


@pytest.fixture
def mock_generation_job():
    """Mock the generation job storage."""
    with patch("wakegen.web.routers.generation._jobs", {}) as mock_jobs:
        yield mock_jobs
