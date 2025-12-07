
from unittest.mock import patch, MagicMock
from wakegen.core.types import ProviderType

def test_list_providers(client):
    """Test the providers listing endpoint."""
    response = client.get("/api/providers")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # Check for core providers we expect
    provider_ids = [p["id"] for p in data]
    assert "edge_tts" in provider_ids

def test_provider_voices(client, mock_provider_registry):
    """Test fetching voices for a provider."""
    # We patch the availability check to ensure we can list voices even if not installed
    with patch("wakegen.web.routers.providers.check_provider_availability") as mock_check, \
         patch("wakegen.web.routers.providers.get_provider") as mock_get_provider: # Must patch usage
        
        mock_check.return_value.is_available = True
        
        # Configure the mock provider returned by get_provider
        mock_provider = MagicMock()
        mock_voice = MagicMock()
        mock_voice.id = "voice_1"
        mock_voice.name = "Test Voice"
        mock_voice.language = "en-US"
        mock_voice.gender = MagicMock()
        mock_voice.gender.value = "female" # Needs to match Voice response schema
        
        async def async_list_voices():
            return [mock_voice]
        mock_provider.list_voices.side_effect = async_list_voices
        
        mock_get_provider.return_value = mock_provider
        
        response = client.get("/api/providers/edge_tts/voices")
        assert response.status_code == 200
        voices = response.json()
        assert isinstance(voices, list)
        
        assert len(voices) > 0
        assert voices[0]["id"] == "voice_1"

def test_provider_voices_invalid_provider(client):
    """Test fetching voices for a non-existent provider."""
    response = client.get("/api/providers/invalid_provider/voices")
    assert response.status_code == 404

def test_provider_status(client):
    """Test checking provider status."""
    response = client.get("/api/providers/edge_tts/status")
    assert response.status_code == 200
    data = response.json()
    assert "is_available" in data
