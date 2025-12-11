from unittest.mock import AsyncMock, patch

import pytest

from wakegen.web.routers.generation import JobStatus


def test_start_generation(client, mock_provider_registry):
    """Test starting a generation job."""
    # Patch the background task so it doesn't actually run forever or fail
    with patch("wakegen.web.routers.generation.run_generation_job") as mock_run:
        payload = {
            "wake_words": ["hey testing"],
            "count": 5,
            "provider": "edge_tts",
            "output_dir": "./test_output",
        }
        response = client.post("/api/generate/start", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

        # Ensure background task was scheduled
        mock_run.assert_called_once()


def test_get_job_status_not_found(client, mock_generation_job):
    """Test getting status for a non-existent job."""
    response = client.get("/api/generate/status/invalid_id")
    assert response.status_code == 404


def test_cancel_job(client, mock_generation_job):
    """Test canceling a job."""
    # Create a mock job first (by using the manual job creation in endpoint or mocking the store)
    # Easiest way is to just inject directly into the mocked dictionary
    from datetime import datetime

    from wakegen.web.routers.generation import GenerationJob

    job_id = "test_job_123"
    job = GenerationJob(
        id=job_id,
        status=JobStatus.RUNNING,
        created_at=datetime.now(),
        wake_words=["test"],
        count=1,
        provider="edge_tts",
        output_dir="./out",
    )
    mock_generation_job[job_id] = job

    # Now cancel it
    response = client.post(f"/api/generate/cancel/{job_id}")
    assert response.status_code == 200

    # Verify status changed
    assert mock_generation_job[job_id].status == JobStatus.CANCELLED
