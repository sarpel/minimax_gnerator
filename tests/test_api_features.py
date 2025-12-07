
import pytest
from unittest.mock import patch
from pathlib import Path

# =============================================================================
# QUALITY API TESTS
# =============================================================================

def test_validate_dataset(client):
    """Test the quality validation endpoint."""
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.rglob", return_value=[Path("test.wav")]):
        
        # Use GET for queries as per implementation
        response = client.get("/api/quality/validate?directory=./data")
        assert response.status_code == 200
        data = response.json()
        # Health score calculation depends on internal logic, but should be 100 if no issues found
        # (Assuming validate_file returns no issues by default or if mocked)
        # We might need to mock validate_file too to control issues
        # But let's check basic success first
        assert "health_score" in data

# =============================================================================
# AUGMENTATION API TESTS
# =============================================================================

def test_apply_augmentation(client):
    """Test the augmentation application endpoint."""
    with patch("wakegen.web.routers.augmentation.run_augmentation") as mock_run:
        # Need to provide a valid profile object and existing input directory
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.rglob", return_value=[Path("test.wav")]):
            
            payload = {
                "input_dir": "./in",
                "output_dir": "./out",
                "profile": {
                    "name": "custom",
                    "noise": {"enabled": True},
                    "reverb": {"enabled": False},
                    "pitch": {"enabled": False},
                    "device": {"enabled": False}
                },
                "copies_per_file": 1
            }
            response = client.post("/api/augmentation/apply", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data

# =============================================================================
# EXPORT API TESTS
# =============================================================================

def test_start_export(client):
    """Test starting an export job."""
    with patch("wakegen.web.routers.export.run_export") as mock_run, \
         patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.rglob", return_value=[Path("test.wav")]), \
         patch("pathlib.Path.mkdir"):
        
        payload = {
            "format": "openwakeword",
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
            "input_dir": "./in",
            "output_dir": "./out",
            "stratify": True,
            "generate_manifest": True,
            "copy_files": False
        }
        response = client.post("/api/export/start", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data

def test_export_status(client):
    """Test checking export job status."""
    with patch("wakegen.web.routers.export._export_jobs", {}) as mock_jobs:
        from wakegen.web.routers.export import ExportJob, ExportStatus
        
        job_id = "exp_123"
        mock_jobs[job_id] = ExportJob(
            id=job_id,
            status=ExportStatus.COMPLETED,
            format="openwakeword",
            input_dir="./in",
            output_dir="./out"
        )
        
        response = client.get(f"/api/export/status/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
