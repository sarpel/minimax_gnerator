from unittest.mock import mock_open, patch

import pytest


def test_get_config_template(client):
    """Test fetching the config template."""
    response = client.get("/api/config/template")
    assert response.status_code == 200
    data = response.json()
    assert "yaml_content" in data
    # Check if we got content
    assert len(data["yaml_content"]) > 0


def test_validate_config_valid(client):
    """Test validating a valid configuration."""
    valid_yaml = """
    project:
      name: test
    generation:
      wake_words: ["test"]
    """
    response = client.post("/api/config/validate", json={"content": valid_yaml})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True


def test_validate_config_invalid(client):
    """Test validating an invalid configuration."""
    invalid_yaml = """
    project:
      name: test
    generation:
      wake_words: "not a list"
    """
    response = client.post("/api/config/validate", json={"content": invalid_yaml})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert len(data["errors"]) > 0


def test_save_config(client):
    """Test saving configuration to disk."""
    yaml_content = "project:\n  name: test\ngeneration:\n  wake_words: [test]"

    # Mock file writing
    with patch("builtins.open", mock_open()) as mock_file:
        # Also need to patch Path.exists/mkdir inside router, but mock_open usually handles basic writes
        # However, save_config calls validate_config which calls load_config which uses open()
        # It's getting complex to mock everything for an integration test.
        # Let's mock validate_config instead to focus on save logic?
        # Or just allow validation to pass since valid_yaml is provided.

        # We also need to patch Path.exists and mkdir
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("pathlib.Path.mkdir"),
        ):
            response = client.post(
                "/api/config/save",
                json={"content": yaml_content, "path": "configs/test.yaml"},
            )
            assert response.status_code == 200
