import os
import re
from typing import Any

import yaml

from wakegen.core.exceptions import ConfigError
from wakegen.models.config import GenerationConfig, ProviderConfig

# This file handles loading configuration.
# We can load "presets" from YAML files (e.g., "quick_test.yaml").


def load_preset(preset_name: str) -> dict[str, Any]:
    """
    Loads a YAML preset file.

    Args:
        preset_name: The name of the preset (e.g., "quick_test").
                     It looks for a file named "wakegen/config/presets/{preset_name}.yaml".

    Returns:
        A dictionary containing the configuration from the file.

    Raises:
        ConfigError: If the file cannot be found or parsed, or if path traversal is detected.
    """
    # SEC-006 Fix: Validate preset name to prevent path traversal attacks
    # ELI5: We check that the preset name only contains safe characters.
    # This prevents someone from using "../../../etc/passwd" as a preset name
    # to access files outside the presets directory.
    if not re.match(r"^[a-zA-Z0-9_-]+$", preset_name):
        raise ConfigError(
            f"Invalid preset name: '{preset_name}'. "
            f"Preset names can only contain letters, numbers, underscores, and hyphens."
        )

    # Construct the full path to the preset file
    # We assume the presets are stored in wakegen/config/presets/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    preset_path = os.path.join(base_dir, "presets", f"{preset_name}.yaml")

    # SEC-006 Fix: Verify the resolved path is within the presets directory
    # ELI5: Even after constructing the path, we double-check that the final
    # location is actually inside our presets folder, not somewhere else.
    real_preset_path = os.path.realpath(preset_path)
    real_presets_dir = os.path.realpath(os.path.join(base_dir, "presets"))

    if not real_preset_path.startswith(real_presets_dir + os.sep):
        raise ConfigError(
            f"Path traversal detected: preset '{preset_name}' resolves outside presets directory"
        )

    if not os.path.exists(preset_path):
        raise ConfigError(f"Preset '{preset_name}' not found at {preset_path}")

    try:
        with open(preset_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise ConfigError(f"Failed to load preset '{preset_name}': {e!s}") from e


def get_generation_config(preset_name: str | None = None) -> GenerationConfig:
    """
    Creates a GenerationConfig object.
    If a preset is provided, it loads values from there.
    Otherwise, it uses defaults and environment variables.
    """
    if preset_name:
        data = load_preset(preset_name)
        # We only want the 'generation' section from the YAML
        gen_data = data.get("generation", {})
        return GenerationConfig(**gen_data)

    return GenerationConfig()


def get_provider_config() -> ProviderConfig:
    """
    Creates a ProviderConfig object from environment variables.
    """
    return ProviderConfig()
