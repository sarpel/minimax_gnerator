import yaml
import os
from typing import Any, Dict
from wakegen.core.exceptions import ConfigError
from wakegen.models.config import GenerationConfig, ProviderConfig

# This file handles loading configuration.
# We can load "presets" from YAML files (e.g., "quick_test.yaml").

def load_preset(preset_name: str) -> Dict[str, Any]:
    """
    Loads a YAML preset file.

    Args:
        preset_name: The name of the preset (e.g., "quick_test").
                     It looks for a file named "wakegen/config/presets/{preset_name}.yaml".

    Returns:
        A dictionary containing the configuration from the file.

    Raises:
        ConfigError: If the file cannot be found or parsed.
    """
    # Construct the full path to the preset file
    # We assume the presets are stored in wakegen/config/presets/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    preset_path = os.path.join(base_dir, "presets", f"{preset_name}.yaml")

    if not os.path.exists(preset_path):
        raise ConfigError(f"Preset '{preset_name}' not found at {preset_path}")

    try:
        with open(preset_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise ConfigError(f"Failed to load preset '{preset_name}': {str(e)}") from e

def get_generation_config(preset_name: str = None) -> GenerationConfig:
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