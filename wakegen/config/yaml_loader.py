"""
YAML Configuration Loader

This module handles loading and validating wakegen project configurations from YAML files.
It uses Pydantic v2 for type-safe validation with helpful error messages.

Think of this like a recipe validator:
- You write your recipe (config) in a YAML file
- This module reads it, checks everything is correct, and gives you a nice object to work with
- If something is wrong (like "cook for -5 minutes"), it tells you exactly what's broken

Key Features:
- Environment variable substitution: ${VAR_NAME} gets replaced with actual values
- Sensible defaults so you don't have to specify everything
- Clear error messages when validation fails
- Type-safe access to all configuration values
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wakegen.core.exceptions import ConfigError
from wakegen.core.types import EnvironmentProfile


# =============================================================================
# ENVIRONMENT VARIABLE SUBSTITUTION
# =============================================================================
# This pattern matches ${VAR_NAME} or ${VAR_NAME:default_value} in strings
# Example: "${API_KEY}" or "${PORT:8080}"
ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def substitute_env_vars(value: Any) -> Any:
    """
    Replace ${VAR_NAME} patterns with actual environment variable values.
    
    This is like a find-and-replace that looks up values from your system.
    
    Examples:
        - "${HOME}" -> "/home/user" (whatever your home directory is)
        - "${API_KEY:default123}" -> the value of API_KEY, or "default123" if not set
        - "${MISSING_VAR}" -> raises error if not set and no default
    
    Args:
        value: Can be a string (we'll substitute), dict (we'll recurse), 
               list (we'll recurse), or anything else (returned as-is).
    
    Returns:
        The value with all ${VAR_NAME} patterns replaced.
    
    Raises:
        ConfigError: If a variable is not set and has no default value.
    """
    # If it's a string, look for ${VAR_NAME} patterns
    if isinstance(value, str):
        def replace_match(match: re.Match[str]) -> str:
            var_name = match.group(1)  # The variable name (e.g., "API_KEY")
            default_value = match.group(2)  # Optional default (e.g., "default123")
            
            # Try to get the value from environment
            env_value = os.environ.get(var_name)
            
            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                # Variable not found and no default - this is an error
                raise ConfigError(
                    f"Environment variable '{var_name}' is not set and has no default value. "
                    f"Either set it (export {var_name}=...) or provide a default: ${{{var_name}:default}}"
                )
        
        # Replace all ${...} patterns in the string
        return ENV_VAR_PATTERN.sub(replace_match, value)
    
    # If it's a dictionary, recursively process all values
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    
    # If it's a list, recursively process all items
    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    
    # For everything else (numbers, booleans, None), return as-is
    return value


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================
# These Pydantic models define the structure and validation rules for each
# section of the YAML config file. Think of them as "forms" that check your input.


class ProjectConfig(BaseModel):
    """
    Metadata about the project itself.
    
    This is like the cover page of your project - name, version, etc.
    
    Example YAML:
        project:
          name: "hey_katya"
          version: "1.0.0"
          description: "Wake word for Katya assistant"
    """
    # Pydantic v2 uses ConfigDict instead of inner Config class
    model_config = ConfigDict(
        extra="forbid",  # Don't allow unknown fields (catches typos)
        str_strip_whitespace=True,  # Remove leading/trailing spaces from strings
    )
    
    name: str = Field(
        ...,  # Required field (no default)
        min_length=1,
        max_length=100,
        description="Project name (used for folder names, should be valid filename)"
    )
    version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+$",  # Must be like "1.0.0"
        description="Semantic version (e.g., 1.0.0)"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional description of the project"
    )
    
    @field_validator("name")
    @classmethod
    def validate_project_name(cls, v: str) -> str:
        """
        Ensure project name is safe for use in file paths.
        
        We don't want names like "../../../etc" or "my:project" that could
        cause problems on different operating systems.
        """
        # Check for characters that are problematic in file paths
        invalid_chars = set('<>:"/\\|?*')
        found_invalid = [c for c in v if c in invalid_chars]
        if found_invalid:
            raise ValueError(
                f"Project name contains invalid characters: {found_invalid}. "
                f"Use only letters, numbers, underscores, hyphens, and spaces."
            )
        return v


class ProviderConfig(BaseModel):
    """
    Configuration for a single TTS provider.
    
    Each provider generates audio differently. This config tells wakegen
    which provider to use, which voices, and how much of the total output
    should come from this provider.
    
    Example YAML:
        providers:
          - type: kokoro
            voices: [af_bella, am_adam]
            weight: 0.4
    """
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )
    
    type: str = Field(
        ...,
        description="Provider type (e.g., 'kokoro', 'piper', 'edge_tts')"
    )
    voices: list[str] = Field(
        default_factory=list,
        description="List of voice IDs to use from this provider"
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,  # Greater than or equal to 0
        le=1.0,  # Less than or equal to 1
        description="Relative weight for this provider (0.0-1.0)"
    )
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific settings (e.g., API keys, model paths)"
    )
    languages: list[str] = Field(
        default_factory=list,
        description="List of languages to filter voices by (e.g., 'tr-TR', 'en-US')"
    )
    
    @field_validator("type")
    @classmethod
    def validate_provider_type(cls, v: str) -> str:
        """
        Ensure the provider type is lowercase (consistent format).
        """
        return v.lower()


class GenerationConfig(BaseModel):
    """
    Configuration for the audio generation process.
    
    This is the main part - what wake words to generate, how many,
    and where to put them.
    
    Example YAML:
        generation:
          wake_words:
            - "hey katya"
            - "katya"
          count: 1000
          output_dir: "./output/hey_katya"
    """
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )
    
    wake_words: list[str] = Field(
        ...,
        min_length=1,  # Need at least one wake word
        description="List of wake word phrases to generate"
    )
    count: int = Field(
        default=100,
        ge=1,  # At least 1 sample
        le=100000,  # Reasonable upper limit
        description="Number of samples to generate per wake word"
    )
    output_dir: str = Field(
        default="./output",
        description="Directory to save generated audio files"
    )
    sample_rate: int = Field(
        default=16000,
        ge=8000,
        le=48000,
        description="Sample rate for output audio (Hz)"
    )
    audio_format: str = Field(
        default="wav",
        pattern=r"^(wav|mp3|flac)$",
        description="Output audio format (wav, mp3, flac)"
    )
    
    @field_validator("wake_words")
    @classmethod
    def validate_wake_words(cls, v: list[str]) -> list[str]:
        """
        Clean up wake words and validate them.
        
        We strip whitespace and ensure they're not empty after stripping.
        """
        cleaned = []
        for i, word in enumerate(v):
            word = word.strip()
            if not word:
                raise ValueError(f"Wake word at index {i} is empty after stripping whitespace")
            if len(word) > 100:
                raise ValueError(f"Wake word '{word[:20]}...' is too long (max 100 chars)")
            cleaned.append(word)
        return cleaned


class AugmentationConfig(BaseModel):
    """
    Configuration for audio augmentation (making samples more realistic).
    
    Augmentation adds effects like background noise, room reverb, and 
    microphone characteristics to make the training data more robust.
    
    Example YAML:
        augmentation:
          enabled: true
          profiles:
            - morning_kitchen
            - car_interior
          augmented_per_original: 3
    """
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )
    
    enabled: bool = Field(
        default=True,
        description="Whether to apply augmentation"
    )
    profiles: list[str] = Field(
        default_factory=lambda: ["morning_kitchen", "car_interior"],
        description="List of environment profiles to use for augmentation"
    )
    augmented_per_original: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Number of augmented versions to create per original sample"
    )
    variation_strength: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="How much to vary augmentation parameters (0.0-1.0)"
    )
    
    @field_validator("profiles")
    @classmethod
    def validate_profiles(cls, v: list[str]) -> list[str]:
        """
        Validate that profile names match known environment profiles.
        """
        # Get valid profile names from the enum
        valid_profiles = {e.value for e in EnvironmentProfile}
        
        invalid = [p for p in v if p.lower() not in valid_profiles]
        if invalid:
            raise ValueError(
                f"Unknown augmentation profiles: {invalid}. "
                f"Valid options are: {sorted(valid_profiles)}"
            )
        
        # Return lowercase versions to match enum values
        return [p.lower() for p in v]


class ExportConfig(BaseModel):
    """
    Configuration for exporting the dataset for training.
    
    This controls how the final dataset is packaged - format, splits, etc.
    
    Example YAML:
        export:
          format: openwakeword
          split_ratio: [0.8, 0.1, 0.1]
    """
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )
    
    format: str = Field(
        default="openwakeword",
        pattern=r"^(openwakeword|speechbrain|wav2vec)$",
        description="Export format for the training framework"
    )
    split_ratio: list[float] = Field(
        default=[0.8, 0.1, 0.1],
        min_length=3,
        max_length=3,
        description="Train/validation/test split ratios (must sum to 1.0)"
    )
    include_manifest: bool = Field(
        default=True,
        description="Whether to generate a manifest file"
    )
    
    @field_validator("split_ratio")
    @classmethod
    def validate_split_ratio(cls, v: list[float]) -> list[float]:
        """
        Ensure split ratios are valid (positive and sum to 1.0).
        """
        if len(v) != 3:
            raise ValueError("split_ratio must have exactly 3 values (train, val, test)")
        
        # Check each value is valid
        for i, ratio in enumerate(v):
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"split_ratio[{i}] must be between 0.0 and 1.0, got {ratio}")
        
        # Check they sum to 1.0 (with small tolerance for float precision)
        total = sum(v)
        if not 0.99 <= total <= 1.01:
            raise ValueError(
                f"split_ratio values must sum to 1.0, got {total:.4f}. "
                f"Example: [0.8, 0.1, 0.1]"
            )
        
        return v


class WakegenConfig(BaseModel):
    """
    The main configuration model that combines all sections.
    
    This is the "root" model that contains everything else. When you call
    load_config(), you get one of these back.
    
    Example YAML:
        project:
          name: "hey_katya"
          version: "1.0.0"
        
        generation:
          wake_words: ["hey katya"]
          count: 1000
        
        providers:
          - type: kokoro
            weight: 0.5
        
        augmentation:
          enabled: true
        
        export:
          format: openwakeword
    """
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )
    
    project: ProjectConfig
    generation: GenerationConfig
    providers: list[ProviderConfig] = Field(
        default_factory=lambda: [ProviderConfig(type="edge_tts")],
        min_length=1,
        description="List of TTS providers to use"
    )
    augmentation: AugmentationConfig = Field(
        default_factory=AugmentationConfig,
        description="Augmentation settings"
    )
    export: ExportConfig = Field(
        default_factory=ExportConfig,
        description="Export/dataset preparation settings"
    )
    
    @model_validator(mode="after")
    def validate_provider_weights(self) -> "WakegenConfig":
        """
        Ensure provider weights sum to approximately 1.0 (if multiple providers).
        
        This validator runs after all individual field validators.
        """
        if len(self.providers) > 1:
            total_weight = sum(p.weight for p in self.providers)
            if not 0.99 <= total_weight <= 1.01:
                raise ValueError(
                    f"Provider weights must sum to 1.0 when using multiple providers. "
                    f"Got {total_weight:.4f}. Adjust weights so they add up correctly."
                )
        return self


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================


def load_config(path: str | Path) -> WakegenConfig:
    """
    Load and validate a wakegen configuration from a YAML file.
    
    This is the main function you'll use. It:
    1. Reads the YAML file
    2. Substitutes any ${ENV_VAR} patterns with actual values
    3. Validates everything against our Pydantic models
    4. Returns a nice, type-safe WakegenConfig object
    
    Args:
        path: Path to the YAML configuration file (e.g., "wakegen.yaml")
    
    Returns:
        Validated WakegenConfig object ready to use.
    
    Raises:
        ConfigError: If the file doesn't exist, can't be parsed, or fails validation.
    
    Example:
        config = load_config("wakegen.yaml")
        print(config.project.name)  # Type-safe access!
        print(config.generation.wake_words)
    """
    # Convert to Path object for easier handling
    path = Path(path)
    
    # Check file exists
    if not path.exists():
        raise ConfigError(
            f"Configuration file not found: {path}\n"
            f"Create one with 'wakegen config init' or specify a valid path."
        )
    
    # Check it's a file (not a directory)
    if not path.is_file():
        raise ConfigError(f"Path is not a file: {path}")
    
    # Read the YAML file
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Failed to parse YAML file: {path}\n"
            f"YAML error: {e}\n"
            f"Check for syntax errors like missing colons or incorrect indentation."
        ) from e
    except Exception as e:
        raise ConfigError(f"Failed to read configuration file: {path}\n{e}") from e
    
    # Handle empty file
    if raw_data is None:
        raise ConfigError(
            f"Configuration file is empty: {path}\n"
            f"Add at least 'project' and 'generation' sections."
        )
    
    # Ensure it's a dictionary
    if not isinstance(raw_data, dict):
        raise ConfigError(
            f"Configuration must be a YAML dictionary/object, got: {type(raw_data).__name__}\n"
            f"Make sure your YAML starts with a section like 'project:'"
        )
    
    # Substitute environment variables
    try:
        processed_data = substitute_env_vars(raw_data)
    except ConfigError:
        raise  # Re-raise our own errors
    except Exception as e:
        raise ConfigError(f"Failed to process environment variables: {e}") from e
    
    # Validate against Pydantic models
    try:
        config = WakegenConfig(**processed_data)
    except Exception as e:
        # Format Pydantic validation errors nicely
        raise ConfigError(
            f"Configuration validation failed for: {path}\n\n{format_validation_error(e)}"
        ) from e
    
    return config


def format_validation_error(error: Exception) -> str:
    """
    Format Pydantic validation errors into human-readable messages.
    
    Pydantic v2 errors have a different structure than v1, so we handle that here.
    """
    from pydantic import ValidationError
    
    if isinstance(error, ValidationError):
        lines = ["Validation errors:"]
        for err in error.errors():
            # Build the field path (e.g., "generation.wake_words.0")
            loc = " → ".join(str(part) for part in err["loc"])
            msg = err["msg"]
            err_type = err["type"]
            
            lines.append(f"  • {loc}: {msg} ({err_type})")
        return "\n".join(lines)
    
    # For other errors, just return the string representation
    return str(error)


# =============================================================================
# TEMPLATE GENERATION
# =============================================================================


def get_template_config() -> str:
    """
    Generate a template YAML configuration with helpful comments.
    
    This is used by 'wakegen config init' to create a starter config file.
    
    Returns:
        A string containing a complete, commented YAML template.
    """
    return '''# =============================================================================
# Wakegen Configuration File
# =============================================================================
# This file configures wake word dataset generation.
# Documentation: https://github.com/sarpel/wakegen
#
# Environment variables can be used with ${VAR_NAME} syntax:
#   output_dir: "${OUTPUT_PATH}/datasets"
# With optional default: ${VAR_NAME:default_value}
# =============================================================================

# -----------------------------------------------------------------------------
# Project Metadata
# -----------------------------------------------------------------------------
project:
  name: "my_wake_word"           # Project name (used for folder names)
  version: "1.0.0"               # Semantic version
  description: "Custom wake word for my assistant"

# -----------------------------------------------------------------------------
# Generation Settings
# -----------------------------------------------------------------------------
generation:
  wake_words:                    # List of wake word phrases to generate
    - "hey assistant"
    - "assistant"
  count: 1000                    # Samples per wake word
  output_dir: "./output"         # Where to save generated files
  sample_rate: 16000             # Output sample rate in Hz
  audio_format: "wav"            # Output format: wav, mp3, or flac

# -----------------------------------------------------------------------------
# TTS Providers
# -----------------------------------------------------------------------------
# Configure which text-to-speech providers to use and their relative weights.
# Weights must sum to 1.0 when using multiple providers.
providers:
  - type: edge_tts               # Microsoft Edge TTS (free, online)
    voices:
      - en-US-JennyNeural
      - en-US-GuyNeural
    weight: 0.4

  - type: kokoro                 # Kokoro TTS (local, fast)
    voices:
      - af_bella
      - am_adam
    weight: 0.3

  - type: piper                  # Piper TTS (local, lightweight)
    voices:
      - en_US-lessac-medium
    weight: 0.3

# -----------------------------------------------------------------------------
# Augmentation Settings
# -----------------------------------------------------------------------------
# Augmentation makes samples more realistic by adding noise, reverb, etc.
augmentation:
  enabled: true
  profiles:                      # Environment profiles to simulate
    - morning_kitchen            # Kitchen with cooking sounds
    - car_interior               # Inside a car with road noise
    - office_space               # Office background
    - bedroom_night              # Quiet bedroom
  augmented_per_original: 3      # Augmented copies per original
  variation_strength: 0.3        # How much to vary parameters (0.0-1.0)

# -----------------------------------------------------------------------------
# Export Settings
# -----------------------------------------------------------------------------
# Configure how the final dataset is packaged for training.
export:
  format: openwakeword           # Export format: openwakeword, speechbrain, wav2vec
  split_ratio: [0.8, 0.1, 0.1]   # Train/validation/test split
  include_manifest: true         # Generate manifest file
'''


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main function
    "load_config",
    # Config models
    "WakegenConfig",
    "ProjectConfig",
    "GenerationConfig",
    "ProviderConfig",
    "AugmentationConfig",
    "ExportConfig",
    # Utilities
    "get_template_config",
    "substitute_env_vars",
]
