import os
import re
import yaml
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, ValidationError, field_validator
from wakegen.core.types import ProviderType, AudioFormat, EnvironmentProfile

# We use Pydantic models to define the structure of our configuration.
# This ensures that the data we load from the YAML file is valid and has the correct types.
# Think of it as a blueprint for our settings.

class ProjectConfig(BaseModel):
    """
    Configuration related to the project identity.
    """
    # 'model_config' is a special Pydantic attribute to configure the model's behavior.
    # 'extra="forbid"' means that if the YAML contains keys not defined here, it will raise an error.
    # This helps catch typos in the config file.
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the project")
    version: str = Field("0.1.0", description="Version of the project")
    description: Optional[str] = Field(None, description="Brief description of the project")

class GenerationConfig(BaseModel):
    """
    Configuration for the wake word generation process.
    """
    model_config = ConfigDict(extra="forbid")

    wake_words: List[str] = Field(..., description="List of wake words to generate")
    count: int = Field(100, ge=1, description="Number of samples to generate per wake word")
    output_dir: str = Field("./output", description="Directory to save generated audio")
    
    # We use the 'AudioFormat' enum to ensure only valid formats are used.
    audio_format: AudioFormat = Field(AudioFormat.WAV, description="Format of the output audio files")
    sample_rate: int = Field(16000, description="Sample rate in Hz (e.g., 16000, 22050, 44100)")

class ProviderConfig(BaseModel):
    """
    Configuration for a specific TTS provider.
    """
    model_config = ConfigDict(extra="forbid")

    type: ProviderType = Field(..., description="Type of the TTS provider (e.g., 'piper', 'edge_tts')")
    voices: List[str] = Field(..., description="List of voice IDs to use for this provider")
    
    # 'weight' determines how often this provider is used relative to others.
    # A higher weight means more samples will be generated using this provider.
    weight: float = Field(1.0, ge=0.0, description="Weight for this provider in the generation mix")
    
    # Optional provider-specific settings (e.g., API keys, URLs)
    # We use Dict[str, Any] to allow flexibility for different provider requirements.
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Provider-specific options")

class AugmentationConfig(BaseModel):
    """
    Configuration for audio augmentation (adding noise, reverb, etc.).
    """
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(False, description="Whether to apply augmentation")
    
    # We use the 'EnvironmentProfile' enum to ensure valid profile names.
    profiles: List[EnvironmentProfile] = Field(default_factory=list, description="List of environment profiles to apply")
    
    augmented_per_original: int = Field(1, ge=1, description="Number of augmented versions to create for each original sample")

class ExportConfig(BaseModel):
    """
    Configuration for exporting the dataset.
    """
    model_config = ConfigDict(extra="forbid")

    format: str = Field("openwakeword", description="Format to export the dataset in")
    
    # The split ratio defines how data is divided into Train, Validation, and Test sets.
    # For example, [0.8, 0.1, 0.1] means 80% training, 10% validation, 10% testing.
    split_ratio: List[float] = Field([0.8, 0.1, 0.1], description="Train/Val/Test split ratios")

    @field_validator('split_ratio')
    @classmethod
    def validate_split_ratio(cls, v: List[float]) -> List[float]:
        """
        Validates that the split ratios sum up to approximately 1.0.
        """
        if not (0.99 <= sum(v) <= 1.01):
            raise ValueError("Split ratios must sum to 1.0")
        return v

class WakegenConfig(BaseModel):
    """
    The main configuration class that aggregates all other configs.
    This represents the structure of the entire 'wakegen.yaml' file.
    """
    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig = Field(..., description="Project metadata")
    generation: GenerationConfig = Field(..., description="Generation settings")
    providers: List[ProviderConfig] = Field(..., description="List of TTS providers to use")
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig, description="Augmentation settings")
    export: ExportConfig = Field(default_factory=ExportConfig, description="Export settings")

def _substitute_env_vars(content: str) -> str:
    """
    Helper function to substitute environment variables in the YAML content.
    It looks for patterns like ${VAR_NAME} and replaces them with the value of the environment variable.
    
    Args:
        content: The raw YAML string.
        
    Returns:
        The YAML string with environment variables substituted.
    """
    # The regex pattern \$\{([^}]+)\} matches strings like ${VAR_NAME}.
    # The group ([^}]+) captures the variable name inside the braces.
    pattern = re.compile(r'\$\{([^}]+)\}')
    
    def replace(match):
        env_var_name = match.group(1)
        # os.environ.get returns the value of the environment variable, or an empty string if not found.
        # You might want to raise an error if a required variable is missing, but for now we return empty string.
        return os.environ.get(env_var_name, "")
    
    return pattern.sub(replace, content)

def load_config(path: str) -> WakegenConfig:
    """
    Loads and validates the configuration from a YAML file.
    
    Args:
        path: Path to the YAML configuration file.
        
    Returns:
        A validated WakegenConfig object.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is invalid.
        ValidationError: If the config data does not match the Pydantic models.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        # 1. Substitute environment variables
        # This allows users to keep secrets (like API keys) out of the config file.
        content_with_env = _substitute_env_vars(raw_content)
        
        # 2. Parse YAML
        # yaml.safe_load is safer than yaml.load as it avoids executing arbitrary code.
        config_dict = yaml.safe_load(content_with_env)
        
        if not config_dict:
            raise ValueError("Configuration file is empty")

        # 3. Validate with Pydantic
        # This step converts the dictionary into our typed Pydantic models.
        # If any data is missing or invalid, Pydantic will raise a ValidationError.
        config = WakegenConfig(**config_dict)
        
        return config

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except ValidationError as e:
        # We re-raise the validation error so the caller can handle it (e.g., print a nice message).
        raise e
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading config: {e}")