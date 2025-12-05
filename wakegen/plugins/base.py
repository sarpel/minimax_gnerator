"""
Plugin Base Module

This module defines the interface (protocol) that all TTS plugins must implement.
Think of it as a contract: any plugin that wants to work with wakegen must follow
these rules.

Why use plugins?
- Extend wakegen without modifying the core code
- Let the community create new TTS providers
- Keep optional/experimental providers separate
- Easy installation: pip install wakegen-plugin-xyz

Example plugin implementation:
```python
from wakegen.plugins import TTSPlugin, PluginMetadata
from wakegen.core.protocols import Voice
from wakegen.core.types import Gender

class MyTTSPlugin(TTSPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my-tts",
            version="1.0.0",
            description="My custom TTS provider",
            author="Your Name",
        )
    
    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        # Your TTS implementation here
        pass
    
    async def list_voices(self) -> list[Voice]:
        return [Voice(id="voice1", name="Voice 1", language="en-US", gender=Gender.NEUTRAL)]
    
    async def validate_config(self) -> None:
        # Check if plugin is properly configured
        pass
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, List, Any, Optional, runtime_checkable
from abc import abstractmethod


# =============================================================================
# EXCEPTIONS
# =============================================================================


class PluginLoadError(Exception):
    """
    Raised when a plugin fails to load.
    
    This could happen because:
    - The plugin package is not installed correctly
    - The plugin doesn't implement the required interface
    - The plugin's dependencies are missing
    - There's an error in the plugin's initialization code
    """
    pass


class PluginValidationError(Exception):
    """
    Raised when a plugin fails validation checks.
    
    This could happen because:
    - Required metadata is missing
    - The plugin doesn't implement required methods
    - Configuration is invalid
    """
    pass


# =============================================================================
# METADATA
# =============================================================================


@dataclass
class PluginMetadata:
    """
    Metadata about a plugin.
    
    This information is displayed to users when listing plugins and helps
    identify which plugin is which.
    
    Attributes:
        name: A unique identifier for the plugin (e.g., "my-tts-plugin").
              Should be lowercase with hyphens.
        version: Semantic version string (e.g., "1.0.0").
        description: Brief description of what the plugin provides.
        author: Who created the plugin.
        homepage: Optional URL for documentation or source code.
        requires_api_key: Whether the plugin needs an API key to work.
        requires_gpu: Whether the plugin needs GPU for reasonable performance.
        supported_languages: List of supported language codes (e.g., ["en", "es"]).
    """
    name: str
    version: str
    description: str
    author: str = "Unknown"
    homepage: Optional[str] = None
    requires_api_key: bool = False
    requires_gpu: bool = False
    supported_languages: List[str] = field(default_factory=lambda: ["en"])
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        if not self.name:
            raise PluginValidationError("Plugin name is required")
        if not self.version:
            raise PluginValidationError("Plugin version is required")
        if not self.description:
            raise PluginValidationError("Plugin description is required")
        
        # Normalize name to lowercase with hyphens
        self.name = self.name.lower().replace("_", "-").replace(" ", "-")


# =============================================================================
# PLUGIN PROTOCOL
# =============================================================================


@runtime_checkable
class TTSPlugin(Protocol):
    """
    Protocol (interface) that all TTS plugins must implement.
    
    This is the contract between wakegen and plugins. Any class that implements
    all these methods/properties can be used as a TTS plugin.
    
    The @runtime_checkable decorator allows us to use isinstance() to check
    if an object implements this protocol.
    
    Required implementations:
        - metadata property: Returns plugin information
        - generate method: Creates audio from text
        - list_voices method: Returns available voices
        - validate_config method: Checks if plugin is ready to use
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        """
        Returns metadata about this plugin.
        
        This is called when listing plugins and provides information like
        the plugin name, version, and description.
        
        Returns:
            PluginMetadata object with plugin information.
        """
        ...
    
    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generate audio from text and save to a file.
        
        This is the main method that does the actual text-to-speech conversion.
        
        Args:
            text: The text to convert to speech (e.g., "hey assistant").
            voice_id: The ID of the voice to use (from list_voices).
            output_path: Full path where the audio file should be saved.
        
        Raises:
            Exception: If generation fails for any reason.
        """
        ...
    
    async def list_voices(self) -> List[Any]:
        """
        List all available voices for this plugin.
        
        Returns a list of Voice objects (or similar) that can be used
        with the generate() method.
        
        Returns:
            List of available voices. Each voice should have at minimum:
            - id: Unique identifier to pass to generate()
            - name: Human-readable name
            - language: Language code (e.g., "en-US")
        """
        ...
    
    async def validate_config(self) -> None:
        """
        Validate that the plugin is properly configured and ready to use.
        
        This method should check things like:
        - Required API keys are present
        - Required dependencies are installed
        - Model files exist (if needed)
        
        Raises:
            PluginValidationError: If configuration is invalid.
        """
        ...


# =============================================================================
# PLUGIN STATE
# =============================================================================


@dataclass
class LoadedPlugin:
    """
    Represents a plugin that has been loaded and is ready to use.
    
    This wraps the actual plugin instance with additional state information.
    
    Attributes:
        instance: The actual plugin object implementing TTSPlugin.
        metadata: Plugin metadata (cached for quick access).
        entry_point: The entry point name this plugin was loaded from.
        is_enabled: Whether the plugin is currently enabled.
        load_error: If loading failed, the error message.
    """
    instance: TTSPlugin
    metadata: PluginMetadata
    entry_point: str
    is_enabled: bool = True
    load_error: Optional[str] = None
    
    @property
    def name(self) -> str:
        """Shortcut to get plugin name."""
        return self.metadata.name


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    "TTSPlugin",
    "PluginMetadata",
    "LoadedPlugin",
    "PluginLoadError",
    "PluginValidationError",
]
