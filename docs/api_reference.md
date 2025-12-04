# API Reference

This reference documents the main classes and functions for developers who want to use WakeGen as a Python library.

**Current Status**: Phase 1A - Basic generation functionality is implemented. Other features are planned but not yet available.

## Core Components

### `wakegen.providers.registry.get_provider`

```python
def get_provider(provider_type: ProviderType, config: ProviderConfig) -> TTSProvider
```

Factory function to get a TTS provider instance.

*   **provider_type**: Enum (`EDGE_TTS`, `PIPER`, `MINIMAX`, `XTTS`).
*   **config**: Configuration object containing API keys and settings.
*   **Returns**: An instance of a class implementing `TTSProvider`.

**Current Status**: Only `EDGE_TTS` is fully implemented. Other providers are planned.

### `wakegen.core.protocols.TTSProvider`

The interface that all TTS providers must implement.

```python
class TTSProvider(Protocol):
    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """Generates audio from text."""
        ...

    async def list_voices(self) -> List[Voice]:
        """Returns a list of available voices."""
        ...
```

## Configuration

### `wakegen.config.settings.get_generation_config`

```python
def get_generation_config(preset_name: Optional[str] = None) -> GenerationConfig
```

Loads generation settings from a preset file or defaults.

**Available Presets**: Currently only `quick_test` preset is available.

## Augmentation

### `wakegen.augmentation.pipeline.AugmentationPipeline`

```python
class AugmentationPipeline:
    def __init__(self, config: AugmentationConfig):
        ...

    def process(self, audio_path: str, output_path: str) -> None:
        """Applies configured effects to an audio file."""
        ...
```

**Current Status**: Augmentation pipeline is defined but not yet fully implemented in the CLI.

## Utilities

### `wakegen.utils.audio.resample_audio`

```python
def resample_audio(file_path: str, target_sr: int) -> None:
    """Resamples an audio file to the target sample rate (e.g., 16000 Hz)."""
    ...
```

**Status**: Fully implemented and working.

## Implementation Notes

1. **Current Working Features**:
   - Edge TTS provider integration
   - Basic audio generation
   - Configuration system
   - Audio resampling

2. **Planned Features**:
   - Full augmentation pipeline
   - Additional TTS providers (Piper, Minimax, XTTS)
   - Export functionality
   - Validation tools
   - Training script generation

3. **Usage Example**:
   ```python
   from wakegen.providers.registry import get_provider
   from wakegen.core.types import ProviderType
   from wakegen.models.config import ProviderConfig

   # Get Edge TTS provider
   provider = get_provider(ProviderType.EDGE_TTS, ProviderConfig())

   # Generate audio
   await provider.generate("Hello world", "en-US-AriaNeural", "output.wav")
   ```