# API Reference

This reference documents the main classes and functions for developers who want to use WakeGen as a Python library.

## Core Components

### `wakegen.providers.registry.get_provider`

```python
def get_provider(provider_type: ProviderType, config: ProviderConfig) -> TTSProvider
```

Factory function to get a TTS provider instance.

*   **provider_type**: Enum (`EDGE_TTS`, `PIPER`, `MINIMAX`, `XTTS`).
*   **config**: Configuration object containing API keys and settings.
*   **Returns**: An instance of a class implementing `TTSProvider`.

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

## Utilities

### `wakegen.utils.audio.resample_audio`

```python
def resample_audio(file_path: str, target_sr: int) -> None:
    """Resamples an audio file to the target sample rate (e.g., 16000 Hz)."""
    ...
```