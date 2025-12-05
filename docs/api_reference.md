# API Reference

This reference documents the main classes and functions for developers who want to use WakeGen as a Python library.

---

## Quick Example

```python
import asyncio
from wakegen.providers.registry import get_provider
from wakegen.core.types import ProviderType
from wakegen.models.config import ProviderConfig

async def main():
    # Get a TTS provider
    provider = get_provider(ProviderType.EDGE_TTS, ProviderConfig())
    
    # Generate audio
    await provider.generate("hey assistant", "en-US-AriaNeural", "output.wav")
    
    # List available voices
    voices = await provider.list_voices()
    for voice in voices[:5]:
        print(f"{voice.id}: {voice.name} ({voice.language})")

asyncio.run(main())
```

---

## Core Components

### Provider Registry

```python
from wakegen.providers.registry import (
    get_provider,
    list_available_providers,
    is_provider_available,
    get_provider_requirements,
)
from wakegen.core.types import ProviderType
from wakegen.models.config import ProviderConfig
```

#### `get_provider(provider_type, config) -> TTSProvider`

Factory function to get a TTS provider instance.

**Parameters:**
- `provider_type: ProviderType` - The provider to instantiate
- `config: ProviderConfig` - Configuration for the provider

**Returns:** Instance of `TTSProvider`

**Example:**
```python
provider = get_provider(ProviderType.KOKORO, ProviderConfig())
```

#### `list_available_providers() -> List[ProviderInfo]`

Returns information about all providers with availability status.

```python
providers = list_available_providers()
for p in providers:
    print(f"{p.type.value}: {'✓' if p.available else '✗'}")
```

#### `is_provider_available(provider_type) -> bool`

Check if a provider's dependencies are installed.

```python
if is_provider_available(ProviderType.BARK):
    provider = get_provider(ProviderType.BARK, config)
```

---

### Provider Types

```python
from wakegen.core.types import ProviderType

class ProviderType(Enum):
    # Cloud providers
    EDGE_TTS = "edge_tts"
    MINIMAX = "minimax"
    
    # Lightweight local providers
    KOKORO = "kokoro"
    PIPER = "piper"
    MIMIC3 = "mimic3"
    
    # High-quality local providers
    COQUI_XTTS = "coqui_xtts"
    F5_TTS = "f5_tts"
    STYLETTS2 = "styletts2"
    ORPHEUS = "orpheus"
    
    # Expressive providers
    BARK = "bark"
    CHATTTS = "chattts"
```

---

### TTSProvider Protocol

All TTS providers implement this interface:

```python
from wakegen.core.protocols import TTSProvider

class TTSProvider(Protocol):
    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type enum."""
        ...
    
    async def generate(
        self,
        text: str,
        voice_id: str,
        output_path: str,
        **kwargs
    ) -> None:
        """Generate audio from text.
        
        Args:
            text: The text to synthesize
            voice_id: Voice identifier
            output_path: Path to save the audio file
            **kwargs: Provider-specific options
        """
        ...
    
    async def list_voices(
        self,
        language: Optional[str] = None
    ) -> List[Voice]:
        """List available voices.
        
        Args:
            language: Optional language filter (e.g., "en-US")
            
        Returns:
            List of available Voice objects
        """
        ...
```

---

### Voice Model

```python
from wakegen.models.audio import Voice, Gender

@dataclass
class Voice:
    id: str                      # Unique voice identifier
    name: str                    # Human-readable name
    language: str                # Language code (e.g., "en-US")
    gender: Gender               # MALE, FEMALE, or NEUTRAL
    provider: ProviderType       # Source provider
    description: Optional[str]   # Voice description
    sample_rate: int = 16000     # Default sample rate

class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"
```

---

## Augmentation

### AugmentationPipeline

```python
from wakegen.augmentation import AugmentationPipeline
from wakegen.augmentation.profiles import EnvironmentProfile

# Create pipeline with profile
pipeline = AugmentationPipeline(profile=EnvironmentProfile.MORNING_KITCHEN)

# Process audio
import soundfile as sf
audio, sr = sf.read("input.wav")
augmented = pipeline.process(audio, sr)
sf.write("augmented.wav", augmented, sr)
```

### Environment Profiles

```python
from wakegen.augmentation.profiles import (
    EnvironmentProfile,
    get_profile,
    list_profiles,
)

class EnvironmentProfile(Enum):
    MORNING_KITCHEN = "morning_kitchen"
    CAR_INTERIOR = "car_interior"
    OFFICE_SPACE = "office_space"
    OUTDOOR_PARK = "outdoor_park"
    BEDROOM_NIGHT = "bedroom_night"
    EVENING_LIVING_ROOM = "evening_living_room"

# Get profile settings
profile = get_profile(EnvironmentProfile.CAR_INTERIOR)
print(profile.noise_profile)
print(profile.room_params)
```

### Device Presets

```python
from wakegen.augmentation import (
    TargetDevice,
    get_device_preset,
    list_device_presets,
    DevicePresetManager,
)

class TargetDevice(Enum):
    ESP32_PDM = "esp32_pdm"
    ESP32_I2S = "esp32_i2s"
    ESP32_S3 = "esp32_s3"
    RPI_USB = "rpi_usb"
    RPI_HAT = "rpi_hat"
    SMART_SPEAKER_MID = "smart_speaker_mid"
    SMART_SPEAKER_PREMIUM = "smart_speaker_premium"
    MOBILE_IOS = "mobile_ios"
    MOBILE_ANDROID = "mobile_android"
    FAR_FIELD = "far_field"
    NEAR_FIELD = "near_field"
    # ... more devices

# Get preset for device
preset = get_device_preset(TargetDevice.ESP32_I2S)
print(preset.audio_spec)        # Device audio specs
print(preset.environment)       # Typical environment
print(preset.priority_augmentations)  # Key augmentations
```

### Individual Effects

```python
from wakegen.augmentation.effects import (
    TelephonySimulator,
    DistanceSimulator,
    PhoneType,
)

# Telephony simulation
telephony = TelephonySimulator()
audio = telephony.apply_telephony_effect(
    audio, sr,
    phone_type=PhoneType.MOBILE_GSM
)

# Distance simulation
distance = DistanceSimulator()
audio = distance.apply_distance_effect(
    audio, sr,
    distance_meters=3.0,
    room_size="medium"
)
```

---

## Export

### Export Formats

```python
from wakegen.export.formats import (
    ExportFormat,
    export_to_format,
    list_export_formats,
    DatasetMetadata,
    SampleMetadata,
)

class ExportFormat(Enum):
    OPENWAKEWORD = "openwakeword"
    MYCROFT_PRECISE = "mycroft_precise"
    PICOVOICE = "picovoice"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    HUGGINGFACE = "huggingface"

# Export dataset
await export_to_format(
    input_dir="./augmented",
    output_dir="./dataset",
    format=ExportFormat.OPENWAKEWORD,
    split_ratio=(0.8, 0.1, 0.1),
    metadata=DatasetMetadata(
        name="hey_assistant",
        wake_word="hey assistant",
        version="1.0.0"
    )
)
```

### Dataset Splitter

```python
from wakegen.export.splitter import DatasetSplitter

splitter = DatasetSplitter(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
)

splits = splitter.split(
    input_dir="./samples",
    output_dir="./dataset"
)
```

---

## Quality

### Quality Scoring

```python
from wakegen.quality import (
    QualityScorer,
    calculate_quality_score,
    QualityReport,
)

# Score a single sample
score = await calculate_quality_score("sample.wav")
print(f"Overall: {score.overall_score:.2f}")
print(f"SNR: {score.snr_score:.2f}")
print(f"Clarity: {score.clarity_score:.2f}")

# Score multiple samples
scorer = QualityScorer()
reports = await scorer.score_directory("./samples")
```

### ASR Verification

```python
from wakegen.quality.asr_check import ASRVerifier

verifier = ASRVerifier(engine="whisper")

result = await verifier.verify(
    audio_path="sample.wav",
    expected_text="hey assistant"
)

print(f"Transcription: {result.transcription}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Match: {result.is_match}")
```

### Validation

```python
from wakegen.quality.validator import AudioValidator

validator = AudioValidator(
    min_duration_ms=200,
    max_duration_ms=3000,
    min_snr_db=10
)

is_valid, issues = await validator.validate("sample.wav")
if not is_valid:
    for issue in issues:
        print(f"Issue: {issue}")
```

---

## Configuration

### Loading Configuration

```python
from wakegen.config import (
    load_yaml_config,
    get_generation_config,
    GenerationConfig,
)

# Load from YAML file
config = load_yaml_config("wakegen.yaml")

# Load preset
config = get_generation_config(preset_name="quick_test")

# Access settings
print(config.output_dir)
print(config.sample_rate)
print(config.providers)
```

### GenerationConfig

```python
from wakegen.models.config import GenerationConfig

@dataclass
class GenerationConfig:
    # Output settings
    output_dir: str = "./output"
    sample_rate: int = 16000
    audio_format: str = "wav"
    
    # Generation settings
    max_concurrent_tasks: int = 5
    retry_attempts: int = 3
    task_timeout_seconds: int = 300
    
    # Voice variation
    speed_range: Tuple[float, float] = (0.9, 1.1)
    pitch_range: Tuple[float, float] = (-2, 2)
    
    # Checkpointing
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 100
```

---

## Utilities

### Audio Utilities

```python
from wakegen.utils.audio import (
    resample_audio,
    normalize_audio,
    get_audio_duration,
    calculate_snr,
)

# Resample to 16kHz
resample_audio("input.wav", "output.wav", target_sr=16000)

# Normalize volume
normalize_audio("input.wav", "output.wav", target_db=-20)

# Get duration
duration_ms = get_audio_duration("sample.wav")

# Calculate SNR
snr_db = calculate_snr(signal, noise)
```

### Caching

```python
from wakegen.utils.caching import GenerationCache

cache = GenerationCache(cache_dir=".wakegen_cache")

# Check if cached
cached_path = cache.get("hey assistant", "af_bella", "kokoro")
if cached_path:
    print(f"Using cached: {cached_path}")
else:
    # Generate and cache
    await provider.generate(text, voice, output)
    cache.put(text, voice, "kokoro", output)
```

### Performance

```python
from wakegen.utils.performance import GPUManager

gpu = GPUManager()

# Check GPU availability
print(f"CUDA: {gpu.cuda_available}")
print(f"MPS: {gpu.mps_available}")
print(f"Device: {gpu.device}")

# Get optimal batch size
batch_size = gpu.get_optimal_batch_size(model_size_mb=500)
```

---

## Training Integration

### Model Testing

```python
from wakegen.training import WakeWordTester, TestResults

tester = WakeWordTester()

results: TestResults = await tester.test_model(
    model_path="./models/hey_assistant.onnx",
    test_dataset="./dataset/test",
    framework="openwakeword"
)

print(f"TPR: {results.true_positive_rate:.2%}")
print(f"FPR: {results.false_positive_rate:.2%}")
print(f"Latency: {results.latency_ms:.1f}ms")
```

### A/B Comparison

```python
from wakegen.training import ModelComparison

comparison = ModelComparison()

results = await comparison.compare_models(
    model_a="./models/v1.onnx",
    model_b="./models/v2.onnx",
    test_dataset="./dataset/test"
)

print(results.winner)
print(results.summary())
```

---

## CLI Integration

You can also access WakeGen features from Python:

```python
from wakegen.ui.cli import commands
import click

# Programmatic CLI invocation
from click.testing import CliRunner

runner = CliRunner()
result = runner.invoke(commands.generate, [
    "--text", "hey assistant",
    "--count", "10",
    "--provider", "kokoro"
])
print(result.output)
```
