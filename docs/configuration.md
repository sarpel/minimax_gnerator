# Configuration Guide

WakeGen offers flexible configuration through YAML files, environment variables, and CLI arguments.

## Configuration Priority

Settings are applied in this order (highest priority first):

1. **CLI Arguments** - `--output-dir ./my_output`
2. **YAML Config File** - `wakegen.yaml`
3. **Environment Variables** - `WAKEGEN_OUTPUT_DIR`
4. **Preset Files** - `quick_test.yaml`
5. **Default Values** - Built into the code

---

## YAML Configuration

### Basic Configuration

Create a `wakegen.yaml` file in your project:

```yaml
project:
  name: "my_wake_word"
  version: "1.0.0"
  description: "Wake word dataset for smart home assistant"

generation:
  wake_words:
    - "hey assistant"
    - "ok assistant"
  count: 1000
  output_dir: "./output/my_wake_word"
  sample_rate: 16000
  audio_format: "wav"

providers:
  - type: edge_tts
    voices:
      - en-US-AriaNeural
      - en-US-GuyNeural
    weight: 0.3
  - type: kokoro
    voices:
      - af_bella
      - am_adam
    weight: 0.4
  - type: piper
    voices:
      - en_US-lessac-medium
    weight: 0.3

augmentation:
  enabled: true
  profiles:
    - morning_kitchen
    - car_interior
  augmented_per_original: 3

export:
  format: openwakeword
  split_ratio: [0.8, 0.1, 0.1]  # train/val/test
```

### Run with Config

```bash
wakegen generate --config wakegen.yaml
```

---

## Generation Settings

```yaml
generation:
  # Text settings
  wake_words:
    - "hey assistant"
    - "assistant"
  
  # Output settings
  count: 1000                    # Samples per wake word
  output_dir: "./output"         # Output directory
  sample_rate: 16000             # Audio sample rate (Hz)
  audio_format: "wav"            # wav, mp3, flac
  
  # Voice variation
  speed_range: [0.9, 1.1]        # Speed variation factor
  pitch_range: [-2, 2]           # Pitch shift (semitones)
  
  # Processing
  max_concurrent: 5              # Parallel generation tasks
  retry_attempts: 3              # Retries on failure
  timeout_seconds: 30            # Per-sample timeout
  
  # Checkpointing
  enable_checkpoints: true       # Resume interrupted generation
  checkpoint_interval: 100       # Save every N samples
```

---

## Provider Configuration

### Multi-Provider Setup

```yaml
providers:
  # Cloud provider (no local resources)
  - type: edge_tts
    weight: 0.2
    voices:
      - en-US-AriaNeural
      - en-US-GuyNeural
      - en-GB-SoniaNeural
    
  # Lightweight local provider
  - type: kokoro
    weight: 0.3
    voices:
      - af_bella
      - af_sarah
      - am_adam
      - am_michael
    
  # High-quality local provider
  - type: f5_tts
    weight: 0.3
    gpu_required: true
    voices:
      - default
    
  # Voice cloning provider
  - type: coqui_xtts
    weight: 0.2
    gpu_required: true
    reference_audio: "./references/speaker1.wav"
```

### Provider-Specific Settings

```yaml
providers:
  - type: minimax
    api_key: ${MINIMAX_API_KEY}   # From environment
    group_id: ${MINIMAX_GROUP_ID}
    rate_limit: 10                 # Requests per minute
    
  - type: piper
    model_path: "./models/piper"
    voices:
      - en_US-lessac-medium
      - en_GB-alan-medium
    
  - type: bark
    use_gpu: true
    small_model: false             # Use full model
    voices:
      - v2/en_speaker_0
      - v2/en_speaker_6
```

---

## Augmentation Configuration

### Basic Augmentation

```yaml
augmentation:
  enabled: true
  augmented_per_original: 3       # Create 3 augmented versions per original
```

### Environment Profiles

```yaml
augmentation:
  profiles:
    - morning_kitchen             # Kitchen with appliances
    - car_interior                # Vehicle environment
    - office_space                # Office with HVAC
    - outdoor_park                # Outdoor with wind
```

### Device-Specific Presets

```yaml
augmentation:
  device_preset: esp32_i2s        # Optimize for ESP32 with I2S mic
  
# Available presets:
# - esp32_pdm       : ESP32 with PDM microphone
# - esp32_i2s       : ESP32 with I2S microphone (INMP441)
# - esp32_s3        : ESP32-S3 with improved audio
# - rpi_usb         : Raspberry Pi with USB microphone
# - rpi_hat         : Raspberry Pi with ReSpeaker HAT
# - smart_speaker_mid    : Mid-range smart speaker
# - smart_speaker_premium: Premium smart speaker
# - mobile_ios      : iPhone microphone
# - mobile_android  : Android phone microphone
# - far_field       : Far-field microphone array
# - near_field      : Headset/close-talking
```

### Custom Augmentation

```yaml
augmentation:
  enabled: true
  
  # Noise settings
  noise:
    enabled: true
    snr_range: [5, 25]            # Signal-to-noise ratio (dB)
    noise_types:
      - hvac
      - speech_babble
      - appliances
      - traffic
  
  # Room simulation
  room:
    enabled: true
    room_presets:
      - small_room
      - medium_room
      - large_room
    rt60_range: [0.2, 0.8]        # Reverb time (seconds)
  
  # Microphone simulation
  microphone:
    enabled: true
    profiles:
      - smartphone
      - conference
      - headset
  
  # Telephony effects
  telephony:
    enabled: false
    phone_types:
      - mobile_gsm
      - voip_medium
  
  # Distance simulation
  distance:
    enabled: true
    distance_range: [0.5, 5.0]    # Distance in meters
  
  # Time domain effects
  time_domain:
    pitch_shift_range: [-2, 2]    # Semitones
    time_stretch_range: [0.9, 1.1]
  
  # Audio degradation
  degradation:
    enabled: false
    mp3_bitrate: [64, 96, 128]
```

---

## Export Configuration

```yaml
export:
  format: openwakeword            # Export format
  output_dir: "./dataset"
  
  # Dataset splits
  split_ratio: [0.8, 0.1, 0.1]    # train/val/test
  shuffle: true
  random_seed: 42
  
  # Metadata
  include_metadata: true
  metadata_format: json

# Available formats:
# - openwakeword  : OpenWakeWord format
# - mycroft       : Mycroft Precise format
# - picovoice     : Picovoice format
# - tensorflow    : TensorFlow/Keras format
# - pytorch       : PyTorch DataLoader format
# - huggingface   : HuggingFace Datasets format
```

---

## Quality Settings

```yaml
quality:
  # Validation
  enable_validation: true
  min_duration_ms: 200
  max_duration_ms: 3000
  min_snr_db: 10
  
  # ASR verification
  asr_verification:
    enabled: true
    engine: whisper               # whisper, vosk
    min_confidence: 0.7
    auto_reject_below: 0.5
  
  # Deduplication
  deduplication:
    enabled: true
    similarity_threshold: 0.95
```

---

## Environment Variables

Set in `.env` file or system environment:

```bash
# API Keys
MINIMAX_API_KEY=your_api_key_here
MINIMAX_GROUP_ID=your_group_id

# Directories
WAKEGEN_OUTPUT_DIR=./output
WAKEGEN_CACHE_DIR=./.wakegen_cache
WAKEGEN_MODEL_DIR=./models

# Logging
WAKEGEN_LOG_LEVEL=INFO           # DEBUG, INFO, WARNING, ERROR

# Performance
WAKEGEN_MAX_WORKERS=4
WAKEGEN_GPU_DEVICE=cuda:0        # cuda:0, cuda:1, mps, cpu
```

### Using Environment Variables in YAML

```yaml
providers:
  - type: minimax
    api_key: ${MINIMAX_API_KEY}
    group_id: ${MINIMAX_GROUP_ID}
```

---

## Presets

Presets are pre-configured YAML files in `wakegen/config/presets/`.

### Using a Preset

```bash
wakegen generate --preset quick_test --text "hey assistant"
```

### Available Presets

| Preset | Description |
|--------|-------------|
| `quick_test` | Quick test with 10 samples |
| `production` | Full production dataset |
| `esp32` | Optimized for ESP32 devices |
| `high_quality` | Maximum quality settings |

### Creating Custom Presets

Create a YAML file in `wakegen/config/presets/`:

```yaml
# wakegen/config/presets/my_preset.yaml
generation:
  count: 500
  output_dir: "./output/custom"
  
providers:
  - type: kokoro
    weight: 1.0
    
augmentation:
  enabled: true
  profiles:
    - morning_kitchen
```

Use with:
```bash
wakegen generate --preset my_preset --text "hey assistant"
```

---

## CLI Configuration Reference

```bash
# Generation options
wakegen generate \
  --text "hey assistant" \          # Wake word text
  --count 100 \                     # Number of samples
  --output-dir ./output \           # Output directory
  --provider kokoro \               # Specific provider
  --voice af_bella \                # Specific voice
  --config wakegen.yaml \           # Config file
  --preset quick_test               # Use preset

# Augmentation options
wakegen augment \
  --input ./output \                # Input directory
  --output ./augmented \            # Output directory
  --profile morning_kitchen \       # Environment profile
  --device esp32_i2s \              # Device preset
  --variations 3                    # Variations per sample

# Export options
wakegen export \
  --input ./augmented \             # Input directory
  --output ./dataset \              # Output directory
  --format openwakeword \           # Export format
  --split 0.8 0.1 0.1               # Train/val/test split

# Utility commands
wakegen list-providers              # Show all providers
wakegen list-providers --available  # Show installed providers
wakegen list-voices --provider kokoro  # Show voices
wakegen cache stats                 # Show cache statistics
wakegen cache clear                 # Clear generation cache
```
