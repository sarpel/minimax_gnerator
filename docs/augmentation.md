# Augmentation Guide

WakeGen includes a comprehensive augmentation pipeline that transforms clean audio samples into realistic variations that improve wake word model robustness.

## Why Augmentation Matters

Training on clean studio audio results in models that fail in real-world conditions:
- Background noise (TV, kitchen appliances, traffic)
- Room acoustics (echo, reverb)
- Distance from microphone
- Device-specific characteristics (ESP32, phone, smart speaker)
- Telephony effects (phone calls, VoIP)

Augmentation simulates these conditions to create robust models.

---

## Augmentation Types

### 1. Background Noise Mixing ✅

Adds realistic environmental sounds to your audio.

**Noise Types**:
- White noise, pink noise
- HVAC/ventilation
- Kitchen appliances
- Traffic sounds
- Speech babble (background conversation)
- Office environments
- Music/TV in background

**Configuration**:
```yaml
augmentation:
  noise:
    enabled: true
    snr_range: [5, 25]  # Signal-to-noise ratio in dB
    noise_types:
      - hvac
      - speech_babble
      - appliances
```

### 2. Room Simulation ✅

Simulates room acoustics using impulse responses.

**Room Types**:
- Small room (bedroom, bathroom)
- Medium room (living room, office)
- Large room (conference room, hall)
- Open space / outdoor

**Parameters**:
- RT60 (reverberation time)
- Room size
- Material absorption
- Microphone/speaker positions

```yaml
augmentation:
  room:
    enabled: true
    room_presets:
      - small_room
      - medium_room
      - large_room
    rt60_range: [0.2, 0.8]  # Reverb time in seconds
```

### 3. Microphone Simulation ✅

Simulates frequency response and characteristics of different microphones.

**Microphone Profiles**:
- Studio condenser
- Smartphone
- Conference microphone
- Headset
- Lapel/lavalier
- Far-field array

**Effects**:
- Frequency response filtering
- Self-noise addition
- Distortion simulation
- Sensitivity variations

```yaml
augmentation:
  microphone:
    enabled: true
    profiles:
      - smartphone
      - conference
      - headset
```

### 4. Telephony Simulation ✅

Simulates phone/VoIP audio characteristics.

**Phone Types**:
- PSTN landline
- Mobile GSM
- Mobile HD Voice
- VoIP (low/medium/high quality)
- Cordless DECT

**Effects**:
- Bandwidth limiting (300-3400 Hz for PSTN)
- Codec artifacts (GSM, AMR, Opus)
- Packet loss simulation
- Jitter
- Echo

```yaml
augmentation:
  telephony:
    enabled: true
    phone_types:
      - mobile_gsm
      - voip_medium
    packet_loss_rate: 0.02
    jitter_ms: 20
```

### 5. Distance Simulation ✅

Simulates speaking from different distances.

**Effects**:
- Distance attenuation (inverse square law)
- Air absorption (high frequency rolloff)
- Proximity effect (bass boost when close)
- Room reverb scaling

```yaml
augmentation:
  distance:
    enabled: true
    distance_range: [0.5, 5.0]  # meters
    air_absorption: true
    proximity_effect: true
```

### 6. Time Domain Effects ✅

Modifies temporal characteristics of audio.

**Effects**:
- Pitch shifting (semitones)
- Time stretching (speed change without pitch)
- Speed variation (with pitch change)

```yaml
augmentation:
  time_domain:
    pitch_shift_range: [-2, 2]  # semitones
    time_stretch_range: [0.9, 1.1]  # factor
    speed_range: [0.95, 1.05]  # factor
```

### 7. Dynamics Processing ✅

Adjusts dynamic range of audio.

**Effects**:
- Compression (reduce dynamic range)
- Limiting (prevent clipping)
- Expansion (increase dynamic range)
- Gating (remove quiet sounds)

```yaml
augmentation:
  dynamics:
    compression:
      threshold_db: -20
      ratio: 3.0
    limiting:
      threshold_db: -1
```

### 8. Audio Degradation ✅

Simulates quality loss and artifacts.

**Effects**:
- Bitrate reduction (MP3, AAC)
- Sample rate reduction
- Bit depth reduction (quantization noise)
- Clipping simulation
- DC offset

```yaml
augmentation:
  degradation:
    enabled: true
    mp3_bitrate: [64, 96, 128]  # kbps
    sample_rate_reduction: [8000, 11025, 16000]
```

---

## Environment Profiles ✅

Pre-configured profiles combining multiple augmentation types for realistic scenarios.

### Available Profiles

| Profile | Description | Use Case |
|---------|-------------|----------|
| `morning_kitchen` | Kitchen with appliances, dishes | Smart speaker |
| `evening_living_room` | TV, quiet conversation | Smart home |
| `office_space` | HVAC, keyboards, meetings | Office assistant |
| `car_interior` | Road noise, engine | Car assistant |
| `outdoor_park` | Wind, birds, traffic | Mobile device |
| `bedroom_night` | Very quiet, minimal noise | Bedside device |

**Usage**:
```yaml
augmentation:
  profiles:
    - morning_kitchen
    - car_interior
  augmented_per_original: 3  # Create 3 variations per profile
```

---

## Device-Specific Presets ✅

Optimized augmentation for specific target hardware.

### Available Device Presets

| Device | Sample Rate | Characteristics |
|--------|-------------|-----------------|
| `esp32_pdm` | 16kHz | PDM mic, limited bandwidth |
| `esp32_i2s` | 16kHz | I2S mic (INMP441), better quality |
| `rpi_usb` | 16kHz | USB microphone |
| `rpi_hat` | 16kHz | ReSpeaker 4-mic array |
| `smart_speaker_mid` | 16kHz | Echo Dot style, 4-mic array |
| `smart_speaker_premium` | 16kHz | Echo Studio, 7-mic array |
| `mobile_ios` | 16kHz | iPhone microphone |
| `mobile_android` | 16kHz | Android phone |
| `far_field` | 16kHz | Far-field array, 2-8m distance |
| `near_field` | 16kHz | Headset, close-talking |

**Usage**:
```python
from wakegen.augmentation import get_device_preset, TargetDevice

preset = get_device_preset(TargetDevice.ESP32_I2S)
print(preset.audio_spec)  # Device audio specifications
print(preset.environment)  # Typical deployment environment
print(preset.priority_augmentations)  # Most important augmentations
```

```yaml
augmentation:
  device_preset: esp32_i2s
  # Automatically configures optimal augmentation for ESP32 with I2S mic
```

---

## CLI Usage

### Basic Augmentation

```bash
# Augment samples with default settings
wakegen augment --input ./clean_samples --output ./augmented

# Use specific profile
wakegen augment --input ./clean_samples --output ./augmented --profile morning_kitchen

# Use device preset
wakegen augment --input ./clean_samples --output ./augmented --device esp32_i2s
```

### Advanced Options

```bash
# Control augmentation count
wakegen augment --input ./clean --output ./aug --variations 5

# Specific augmentation types
wakegen augment --input ./clean --output ./aug \
  --noise --room --microphone \
  --snr-range 5 20 \
  --distance-range 0.5 3.0

# Batch processing with config file
wakegen augment --config augmentation.yaml
```

---

## Python API

### Basic Usage

```python
from wakegen.augmentation import AugmentationPipeline
import soundfile as sf

# Load audio
audio, sr = sf.read("clean_sample.wav")

# Create pipeline
pipeline = AugmentationPipeline()

# Apply augmentation
augmented = pipeline.process(audio, sr)

# Save result
sf.write("augmented_sample.wav", augmented, sr)
```

### Using Profiles

```python
from wakegen.augmentation import get_profile, EnvironmentProfile

# Get pre-configured profile
profile = get_profile(EnvironmentProfile.CAR_INTERIOR)

# Access profile settings
print(profile.noise_profile)
print(profile.room_params)
print(profile.microphone_profile)
```

### Using Device Presets

```python
from wakegen.augmentation import (
    get_device_preset,
    TargetDevice,
    DevicePresetManager,
)

# Get preset for ESP32
preset = get_device_preset(TargetDevice.ESP32_I2S)

# Get augmentation config
config = DevicePresetManager().get_augmentation_config(
    TargetDevice.ESP32_I2S,
    include_all=False  # Only priority augmentations
)

# List all presets
from wakegen.augmentation import list_device_presets
for preset in list_device_presets():
    print(f"{preset['name']}: {preset['description']}")
```

### Custom Augmentation

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

## Best Practices

### 1. Match Your Target Environment

Choose augmentations that match where your wake word will be used:
- **Smart speaker** → Room simulation, far-field, background noise
- **Phone app** → Mobile mic, street noise, telephony
- **Car** → Road noise, limited bandwidth
- **Headset** → Near-field, minimal reverb

### 2. Use Realistic SNR Values

| Environment | Typical SNR |
|-------------|-------------|
| Quiet bedroom | 30-50 dB |
| Living room with TV | 15-25 dB |
| Kitchen cooking | 10-20 dB |
| Busy street | 5-15 dB |
| Crowded cafe | 0-10 dB |

### 3. Balance Augmentation

Don't over-augment. Include:
- ~20% clean or lightly augmented
- ~40% moderate augmentation
- ~30% heavy augmentation
- ~10% extreme (edge cases)

### 4. Verify Quality

Use the quality scoring system to ensure augmented samples are still recognizable:

```python
from wakegen.quality import calculate_quality_score

score = await calculate_quality_score("augmented_sample.wav")
if score.overall_score < 0.3:
    print("Warning: Sample may be too degraded")
```

### 5. Use Multiple Variations

Create diverse variations of each sample:

```yaml
augmentation:
  variations_per_sample: 5
  variation_strategy: random  # or: sequential, grid
```

---

## Troubleshooting

### Augmented Audio Sounds Unrealistic

- Check SNR values (too low = unintelligible)
- Verify room simulation RT60 (too high = swimming pool effect)
- Ensure proper sample rate matching

### Performance Issues

- Reduce parallel workers for memory-constrained systems
- Use simpler room simulation (fewer reflections)
- Skip heavy augmentations on CPU-only systems

### Model Not Improving

- Check that augmentation matches deployment environment
- Verify ASR can still transcribe augmented samples
- Include more clean samples in training mix
