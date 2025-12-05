# Training Guide

This guide explains how to use WakeGen-generated datasets to train wake word detection models.

## Overview

WakeGen integrates with popular wake word training frameworks:

- **OpenWakeWord** - Recommended for most use cases
- **Mycroft Precise** - Good for simple wake words
- **Picovoice Porcupine** - Commercial option with good accuracy

---

## Complete Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Generate   │───▶│  Augment    │───▶│   Export    │───▶│   Train     │
│  Samples    │    │  Samples    │    │  Dataset    │    │   Model     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Step 1: Generate Samples

```bash
# Generate 1000 samples with multiple providers
wakegen generate --text "hey assistant" --count 1000 --config generation.yaml
```

### Step 2: Augment Samples

```bash
# Apply augmentation for robustness
wakegen augment --input ./output --output ./augmented \
  --profile morning_kitchen \
  --device esp32_i2s \
  --variations 3
```

### Step 3: Export Dataset

```bash
# Export for OpenWakeWord
wakegen export --input ./augmented --output ./dataset \
  --format openwakeword \
  --split 0.8 0.1 0.1
```

### Step 4: Train Model

See framework-specific sections below.

---

## OpenWakeWord Training

OpenWakeWord is the recommended framework for custom wake word models.

### Prerequisites

```bash
pip install openwakeword
```

### Generate Training Script

WakeGen can generate a training script for your dataset:

```bash
wakegen train-script \
  --dataset ./dataset \
  --model-type openwakeword \
  --output train_hey_assistant.py
```

### Manual Training

```python
from openwakeword import Model
from openwakeword.utils import train_model

# Train custom wake word model
train_model(
    positive_samples="./dataset/train/positive",
    negative_samples="./dataset/train/negative",
    output_path="./models/hey_assistant.onnx",
    model_type="wakeword",
    epochs=50,
    batch_size=32,
)
```

### Training Configuration

```yaml
# openwakeword_config.yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  
  # Model architecture
  model_type: wakeword
  feature_extractor: melspectrogram
  
  # Augmentation during training
  augment_on_fly: true
  mixup_alpha: 0.2
  
  # Validation
  validation_split: 0.1
  early_stopping_patience: 5
```

### Dataset Structure for OpenWakeWord

```
dataset/
├── train/
│   ├── positive/           # Wake word samples
│   │   ├── hey_assistant_001.wav
│   │   ├── hey_assistant_002.wav
│   │   └── ...
│   └── negative/           # Non-wake word samples
│       ├── negative_001.wav
│       └── ...
├── validation/
│   ├── positive/
│   └── negative/
└── test/
    ├── positive/
    └── negative/
```

---

## Mycroft Precise Training

### Prerequisites

```bash
pip install precise-runner
git clone https://github.com/MycroftAI/mycroft-precise
cd mycroft-precise
pip install -e .
```

### Export for Mycroft

```bash
wakegen export --input ./augmented --output ./dataset --format mycroft
```

### Train Model

```bash
# Create model
precise-train hey_assistant.net ./dataset/

# Convert to production format
precise-convert hey_assistant.net
```

### Dataset Structure for Mycroft

```
dataset/
├── wake-word/              # Positive samples
│   ├── hey_assistant_001.wav
│   └── ...
├── not-wake-word/          # Negative samples
│   └── ...
└── test/
    ├── wake-word/
    └── not-wake-word/
```

---

## Model Testing

### Using WakeGen's Model Tester

```python
from wakegen.training import WakeWordTester

tester = WakeWordTester()

results = await tester.test_model(
    model_path="./models/hey_assistant.onnx",
    test_dataset="./dataset/test",
    framework="openwakeword"
)

print(f"True Positive Rate: {results.true_positive_rate:.2%}")
print(f"False Positive Rate: {results.false_positive_rate:.2%}")
print(f"Latency: {results.latency_ms:.1f}ms")
print(f"Memory: {results.memory_mb:.1f}MB")
```

### CLI Testing

```bash
wakegen test-model \
  --model ./models/hey_assistant.onnx \
  --test-dir ./dataset/test \
  --framework openwakeword
```

### Test Results

```
╔══════════════════════════════════════════════════════════════╗
║  Model Test Results: hey_assistant.onnx                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Accuracy Metrics:                                            ║
║    True Positive Rate:   95.3%                                ║
║    False Positive Rate:   1.2%                                ║
║    False Negative Rate:   4.7%                                ║
║    Precision:            97.8%                                ║
║    Recall:               95.3%                                ║
║    F1 Score:             96.5%                                ║
║                                                               ║
║  Performance Metrics:                                         ║
║    Average Latency:      12.3ms                               ║
║    95th Percentile:      18.7ms                               ║
║    Memory Usage:         8.2MB                                ║
║    Model Size:           2.1MB                                ║
║                                                               ║
║  Test Set:                                                    ║
║    Positive Samples:     500                                  ║
║    Negative Samples:     2000                                 ║
║                                                               ║
╚══════════════════════════════════════════════════════════════╝
```

---

## A/B Model Comparison

Compare different model versions:

```python
from wakegen.training import ModelComparison

comparison = ModelComparison()

results = await comparison.compare_models(
    model_a="./models/hey_assistant_v1.onnx",
    model_b="./models/hey_assistant_v2.onnx",
    test_dataset="./dataset/test"
)

print(results.summary())
```

### CLI Comparison

```bash
wakegen compare-models \
  --model-a ./models/v1.onnx \
  --model-b ./models/v2.onnx \
  --test-dir ./dataset/test
```

---

## Best Practices

### 1. Dataset Size

| Quality Level | Positive Samples | Negative Samples |
|---------------|------------------|------------------|
| Minimum | 500 | 2,000 |
| Good | 2,000 | 10,000 |
| Excellent | 5,000+ | 50,000+ |

### 2. Voice Diversity

Use multiple TTS providers and voices:

```yaml
providers:
  - type: edge_tts
    weight: 0.2
  - type: kokoro
    weight: 0.3
  - type: piper
    weight: 0.2
  - type: f5_tts
    weight: 0.2
  - type: bark
    weight: 0.1
```

### 3. Augmentation Balance

```yaml
augmentation:
  clean_ratio: 0.2           # 20% clean samples
  light_augmentation: 0.3    # 30% light augmentation
  medium_augmentation: 0.3   # 30% medium augmentation
  heavy_augmentation: 0.2    # 20% heavy augmentation
```

### 4. Negative Samples

Include diverse negative samples:
- Similar-sounding phrases
- Background speech
- Music and media
- Environmental sounds
- Silence

```bash
# Generate negative samples
wakegen generate-negatives \
  --wake-word "hey assistant" \
  --count 5000 \
  --output ./negatives
```

### 5. Cross-Validation

Test on unseen conditions:
- Different microphones
- Real recordings
- Various environments

---

## Deployment

### Export for Edge Devices

```bash
# Export ONNX for edge deployment
wakegen export-model \
  --input ./models/hey_assistant.onnx \
  --format onnx-quantized \
  --output ./deploy/hey_assistant_int8.onnx
```

### ESP32 Deployment

```python
# Example: Using model on ESP32 with OpenWakeWord
from openwakeword.model import Model

# Load quantized model
model = Model(
    wakeword_models=["./models/hey_assistant_int8.onnx"],
    inference_framework="onnx"
)

# Process audio
prediction = model.predict(audio_frame)
if prediction["hey_assistant"] > 0.5:
    print("Wake word detected!")
```

### Performance Targets

| Device | Target Latency | Target Memory |
|--------|----------------|---------------|
| ESP32 | <100ms | <500KB |
| Raspberry Pi | <50ms | <10MB |
| Desktop | <20ms | <50MB |

---

## Troubleshooting

### Low Accuracy

1. **Increase dataset size** - More diverse samples
2. **Add augmentation** - Simulate real conditions
3. **Check audio quality** - Validate samples
4. **Balance classes** - Proper positive/negative ratio

### High False Positives

1. **Add confusing negatives** - Similar-sounding phrases
2. **Increase negative samples** - 5-10x positive count
3. **Adjust threshold** - Higher detection threshold

### High False Negatives

1. **Add voice variety** - More speakers, accents
2. **Add augmentation** - More environmental conditions
3. **Check sample quality** - ASR verification

### Slow Inference

1. **Quantize model** - INT8 quantization
2. **Reduce model size** - Smaller architecture
3. **Optimize features** - Fewer mel bands
