# Quick Start Guide

This guide will get you up and running with **WakeGen** in under 5 minutes.

## Prerequisites

- Python 3.10+
- pip or uv package manager

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wakegen.git
cd wakegen

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install wakegen
pip install -e .
```

## 2. Generate Your First Samples

### Using the CLI

```bash
# Generate 10 samples with Edge TTS (no setup required)
wakegen generate --text "hey computer" --count 10

# List available voices
wakegen list-voices --provider edge_tts

# Use a specific voice
wakegen generate --text "hey computer" --count 10 --voice en-US-AriaNeural
```

### Using the Interactive Wizard

```bash
wakegen generate --interactive
```

The wizard guides you through:
1. Wake word selection
2. Sample count
3. Provider selection
4. Voice preferences
5. Output directory

## 3. Use Multiple Providers

WakeGen supports 11+ TTS providers for diverse voice samples:

```bash
# List all available providers
wakegen list-providers

# Generate with a specific provider
wakegen generate --text "hey assistant" --provider piper --count 20

# Use Kokoro (lightweight, fast)
wakegen generate --text "hey assistant" --provider kokoro --count 20
```

## 4. Batch Generation with Config File

For production datasets, use a YAML configuration:

```yaml
# wakegen.yaml
project:
  name: "hey_assistant"
  version: "1.0.0"

generation:
  wake_words:
    - "hey assistant"
  count: 1000
  output_dir: "./output/hey_assistant"

providers:
  - type: edge_tts
    weight: 0.3
  - type: kokoro
    weight: 0.4
  - type: piper
    weight: 0.3

augmentation:
  enabled: true
  profiles:
    - morning_kitchen
    - car_interior
  augmented_per_original: 3
```

Run with:
```bash
wakegen generate --config wakegen.yaml
```

## 5. Apply Augmentation

Make your samples robust with augmentation:

```bash
# Augment existing samples
wakegen augment --input ./output --output ./augmented

# Use a specific profile
wakegen augment --input ./output --output ./augmented --profile morning_kitchen

# Target specific device
wakegen augment --input ./output --output ./augmented --device esp32_i2s
```

## 6. Export for Training

Export your dataset for different training frameworks:

```bash
# Export for OpenWakeWord
wakegen export --input ./augmented --output ./dataset --format openwakeword

# Export for PyTorch
wakegen export --input ./augmented --output ./dataset --format pytorch

# Export for HuggingFace
wakegen export --input ./augmented --output ./dataset --format huggingface
```

## 7. Check Your Results

```bash
# View generation statistics
wakegen stats --dir ./output

# Check quality scores
wakegen validate --dir ./output
```

Navigate to your output directory:
```bash
ls output/
```

You'll see files like:
- `hey_computer_edge_tts_001.wav`
- `hey_computer_kokoro_002.wav`
- `hey_computer_piper_003.wav`

## Quick Provider Reference

| Provider | Type | GPU Required | Best For |
|----------|------|--------------|----------|
| Edge TTS | Cloud | No | Quick start, many languages |
| Kokoro | Local | No | Fast CPU generation |
| Piper | Local | No | Offline, embedded devices |
| Mimic 3 | Local | No | Privacy-focused |
| F5-TTS | Local | Recommended | High quality |
| StyleTTS 2 | Local | Recommended | Expressive speech |
| Bark | Local | Yes | Expressive, non-speech sounds |
| ChatTTS | Local | Recommended | Conversational |
| Coqui XTTS | Local | Yes | Voice cloning |
| Orpheus | Local | Varies | Scalable quality |
| MiniMax | Cloud | No | Commercial quality |

## Next Steps

- **[Providers Guide](providers.md)** - Learn about all TTS providers
- **[Augmentation Guide](augmentation.md)** - Make samples robust
- **[Configuration Guide](configuration.md)** - Advanced configuration
- **[Training Guide](training.md)** - Train wake word models

## Common Issues

### "No module named 'kokoro'"

Install the provider's dependencies:
```bash
pip install kokoro-onnx
```

### "Provider not available"

Check if provider is installed:
```bash
wakegen list-providers --available
```

### Audio driver issues

Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1 portaudio19-dev

# macOS
brew install portaudio libsndfile
```
