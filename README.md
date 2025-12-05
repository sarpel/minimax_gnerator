# WakeGen - Wake Word Dataset Generator

A comprehensive tool to generate high-quality synthetic wake word datasets using multiple Text-to-Speech (TTS) providers. Designed for creating training data for wake word detection models like "Hey Assistant", "Jarvis", "Alexa", etc.

**Version:** 1.0.0

## ✨ Features

- **11 TTS Providers**: Edge TTS, Kokoro, Piper, Mimic3, Bark, ChatTTS, StyleTTS2, Orpheus, Coqui XTTS, F5-TTS, MiniMax
- **Advanced Augmentation**: Room simulation, noise injection, telephony effects, distance simulation
- **6 Export Formats**: OpenWakeWord, Mycroft Precise, Picovoice, TensorFlow, PyTorch, HuggingFace
- **Quality Assurance**: ASR verification, SNR scoring, automatic quality filtering
- **Device Presets**: ESP32, Raspberry Pi, smart speakers, conference systems
- **Flexible Configuration**: YAML-based configs, CLI options, environment variables

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/sarpel/wakegen.git
cd wakegen
pip install -e .

# Or install with GPU support
pip install -e ".[gpu]"
```

### Basic Usage

```bash
# Generate samples with Edge TTS (no setup required)
wakegen generate --text "hey assistant" --count 100

# List available voices for a provider
wakegen list-voices --provider edge_tts

# Use multiple providers with YAML config
wakegen generate --config my_config.yaml

# Generate with augmentation
wakegen generate --text "hey assistant" --count 100 --augment --profile kitchen
```

### Interactive Wizard

```bash
wakegen wizard
```

## 📋 Supported Providers

| Provider | Type | CPU | GPU | Quality | Voice Cloning |
|----------|------|-----|-----|---------|---------------|
| Edge TTS | Free | ✅ | - | ⭐⭐⭐⭐ | ❌ |
| Kokoro | Open Source | ✅ | ❌ | ⭐⭐⭐⭐ | ❌ |
| Piper | Open Source | ✅ | ❌ | ⭐⭐⭐ | ❌ |
| Mimic3 | Open Source | ✅ | ❌ | ⭐⭐⭐ | ❌ |
| Bark | Open Source | ❌ | ✅ | ⭐⭐⭐⭐ | ✅ |
| ChatTTS | Open Source | ❌ | ✅ | ⭐⭐⭐⭐ | ❌ |
| StyleTTS2 | Open Source | ❌ | ✅ | ⭐⭐⭐⭐⭐ | ❌ |
| Orpheus | Open Source | ✅/❌ | ✅ | ⭐⭐⭐-⭐⭐⭐⭐⭐ | ❌ |
| Coqui XTTS | Open Source | ❌ | ✅ | ⭐⭐⭐⭐⭐ | ✅ |
| F5-TTS | Open Source | ❌ | ✅ | ⭐⭐⭐⭐⭐ | ✅ |
| MiniMax | Commercial | ✅ | - | ⭐⭐⭐⭐⭐ | ❌ |

## 📂 Project Structure

```
wakegen/
├── augmentation/    # Audio augmentation pipeline
│   ├── effects/     # Time/frequency domain effects
│   ├── noise/       # Noise injection and mixing
│   ├── room/        # Room impulse response simulation
│   └── microphone/  # Microphone simulation
├── config/          # Configuration and presets
├── core/            # Core types and protocols
├── export/          # Export format handlers
├── generation/      # Generation orchestration
├── models/          # Pydantic data models
├── providers/       # TTS provider implementations
├── quality/         # Quality assurance
├── training/        # Training script generation
├── ui/              # CLI and wizards
└── utils/           # Helper utilities
```

## 🔧 Configuration

### YAML Configuration

```yaml
# wakegen.yaml
project:
  name: "my_wake_word"
  wake_word: "hey assistant"

generation:
  count: 1000
  output_dir: "./output"

providers:
  - type: edge_tts
    voices: [en-US-AriaNeural, en-US-GuyNeural]
    weight: 0.5
  - type: kokoro
    voices: [af_bella, am_adam]
    weight: 0.5

augmentation:
  enabled: true
  profiles: [kitchen, living_room, car]
  augmented_per_original: 3

export:
  format: openwakeword
  split_ratio: [0.8, 0.1, 0.1]
```

### Environment Variables

```bash
# .env
WAKEGEN_OUTPUT_DIR=./output
WAKEGEN_SAMPLE_RATE=16000
MINIMAX_API_KEY=your_api_key  # For MiniMax provider
```

## 📖 Documentation

- [Quick Start Guide](docs/quickstart.md)
- [Installation Guide](docs/installation.md)
- [Configuration Reference](docs/configuration.md)
- [Provider Documentation](docs/providers.md)
- [Augmentation Guide](docs/augmentation.md)
- [Training Guide](docs/training.md)
- [API Reference](docs/api_reference.md)

## 🧪 Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=wakegen
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/contributing.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Edge TTS](https://github.com/rany2/edge-tts) for the free Microsoft TTS API
- [Kokoro](https://github.com/hexgrad/kokoro) for the lightweight TTS model
- [OpenWakeWord](https://github.com/dscripka/openWakeWord) for wake word detection framework
- All open source TTS projects that make this tool possible