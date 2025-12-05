# Changelog

All notable changes to WakeGen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-XX

### Added

#### TTS Providers (11 total)
- **Edge TTS**: Free Microsoft neural TTS with 400+ voices
- **Kokoro**: Lightweight 82M parameter model, fast CPU inference
- **Piper**: CPU-friendly local TTS with Turkish support
- **Mimic3**: Mycroft's privacy-friendly offline TTS
- **Bark**: Expressive TTS from Suno with emotion control
- **ChatTTS**: Conversational speech optimized TTS
- **StyleTTS2**: State-of-the-art neural TTS
- **Orpheus**: Scalable TTS with multiple model sizes (150M-3B)
- **Coqui XTTS**: Voice cloning with reference audio
- **F5-TTS**: High-quality voice synthesis with cloning
- **MiniMax**: Commercial API with premium voices

#### Augmentation Pipeline
- Room impulse response simulation (pyroomacoustics)
- Background noise injection (traffic, restaurant, office, home)
- Telephony simulation (landline, mobile, VoIP codecs)
- Distance simulation with attenuation modeling
- Device-specific presets (ESP32, Raspberry Pi, smart speakers)
- Environment profiles (kitchen, living room, car, outdoor)
- Dynamic range compression
- Time-domain effects (pitch shift, speed variation)

#### Export Formats
- **OpenWakeWord**: Native format for wake word training
- **Mycroft Precise**: Mycroft wake word detector format
- **Picovoice**: Porcupine-compatible format
- **TensorFlow**: tf.data pipeline with TFRecords
- **PyTorch**: DataLoader with custom Dataset class
- **HuggingFace**: datasets library compatible format

#### Quality Assurance
- ASR verification using Whisper
- SNR (Signal-to-Noise Ratio) calculation
- Automatic quality scoring
- Clipping detection
- Duration validation
- Speech clarity metrics

#### CLI Features
- `wakegen generate`: Generate samples with any provider
- `wakegen list-voices`: List available voices per provider
- `wakegen list-providers`: Show available/unavailable providers
- `wakegen wizard`: Interactive setup wizard
- `wakegen batch`: Batch processing from file
- `wakegen export`: Export to training formats
- `wakegen augment`: Apply augmentation to existing samples
- `wakegen cache`: Cache management (stats, clear, list)

#### Configuration
- YAML-based configuration files
- Environment variable support
- Provider weight distribution
- Augmentation profiles
- Device presets

#### Performance
- Async parallel generation
- Provider-aware rate limiting
- Generation caching (LRU eviction)
- GPU detection and optimization
- Multi-GPU support
- Memory-efficient batching

#### Documentation
- Comprehensive quick start guide
- Detailed installation instructions
- Provider-specific documentation
- Augmentation guide with examples
- Training integration guide
- Full API reference

### Changed
- Upgraded from Phase 1A to full 1.0.0 release
- Improved error handling across all providers
- Enhanced progress dashboard with Rich

### Fixed
- Voice import bug in Bark and ChatTTS providers
- Orchestrator import errors
- GenerationConfig missing attributes
- Provider registry function-based approach

## [0.1.0] - 2024-XX-XX

### Added
- Initial Phase 1A release
- Edge TTS provider implementation
- Basic generation CLI
- Simple configuration system
- Audio resampling utilities

---

For more details, see the [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for the full development roadmap.
