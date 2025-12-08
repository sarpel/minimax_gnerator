# WakeGen - Product Overview

WakeGen is a comprehensive wake word dataset generator that creates high-quality synthetic audio samples for training wake word detection models (e.g., "Hey Assistant", "Jarvis", "Alexa").

## Core Purpose
Generate diverse, realistic audio datasets using multiple Text-to-Speech (TTS) providers, with advanced augmentation to simulate real-world acoustic conditions.

## Key Capabilities
- **Multi-Provider TTS**: 11 providers (Edge TTS, Kokoro, Piper, Bark, ChatTTS, StyleTTS2, Coqui XTTS, F5-TTS, MiniMax, etc.)
- **Audio Augmentation**: Room simulation, noise injection, telephony effects, microphone simulation, distance effects
- **Export Formats**: OpenWakeWord, Mycroft Precise, Picovoice, TensorFlow, PyTorch, HuggingFace
- **Quality Assurance**: ASR verification, SNR scoring, automatic quality filtering, deduplication
- **Device Presets**: ESP32, Raspberry Pi, smart speakers, conference systems

## Target Users
- ML engineers building wake word detection models
- Researchers working on speech recognition
- Developers creating voice-activated applications

## Interfaces
- **CLI**: Primary interface via `wakegen` command with subcommands (generate, wizard, batch, list-voices, etc.)
- **Web UI**: Optional FastAPI-based dashboard for visual control (install with `pip install wakegen[web]`)
- **YAML Config**: Reproducible generation via configuration files
