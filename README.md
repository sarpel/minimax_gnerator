# Wake Word Dataset Generator (Phase 1A)

A tool to generate synthetic wake word datasets using various Text-to-Speech (TTS) providers. This project is designed to help create training data for wake word detection models (like "Hey Katya", "Jarvis", etc.).

**Current Status**: Phase 1A - Basic generation with Edge TTS is working. Other features (augmentation, export, validation, training) are planned but not yet fully implemented.

## 🚀 Quick Start

### 1. Installation

First, make sure you have Python 3.10 or higher installed.

```bash
# Clone the repository (if you haven't already)
# git clone https://github.com/yourusername/wakegen.git
# cd wakegen

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the project in editable mode
pip install -e .
```

### 2. Configuration

Copy the example environment file and configure your settings:

```bash
# On Windows
copy .env.example .env
# On macOS/Linux
cp .env.example .env
```

For Phase 1A, we are using **Edge TTS**, which is free and doesn't require an API key. You can skip editing `.env` if you just want to test the basic functionality.

### 3. Usage

Generate 10 samples of "Hey Katya" using the default settings:

```bash
python -m wakegen generate --text "hey katya" --count 10
```

This will create 10 WAV files in the `output/` directory.

**Note**: The interactive wizard (`--interactive` flag) is available but currently defaults to Edge TTS provider.

## 🏗️ Project Structure

```
wakegen/
├── config/         # Configuration settings and presets
├── core/           # Core types, exceptions, and protocols
├── models/         # Data models (Pydantic)
├── providers/      # TTS Provider implementations (Edge TTS, etc.)
├── ui/             # Command Line Interface (CLI)
├── utils/          # Helper functions (audio processing, logging)
└── main.py         # Entry point
```

## 📋 Current Implementation Status

### ✅ Working Features
- **Edge TTS Provider**: Free, high-quality neural TTS
- **Basic Generation**: Create WAV files with custom wake words
- **Configuration System**: Presets and environment variables
- **Audio Resampling**: Convert to target sample rates
- **CLI Interface**: Basic generation commands

### 🚧 Planned Features (Not Yet Implemented)
- **Augmentation**: Adding noise, reverb, and degradation effects
- **Export**: Converting datasets for training frameworks
- **Validation**: Quality assurance checks
- **Training Scripts**: Automatic training script generation
- **Additional Providers**: Piper TTS, Minimax, Coqui XTTS

## ️ Development

To run the tests:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## 📝 License

MIT License