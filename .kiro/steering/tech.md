# WakeGen - Technical Stack

## Language & Runtime
- **Python 3.10+** (required minimum)
- **Async/await** patterns throughout (asyncio-based)

## Build System
- **setuptools** (pyproject.toml based)
- Entry point: `wakegen = "wakegen.main:cli"`

## Core Dependencies
| Library | Purpose |
|---------|---------|
| pydantic / pydantic-settings | Data validation, config management |
| click | CLI framework |
| rich | Terminal UI (tables, progress bars, panels) |
| httpx | Async HTTP client |
| soundfile / librosa | Audio I/O and processing |
| scipy | Signal processing |
| pyroomacoustics | Room impulse response simulation |
| torch / torchaudio | Deep learning, audio processing |
| onnxruntime | CPU model inference |
| tenacity | Retry logic |
| aiosqlite | Async checkpoint storage |

## Optional Dependencies
- **Web UI**: `fastapi`, `uvicorn`, `websockets`, `aiofiles` (install with `pip install wakegen[web]`)
- **Dev tools**: `pytest`, `pytest-asyncio`, `black`, `isort`, `mypy`

## TTS Providers (separate installs)
Some providers require manual installation:
- `pip install git+https://github.com/suno-ai/bark.git` (Bark)
- `pip install mycroft-mimic3-tts` (Mimic3, Linux only)
- `pip install ChatTTS`, `f5-tts`, `styletts2`, `kokoro-onnx`, `orpheus-tts`

## Common Commands

```bash
# Installation
pip install -e .              # Standard install
pip install -e ".[dev]"       # With dev tools
pip install -e ".[web]"       # With web UI

# Running
wakegen generate --text "hey assistant" --count 100
wakegen wizard                # Interactive mode
wakegen list-providers -v     # Show available providers
wakegen list-voices --provider edge_tts

# Testing
pytest                        # Run all tests
pytest --cov=wakegen          # With coverage
pytest tests/test_api_config.py  # Specific test file

# Type checking
mypy wakegen

# Formatting
black wakegen tests
isort wakegen tests
```

## Configuration
- **Environment**: `.env` file for API keys and settings
- **YAML configs**: Project-level configuration in `wakegen.yaml`
- **Presets**: `wakegen/config/presets/*.yaml`

## Audio Standards
- Default sample rate: 16000 Hz (speech recognition standard)
- Default format: WAV
- Resampling handled automatically via librosa
