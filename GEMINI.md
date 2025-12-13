# WakeGen - Project Context & Instructions

## Project Overview
**WakeGen** is a comprehensive Python-based tool designed to generate high-quality synthetic wake word datasets. It orchestrates multiple Text-to-Speech (TTS) providers (both free and commercial) to create diverse audio samples for training wake word detection models.

**Key Features:**
- **Multi-Provider Architecture:** Supports 11+ TTS engines (Edge TTS, Kokoro, Piper, Bark, etc.).
- **Augmentation Pipeline:** Simulates real-world conditions (noise, room acoustics, telephony).
- **Export Formats:** Generates datasets compatible with OpenWakeWord, PyTorch, etc.
- **CLI & Web UI:** Provides both command-line and web interfaces.

## Architecture

The project follows a modular, plugin-based architecture:

*   **`wakegen/`**: Root package.
    *   **`core/`**: Defines core types, protocols, and exceptions.
    *   **`providers/`**: Contains the TTS provider implementations. All providers inherit from `BaseProvider` and must implement `generate`, `list_voices`, etc.
    *   **`config/`**: Handles configuration via YAML files and environment variables (`.env`). Uses `pydantic-settings`.
    *   **`augmentation/`**: Audio processing pipeline using `librosa` and `pyroomacoustics`.
    *   **`ui/cli/`**: Command-line interface implemented with `click`.
    *   **`models/`**: Data models defined using `Pydantic` for validation.

## Development Workflow

### 1. Setup
*   **Windows:** Run `install.bat`
*   **Linux/macOS:** Run `./install.sh`
*   **Manual:** `pip install -e ".[dev,gpu]"` (depending on hardware)

### 2. Running the Application
The application is installed as a CLI tool named `wakegen`.
*   **Generate:** `wakegen generate --text "hey assistant" --count 100`
*   **Wizard:** `wakegen wizard` (Interactive mode)
*   **List Voices:** `wakegen list-voices --provider edge_tts`

### 3. Web UI
*   **Start:** Run `start.bat` (Windows) to launch the web server
*   **URL:** http://127.0.0.1:8005
*   **Port:** The web UI runs on **port 8005** (configured in `start.bat`)
*   Pages: Dashboard, Providers, Generate, Augmentation, Export, Quality, Config, System

### 4. Testing & Quality
*   **Test Runner:** `pytest` (Configured in `pyproject.toml`)
    *   Run all tests: `pytest`
    *   With coverage: `pytest --cov=wakegen`
*   **Type Checking:** `mypy` (Strict mode enabled)
*   **Formatting:** `black`, `isort`
*   **Linting:** `ruff`

## Coding Conventions

### Educational Protocol (MANDATORY)
*   **ELI5 Comments:** Complex logic MUST be explained simply ("Explain Like I'm 5").
*   **Why > What:** Comments should explain *why* a specific approach or security check is used (e.g., "SEC-006 Fix: Validate preset name to prevent path traversal").
*   **Syntax Explanation:** When using advanced Python features, briefly explain how they work.

### Architecture & Safety
*   **Providers:** Must implement the `BaseProvider` abstract base class.
*   **Async/Await:** Heavy use of `asyncio` for I/O-bound operations (TTS generation).
*   **Security:**
    *   **Path Traversal:** strictly validate all file paths and user inputs (as seen in `load_preset`).
    *   **Input Validation:** Use Pydantic models for all configuration and data structures.
*   **Typing:** All code must be fully typed. No `Any` unless absolutely necessary and documented.

### File Structure
*   **`__init__.py`**: Keep minimal, mostly for exposing exports.
*   **Settings**: All configuration logic resides in `wakegen/config/`.
*   **Entry Point**: `wakegen/main.py` is the CLI entry point.
