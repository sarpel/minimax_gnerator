# Contributing to WakeGen

We welcome contributions! Whether it's fixing a bug, adding a new TTS provider, or improving documentation, your help is appreciated.

## Development Setup

1.  **Clone the repo**:
    ```bash
    git clone https://github.com/yourusername/wakegen.git
    cd wakegen
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/Mac
    # .venv\Scripts\activate   # Windows
    ```

3.  **Install dev dependencies**:
    ```bash
    pip install -e .[dev]
    ```

## Running Tests

We use `pytest` for testing. Please ensure all tests pass before submitting a PR.

```bash
pytest
```

## Code Style

*   **Type Hints**: We use strict type hints (`mypy`).
*   **Formatting**: We use `black` and `isort`.
*   **Comments**: We follow the "ELI5" principle. Explain *why* you are doing something, not just *what*.

## Adding a New Provider

To add a new TTS provider:

1.  Create a new file in `wakegen/providers/`.
2.  Implement the `TTSProvider` protocol.
3.  Register it in `wakegen/providers/registry.py`.
4.  Add unit tests in `tests/test_providers/`.

## Submitting a Pull Request

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/my-feature`).
3.  Commit your changes.
4.  Push to your fork.
5.  Open a Pull Request on GitHub.