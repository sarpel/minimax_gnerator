# TTS Providers

WakeGen supports multiple Text-to-Speech (TTS) engines to generate diverse audio samples.

**Current Status**: Phase 1A - Only Edge TTS is currently implemented and working. Other providers are planned for future phases.

## 1. Edge TTS (Currently Available) ✅

*   **Type**: Free, Online
*   **Quality**: High (Neural)
*   **Speed**: Fast (depends on internet)
*   **Best for**: Quick tests, high-quality English/Turkish samples.

**Configuration**:
No API key required. Just select it in the wizard or config.

**Current Status**: Fully implemented and working.

## 2. Piper TTS (Planned) 🚧

*   **Type**: Free, Offline, Local
*   **Quality**: Good (Fast Neural)
*   **Speed**: Very Fast (runs on CPU/Raspberry Pi)
*   **Best for**: Generating massive datasets offline, running on Raspberry Pi.

**Configuration**:
Requires downloading voice models. WakeGen handles this automatically for supported voices.

**Status**: Planned for future phase. Not yet implemented.

## 3. Minimax (Commercial, Planned) 🚧

*   **Type**: Paid, Online
*   **Quality**: Ultra-High (Clone capability)
*   **Speed**: Moderate
*   **Best for**: Professional-grade datasets, specific voice cloning.

**Configuration**:
Requires an API Key.
1.  Get a key from the Minimax dashboard.
2.  Add it to your `.env` file:
    ```bash
    MINIMAX_API_KEY=your_key
    MINIMAX_GROUP_ID=your_group_id
    ```

**Status**: Planned for future phase. Not yet implemented.

## 4. Coqui XTTS (Experimental, Planned) 🚧

*   **Type**: Free/Open Source, Local (GPU recommended)
*   **Quality**: High (Voice Cloning)
*   **Speed**: Slow on CPU
*   **Best for**: High-quality voice cloning if you have a GPU.

**Note**: This provider is resource-intensive and not recommended for Raspberry Pi.

**Status**: Planned for future phase. Not yet implemented.

## Current Implementation

As of Phase 1A, only **Edge TTS** is fully implemented and available for use. The other providers are part of the architectural design but have not been implemented yet.

When using the CLI, you can only generate audio using Edge TTS:

```bash
# This works (Edge TTS)
wakegen generate --text "hello world" --count 5

# Provider selection is not yet available in CLI
# Future: --provider edge_tts (default), --provider piper, etc.
```

The provider system is designed to be extensible, and additional providers will be added in future phases as indicated in the roadmap.