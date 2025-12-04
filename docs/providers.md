# TTS Providers

WakeGen supports multiple Text-to-Speech (TTS) engines to generate diverse audio samples.

## 1. Edge TTS (Default)

*   **Type**: Free, Online
*   **Quality**: High (Neural)
*   **Speed**: Fast (depends on internet)
*   **Best for**: Quick tests, high-quality English/Turkish samples.

**Configuration:**
No API key required. Just select it in the wizard or config.

## 2. Piper TTS

*   **Type**: Free, Offline, Local
*   **Quality**: Good (Fast Neural)
*   **Speed**: Very Fast (runs on CPU/Raspberry Pi)
*   **Best for**: Generating massive datasets offline, running on Raspberry Pi.

**Configuration:**
Requires downloading voice models. WakeGen handles this automatically for supported voices.

## 3. Minimax (Commercial)

*   **Type**: Paid, Online
*   **Quality**: Ultra-High (Clone capability)
*   **Speed**: Moderate
*   **Best for**: Professional-grade datasets, specific voice cloning.

**Configuration:**
Requires an API Key.
1.  Get a key from the Minimax dashboard.
2.  Add it to your `.env` file:
    ```bash
    MINIMAX_API_KEY=your_key
    MINIMAX_GROUP_ID=your_group_id
    ```

## 4. Coqui XTTS (Experimental)

*   **Type**: Free/Open Source, Local (GPU recommended)
*   **Quality**: High (Voice Cloning)
*   **Speed**: Slow on CPU
*   **Best for**: High-quality voice cloning if you have a GPU.

**Note**: This provider is resource-intensive and not recommended for Raspberry Pi.