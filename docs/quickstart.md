# Quick Start Guide

This guide will get you up and running with **WakeGen** in under 5 minutes. We will generate a small dataset for the wake word "Hey Computer".

**Current Status**: Phase 1A - Basic generation is working. Interactive wizard and augmentation features are planned but not yet fully implemented.

## 1. The Interactive Wizard

The interactive wizard is available but currently has limited functionality.

Run this command:

```bash
wakegen generate --interactive
```

**Current Status**: The wizard will guide you through basic options but currently defaults to Edge TTS provider and basic generation.

### Step-by-Step Walkthrough

1.  **Wake Word**: Enter `hey computer` (or your preferred phrase).
2.  **Sample Count**: Enter `10` for a quick test.
3.  **Provider**: Currently defaults to Edge TTS (only available provider).
4.  **Output Directory**: Press Enter to accept the default (`./output`).
5.  **Augmentation**: Augmentation is not yet implemented.

Once you confirm, WakeGen will start generating audio files!

## 2. Using Command Line Arguments

The simplest way to generate audio is using direct command line arguments:

```bash
wakegen generate --text "hey computer" --count 10
```

*   `--text`: The phrase to generate.
*   `--count`: How many samples to create.

**Note**: Preset functionality is available but limited to basic configuration:

```bash
wakegen generate --text "hey computer" --count 10 --preset quick_test
```

## 3. Check Your Results

Navigate to the `output` directory:

```bash
ls output/
```

You should see files like:
*   `hey_computer_1.wav`
*   `hey_computer_2.wav`
*   ...

Play a few to make sure they sound correct!

## Next Steps

Now that you have a basic dataset, you can:

1.  **Wait for Augmentation**: Augmentation features are planned for future phases (see [Augmentation Guide](augmentation.md) for planned features).
2.  **Explore Training**: Training functionality is planned but not yet implemented (see [Training Guide](training.md) for future plans).

## Current Limitations

As of Phase 1A:
- Only Edge TTS provider is available
- Augmentation pipeline is not yet implemented
- Export, validation, and training features are planned but not working
- Interactive wizard has limited functionality

These features will be added in future phases as development progresses.