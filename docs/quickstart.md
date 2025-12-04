# Quick Start Guide

This guide will get you up and running with **WakeGen** in under 5 minutes. We will generate a small dataset for the wake word "Hey Computer".

## 1. The Interactive Wizard

The easiest way to start is using the interactive wizard. It guides you through every step.

Run this command:

```bash
wakegen generate --interactive
```

### Step-by-Step Walkthrough

1.  **Wake Word**: Enter `hey computer` (or your preferred phrase).
2.  **Sample Count**: Enter `10` for a quick test.
3.  **Provider**: Select `1` for Edge TTS (easiest to start with).
4.  **Output Directory**: Press Enter to accept the default (`./output`).
5.  **Augmentation**: Press Enter to enable it (adds realism).

Once you confirm, WakeGen will start generating audio files!

## 2. Using Command Line Arguments

If you prefer to skip the wizard, you can provide all options in one command:

```bash
wakegen generate --text "hey computer" --count 10 --preset quick_test
```

*   `--text`: The phrase to generate.
*   `--count`: How many samples to create.
*   `--preset`: Uses a pre-defined configuration (optimized for speed).

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

1.  **Augment it**: Add more noise and variations (see [Augmentation Guide](augmentation.md)).
2.  **Train a Model**: Use these files to train OpenWakeWord (see [Training Guide](training.md)).