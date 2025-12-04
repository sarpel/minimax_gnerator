# Augmentation Guide

**Current Status**: Augmentation features are planned but not yet implemented in Phase 1A.

Augmentation is the process of adding noise, reverb, and other effects to your clean audio samples. This will be **critical** for training a robust wake word model that works in the real world.

## Why Augment?

If you train only on clean studio audio, your model will fail when:
*   The TV is on.
*   Someone is washing dishes.
*   You are far away from the microphone (reverb).

Augmentation will simulate these conditions during training.

## Planned Features

WakeGen will include a powerful augmentation pipeline in future phases:

### 1. Background Noise (Planned)

Mixes your wake word with environmental sounds.
*   **Types**: White noise, pink noise, household sounds (vacuum, dishes), nature sounds.
*   **Control**: You will be able to set the Signal-to-Noise Ratio (SNR). Lower SNR = louder noise.

### 2. Room Impulse Response (RIR) / Reverb (Planned)

Simulates how sound bounces around a room.
*   **Simulation**: Will use `pyroomacoustics` to simulate different room sizes (small bedroom, large living room).
*   **Real RIRs**: Will support recorded impulse responses for high accuracy.

### 3. Audio Degradation (Planned)

Simulates cheap microphones or bad connections.
*   **Resampling**: Lowers quality (e.g., 8kHz).
*   **Clipping**: Simulates distortion from shouting.
*   **Bandpass Filter**: Simulates phone audio.

## Current Status

As of Phase 1A:
- ✅ **Basic Generation**: Working (clean audio files)
- 🚧 **Augmentation Pipeline**: Defined but not implemented
- 🚧 **CLI Integration**: Planned but not working
- 🚧 **Custom Profiles**: Planned but not available

## What You Can Do Now

While augmentation is not yet available:

1. **Generate Clean Samples**: Use the working generation to create base audio files
2. **Manual Augmentation**: Use external tools like Audacity or SoX to add effects manually
3. **Prepare for Future**: Familiarize yourself with the planned augmentation features

## Future Usage

When implemented, you will be able to use augmentation like this:

```bash
# This command is planned but not yet working
wakegen augment --input-dir ./clean_samples --output-dir ./augmented_samples
```

The augmentation system is part of the architectural design and will be implemented in future phases as development progresses.