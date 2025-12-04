# Augmentation Guide

Augmentation is the process of adding noise, reverb, and other effects to your clean audio samples. This is **critical** for training a robust wake word model that works in the real world.

## Why Augment?

If you train only on clean studio audio, your model will fail when:
*   The TV is on.
*   Someone is washing dishes.
*   You are far away from the microphone (reverb).

Augmentation simulates these conditions during training.

## Available Effects

WakeGen includes a powerful augmentation pipeline:

### 1. Background Noise
Mixes your wake word with environmental sounds.
*   **Types**: White noise, pink noise, household sounds (vacuum, dishes), nature sounds.
*   **Control**: You can set the Signal-to-Noise Ratio (SNR). Lower SNR = louder noise.

### 2. Room Impulse Response (RIR) / Reverb
Simulates how sound bounces around a room.
*   **Simulation**: Uses `pyroomacoustics` to simulate different room sizes (small bedroom, large living room).
*   **Real RIRs**: Can use recorded impulse responses for high accuracy.

### 3. Audio Degradation
Simulates cheap microphones or bad connections.
*   **Resampling**: Lowers quality (e.g., 8kHz).
*   **Clipping**: Simulates distortion from shouting.
*   **Bandpass Filter**: Simulates phone audio.

## Using Augmentation

### In the Wizard
Simply select "Yes" when asked about augmentation. This applies a balanced default profile.

### Via CLI
You can augment existing files:

```bash
wakegen augment --input-dir ./clean_samples --output-dir ./augmented_samples
```

### Custom Profiles
You can define custom augmentation profiles in your preset YAML file:

```yaml
augmentation:
  enabled: true
  pipeline:
    - type: "noise"
      prob: 0.8
      snr_range: [5, 15]
    - type: "reverb"
      prob: 0.5
      room_size: "medium"
```