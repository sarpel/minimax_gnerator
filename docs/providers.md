# TTS Providers

WakeGen supports multiple Text-to-Speech (TTS) engines to generate diverse audio samples. This guide covers all available providers, their features, and how to use them.

## Provider Overview

| Provider | Type | GPU Required | Quality | Best For |
|----------|------|--------------|---------|----------|
| Edge TTS | Free, Online | No | ⭐⭐⭐⭐ | Quick tests, general use |
| Piper | Free, Offline | No | ⭐⭐⭐ | Raspberry Pi, offline generation |
| Kokoro | Free, Offline | No | ⭐⭐⭐⭐ | Fast CPU inference, lightweight |
| Mimic 3 | Free, Offline | No | ⭐⭐⭐ | Privacy-focused, embedded devices |
| Coqui XTTS | Free, Offline | Recommended | ⭐⭐⭐⭐⭐ | Voice cloning |
| F5-TTS | Free, Offline | Recommended | ⭐⭐⭐⭐⭐ | Voice cloning, high quality |
| StyleTTS 2 | Free, Offline | Recommended | ⭐⭐⭐⭐⭐ | Expressive speech |
| Orpheus | Free, Offline | For large models | ⭐⭐⭐-⭐⭐⭐⭐⭐ | Scalable (150M-3B) |
| Bark | Free, Offline | Recommended | ⭐⭐⭐⭐ | Emotions, non-speech sounds |
| ChatTTS | Free, Offline | Recommended | ⭐⭐⭐⭐ | Conversational speech |
| MiniMax | Commercial | No | ⭐⭐⭐⭐⭐ | Professional production |

---

## Free Online Providers

### Edge TTS ✅

Microsoft Edge TTS provides high-quality neural voices for free.

- **Quality**: High (Neural)
- **Speed**: Fast (depends on internet)
- **Languages**: 40+ languages
- **Voices**: 300+ voices

**Installation**: 
```bash
pip install edge-tts
```

**Usage**:
```bash
wakegen generate --text "hey katya" --provider edge_tts --voice en-US-AriaNeural
```

**Configuration** (wakegen.yaml):
```yaml
providers:
  - type: edge_tts
    voices: 
      - en-US-AriaNeural
      - en-US-GuyNeural
```

---

## Free Offline Providers (CPU-Friendly)

### Kokoro TTS ✅

Lightweight 82M parameter model that runs faster than real-time on CPU.

- **Quality**: High
- **Speed**: Very Fast on CPU
- **Size**: 82MB model
- **Languages**: English (British & American)

**Installation**:
```bash
pip install kokoro-onnx
```

**Available Voices**:
- American Female: af_bella, af_nicole, af_sarah, af_sky
- American Male: am_adam, am_michael
- British Female: bf_emma, bf_isabella
- British Male: bm_george, bm_lewis

**Usage**:
```bash
wakegen generate --text "hey katya" --provider kokoro --voice af_bella
```

### Piper TTS ✅

Ultra-fast TTS designed for Raspberry Pi and embedded devices.

- **Quality**: Good
- **Speed**: Extremely Fast
- **Size**: ~15-50MB per voice
- **Languages**: 20+ languages

**Installation**:
```bash
pip install piper-tts
```

**Usage**:
```bash
wakegen generate --text "hey katya" --provider piper --voice en_US-lessac-medium
```

### Mimic 3 ✅

Mycroft's privacy-friendly TTS engine.

- **Quality**: Good
- **Speed**: Fast
- **Privacy**: Fully offline, no data collection
- **Languages**: 10+ languages

**Installation**:
```bash
pip install mycroft-mimic3-tts
```

**Usage**:
```bash
wakegen generate --text "hey katya" --provider mimic3 --voice en_US/ljspeech_low
```

---

## High-Quality Offline Providers (GPU Recommended)

### Coqui XTTS ✅

State-of-the-art voice cloning with zero-shot capability.

- **Quality**: Excellent
- **Speed**: Slow on CPU, Fast on GPU
- **Feature**: Voice cloning from reference audio
- **Languages**: 17+ languages

**Installation**:
```bash
pip install TTS
```

**Voice Cloning**:
```bash
# Provide reference audio for voice cloning
wakegen generate --text "hey katya" --provider coqui_xtts --voice /path/to/reference.wav
```

**Note**: XTTS requires reference audio for voice cloning. It doesn't have preset voices.

### F5-TTS ✅

High-quality TTS with excellent voice cloning.

- **Quality**: Excellent
- **Speed**: Moderate (GPU recommended)
- **Feature**: Voice cloning, natural prosody
- **VRAM**: ~1.5GB

**Installation**:
```bash
pip install f5-tts
```

**Usage**:
```bash
# With preset voice
wakegen generate --text "hey katya" --provider f5_tts --voice default

# With reference audio for cloning
wakegen generate --text "hey katya" --provider f5_tts --voice /path/to/reference.wav
```

### StyleTTS 2 ✅

State-of-the-art expressive speech synthesis with style control.

- **Quality**: State-of-the-art
- **Speed**: Moderate (GPU recommended)
- **Feature**: Style/emotion control
- **Styles**: neutral, happy, sad, angry, surprised, fearful

**Installation**:
```bash
pip install styletts2
```

**Usage**:
```bash
# With style control (format: style:voice)
wakegen generate --text "hey katya" --provider styletts2 --voice happy:default

# Available styles: neutral, happy, sad, angry, surprised, fearful
```

### Orpheus TTS ✅

Scalable TTS with models from 150M to 3B parameters.

- **Quality**: Good to Excellent (depends on model size)
- **Speed**: Varies by model
- **Models**: 150M, 400M, 1B, 3B parameters
- **Auto-select**: Chooses model based on available GPU memory

**Installation**:
```bash
pip install orpheus-tts
```

**Model Sizes**:
| Model | Quality | GPU Memory | CPU Capable |
|-------|---------|------------|-------------|
| 150M | Good | <2GB | Yes |
| 400M | Better | 2-4GB | Yes (slow) |
| 1B | Great | 4-8GB | No |
| 3B | Excellent | 8GB+ | No |

**Usage**:
```bash
# Auto-select best model for hardware
wakegen generate --text "hey katya" --provider orpheus --voice default

# Specify model size
wakegen generate --text "hey katya" --provider orpheus --voice default --model-size 400m
```

### Bark ✅

Expressive TTS from Suno with support for emotions and non-speech sounds.

- **Quality**: High
- **Speed**: Slow (GPU strongly recommended)
- **Feature**: Laughter, sighs, music, emotions
- **Languages**: 13+ languages

**Installation**:
```bash
pip install git+https://github.com/suno-ai/bark.git
```

**Speaker Presets**:
- English: v2/en_speaker_0 through v2/en_speaker_9
- Also available: zh, de, es, fr, hi, it, ja, ko, pl, pt, ru, tr

**Expression Markers**:
- `[laughs]` - Add laughter
- `[sighs]` - Add sighing
- `[clears throat]` - Clear throat sound
- `[gasps]` - Gasping sound
- `...` - Hesitation

**Usage**:
```bash
wakegen generate --text "hey katya" --provider bark --voice v2/en_speaker_3

# With expression
wakegen generate --text "[laughs] hey katya" --provider bark --voice v2/en_speaker_3
```

### ChatTTS ✅

Conversational speech synthesis optimized for dialogue.

- **Quality**: High
- **Speed**: Moderate (GPU recommended)
- **Feature**: Natural conversational prosody
- **Languages**: Chinese, English

**Installation**:
```bash
pip install chattts
```

**Speaker Presets**:
- conversational_1, conversational_2, conversational_3
- friendly_1, friendly_2
- professional_1, professional_2
- casual_1, casual_2

**Usage**:
```bash
wakegen generate --text "hey katya" --provider chattts --voice conversational_1

# With control parameters
wakegen generate --text "hey katya" --provider chattts --voice friendly_1 --speed 1.2
```

---

## Commercial Providers

### MiniMax ✅

Professional-grade commercial TTS with excellent quality.

- **Quality**: Excellent
- **Speed**: Fast
- **Feature**: Commercial-grade reliability
- **Pricing**: Pay per character

**Setup**:
1. Get API key from [MiniMax Console](https://api.minimax.chat/)
2. Set environment variables:
```bash
export MINIMAX_API_KEY=your_api_key
export MINIMAX_GROUP_ID=your_group_id
```

**Usage**:
```bash
wakegen generate --text "hey katya" --provider minimax --voice voice_id
```

---

## Provider Auto-Discovery

WakeGen can automatically discover which providers are available on your system:

```bash
wakegen list-providers --available
```

This checks:
- Required Python packages
- API keys (for commercial providers)
- GPU availability (for GPU-required providers)

## Multi-Provider Generation

Generate samples using multiple providers for maximum diversity:

```yaml
# wakegen.yaml
generation:
  count: 1000

providers:
  - type: edge_tts
    weight: 0.3
    voices: [en-US-AriaNeural, en-US-GuyNeural]
  - type: kokoro
    weight: 0.3
    voices: [af_bella, am_adam]
  - type: piper
    weight: 0.2
    voices: [en_US-lessac-medium]
  - type: bark
    weight: 0.2
    voices: [v2/en_speaker_3, v2/en_speaker_5]
```

```bash
wakegen batch --config wakegen.yaml
```

---

## Choosing the Right Provider

### For Quick Testing
- **Edge TTS**: Fast, free, no setup required

### For Offline/Privacy
- **Kokoro**: Best quality/speed on CPU
- **Piper**: Fastest, works on Raspberry Pi
- **Mimic 3**: Privacy-focused

### For Voice Cloning
- **Coqui XTTS**: Zero-shot cloning
- **F5-TTS**: High-quality cloning

### For Expressive Speech
- **StyleTTS 2**: Style/emotion control
- **Bark**: Non-speech sounds, emotions

### For Conversations
- **ChatTTS**: Natural dialogue prosody

### For Maximum Quality
- **StyleTTS 2**: State-of-the-art
- **Orpheus 3B**: Highest quality (requires GPU)

### For Production
- **MiniMax**: Commercial reliability
- **Edge TTS**: Free, high availability

---

## Troubleshooting

### Provider Not Found
```bash
pip install <package-name>  # Install required package
wakegen list-providers --available  # Verify installation
```

### GPU Not Detected
```bash
# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode
wakegen generate --text "test" --provider styletts2 --cpu-only
```

### Model Download Issues
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/torch/

# Retry generation
wakegen generate --text "test" --provider kokoro
```
