# Configuration Guide

WakeGen is highly configurable. You can control almost every aspect of the generation process using **Presets** and **Environment Variables**.

## Presets

Presets are YAML files that define a collection of settings. They are located in `wakegen/config/presets/`.

### Using a Preset

To use a preset, pass its name (without `.yaml`) to the CLI:

```bash
wakegen generate --preset quick_test ...
```

### Creating a Custom Preset

You can create your own preset file (e.g., `my_config.yaml`) and place it in the presets folder. Here is an example structure:

```yaml
# my_config.yaml

# Generation Settings
output_dir: "./my_dataset"
sample_rate: 16000  # Standard for most ML models
audio_format: "wav"

# Augmentation Settings
augmentation:
  enabled: true
  noise_prob: 0.5   # 50% chance to add background noise
  reverb_prob: 0.3  # 30% chance to add reverb
```

## Environment Variables

For sensitive information (like API keys) or system-wide settings, use environment variables. You can put these in a `.env` file in the project root.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `WAKEGEN_OUTPUT_DIR` | Default directory for saving files | `./output` |
| `WAKEGEN_LOG_LEVEL` | Logging verbosity (DEBUG, INFO, WARNING) | `INFO` |
| `MINIMAX_API_KEY` | API Key for Minimax provider (if used) | *None* |
| `MINIMAX_GROUP_ID` | Group ID for Minimax provider | *None* |

### Example .env File

```bash
WAKEGEN_LOG_LEVEL=DEBUG
MINIMAX_API_KEY=your_secret_key_here
```

## Priority Order

Settings are applied in this order (highest priority first):

1.  **CLI Arguments** (e.g., `--output-dir`)
2.  **Preset File** (e.g., `quick_test.yaml`)
3.  **Environment Variables** (e.g., `WAKEGEN_OUTPUT_DIR`)
4.  **Default Values** (Hardcoded in the code)