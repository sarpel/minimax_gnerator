# Configuration Guide

WakeGen is highly configurable. You can control almost every aspect of the generation process using **Presets** and **Environment Variables**.

**Current Status**: Basic configuration is working. Advanced features are planned for future phases.

## Presets

Presets are YAML files that define a collection of settings. They are located in `wakegen/config/presets/`.

### Using a Preset

To use a preset, pass its name (without `.yaml`) to the CLI:

```bash
wakegen generate --preset quick_test ...
```

**Current Status**: Only the `quick_test` preset is available and working.

### Creating a Custom Preset

You can create your own preset file (e.g., `my_config.yaml`) and place it in the presets folder. Here is an example structure:

```yaml
# my_config.yaml

# Generation Settings
output_dir: "./my_dataset"
sample_rate: 16000  # Standard for most ML models
audio_format: "wav"

# Augmentation Settings (Planned for future phases)
augmentation:
  enabled: true
  noise_prob: 0.5   # 50% chance to add background noise
  reverb_prob: 0.3  # 30% chance to add reverb
```

**Note**: Augmentation settings are defined but not yet implemented.

## Environment Variables

For sensitive information (like API keys) or system-wide settings, use environment variables. You can put these in a `.env` file in the project root.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `WAKEGEN_OUTPUT_DIR` | Default directory for saving files | `./output` |
| `WAKEGEN_LOG_LEVEL` | Logging verbosity (DEBUG, INFO, WARNING) | `INFO` |
| `MINIMAX_API_KEY` | API Key for Minimax provider (if used) | *None* |
| `MINIMAX_GROUP_ID` | Group ID for Minimax provider | *None* |

**Current Status**: Basic environment variables are working. Minimax-related variables are defined but not yet used since Minimax provider is not implemented.

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

## Current Implementation Status

### ✅ Working Features
- Basic preset loading (`quick_test` preset)
- Environment variable support
- CLI argument overrides
- Output directory configuration
- Sample rate and audio format settings

### 🚧 Planned Features
- Advanced augmentation configuration
- Multiple provider configurations
- Custom voice selection
- Advanced noise and reverb profiles

The configuration system is designed to be extensible and will support more advanced features as they are implemented in future phases.