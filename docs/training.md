# Training Guide

**Current Status**: Training functionality is planned but not yet implemented in Phase 1A.

Once you have generated and augmented your dataset, the next step will be to train a wake word model. WakeGen is designed to work seamlessly with **OpenWakeWord** in future phases.

## Current Status

As of Phase 1A, training-related features are not yet implemented:

- ✅ **Dataset Generation**: Working (basic generation with Edge TTS)
- 🚧 **Augmentation**: Planned but not implemented
- 🚧 **Export**: Planned but not implemented
- 🚧 **Training Scripts**: Planned but not implemented
- 🚧 **Validation**: Planned but not implemented

## Planned Features (Future Phases)

### 1. Exporting the Dataset

**Planned**: Export functionality will organize data into formats that training frameworks understand.

```bash
# This command is planned but not yet working
wakegen export --data-dir ./output --format openwakeword --output-path ./training_data
```

### 2. Generating a Training Script

**Planned**: WakeGen will simplify OpenWakeWord training by generating scripts.

```bash
# This command is planned but not yet working
wakegen train-script --model-type openwakeword --output-script train.sh
```

### 3. Running the Training

**Planned**: Integration with OpenWakeWord library.

### 4. Testing the Model

**Planned**: Validation tools for trained models.

## What You Can Do Now

While training features are not yet available, you can:

1. **Generate Basic Datasets**: Use the working generation functionality to create WAV files
2. **Prepare for Future Features**: Familiarize yourself with the planned workflow
3. **Explore OpenWakeWord**: Research the target training framework independently

## Development Roadmap

Training functionality is part of the planned roadmap and will be implemented in future phases. The current focus is on completing the core generation and augmentation features first.

For now, you can use the generated WAV files as a starting point and manually prepare them for training using other tools.