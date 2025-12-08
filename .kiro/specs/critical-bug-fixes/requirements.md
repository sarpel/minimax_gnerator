# Requirements Document

## Introduction

This specification addresses critical and high-priority bugs identified in the WakeGen codebase analysis. WakeGen is a wake word dataset generator that uses multiple TTS providers to create synthetic audio samples. The analysis revealed 8 critical issues and several major issues that cause runtime failures, prevent provider instantiation, and break core functionality. This spec focuses on fixing these bugs to restore system stability and correctness.

## Glossary

- **WakeGen**: The wake word dataset generator application
- **TTS Provider**: A Text-to-Speech service implementation (e.g., Bark, ChatTTS, Edge TTS)
- **BaseProvider**: The abstract base class that all TTS providers must extend
- **ProviderConfig**: Pydantic model containing provider configuration settings
- **Voice**: Pydantic model representing a TTS voice with id, name, gender, and language
- **CheckpointManager**: Async context manager for saving/resuming generation state
- **Augmentation Pipeline**: System for applying audio effects to generated samples
- **Pydantic v2**: The data validation library used for models (version 2.x)

## Requirements

### Requirement 1

**User Story:** As a developer, I want the augmentation pipeline batch processing to work without crashing, so that I can process multiple audio files for augmentation.

#### Acceptance Criteria

1. WHEN the batch_augment function processes multiple files THEN the WakeGen system SHALL initialize a failed_files list before the processing loop
2. WHEN a file fails augmentation THEN the WakeGen system SHALL append the failure information to the failed_files list
3. WHEN batch processing completes THEN the WakeGen system SHALL return accurate counts of successful and failed files

### Requirement 2

**User Story:** As a developer, I want to instantiate the BarkProvider without errors, so that I can use Bark TTS for voice generation.

#### Acceptance Criteria

1. WHEN BarkProvider is instantiated THEN the BarkProvider SHALL pass a valid config parameter to BaseProvider.__init__
2. WHEN BarkProvider is instantiated without explicit config THEN the BarkProvider SHALL create a default ProviderConfig
3. WHEN BarkProvider.list_voices is called THEN the BarkProvider SHALL create Voice objects using the 'id' attribute instead of 'voice_id'

### Requirement 3

**User Story:** As a developer, I want to instantiate the ChatTTSProvider without errors, so that I can use ChatTTS for voice generation.

#### Acceptance Criteria

1. WHEN ChatTTSProvider is instantiated THEN the ChatTTSProvider SHALL pass a valid config parameter to BaseProvider.__init__
2. WHEN ChatTTSProvider is instantiated without explicit config THEN the ChatTTSProvider SHALL create a default ProviderConfig
3. WHEN ChatTTSProvider.list_voices is called THEN the ChatTTSProvider SHALL create Voice objects using the 'id' attribute instead of 'voice_id'

### Requirement 4

**User Story:** As a developer, I want the generation orchestrator to handle configuration errors properly, so that error handling works correctly during generation.

#### Acceptance Criteria

1. WHEN the orchestrator module is loaded THEN the orchestrator SHALL have ConfigError imported from wakegen.core.exceptions
2. WHEN a configuration error occurs during generation THEN the orchestrator SHALL raise ConfigError with a descriptive message

### Requirement 5

**User Story:** As a developer, I want the CheckpointManager to have clean async context manager implementation, so that checkpoint save/restore works reliably.

#### Acceptance Criteria

1. WHEN CheckpointManager is defined THEN the CheckpointManager SHALL have exactly one __aenter__ method definition
2. WHEN CheckpointManager is defined THEN the CheckpointManager SHALL have exactly one __aexit__ method definition
3. WHEN CheckpointManager serializes task state THEN the CheckpointManager SHALL use Pydantic v2 model_dump method instead of deprecated dict method

### Requirement 6

**User Story:** As a developer, I want all Pydantic model serialization to use v2-compatible methods, so that the codebase is forward-compatible and deprecation-warning-free.

#### Acceptance Criteria

1. WHEN serializing Pydantic models in checkpoint.py THEN the WakeGen system SHALL use model_dump() instead of dict()
2. WHEN serializing Pydantic models in orchestrator.py THEN the WakeGen system SHALL use model_dump() instead of dict()
3. WHEN any Pydantic model needs JSON serialization THEN the WakeGen system SHALL use model_dump(mode='json') for JSON-compatible output
