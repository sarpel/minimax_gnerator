# Implementation Plan

- [x] 1. Fix augmentation pipeline batch processing bug





  - [x] 1.1 Initialize failed_files variable in batch_augment function


    - Add `failed_files = []` before the processing loop in `wakegen/augmentation/pipeline.py`
    - Ensure failed files are appended with error information
    - _Requirements: 1.1, 1.2, 1.3_
  - [x] 1.2 Write property test for batch augmentation count accuracy






    - **Property 1: Batch augmentation returns accurate counts**
    - **Validates: Requirements 1.3**





- [-] 2. Fix BarkProvider constructor and Voice attribute bugs




  - [x] 2.1 Update BarkProvider constructor to accept and pass config



    - Modify `__init__` signature to accept optional `config: ProviderConfig = None`
    - Call `super().__init__(config or ProviderConfig())` 

    - Add necessary imports for ProviderConfig
    - _Requirements: 2.1, 2.2_


  - [ ] 2.2 Fix Voice object creation in BarkProvider.list_voices

    - Change `voice_id=preset` to `id=preset`
    - Add `provider=ProviderType.BARK` to Voice constructor
    - _Requirements: 2.3_
  - [-] 2.3 Write property test for BarkProvider voice listing



    - **Property 2: Provider voice listing returns valid Voice objects**
    - **Validates: Requirements 2.3**

- [ ] 3. Fix ChatTTSProvider constructor and Voice attribute bugs

  - [ ] 3.1 Update ChatTTSProvider constructor to accept and pass config

    - Modify `__init__` signature to accept optional `config: ProviderConfig = None`
    - Call `super().__init__(config or ProviderConfig())`
    - Add necessary imports for ProviderConfig
    - _Requirements: 3.1, 3.2_
  - [ ] 3.2 Fix Voice object creation in ChatTTSProvider.list_voices

    - Change `voice_id=preset_name` to `id=preset_name`
    - Add `provider=ProviderType.CHATTTS` to Voice constructor
    - _Requirements: 3.3_
  - [ ] 3.3 Write property test for ChatTTSProvider voice listing


    - **Property 2: Provider voice listing returns valid Voice objects**
    - **Validates: Requirements 3.3**

- [ ] 4. Checkpoint - Ensure provider fixes work

  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Fix generation orchestrator bugs

  - [ ] 5.1 Add missing ConfigError import to orchestrator

    - Add `ConfigError` to the import from `wakegen.core.exceptions`
    - _Requirements: 4.1, 4.2_
  - [ ] 5.2 Update Pydantic serialization in orchestrator

    - Replace `variation_params.dict()` with `variation_params.model_dump()`
    - _Requirements: 6.2_

- [ ] 6. Fix CheckpointManager bugs

  - [ ] 6.1 Remove duplicate __aenter__ and __aexit__ methods

    - Remove the second definition of `__aenter__` (around line 450)
    - Remove the second definition of `__aexit__` (around line 455)
    - Keep the first definitions with proper docstrings
    - _Requirements: 5.1, 5.2_
  - [ ] 6.2 Update Pydantic serialization in checkpoint manager

    - Replace `parameters.dict()` with `parameters.model_dump()`
    - Replace `result.dict()` with `result.model_dump()`
    - _Requirements: 5.3, 6.1_

- [ ] 7. Final Checkpoint - Ensure all tests pass

  - Ensure all tests pass, ask the user if questions arise.
