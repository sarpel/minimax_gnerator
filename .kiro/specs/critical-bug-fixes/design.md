# Design Document: Critical Bug Fixes

## Overview

This design document addresses critical and high-priority bugs identified in the WakeGen codebase analysis. The fixes target runtime errors, provider instantiation failures, and Pydantic v2 compatibility issues that prevent core functionality from working correctly.

The bugs fall into three categories:
1. **Undefined Variables** - Variables used before initialization causing NameError
2. **Constructor Signature Mismatches** - Provider classes not properly calling parent constructors
3. **API Compatibility** - Using deprecated Pydantic v1 methods in a v2 codebase

## Architecture

The fixes maintain the existing architecture while correcting implementation errors:

```
┌─────────────────────────────────────────────────────────────────┐
│                        WakeGen Core                              │
├─────────────────────────────────────────────────────────────────┤
│  augmentation/pipeline.py    │  Fix: Initialize failed_files    │
├─────────────────────────────────────────────────────────────────┤
│  providers/opensource/       │  Fix: Constructor signatures     │
│    bark.py                   │       Voice model attributes     │
│    chattts.py                │                                  │
├─────────────────────────────────────────────────────────────────┤
│  generation/                 │  Fix: Missing import             │
│    orchestrator.py           │       Pydantic v2 methods        │
│    checkpoint.py             │       Duplicate methods          │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Augmentation Pipeline (pipeline.py)

**Current Issue:** `failed_files` variable used in `batch_augment()` without initialization.

**Fix:** Initialize `failed_files = []` before the processing loop at line ~290.

```python
# Before
results = []
for i, input_path in enumerate(input_paths):

# After  
results = []
failed_files = []  # Track failed files for reporting
for i, input_path in enumerate(input_paths):
```

### 2. BarkProvider (bark.py)

**Current Issues:**
- `__init__` calls `super().__init__()` without passing `config` parameter
- `list_voices()` creates Voice objects with `voice_id` instead of `id`

**Fix Constructor:**
```python
# Before
def __init__(self, use_gpu: bool = True, ...):
    super().__init__()

# After
def __init__(self, config: Optional[ProviderConfig] = None, use_gpu: bool = True, ...):
    super().__init__(config or ProviderConfig())
```

**Fix Voice Creation:**
```python
# Before
Voice(voice_id=preset, name=..., gender=..., language=...)

# After
Voice(id=preset, name=..., gender=..., language=..., provider=ProviderType.BARK)
```

### 3. ChatTTSProvider (chattts.py)

**Current Issues:** Same as BarkProvider - constructor and Voice attribute issues.

**Fix:** Apply identical patterns as BarkProvider fixes.

### 4. Generation Orchestrator (orchestrator.py)

**Current Issues:**
- `ConfigError` used but not imported
- Uses deprecated `.dict()` method

**Fix Imports:**
```python
from wakegen.core.exceptions import GenerationError, ConfigError
```

**Fix Pydantic Method:**
```python
# Before
"variation_params": variation_params.dict()

# After
"variation_params": variation_params.model_dump()
```

### 5. CheckpointManager (checkpoint.py)

**Current Issues:**
- Duplicate `__aenter__` and `__aexit__` method definitions
- Uses deprecated `.dict()` method for serialization

**Fix Duplicates:** Remove the second set of `__aenter__`/`__aexit__` definitions (lines ~450-460).

**Fix Pydantic Method:**
```python
# Before
parameters_json = json.dumps(parameters.dict()) if parameters else None
result_json = json.dumps(result.dict()) if result else None

# After
parameters_json = json.dumps(parameters.model_dump()) if parameters else None
result_json = json.dumps(result.model_dump()) if result else None
```

## Data Models

No changes to data models are required. The `Voice` model already uses `id` as the field name - the bug is in the provider code that incorrectly uses `voice_id`.

**Voice Model (unchanged):**
```python
class Voice(BaseModel):
    id: str = Field(..., description="The unique identifier for the voice")
    name: str
    gender: Gender
    language: str
    provider: ProviderType
    supports_cloning: bool = False
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Based on the prework analysis, the following correctness properties have been identified:

### Property 1: Batch augmentation returns accurate counts
*For any* list of input files (valid and invalid), when batch_augment completes, the sum of successful results and failed files should equal the total number of input files.
**Validates: Requirements 1.3**

### Property 2: Provider voice listing returns valid Voice objects
*For any* provider (BarkProvider or ChatTTSProvider), when list_voices is called, all returned Voice objects should have a non-empty `id` attribute that matches the expected voice identifier format.
**Validates: Requirements 2.3, 3.3**

## Error Handling

The fixes improve error handling in the following ways:

1. **Augmentation Pipeline:** Failed files are now tracked and reported instead of causing crashes
2. **Provider Instantiation:** Providers now properly initialize with default configs, preventing TypeError
3. **Orchestrator:** ConfigError is now properly imported and can be caught/raised
4. **Checkpoint Manager:** Clean async context manager implementation ensures proper resource cleanup

## Testing Strategy

### Unit Tests

Unit tests will verify:
- BarkProvider instantiation without config parameter
- ChatTTSProvider instantiation without config parameter
- Voice objects have correct `id` attribute
- ConfigError is importable from orchestrator module
- CheckpointManager has single context manager methods
- No deprecation warnings from Pydantic serialization

### Property-Based Tests

Property-based testing will be implemented using **pytest** with **hypothesis** library:

1. **Property 1 Test:** Generate random lists of file paths (some valid, some invalid), run batch_augment, verify count invariant
2. **Property 2 Test:** For each provider, call list_voices and verify all Voice objects have valid `id` attributes

Each property-based test will:
- Run a minimum of 100 iterations
- Be tagged with the format: `**Feature: critical-bug-fixes, Property {number}: {property_text}**`
- Reference the specific correctness property from this design document
