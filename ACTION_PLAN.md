# Wake Word Generator (wakegen) - Action Plan

> **Generated:** December 4, 2025  
> **Status:** Pending Review  
> **Total Issues:** 29 identified

---

## Overview

This action plan addresses issues discovered during code review, organized by priority. Each section includes specific tasks with estimated effort and dependencies.

---

## 🔴 Phase 1: Critical Fixes (Blocking - Code Won't Run)

**Estimated Effort:** 2-3 hours  
**Priority:** MUST FIX before any testing

### 1.1 Fix `orchestrator.py` Import Errors

**File:** `wakegen/generation/orchestrator.py`

- [ ] Add missing `import time` at top of file
- [ ] Add `AsyncIterator` to typing imports: `from typing import AsyncIterator`
- [ ] Remove non-existent `ProviderRegistry` class usage
- [ ] Replace with function-based registry calls:
  ```python
  # Replace:
  from wakegen.providers.registry import ProviderRegistry
  registry = ProviderRegistry()
  registry.get_commercial_provider()
  
  # With:
  from wakegen.providers.registry import get_provider
  from wakegen.core.types import ProviderType
  get_provider(ProviderType.MINIMAX, config)
  ```

### 1.2 Add Missing Attributes to `GenerationConfig`

**File:** `wakegen/models/config.py`

Add the following fields to `GenerationConfig` class:

- [ ] Checkpoint settings:
  - `checkpoint_db_path: str = "checkpoints.db"`
  - `checkpoint_cleanup_interval: int = 3600`
  - `max_checkpoints: int = 10`

- [ ] Progress settings:
  - `progress_refresh_rate: float = 0.1`
  - `show_task_details: bool = True`
  - `console_width: int = 80`

- [ ] Batch processing settings:
  - `max_concurrent_tasks: int = 5`
  - `retry_attempts: int = 3`
  - `task_timeout_seconds: int = 300`
  - `rate_limits: Dict[str, Tuple[int, int]] = {"commercial": (10, 60), "free": (5, 60)}`

- [ ] Voice settings:
  - `default_voice_ids: Optional[List[str]] = None`
  - `speed_range: Optional[Tuple[float, float]] = (0.8, 1.2)`
  - `pitch_range: Optional[Tuple[float, float]] = (0.9, 1.1)`
  - `use_commercial_providers: bool = False`

### 1.3 Fix Provider Registry Integration

**File:** `wakegen/generation/orchestrator.py`

- [ ] Refactor `_get_primary_provider()` method to use existing registry functions
- [ ] Refactor `generate_with_fallback()` method similarly
- [ ] Add proper provider fallback logic using `get_provider()` function

---

## 🟠 Phase 2: High Priority Fixes (Functionality Issues)

**Estimated Effort:** 4-6 hours  
**Priority:** Required for providers to work correctly

### 2.1 Fix Piper TTS Provider Implementation

**File:** `wakegen/providers/opensource/piper.py`

- [ ] Research actual `piper-tts` package API (verify imports exist)
- [ ] Update imports to match actual package structure
- [ ] Fix `_ensure_piper_available()` method
- [ ] Fix `generate()` method to use correct API calls
- [ ] Test with actual Piper installation
- [ ] Consider alternative: Use subprocess to call piper CLI directly

**Alternative approach if piper-tts API doesn't match:**
```python
# Use subprocess-based approach
async def generate(self, text: str, voice_id: str, output_path: str) -> None:
    cmd = ["piper", "--model", voice_id, "--output_file", output_path]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await process.communicate(input=text.encode())
```

### 2.2 Fix Coqui XTTS Provider Implementation

**File:** `wakegen/providers/opensource/coqui_xtts.py`

- [ ] Remove fictional preset voices (`tr_female_1`, etc.)
- [ ] Update `list_voices()` to return only voice cloning option
- [ ] Fix `_generate_with_preset_voice()` - XTTS doesn't support presets
- [ ] Update documentation to clarify voice cloning requirement
- [ ] Add validation that reference audio must be provided

### 2.3 Verify MiniMax API Implementation

**File:** `wakegen/providers/commercial/minimax.py`

- [ ] Verify API endpoint URL against official documentation
- [ ] Verify request/response model structure
- [ ] Test with actual API key (if available)
- [ ] Fix deprecated `asyncio.get_event_loop()` usage:
  ```python
  # Replace:
  current_time = asyncio.get_event_loop().time()
  
  # With:
  current_time = time.time()
  ```

### 2.4 Add Missing Generation Components

**Files to verify exist and are complete:**

- [ ] `wakegen/generation/variation_engine.py` - Verify `VariationEngine` class exists
- [ ] `wakegen/generation/batch_processor.py` - Verify `BatchProcessor` class exists
- [ ] `wakegen/generation/checkpoint.py` - Verify `CheckpointManager` class exists
- [ ] `wakegen/generation/progress.py` - Verify `ProgressTracker` class exists
- [ ] `wakegen/generation/rate_limiter.py` - Verify `RateLimiter` class exists

---

## 🟡 Phase 3: Medium Priority Improvements

**Estimated Effort:** 3-4 hours  
**Priority:** Code quality and maintainability

### 3.1 Project Structure Cleanup

- [ ] Rename project directory from `minimax_gnerator` to `minimax_generator` (optional - may affect git history)
- [ ] Move `test_new_providers.py` to `tests/test_providers/`
- [ ] Move `test_quality_assurance.py` to `tests/test_quality/`
- [ ] Add `LICENSE` file (MIT as specified in pyproject.toml)
- [ ] Add `CHANGELOG.md` file
- [ ] Add `py.typed` marker file to `wakegen/`

### 3.2 Fix Type Annotations

**File:** `wakegen/core/protocols.py`

- [ ] Change `list_voices()` return type from `List[Any]` to `List[Voice]`
- [ ] Add import: `from wakegen.models.audio import Voice`

**File:** `wakegen/config/settings.py`

- [ ] Add return type to `get_generation_config()`: `-> GenerationConfig`
- [ ] Add return type to `get_provider_config()`: `-> ProviderConfig`

### 3.3 Improve `__init__.py` Exports

**File:** `wakegen/models/__init__.py`

```python
from wakegen.models.audio import Voice, AudioSample, ProviderCapabilities
from wakegen.models.config import ProviderConfig, GenerationConfig
from wakegen.models.generation import (
    GenerationRequest,
    GenerationResponse,
    GenerationParameters,
    GenerationResult
)

__all__ = [
    "Voice",
    "AudioSample", 
    "ProviderCapabilities",
    "ProviderConfig",
    "GenerationConfig",
    "GenerationRequest",
    "GenerationResponse",
    "GenerationParameters",
    "GenerationResult"
]
```

**File:** `wakegen/core/__init__.py`

```python
from wakegen.core.types import (
    ProviderType,
    AudioFormat,
    QualityLevel,
    Gender,
    AugmentationType,
    EnvironmentProfile
)
from wakegen.core.exceptions import (
    WakeGenError,
    ProviderError,
    ConfigError,
    AudioError,
    GenerationError,
    AugmentationError
)
from wakegen.core.protocols import TTSProvider

__all__ = [
    "ProviderType",
    "AudioFormat",
    "QualityLevel",
    "Gender",
    "AugmentationType",
    "EnvironmentProfile",
    "WakeGenError",
    "ProviderError",
    "ConfigError",
    "AudioError",
    "GenerationError",
    "AugmentationError",
    "TTSProvider"
]
```

### 3.4 Remove Unused Imports

- [ ] `wakegen/providers/commercial/minimax.py`: Remove `TYPE_CHECKING` import
- [ ] Run `isort` and `autoflake` across project

### 3.5 Add Constants for Magic Numbers

**File:** `wakegen/providers/commercial/minimax.py`

```python
# Add at module level
DEFAULT_MAX_REQUESTS_PER_MINUTE = 60
DEFAULT_API_TIMEOUT_SECONDS = 30.0
RATE_LIMIT_WINDOW_SECONDS = 60
```

---

## 🟢 Phase 4: Testing Improvements

**Estimated Effort:** 6-8 hours  
**Priority:** Ensures reliability

### 4.1 Add Unit Tests for Core Components

- [ ] `tests/test_core/test_types.py` - Test enum values
- [ ] `tests/test_core/test_exceptions.py` - Test exception hierarchy
- [ ] `tests/test_models/test_config.py` - Test Pydantic models
- [ ] `tests/test_models/test_audio.py` - Test audio models
- [ ] `tests/test_utils/test_audio.py` - Test audio utilities

### 4.2 Add Provider Tests with Mocking

**File:** `tests/test_providers/test_edge_tts.py`

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_edge_tts_generate():
    with patch('edge_tts.Communicate') as mock_communicate:
        mock_instance = AsyncMock()
        mock_communicate.return_value = mock_instance
        
        provider = EdgeTTSProvider(ProviderConfig())
        await provider.generate("test", "en-US-AriaNeural", "/tmp/test.wav")
        
        mock_communicate.assert_called_once_with("test", "en-US-AriaNeural")
        mock_instance.save.assert_called_once()
```

### 4.3 Add Integration Tests

- [ ] `tests/integration/test_generation_pipeline.py`
- [ ] `tests/integration/test_augmentation_pipeline.py`
- [ ] `tests/integration/test_cli.py`

### 4.4 Add Test Configuration

**File:** `tests/conftest.py` (expand existing)

```python
@pytest.fixture
def mock_provider_config():
    return ProviderConfig(
        minimax_api_key="test_key",
        minimax_group_id="test_group"
    )

@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a sample WAV file for testing."""
    import numpy as np
    import soundfile as sf
    
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    
    file_path = tmp_path / "test.wav"
    sf.write(file_path, audio, sample_rate)
    return file_path
```

---

## 🔧 Phase 5: Architecture Improvements (Optional)

**Estimated Effort:** 8-12 hours  
**Priority:** Long-term maintainability

### 5.1 Implement Proper Dependency Injection

- [ ] Create `ProviderFactory` class
- [ ] Remove global registry side effects
- [ ] Update CLI to use factory pattern

### 5.2 Add Structured Logging

- [ ] Create logging configuration module
- [ ] Add context to all log messages
- [ ] Consider adding log correlation IDs for async operations

### 5.3 Add Configuration Validation at Startup

- [ ] Create `validate_system_config()` function
- [ ] Check writable directories
- [ ] Validate optional dependencies
- [ ] Add to CLI startup

### 5.4 Use `pathlib.Path` Consistently

- [ ] Audit all file path handling
- [ ] Replace `os.path` with `pathlib.Path` where appropriate
- [ ] Ensure cross-platform compatibility

---

## 📋 Execution Order

### Week 1: Critical & High Priority
1. ✅ Phase 1.1 - Fix orchestrator imports
2. ✅ Phase 1.2 - Add missing config attributes  
3. ✅ Phase 1.3 - Fix registry integration
4. ✅ Phase 2.1 - Fix Piper provider
5. ✅ Phase 2.2 - Fix Coqui XTTS provider
6. ✅ Phase 2.3 - Verify MiniMax API

### Week 2: Medium Priority & Testing
7. ✅ Phase 3.1 - Project structure cleanup
8. ✅ Phase 3.2 - Fix type annotations
9. ✅ Phase 3.3 - Improve exports
10. ✅ Phase 4.1 - Add unit tests
11. ✅ Phase 4.2 - Add provider tests

### Week 3: Polish & Architecture (Optional)
12. ⬜ Phase 4.3 - Integration tests
13. ⬜ Phase 5.x - Architecture improvements

---

## 📝 Notes

### Dependencies to Verify
Before implementing provider fixes, verify these packages work as expected:
- `piper-tts>=1.2.0` - Check actual API
- `TTS>=0.22.0` - Check XTTS v2 API
- MiniMax API documentation

### Testing Environment
- Python 3.10+ required
- GPU recommended for Coqui XTTS
- Internet required for Edge TTS and MiniMax

### Breaking Changes
- Adding required fields to `GenerationConfig` may break existing presets
- Removing preset voices from Coqui XTTS changes the API

---

## ✅ Approval Checklist

Before proceeding with implementation:

- [ ] Review and approve Phase 1 (Critical Fixes)
- [ ] Review and approve Phase 2 (High Priority)
- [ ] Review and approve Phase 3 (Medium Priority)  
- [ ] Review and approve Phase 4 (Testing)
- [ ] Review and approve Phase 5 (Optional Architecture)
- [ ] Confirm execution order
- [ ] Identify any additional requirements

---

**Please review this action plan and let me know:**
1. Which phases to proceed with
2. Any items to add, remove, or reprioritize
3. Any concerns about the proposed changes
