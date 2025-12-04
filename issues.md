# Code Review: Wake Word Dataset Generator (wakegen)

## Executive Summary

This is a well-structured Python project for generating synthetic wake word datasets using various TTS providers. The architecture is clean and extensible, with good separation of concerns. However, there are several issues ranging from critical bugs to code quality improvements.

---

## 🔴 Critical Issues

### 1. **Missing Import in `orchestrator.py`**
```python
# Line ~340 in wakegen/generation/orchestrator.py
timestamp = int(time.time() * 1000)  # 'time' is not imported!
```
**Fix:** Add `import time` at the top of the file.

### 2. **Missing `AsyncIterator` Import in `orchestrator.py`**
```python
async def _generate_parameters_with_checkpoint(...) -> AsyncIterator[GenerationParameters]:
```
**Fix:** Add `from typing import AsyncIterator` to imports.

### 3. **Incorrect `ProviderRegistry` Usage in `orchestrator.py`**
```python
from wakegen.providers.registry import ProviderRegistry  # This class doesn't exist!
registry = ProviderRegistry()
```
The registry module uses functions (`get_provider`, `register_provider`), not a class.

### 4. **Missing Model Attributes in `GenerationConfig`**
`orchestrator.py` references many attributes that don't exist in `GenerationConfig`:
- `checkpoint_db_path`, `checkpoint_cleanup_interval`, `max_checkpoints`
- `progress_refresh_rate`, `show_task_details`, `console_width`
- `max_concurrent_tasks`, `retry_attempts`, `task_timeout_seconds`, `rate_limits`
- `default_voice_ids`, `speed_range`, `pitch_range`, `use_commercial_providers`

---

## 🟠 High Priority Issues

### 5. **Piper TTS Implementation Issues**
```python
# wakegen/providers/opensource/piper.py
from piper_tts import PiperVoice, synthesize  # These may not exist in piper-tts package
```
The `piper-tts` package API differs from what's implemented. The actual API uses `PiperVoice.load()` and different method signatures.

### 6. **Coqui XTTS Preset Voices Don't Exist**
```python
speaker_mapping = {
    "tr_female_1": "tr_female",  # These speakers don't exist in XTTS
    "tr_male_1": "tr_male",
    ...
}
```
XTTS requires reference audio for voice cloning; these preset speakers are fictional.

### 7. **MiniMax API Response Model May Be Incorrect**
The `MiniMaxTTSResponse` model assumes a specific API response structure that may not match the actual MiniMax API. The `success` and `audio_data` fields should be verified against actual API documentation.

### 8. **Race Condition in Rate Limiter**
```python
# minimax.py rate limiter
current_time = asyncio.get_event_loop().time()  # Deprecated in Python 3.10+
```
Use `asyncio.get_running_loop().time()` or `time.time()` instead.

---

## 🟡 Medium Priority Issues

### 9. **Typo in Project Directory Name**
The project is in `minimax_gnerator` (missing 'e' in generator).

### 10. **Inconsistent Return Type Annotations**
```python
# wakegen/core/protocols.py
async def list_voices(self) -> List[Any]:  # Returns List[Voice] in implementations
```
Should be `List[Voice]` for consistency.

### 11. **Missing `__init__.py` Content**
Several `__init__.py` files only have comments without proper exports:
- `wakegen/models/__init__.py`
- `wakegen/core/__init__.py`

### 12. **Inconsistent Error Handling in Providers**
```python
# edge_tts.py
except Exception as e:
    raise ProviderError(...) from e

# piper.py  
except Exception as synth_error:  # Different variable name
    raise ProviderError(...) from synth_error
```

### 13. **Unused Imports**
```python
# wakegen/providers/commercial/minimax.py
from typing import TYPE_CHECKING  # Never used
```

### 14. **Magic Numbers Without Constants**
```python
# minimax.py
self.max_requests_per_minute = 60  # Should be a constant
timeout=30.0  # Should be configurable
```

### 15. **Test Files in Root Directory**
`test_new_providers.py` and `test_quality_assurance.py` should be in the `tests/` directory.

---

## 🟢 Code Quality Improvements

### 16. **Add Type Hints to All Functions**
Several functions lack complete type hints:
```python
# wakegen/config/settings.py
def load_preset(preset_name: str) -> Dict[str, Any]:  # Good
def get_generation_config(preset_name: str = None):    # Missing return type
```

### 17. **Use `pathlib.Path` Consistently**
Mix of `os.path` and `pathlib.Path`:
```python
# Some files use:
from pathlib import Path
file_path = Path(file_path)

# Others use:
import os
file_path = os.path.join(dir, filename)
```

### 18. **Add Docstrings to All Public Methods**
Some methods lack docstrings:
```python
# wakegen/providers/base.py
def __init__(self, config: ProviderConfig):
    """Initialize the provider with configuration."""  # Good!
    
# But cli/commands.py has many undocumented internal functions
```

### 19. **Consider Using `asyncio.to_thread` for Blocking Operations**
```python
# In Piper provider - subprocess calls block the event loop
result = subprocess.run(...)  # Should be run in thread pool
```

### 20. **Add Validation to Pydantic Models**
```python
# wakegen/models/generation.py
class GenerationParameters(BaseModel):
    speed: float = Field(1.0, ge=0.1, le=3.0)  # Good!
    prosody: str = Field("normal")  # Should validate allowed values
```

---

## 📋 Architecture Recommendations

### 21. **Dependency Injection for Providers**
Consider using a proper DI pattern instead of the global registry:
```python
# Current approach
from wakegen.providers import *  # Side effects on import

# Better approach
class ProviderFactory:
    def __init__(self, config: ProviderConfig):
        self._providers = {}
        
    def register(self, provider_type: ProviderType, provider_class: Type[BaseProvider]):
        self._providers[provider_type] = provider_class
```

### 22. **Add Structured Logging**
```python
# Current
logger.info(f"Successfully generated MiniMax TTS audio: {output_path}")

# Better - structured logging
logger.info("TTS audio generated", extra={
    "provider": "minimax",
    "output_path": output_path,
    "voice_id": voice_id
})
```

### 23. **Add Configuration Validation**
Create a startup validation that checks all configurations:
```python
async def validate_system_config():
    """Validate all system configurations at startup."""
    errors = []
    
    # Check output directory is writable
    # Check required dependencies are installed
    # Validate API keys if commercial providers enabled
    
    if errors:
        raise ConfigError(f"Configuration errors: {errors}")
```

---

## 🧪 Testing Gaps

### 24. **Missing Unit Tests**
Current test coverage is minimal:
- Only 1 test file in `tests/test_providers/`
- No tests for augmentation pipeline
- No tests for quality scoring
- No tests for CLI commands

### 25. **Integration Tests Needed**
Add integration tests for:
- Full generation pipeline
- Augmentation with real audio
- Export functionality

### 26. **Mock External Services**
Tests should mock external services:
```python
# Example
@pytest.fixture
def mock_edge_tts():
    with patch('edge_tts.Communicate') as mock:
        mock.return_value.save = AsyncMock()
        yield mock
```

---

## 📁 Project Structure Issues

### 27. **Missing `py.typed` Marker**
For proper type checking support, add an empty `py.typed` file to `wakegen/`.

### 28. **Missing `CHANGELOG.md`**
Add a changelog to track versions.

### 29. **Missing `LICENSE` File**
The pyproject.toml mentions MIT license but there's no LICENSE file.

---

## 🔧 Recommended Immediate Fixes

Here's a prioritized list of fixes:

1. **Fix the `orchestrator.py` import errors** (critical - code won't run)
2. **Fix `ProviderRegistry` class that doesn't exist** (critical)
3. **Add missing attributes to `GenerationConfig`** (critical)
4. **Move test files to proper location** (minor cleanup)
5. **Add proper exports to `__init__.py` files** (improves usability)
6. **Verify Piper and Coqui XTTS implementations** against actual APIs
