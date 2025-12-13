# ADR-003: Lazy Imports for Optional TTS Providers

**Status:** Accepted  
**Date:** 2024-12-13

## Context

WakeGen supports 11+ TTS providers, each with different dependencies:

| Provider | Dependencies | Size |
|----------|-------------|------|
| Edge TTS | `edge-tts` | ~1MB |
| Piper | `piper-tts`, `onnxruntime` | ~50MB |
| Bark | `bark`, `torch` | ~2GB |
| F5-TTS | `f5-tts`, `torch` | ~1GB |

Installing ALL providers is:
- Time consuming (~10+ minutes)
- Storage intensive (~5GB+)
- Conflict-prone (numpy version conflicts)

## Decision

We use **lazy imports** inside provider methods instead of module-level imports.

```python
# PATTERN: Lazy import inside initialize()
class BarkProvider(BaseProvider):
    async def initialize(self) -> None:
        # Only import when actually used
        from bark import generate_audio, preload_models
        preload_models()
```

## Rationale

### Why Lazy Imports?

1. **Graceful Degradation**: Unavailable providers show as "not installed" instead of crashing the entire application
2. **Fast Startup**: Application starts instantly without loading heavy ML models
3. **User Choice**: Users install only the providers they need
4. **Conflict Isolation**: Import errors are contained to specific providers

### Trade-offs

| Benefit | Cost |
|---------|------|
| Fast startup | First-use delay for model loading |
| No crashes for missing deps | Slightly more complex code |
| Memory efficient | Marginally slower imports |

### Alternative Considered: Separate Packages

We could have split each provider into its own PyPI package. Rejected because:
- Increases maintenance burden
- Harder for users to discover providers
- Version synchronization issues

## Consequences

### Positive
- Works out-of-box with just Edge TTS (cloud, no deps)
- Users see available/unavailable status in UI
- No import-time crashes

### Negative
- PEP 8 prefers module-level imports (style violation)
- Static type checkers may not catch import errors
- First generation with a provider is slower

## Implementation Notes

All providers follow this pattern:
```python
async def generate(self, text, voice_id, output_path):
    # Lazy import
    from heavy_library import synthesize
    
    # Use the library
    await synthesize(text, output_path)
```
