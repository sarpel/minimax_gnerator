# Wakegen Test Coverage Analysis & Strategy Report

**Analysis Date**: 2025-12-09
**Project**: wakegen (Wake Word Generation Framework)
**Source Files**: 101 Python modules
**Test Files**: 7 test modules
**Test-to-Source Ratio**: 6.9% (critically low)

---

## Executive Summary

**Overall Assessment**: INSUFFICIENT COVERAGE - CRITICAL GAPS IDENTIFIED

The wakegen project demonstrates a **minimal testing foundation** with significant coverage gaps across core functionality. Current testing focuses primarily on API endpoints and property-based validation for specific providers, leaving critical generation, augmentation, quality, and export modules untested.

**Key Findings**:
- **Coverage Estimate**: ~15-20% (based on file and functionality analysis)
- **Test Type Distribution**: Heavy API bias, minimal unit/integration testing
- **Critical Gaps**: Core orchestration, batch processing, quality validation, export formats
- **Test Quality**: Good property-based tests, but insufficient negative testing and edge cases
- **CI/CD Readiness**: Moderate - tests are async-aware but lack parallelization optimization

---

## 1. Test Coverage Assessment

### 1.1 Coverage Map by Module

#### CORE MODULES (wakegen/core/)
| Module | Test Coverage | Status | Priority |
|--------|--------------|--------|----------|
| protocols.py | 0% | ❌ UNTESTED | P1 |
| types.py | 0% | ❌ UNTESTED | P2 |
| exceptions.py | 0% | ❌ UNTESTED | P3 |

**Analysis**: Core abstractions lack validation. Protocol compliance, type safety, and exception hierarchies are untested.

#### PROVIDER MODULES (wakegen/providers/)
| Module | Test Coverage | Status | Priority |
|--------|--------------|--------|----------|
| registry.py | ~30% | ⚠️ PARTIAL | P1 |
| base.py | 0% | ❌ UNTESTED | P2 |
| commercial/minimax.py | 0% | ❌ UNTESTED | P1 |
| free/edge_tts.py | ~20% | ⚠️ PARTIAL | P2 |
| opensource/bark.py | ~60% | ✅ GOOD | - |
| opensource/chattts.py | ~60% | ✅ GOOD | - |
| opensource/coqui_xtts.py | 0% | ❌ UNTESTED | P2 |
| opensource/f5_tts.py | 0% | ❌ UNTESTED | P3 |
| opensource/kokoro.py | 0% | ❌ UNTESTED | P3 |
| opensource/mimic3.py | 0% | ❌ UNTESTED | P3 |
| opensource/orpheus.py | 0% | ❌ UNTESTED | P3 |
| opensource/piper.py | 0% | ❌ UNTESTED | P3 |
| opensource/styletts2.py | 0% | ❌ UNTESTED | P3 |

**Analysis**: Strong property-based tests for Bark and ChatTTS providers validate voice listing. Registry has basic API-level testing. Commercial Minimax provider (newly added) completely untested. Base provider abstractions lack validation.

**Tested Functionality**:
- ✅ Voice listing returns valid Voice objects (Bark, ChatTTS)
- ✅ Voice ID format validation (property-based)
- ✅ Provider enumeration counts
- ✅ Unique voice ID constraints

**Untested Functionality**:
- ❌ Actual audio generation workflows
- ❌ Provider initialization failures
- ❌ API key validation and error handling
- ❌ Rate limiting enforcement
- ❌ Provider fallback mechanisms
- ❌ Resource cleanup after generation

#### GENERATION MODULES (wakegen/generation/)
| Module | Test Coverage | Status | Priority |
|--------|--------------|--------|----------|
| orchestrator.py | ~15% | ⚠️ MINIMAL | P1 |
| batch_processor.py | 0% | ❌ UNTESTED | P1 |
| variation_engine.py | 0% | ❌ UNTESTED | P1 |
| checkpoint.py | 0% | ❌ UNTESTED | P1 |
| progress.py | 0% | ❌ UNTESTED | P2 |
| rate_limiter.py | 0% | ❌ UNTESTED | P1 |

**Analysis**: CRITICAL GAP - Core generation orchestration is untested. API endpoint tests provide minimal coverage of job lifecycle, but underlying batch processing, checkpointing, and rate limiting have zero validation.

**Tested Functionality**:
- ✅ Job creation via API endpoint
- ✅ Job status queries
- ✅ Job cancellation

**Untested Functionality**:
- ❌ Batch processing with concurrency limits
- ❌ Rate limiter token bucket algorithm
- ❌ Checkpoint save/restore workflows
- ❌ Progress tracking accuracy
- ❌ Variation parameter generation
- ❌ Error recovery and retry logic
- ❌ Task timeout handling
- ❌ Resource cleanup on job failure
- ❌ Concurrent job execution

#### AUGMENTATION MODULES (wakegen/augmentation/)
| Module | Test Coverage | Status | Priority |
|--------|--------------|--------|----------|
| pipeline.py | ~35% | ⚠️ PARTIAL | P1 |
| profiles.py | ~10% | ⚠️ MINIMAL | P2 |
| effects/degradation.py | 0% | ❌ UNTESTED | P2 |
| effects/dynamics.py | 0% | ❌ UNTESTED | P2 |
| effects/telephony.py | 0% | ❌ UNTESTED | P3 |
| effects/time_domain.py | 0% | ❌ UNTESTED | P2 |
| microphone/simulator.py | 0% | ❌ UNTESTED | P2 |
| noise/mixer.py | 0% | ❌ UNTESTED | P2 |
| noise/events.py | 0% | ❌ UNTESTED | P3 |
| noise/profiles.py | 0% | ❌ UNTESTED | P2 |
| room/simulator.py | 0% | ❌ UNTESTED | P2 |
| room/convolver.py | 0% | ❌ UNTESTED | P3 |
| device_presets.py | 0% | ❌ UNTESTED | P3 |

**Analysis**: Property-based tests validate batch count accuracy (successful + failed = total), but actual augmentation effects are completely untested. Critical gap for audio quality assurance.

**Tested Functionality**:
- ✅ Batch augmentation count property (success + failure = total input)
- ✅ Empty input handling
- ✅ All-valid and all-invalid file scenarios

**Untested Functionality**:
- ❌ Individual augmentation effects (noise, room, mic, dynamics)
- ❌ Audio quality after augmentation
- ❌ Profile-based augmentation selection
- ❌ Augmentation parameter ranges
- ❌ Effect chaining order
- ❌ Audio degradation accuracy
- ❌ Convolution correctness
- ❌ Device preset application
- ❌ SNR calculations
- ❌ Frequency response simulation

#### QUALITY MODULES (wakegen/quality/)
| Module | Test Coverage | Status | Priority |
|--------|--------------|--------|----------|
| validator.py | ~15% | ⚠️ MINIMAL | P1 |
| scorer.py | 0% | ❌ UNTESTED | P1 |
| statistics.py | 0% | ❌ UNTESTED | P2 |
| asr_check.py | 0% | ❌ UNTESTED | P1 |
| deduplication.py | 0% | ❌ UNTESTED | P2 |
| report_generator.py | 0% | ❌ UNTESTED | P3 |

**Analysis**: CRITICAL GAP - Quality validation is a key differentiator, yet almost completely untested. API endpoint provides basic coverage, but scoring algorithms, ASR verification, and deduplication are unvalidated.

**Tested Functionality**:
- ✅ Dataset validation endpoint returns health score

**Untested Functionality**:
- ❌ Sample validation logic (duration, SNR, amplitude)
- ❌ Quality scoring algorithms
- ❌ Statistical analysis accuracy
- ❌ ASR transcription verification
- ❌ Deduplication algorithms (hash-based, perceptual)
- ❌ Report generation formats
- ❌ Threshold enforcement
- ❌ Quality metrics calculation (ZCR, RMS, peak)

#### EXPORT MODULES (wakegen/export/)
| Module | Test Coverage | Status | Priority |
|--------|--------------|--------|----------|
| formats.py | ~10% | ⚠️ MINIMAL | P1 |
| openwakeword.py | 0% | ❌ UNTESTED | P1 |
| splitter.py | 0% | ❌ UNTESTED | P1 |
| manifest.py | 0% | ❌ UNTESTED | P2 |

**Analysis**: Export functionality has minimal API-level testing, but actual format conversion, splitting algorithms, and manifest generation are untested.

**Tested Functionality**:
- ✅ Export job creation via API
- ✅ Export job status queries

**Untested Functionality**:
- ❌ OpenWakeWord format conversion
- ❌ Train/val/test splitting stratification
- ❌ Manifest generation accuracy
- ❌ File copying vs symlinking
- ❌ Split ratio validation
- ❌ Dataset structure creation

#### WEB API MODULES (wakegen/web/)
| Module | Test Coverage | Status | Priority |
|--------|--------------|--------|----------|
| routers/providers.py | ~40% | ✅ GOOD | - |
| routers/generation.py | ~35% | ⚠️ PARTIAL | P2 |
| routers/config_router.py | ~45% | ✅ GOOD | - |
| routers/augmentation.py | ~15% | ⚠️ MINIMAL | P2 |
| routers/export.py | ~15% | ⚠️ MINIMAL | P2 |
| routers/quality.py | ~10% | ⚠️ MINIMAL | P2 |
| routers/system.py | 0% | ❌ UNTESTED | P3 |
| routers/audio.py | 0% | ❌ UNTESTED | P3 |
| app.py | 0% | ❌ UNTESTED | P2 |
| config.py | 0% | ❌ UNTESTED | P3 |
| websocket.py | 0% | ❌ UNTESTED | P1 |

**Analysis**: API endpoints have moderate coverage for basic CRUD operations, but error paths, validation, and real-time features (WebSocket) are untested.

**Tested Functionality**:
- ✅ Provider listing and voice enumeration
- ✅ Config template retrieval and validation
- ✅ Generation job lifecycle (start, status, cancel)
- ✅ Basic augmentation and export job creation

**Untested Functionality**:
- ❌ WebSocket real-time updates
- ❌ Error response formats
- ❌ Request validation edge cases
- ❌ Authentication/authorization (if implemented)
- ❌ File upload handling
- ❌ Streaming audio responses
- ❌ CORS configuration
- ❌ Rate limiting middleware

#### MODELS & CONFIG (wakegen/models/, wakegen/config/)
| Module | Test Coverage | Status | Priority |
|--------|--------------|--------|----------|
| models/audio.py | ~15% | ⚠️ MINIMAL | P2 |
| models/config.py | ~20% | ⚠️ PARTIAL | P2 |
| models/generation.py | ~10% | ⚠️ MINIMAL | P2 |
| config/settings.py | 0% | ❌ UNTESTED | P2 |
| config/yaml_loader.py | ~20% | ⚠️ PARTIAL | P2 |

**Analysis**: Configuration validation has minimal coverage through API tests. Model serialization, validation, and YAML loading edge cases are untested.

#### UTILITIES (wakegen/utils/)
| Module | Test Coverage | Status | Priority |
|--------|--------------|--------|----------|
| audio.py | 0% | ❌ UNTESTED | P1 |
| async_helpers.py | 0% | ❌ UNTESTED | P1 |
| caching.py | 0% | ❌ UNTESTED | P2 |
| gpu.py | 0% | ❌ UNTESTED | P2 |
| logging.py | 0% | ❌ UNTESTED | P3 |
| performance.py | 0% | ❌ UNTESTED | P3 |

**Analysis**: CRITICAL GAP - Utility functions are foundational but completely untested. Audio loading/processing, async helpers, caching, and GPU detection lack validation.

#### PLUGINS (wakegen/plugins/)
| Module | Test Coverage | Status | Priority |
|--------|--------------|--------|----------|
| base.py | 0% | ❌ UNTESTED | P2 |
| discovery.py | 0% | ❌ UNTESTED | P2 |

**Analysis**: Plugin system is untested, risking third-party integration failures.

#### TRAINING (wakegen/training/)
| Module | Test Coverage | Status | Priority |
|--------|--------------|--------|----------|
| All modules | 0% | ❌ UNTESTED | P3 |

**Analysis**: Training utilities are untested but lower priority than core generation pipeline.

---

## 2. Test Type Distribution Analysis

### 2.1 Current Test Distribution

```
Test Type              Count    Percentage    Target    Gap
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unit Tests             ~8       20%           70%       -50%
Integration Tests      ~12      30%           20%       +10%
API Tests              ~15      38%           5%        +33%
Property Tests         ~5       12%           5%        +7%
E2E Tests              0        0%            <5%       -5%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total                  ~40
```

**Analysis**:
- **Inverted pyramid**: Heavy focus on API/integration tests, minimal unit test foundation
- **Property-based testing**: Excellent use for provider validation, but only 2 modules covered
- **Missing E2E**: No full workflow tests (generate → augment → validate → export)

### 2.2 Mock Usage Assessment

**Appropriateness**: GOOD
- Providers properly mocked in API tests to avoid external dependencies
- Async mocking correctly implemented for async provider methods
- File system operations mocked to enable fast, isolated testing

**Issues**:
- Over-reliance on mocking may hide integration issues
- No "real provider" integration tests (marked with pytest.mark.slow)
- Mock completeness varies (some provider methods not fully mocked)

### 2.3 Async Test Patterns

**Quality**: GOOD
- Consistent use of `@pytest.mark.asyncio` decorator
- Proper async/await patterns throughout tests
- Async mock implementations are correct

**Gaps**:
- No concurrency/race condition testing
- No timeout testing for async operations
- Missing deadlock/livelock scenarios

---

## 3. Test Quality Metrics

### 3.1 Assertion Density

**Average Assertions per Test**: 2.8
**Assessment**: ADEQUATE but could be improved

**Examples**:

**Good** (test_bark_voice_listing_returns_valid_voices):
```python
- 8+ assertions per test
- Validates multiple properties (id, format, provider, gender, language)
- Checks both positive and filter-based constraints
```

**Weak** (test_list_providers):
```python
- Only 3 assertions
- Basic structure validation only
- No edge case validation
```

### 3.2 Test Isolation

**Quality**: GOOD
- Fixtures properly scoped (function-level by default)
- Each test creates fresh TestClient and app instances
- Mock patch contexts properly managed with `with` statements
- Temporary directories used for file-based tests

**Issues**:
- Some tests may share global state through registry
- No explicit cleanup verification in async tests

### 3.3 Edge Case Coverage

**Assessment**: WEAK

**Missing Edge Cases**:
- Empty string inputs
- Null/None value handling
- Invalid file paths (permissions, special characters)
- Extremely large file counts
- Unicode and international character handling
- Concurrent access to shared resources
- Resource exhaustion scenarios
- Malformed configuration files
- Network failures and timeouts
- Partial file writes/corrupted audio

### 3.4 Error Handling Path Testing

**Coverage**: ~20% (critically low)

**Tested Error Paths**:
- ✅ Invalid provider lookup (404 responses)
- ✅ Invalid configuration validation
- ✅ Non-existent job queries

**Untested Error Paths**:
- ❌ Provider initialization failures (missing dependencies)
- ❌ API key validation errors
- ❌ Rate limit exceeded scenarios
- ❌ Disk space exhaustion
- ❌ Audio file corruption handling
- ❌ Timeout handling in generation
- ❌ Checkpoint restoration failures
- ❌ Concurrent job conflicts
- ❌ Invalid audio format handling
- ❌ Out-of-memory conditions

---

## 4. Testing Gaps - Prioritized List

### PRIORITY 1 (CRITICAL - Implement Immediately)

#### P1.1 Core Generation Orchestration
**Missing Tests**:
1. `test_orchestrator_full_workflow` - End-to-end generation session
2. `test_batch_processor_concurrency_limits` - Verify max concurrent tasks
3. `test_rate_limiter_token_bucket` - Validate rate limiting algorithm
4. `test_checkpoint_save_restore` - Checkpoint persistence and recovery
5. `test_generation_error_recovery` - Retry logic and error handling
6. `test_provider_fallback_mechanism` - Provider selection and fallback

**Rationale**: Core value proposition untested, high risk of production failures.

#### P1.2 Quality Validation
**Missing Tests**:
1. `test_sample_validator_duration_checks` - Min/max duration enforcement
2. `test_sample_validator_snr_calculation` - SNR accuracy
3. `test_quality_scorer_algorithms` - Scoring logic validation
4. `test_asr_transcription_verification` - ASR accuracy checking
5. `test_deduplication_hash_collision` - Deduplication correctness

**Rationale**: Quality differentiator, untested algorithms risk incorrect validation.

#### P1.3 Audio Utilities
**Missing Tests**:
1. `test_load_audio_formats` - Support for WAV, MP3, FLAC, OGG
2. `test_audio_resampling_accuracy` - Resampling quality
3. `test_audio_normalization` - Amplitude normalization
4. `test_audio_silence_detection` - Silence trimming
5. `test_audio_format_conversion` - Format conversion correctness

**Rationale**: Foundation for all audio processing, errors cascade through system.

#### P1.4 Export Workflows
**Missing Tests**:
1. `test_openwakeword_export_structure` - Directory structure validation
2. `test_dataset_splitter_stratification` - Split ratio accuracy
3. `test_manifest_generation_format` - Manifest JSON correctness
4. `test_export_file_integrity` - File copying/symlinking

**Rationale**: Export failures invalidate all previous work, critical for user workflows.

#### P1.5 Minimax Provider (New Feature)
**Missing Tests**:
1. `test_minimax_provider_initialization` - Provider setup
2. `test_minimax_api_key_validation` - API key handling
3. `test_minimax_voice_listing` - Voice enumeration
4. `test_minimax_generation_workflow` - Audio generation
5. `test_minimax_error_handling` - API error responses
6. `test_minimax_rate_limiting` - Commercial rate limits

**Rationale**: New commercial provider, zero test coverage is unacceptable for production.

### PRIORITY 2 (IMPORTANT - Implement Soon)

#### P2.1 Augmentation Effects
**Missing Tests**:
1. `test_noise_mixer_snr_accuracy` - SNR mixing validation
2. `test_room_simulator_convolution` - Room acoustics simulation
3. `test_microphone_simulator_frequency_response` - Mic characteristics
4. `test_dynamics_processor_compression` - Compression/limiting
5. `test_time_domain_pitch_shift` - Pitch shifting accuracy
6. `test_audio_degradation_effects` - Codec/bandwidth simulation

#### P2.2 Configuration Management
**Missing Tests**:
1. `test_yaml_loader_malformed_input` - Parsing error handling
2. `test_config_schema_validation` - Pydantic model validation
3. `test_config_environment_overrides` - Environment variable handling
4. `test_config_defaults` - Default value application

#### P2.3 Async Helpers & Concurrency
**Missing Tests**:
1. `test_async_batch_processing` - Batch async task execution
2. `test_async_timeout_handling` - Timeout enforcement
3. `test_async_cancellation` - Task cancellation
4. `test_async_resource_cleanup` - Context manager cleanup

#### P2.4 WebSocket Real-Time Updates
**Missing Tests**:
1. `test_websocket_connection_lifecycle` - Connect/disconnect
2. `test_websocket_progress_updates` - Real-time progress
3. `test_websocket_multiple_clients` - Concurrent connections
4. `test_websocket_error_handling` - Connection errors

### PRIORITY 3 (NICE-TO-HAVE - Implement When Capacity Allows)

#### P3.1 Training Utilities
- Script generation tests
- Model testing framework tests
- A/B comparison tests

#### P3.2 Device Presets
- Device profile application tests
- Preset validation tests

#### P3.3 Performance Monitoring
- GPU detection tests
- Memory profiling tests
- Performance metrics tests

---

## 5. Test Pyramid Adherence Analysis

### Current Pyramid (INVERTED - Problematic)
```
        ╱╲
       ╱  ╲    E2E Tests: 0
      ╱────╲
     ╱      ╲  Integration/API Tests: ~27 (67%)
    ╱────────╲
   ╱          ╲
  ╱────────────╲ Unit Tests: ~8 (20%)
 ╱──────────────╲
╱────────────────╲ Foundation (untested): ~5 (13%)
```

### Target Pyramid (Recommended)
```
        ╱╲
       ╱  ╲    E2E Tests: 2-3 (2%)
      ╱────╲
     ╱      ╲  Integration Tests: 20-25 (18%)
    ╱────────╲
   ╱          ╲
  ╱────────────╲ Unit Tests: 85-90 (75%)
 ╱──────────────╲
╱────────────────╲
```

**Action Items**:
1. Build unit test foundation (target: 85-90 tests)
2. Reduce API test proportion (consolidate similar tests)
3. Add 2-3 critical E2E workflow tests
4. Maintain property-based tests as supplementary validation

---

## 6. CI/CD Readiness Assessment

### 6.1 Test Speed
**Current**: Unknown (no CI/CD metrics available)
**Estimated**: ~5-10 seconds for current suite
**Target**: <30 seconds for full suite

**Optimization Opportunities**:
- Parallel test execution (pytest-xdist)
- Mock all external I/O by default
- Separate slow tests with markers (@pytest.mark.slow)

### 6.2 Flaky Test Potential
**Risk Level**: MODERATE

**Identified Risks**:
- Temporary file cleanup may fail on Windows
- Async timing dependencies could cause intermittent failures
- Property-based tests with insufficient examples may be unstable
- File system race conditions in concurrent tests

**Mitigation**:
- Use pytest-timeout for test duration limits
- Implement proper async cleanup in fixtures
- Increase Hypothesis example counts for stability
- Use proper temp directory isolation

### 6.3 Environment Dependencies
**Assessment**: GOOD

**Strengths**:
- Providers properly mocked to avoid external API dependencies
- No hardcoded file paths detected
- Environment-agnostic test design

**Gaps**:
- No tests for actual provider installations (integration tests needed)
- GPU detection logic untested (may fail in CI without GPU)
- Audio codec dependencies not validated in tests

---

## 7. Recommended Testing Strategy

### 7.1 Immediate Actions (Sprint 1)

**Week 1-2: Foundation Building**
1. Implement P1.3 Audio Utilities tests (20 tests)
2. Implement P1.5 Minimax Provider tests (15 tests)
3. Set up pytest-xdist for parallel execution
4. Configure pytest-cov for coverage reporting

**Week 3-4: Core Workflows**
1. Implement P1.1 Generation Orchestration tests (25 tests)
2. Implement P1.2 Quality Validation tests (20 tests)
3. Add pytest.mark.slow for integration tests
4. Establish CI/CD pipeline with coverage gates (target: 70%)

### 7.2 Mid-Term Strategy (Sprint 2-3)

**Month 2: Augmentation & Export**
1. Implement P1.4 Export Workflow tests (15 tests)
2. Implement P2.1 Augmentation Effects tests (30 tests)
3. Add E2E workflow tests (3 tests)
4. Implement mutation testing with mutmut

**Month 3: Completeness & Quality**
1. Implement P2.2-P2.4 tests (40 tests)
2. Add negative testing coverage (20 tests)
3. Implement load testing for concurrent operations
4. Add security testing (input validation, injection)

### 7.3 Long-Term Goals (Month 4+)

1. Achieve 80%+ line coverage, 70%+ branch coverage
2. Implement contract testing for API stability
3. Add performance regression testing
4. Establish property-based testing for all providers
5. Create test data generation framework

---

## 8. Specific Test Recommendations

### 8.1 Critical Missing Tests (Write First)

```python
# Test: test_generation_orchestrator_full_workflow.py
@pytest.mark.asyncio
async def test_full_generation_workflow_success():
    """E2E test: wake words → generation → augmentation → validation → export"""
    config = GenerationConfig(...)
    orchestrator = GenerationOrchestrator(config)

    results = await orchestrator.generate(
        wake_words=["hey katya"],
        count=10,
        provider="edge_tts"
    )

    assert len(results) == 10
    assert all(result.success for result in results)
    assert all(Path(result.output_path).exists() for result in results)
    # Validate audio properties
    # Validate augmentation applied
    # Validate quality metrics


# Test: test_rate_limiter_token_bucket.py
@pytest.mark.asyncio
async def test_rate_limiter_enforces_limits():
    """Verify token bucket rate limiting prevents exceeding max_requests"""
    limiter = RateLimiter(max_requests=5, period_seconds=10)

    start_time = time.time()
    for _ in range(10):
        await limiter.wait_for_token()
    elapsed = time.time() - start_time

    # Should take ~10 seconds for 10 requests with limit of 5/10s
    assert 9.5 < elapsed < 11.0, f"Expected ~10s, got {elapsed}s"


# Test: test_sample_validator_comprehensive.py
@pytest.mark.asyncio
@pytest.mark.parametrize("duration,expected_valid", [
    (0.3, False),  # Too short
    (0.5, True),   # Min duration
    (5.0, True),   # Normal
    (10.0, True),  # Max duration
    (11.0, False), # Too long
])
async def test_validator_duration_checks(duration, expected_valid):
    """Validate duration thresholds are enforced"""
    audio_file = create_test_audio(duration=duration)
    result = await validate_sample(audio_file)
    assert result.is_valid == expected_valid


# Test: test_minimax_provider_integration.py
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("MINIMAX_API_KEY"), reason="No API key")
@pytest.mark.asyncio
async def test_minimax_real_generation():
    """Integration test with real Minimax API (requires API key)"""
    provider = MinimaxProvider(config=ProviderConfig())
    result = await provider.generate(
        text="hey katya",
        voice_id="female-en-us-01",
        output_path="/tmp/test.wav"
    )
    assert Path(result).exists()
    # Validate audio format
    # Validate duration


# Test: test_augmentation_effects_quality.py
@pytest.mark.asyncio
async def test_noise_mixer_snr_accuracy():
    """Verify SNR calculation accuracy in noise mixing"""
    original = create_sine_wave(freq=440, duration=1.0)
    noise = create_white_noise(duration=1.0)

    mixer = NoiseMixer(sample_rate=16000)
    mixed = mixer.mix(original, noise, target_snr_db=20.0)

    actual_snr = calculate_snr(mixed, noise)
    assert 19.0 < actual_snr < 21.0, f"Expected SNR ~20dB, got {actual_snr}dB"
```

### 8.2 Negative Testing Examples

```python
# Test: test_error_handling_negative.py
@pytest.mark.asyncio
async def test_orchestrator_handles_provider_failure():
    """Verify graceful degradation when provider fails"""
    with patch("wakegen.providers.registry.get_provider") as mock:
        mock.side_effect = ProviderError("API key invalid")

        orchestrator = GenerationOrchestrator(config)
        with pytest.raises(GenerationError) as exc_info:
            await orchestrator.generate(...)

        assert "API key invalid" in str(exc_info.value)


@pytest.mark.asyncio
async def test_batch_processor_timeout_handling():
    """Verify tasks are cancelled on timeout"""
    config = BatchConfig(timeout_seconds=1)
    processor = BatchProcessor(config)

    async def slow_task():
        await asyncio.sleep(10)  # Exceeds timeout

    with pytest.raises(asyncio.TimeoutError):
        await processor.process([slow_task])


def test_config_validation_rejects_invalid_yaml():
    """Verify config validation catches malformed YAML"""
    invalid_yaml = "project: {{"  # Malformed

    result = validate_config(invalid_yaml)
    assert not result.valid
    assert "YAML parsing error" in result.errors[0]
```

### 8.3 Property-Based Testing Expansion

```python
# Test: test_properties_augmentation.py
from hypothesis import given, strategies as st

@given(
    snr_db=st.floats(min_value=0.0, max_value=40.0),
    duration=st.floats(min_value=0.5, max_value=5.0)
)
@pytest.mark.asyncio
async def test_noise_mixer_maintains_duration(snr_db, duration):
    """Property: Noise mixing should not change audio duration"""
    original = create_test_audio(duration=duration)
    mixer = NoiseMixer(sample_rate=16000)

    mixed = await mixer.apply(original, snr_db=snr_db)

    assert abs(len(mixed) - len(original)) < 100, "Duration changed significantly"


@given(
    pitch_steps=st.floats(min_value=-12.0, max_value=12.0)
)
def test_pitch_shift_reversibility(pitch_steps):
    """Property: Pitch shift should be reversible (shift up then down = original)"""
    original = create_sine_wave(freq=440, duration=1.0)
    effects = TimeDomainEffects(sample_rate=16000)

    shifted_up = effects.pitch_shift(original, pitch_steps)
    restored = effects.pitch_shift(shifted_up, -pitch_steps)

    correlation = np.corrcoef(original, restored)[0, 1]
    assert correlation > 0.95, f"Poor reversibility: correlation={correlation}"
```

---

## 9. Test Infrastructure Recommendations

### 9.1 pytest Configuration

```ini
# pytest.ini
[tool:pytest]
minversion = 7.0
addopts =
    --strict-markers
    --cov=wakegen
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=70
    -v
    -ra
    --durations=10
    --maxfail=5
    --tb=short

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (may use external resources)
    slow: Slow tests (>1 second)
    asyncio: Async tests
    property: Property-based tests

timeout = 300
asyncio_mode = auto
```

### 9.2 Coverage Configuration

```ini
# .coveragerc
[run]
source = wakegen
omit =
    */tests/*
    */migrations/*
    */__pycache__/*
    */site-packages/*

[report]
precision = 2
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    @abstractmethod
```

### 9.3 CI/CD Pipeline (GitHub Actions Example)

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Run unit tests
      run: pytest -m "unit" -n auto

    - name: Run integration tests
      run: pytest -m "integration"
      env:
        MINIMAX_API_KEY: ${{ secrets.MINIMAX_API_KEY }}

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
```

---

## 10. Conclusion & Action Plan

### Summary
The wakegen project has a **minimal testing foundation** with critical gaps in core functionality. Immediate action is required to:

1. **Build unit test foundation** (target: 85+ tests)
2. **Test critical workflows** (generation, quality, export)
3. **Validate new Minimax provider** (0% coverage unacceptable)
4. **Establish CI/CD pipeline** with coverage gates

### Success Metrics (3 Month Timeline)

| Metric | Current | Month 1 | Month 2 | Month 3 |
|--------|---------|---------|---------|---------|
| Line Coverage | ~15% | 45% | 65% | 80% |
| Branch Coverage | ~10% | 35% | 55% | 70% |
| Test Count | 40 | 100 | 175 | 220 |
| Unit/Integration Ratio | 0.3 | 1.5 | 2.5 | 3.5 |
| Flaky Test Rate | Unknown | <2% | <1% | <0.5% |
| CI/CD Build Time | N/A | <2min | <3min | <4min |

### Investment Required
- **Engineer Time**: ~2-3 weeks full-time for P1 tests
- **Infrastructure**: CI/CD setup, coverage reporting
- **Training**: Property-based testing, async testing patterns
- **Tools**: pytest-xdist, pytest-cov, mutmut, hypothesis

### Risk Assessment
**Without Immediate Action**:
- High risk of production defects in core generation workflows
- Quality validation may produce incorrect results
- New Minimax provider may have undetected bugs
- Technical debt will compound as codebase grows

**With Recommended Strategy**:
- 80%+ defect detection before production
- Confident refactoring and feature additions
- Faster debugging through comprehensive test suite
- Professional-grade quality assurance
