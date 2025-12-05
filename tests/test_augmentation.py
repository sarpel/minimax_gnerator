"""
Unit tests for augmentation modules.

These tests verify augmentation effects and pipelines work correctly.
Some tests require optional dependencies like pyroomacoustics.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from typing import Generator


# Check if augmentation dependencies are available
try:
    import pyroomacoustics
    HAS_PYROOMACOUSTICS = True
except ImportError:
    HAS_PYROOMACOUSTICS = False

# Skip marker for tests requiring pyroomacoustics
requires_pyroomacoustics = pytest.mark.skipif(
    not HAS_PYROOMACOUSTICS,
    reason="pyroomacoustics not installed"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_audio() -> tuple[np.ndarray, int]:
    """Create sample audio data for testing."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # Generate a simple sine wave with some variation
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
    return audio, sr


@pytest.fixture
def sample_audio_file(temp_dir: Path, sample_audio: tuple) -> Path:
    """Create a sample audio file."""
    import soundfile as sf
    
    audio, sr = sample_audio
    output_path = temp_dir / "sample.wav"
    sf.write(output_path, audio, sr)
    return output_path


# ============================================================================
# Device Preset Tests
# ============================================================================

@requires_pyroomacoustics
class TestDevicePresets:
    """Tests for device-specific presets."""
    
    def test_target_device_enum(self):
        """Test TargetDevice enum has expected values."""
        from wakegen.augmentation import TargetDevice
        
        # Check key devices exist
        assert hasattr(TargetDevice, 'ESP32_PDM')
        assert hasattr(TargetDevice, 'ESP32_I2S')
        assert hasattr(TargetDevice, 'ESP32_S3')
        assert hasattr(TargetDevice, 'RPI_USB')
        assert hasattr(TargetDevice, 'RPI_HAT')
        assert hasattr(TargetDevice, 'SMART_SPEAKER_MID')
        assert hasattr(TargetDevice, 'MOBILE_IOS')
        assert hasattr(TargetDevice, 'FAR_FIELD')
        assert hasattr(TargetDevice, 'NEAR_FIELD')
    
    def test_get_device_preset(self):
        """Test getting a device preset."""
        from wakegen.augmentation import TargetDevice, get_device_preset
        
        preset = get_device_preset(TargetDevice.ESP32_I2S)
        
        assert preset is not None
        assert preset.device == TargetDevice.ESP32_I2S
        assert preset.audio_spec is not None
        assert preset.environment is not None
    
    def test_list_device_presets(self):
        """Test listing all device presets."""
        from wakegen.augmentation import list_device_presets
        
        presets = list_device_presets()
        
        assert len(presets) > 0
        
        # Each preset should have required fields
        for preset in presets:
            assert 'name' in preset or 'device' in preset
    
    def test_device_audio_spec(self):
        """Test DeviceAudioSpec dataclass."""
        from wakegen.augmentation.device_presets import DeviceAudioSpec
        
        spec = DeviceAudioSpec(
            sample_rate=16000,
            bit_depth=16,
            channels=1,
            frequency_response_low=100,
            frequency_response_high=8000,
            dynamic_range_db=60,
            self_noise_db=-70,
        )
        
        assert spec.sample_rate == 16000
        assert spec.bit_depth == 16
    
    def test_device_preset_manager(self):
        """Test DevicePresetManager."""
        from wakegen.augmentation import TargetDevice, DevicePresetManager
        
        manager = DevicePresetManager()
        
        # Get augmentation config for device
        config = manager.get_augmentation_config(
            TargetDevice.ESP32_I2S,
            include_all=False
        )
        
        assert config is not None


# ============================================================================
# Telephony Simulator Tests
# ============================================================================

@requires_pyroomacoustics
class TestTelephonySimulator:
    """Tests for telephony simulation."""
    
    def test_phone_type_enum(self):
        """Test PhoneType enum values."""
        from wakegen.augmentation.effects import PhoneType
        
        assert hasattr(PhoneType, 'PSTN')
        assert hasattr(PhoneType, 'MOBILE_GSM')
        assert hasattr(PhoneType, 'MOBILE_HD')
        assert hasattr(PhoneType, 'VOIP_LOW')
        assert hasattr(PhoneType, 'VOIP_MEDIUM')
        assert hasattr(PhoneType, 'VOIP_HIGH')
    
    def test_simulator_creation(self):
        """Test creating telephony simulator."""
        from wakegen.augmentation.effects import TelephonySimulator
        
        simulator = TelephonySimulator()
        assert simulator is not None
    
    def test_apply_telephony_effect(self, sample_audio: tuple):
        """Test applying telephony effect."""
        from wakegen.augmentation.effects import TelephonySimulator, PhoneType
        
        audio, sr = sample_audio
        simulator = TelephonySimulator()
        
        augmented = simulator.apply_telephony_effect(
            audio, sr,
            phone_type=PhoneType.MOBILE_GSM
        )
        
        assert augmented is not None
        assert len(augmented) > 0
        assert isinstance(augmented, np.ndarray)
    
    def test_different_phone_types(self, sample_audio: tuple):
        """Test different phone types produce different results."""
        from wakegen.augmentation.effects import TelephonySimulator, PhoneType
        
        audio, sr = sample_audio
        simulator = TelephonySimulator()
        
        results = {}
        for phone_type in [PhoneType.PSTN, PhoneType.MOBILE_GSM, PhoneType.VOIP_HIGH]:
            results[phone_type] = simulator.apply_telephony_effect(
                audio.copy(), sr,
                phone_type=phone_type
            )
        
        # Each phone type should produce different output
        assert not np.allclose(results[PhoneType.PSTN], results[PhoneType.MOBILE_GSM])


# ============================================================================
# Distance Simulator Tests
# ============================================================================

@requires_pyroomacoustics
class TestDistanceSimulator:
    """Tests for distance simulation."""
    
    def test_simulator_creation(self):
        """Test creating distance simulator."""
        from wakegen.augmentation.effects import DistanceSimulator
        
        simulator = DistanceSimulator()
        assert simulator is not None
    
    def test_apply_distance_effect(self, sample_audio: tuple):
        """Test applying distance effect."""
        from wakegen.augmentation.effects import DistanceSimulator
        
        audio, sr = sample_audio
        simulator = DistanceSimulator()
        
        augmented = simulator.apply_distance_effect(
            audio, sr,
            distance_meters=3.0,
            room_size="medium"
        )
        
        assert augmented is not None
        assert len(augmented) > 0
    
    def test_distance_attenuation(self, sample_audio: tuple):
        """Test that farther distance produces quieter audio."""
        from wakegen.augmentation.effects import DistanceSimulator
        
        audio, sr = sample_audio
        simulator = DistanceSimulator()
        
        close = simulator.apply_distance_effect(audio.copy(), sr, distance_meters=0.5)
        far = simulator.apply_distance_effect(audio.copy(), sr, distance_meters=5.0)
        
        # Far audio should be quieter (lower RMS)
        close_rms = np.sqrt(np.mean(close ** 2))
        far_rms = np.sqrt(np.mean(far ** 2))
        
        assert far_rms < close_rms


# ============================================================================
# Augmentation Pipeline Tests
# ============================================================================

@requires_pyroomacoustics
class TestAugmentationPipeline:
    """Tests for augmentation pipeline."""
    
    def test_pipeline_creation(self):
        """Test creating augmentation pipeline."""
        from wakegen.augmentation import AugmentationPipeline
        
        pipeline = AugmentationPipeline()
        assert pipeline is not None
    
    def test_pipeline_process(self, sample_audio: tuple):
        """Test processing audio through pipeline."""
        from wakegen.augmentation import AugmentationPipeline
        
        audio, sr = sample_audio
        pipeline = AugmentationPipeline()
        
        augmented = pipeline.process(audio, sr)
        
        assert augmented is not None
        assert len(augmented) > 0


# ============================================================================
# Environment Profile Tests
# ============================================================================

@requires_pyroomacoustics
class TestEnvironmentProfiles:
    """Tests for environment profiles."""
    
    def test_profile_enum(self):
        """Test EnvironmentProfile enum exists."""
        from wakegen.augmentation.profiles import EnvironmentProfile
        
        # Check some profiles exist
        profiles = list(EnvironmentProfile)
        assert len(profiles) > 0
    
    def test_get_profile(self):
        """Test getting an environment profile."""
        from wakegen.augmentation.profiles import EnvironmentProfile, get_profile
        
        # Get first available profile
        profile_enum = list(EnvironmentProfile)[0]
        profile = get_profile(profile_enum)
        
        assert profile is not None


# ============================================================================
# Effects Module Tests
# ============================================================================

@requires_pyroomacoustics
class TestEffectsModule:
    """Tests for effects module imports."""
    
    def test_import_all_effects(self):
        """Test importing all effects."""
        from wakegen.augmentation.effects import (
            TelephonySimulator,
            DistanceSimulator,
            PhoneType,
        )
        
        assert TelephonySimulator is not None
        assert DistanceSimulator is not None
        assert PhoneType is not None
    
    def test_degradation_module(self):
        """Test degradation effects module."""
        from wakegen.augmentation.effects import degradation
        
        assert degradation is not None
    
    def test_dynamics_module(self):
        """Test dynamics effects module."""
        from wakegen.augmentation.effects import dynamics
        
        assert dynamics is not None
    
    def test_time_domain_module(self):
        """Test time domain effects module."""
        from wakegen.augmentation.effects import time_domain
        
        assert time_domain is not None


# ============================================================================
# Noise Module Tests
# ============================================================================

@requires_pyroomacoustics
class TestNoiseModule:
    """Tests for noise modules."""
    
    def test_import_noise_modules(self):
        """Test importing noise modules."""
        from wakegen.augmentation.noise import (
            mixer,
            profiles,
            events,
        )
        
        assert mixer is not None
        assert profiles is not None


# ============================================================================
# Room Module Tests
# ============================================================================

@requires_pyroomacoustics
class TestRoomModule:
    """Tests for room simulation modules."""
    
    def test_import_room_modules(self):
        """Test importing room modules."""
        from wakegen.augmentation.room import (
            convolver,
            simulator,
        )
        
        assert convolver is not None
        assert simulator is not None


# ============================================================================
# Microphone Module Tests
# ============================================================================

@requires_pyroomacoustics
class TestMicrophoneModule:
    """Tests for microphone simulation modules."""
    
    def test_import_microphone_modules(self):
        """Test importing microphone modules."""
        from wakegen.augmentation.microphone import simulator
        
        assert simulator is not None


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
