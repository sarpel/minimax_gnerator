"""
Device-Specific Augmentation Presets

This module provides augmentation profiles optimized for different target hardware
devices commonly used for wake word detection. Each preset is tuned to match the
acoustic characteristics and constraints of the target device.

Target Devices:
- ESP32 microcontrollers with PDM/I2S microphones
- Raspberry Pi with USB/I2S microphones  
- Smart speakers (Echo-like devices)
- Mobile phones (iOS/Android)
- Embedded Linux devices (Jetson, etc.)

Key Considerations:
- Microphone frequency response limitations
- Sample rate and bit depth constraints
- Typical deployment environments
- Background noise profiles
- Processing power constraints affecting real-time augmentation
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class TargetDevice(str, Enum):
    """Target hardware devices for wake word deployment."""
    
    # Microcontrollers
    ESP32_PDM = "esp32_pdm"  # ESP32 with PDM microphone
    ESP32_I2S = "esp32_i2s"  # ESP32 with I2S microphone (e.g., INMP441)
    ESP32_S3 = "esp32_s3"  # ESP32-S3 with enhanced audio features
    
    # Single-board computers
    RPI_USB = "rpi_usb"  # Raspberry Pi with USB microphone
    RPI_I2S = "rpi_i2s"  # Raspberry Pi with I2S microphone
    RPI_HAT = "rpi_hat"  # Raspberry Pi with audio HAT (ReSpeaker, etc.)
    
    # Smart speakers
    SMART_SPEAKER_BUDGET = "smart_speaker_budget"  # Low-cost smart speakers
    SMART_SPEAKER_MID = "smart_speaker_mid"  # Mid-range devices (Echo Dot)
    SMART_SPEAKER_PREMIUM = "smart_speaker_premium"  # High-end (Echo Studio, HomePod)
    
    # Mobile devices
    MOBILE_IOS = "mobile_ios"  # iPhone/iPad
    MOBILE_ANDROID = "mobile_android"  # Android phones
    
    # Embedded Linux
    JETSON_NANO = "jetson_nano"  # NVIDIA Jetson Nano
    CORAL_DEV = "coral_dev"  # Google Coral Dev Board
    
    # Generic profiles
    LOW_POWER = "low_power"  # Battery-powered devices
    FAR_FIELD = "far_field"  # Far-field microphone arrays
    NEAR_FIELD = "near_field"  # Close-talking scenarios


@dataclass
class DeviceAudioSpec:
    """Audio specifications for a target device."""
    
    sample_rate: int
    bit_depth: int
    channels: int
    frequency_response: tuple[int, int]  # (low_hz, high_hz)
    noise_floor_db: float  # Self-noise level
    sensitivity_dbfs: float  # Reference sensitivity
    max_spl_db: float  # Maximum sound pressure level
    adc_resolution: int  # ADC bits (affects quantization noise)


@dataclass
class DeviceEnvironment:
    """Typical deployment environment for a device."""
    
    typical_distances_m: tuple[float, float]  # (min, max) speaking distance
    typical_snr_db: tuple[float, float]  # (min, max) signal-to-noise ratio
    reverb_level: str  # "low", "medium", "high"
    background_noise_types: list[str]
    interference_sources: list[str]


@dataclass
class DevicePreset:
    """Complete preset for a target device."""
    
    device: TargetDevice
    name: str
    description: str
    audio_spec: DeviceAudioSpec
    environment: DeviceEnvironment
    
    # Augmentation parameters
    microphone_filter: Dict[str, Any] = field(default_factory=dict)
    room_simulation: Dict[str, Any] = field(default_factory=dict)
    noise_mixing: Dict[str, Any] = field(default_factory=dict)
    telephony_effects: Dict[str, Any] = field(default_factory=dict)
    degradation_effects: Dict[str, Any] = field(default_factory=dict)
    
    # Training recommendations
    recommended_variations: int = 10
    priority_augmentations: list[str] = field(default_factory=list)


# Device audio specifications
DEVICE_SPECS: Dict[TargetDevice, DeviceAudioSpec] = {
    TargetDevice.ESP32_PDM: DeviceAudioSpec(
        sample_rate=16000,
        bit_depth=16,
        channels=1,
        frequency_response=(100, 7000),
        noise_floor_db=-60,
        sensitivity_dbfs=-26,
        max_spl_db=120,
        adc_resolution=12,
    ),
    TargetDevice.ESP32_I2S: DeviceAudioSpec(
        sample_rate=16000,
        bit_depth=24,  # INMP441 is 24-bit
        channels=1,
        frequency_response=(60, 8000),
        noise_floor_db=-87,  # Better than PDM
        sensitivity_dbfs=-26,
        max_spl_db=120,
        adc_resolution=24,
    ),
    TargetDevice.ESP32_S3: DeviceAudioSpec(
        sample_rate=16000,
        bit_depth=24,
        channels=1,
        frequency_response=(50, 8000),
        noise_floor_db=-90,
        sensitivity_dbfs=-26,
        max_spl_db=122,
        adc_resolution=24,
    ),
    TargetDevice.RPI_USB: DeviceAudioSpec(
        sample_rate=16000,
        bit_depth=16,
        channels=1,
        frequency_response=(80, 15000),
        noise_floor_db=-65,
        sensitivity_dbfs=-30,
        max_spl_db=115,
        adc_resolution=16,
    ),
    TargetDevice.RPI_I2S: DeviceAudioSpec(
        sample_rate=16000,
        bit_depth=24,
        channels=1,
        frequency_response=(50, 15000),
        noise_floor_db=-85,
        sensitivity_dbfs=-26,
        max_spl_db=120,
        adc_resolution=24,
    ),
    TargetDevice.RPI_HAT: DeviceAudioSpec(
        sample_rate=16000,
        bit_depth=24,
        channels=4,  # ReSpeaker 4-mic array
        frequency_response=(40, 16000),
        noise_floor_db=-90,
        sensitivity_dbfs=-26,
        max_spl_db=120,
        adc_resolution=24,
    ),
    TargetDevice.SMART_SPEAKER_BUDGET: DeviceAudioSpec(
        sample_rate=16000,
        bit_depth=16,
        channels=2,
        frequency_response=(100, 12000),
        noise_floor_db=-60,
        sensitivity_dbfs=-28,
        max_spl_db=115,
        adc_resolution=16,
    ),
    TargetDevice.SMART_SPEAKER_MID: DeviceAudioSpec(
        sample_rate=16000,
        bit_depth=16,
        channels=4,
        frequency_response=(60, 14000),
        noise_floor_db=-72,
        sensitivity_dbfs=-26,
        max_spl_db=118,
        adc_resolution=24,
    ),
    TargetDevice.SMART_SPEAKER_PREMIUM: DeviceAudioSpec(
        sample_rate=16000,
        bit_depth=24,
        channels=7,  # 7-mic array
        frequency_response=(40, 16000),
        noise_floor_db=-90,
        sensitivity_dbfs=-26,
        max_spl_db=120,
        adc_resolution=24,
    ),
    TargetDevice.MOBILE_IOS: DeviceAudioSpec(
        sample_rate=16000,
        bit_depth=16,
        channels=1,
        frequency_response=(60, 15000),
        noise_floor_db=-80,
        sensitivity_dbfs=-26,
        max_spl_db=120,
        adc_resolution=16,
    ),
    TargetDevice.MOBILE_ANDROID: DeviceAudioSpec(
        sample_rate=16000,
        bit_depth=16,
        channels=1,
        frequency_response=(80, 14000),
        noise_floor_db=-70,  # Varies by device
        sensitivity_dbfs=-28,
        max_spl_db=118,
        adc_resolution=16,
    ),
    TargetDevice.JETSON_NANO: DeviceAudioSpec(
        sample_rate=16000,
        bit_depth=16,
        channels=2,
        frequency_response=(50, 15000),
        noise_floor_db=-80,
        sensitivity_dbfs=-26,
        max_spl_db=120,
        adc_resolution=24,
    ),
    TargetDevice.CORAL_DEV: DeviceAudioSpec(
        sample_rate=16000,
        bit_depth=16,
        channels=1,
        frequency_response=(60, 14000),
        noise_floor_db=-75,
        sensitivity_dbfs=-26,
        max_spl_db=118,
        adc_resolution=16,
    ),
    TargetDevice.LOW_POWER: DeviceAudioSpec(
        sample_rate=8000,  # Lower sample rate for power savings
        bit_depth=16,
        channels=1,
        frequency_response=(200, 3400),  # Narrowband for power savings
        noise_floor_db=-55,
        sensitivity_dbfs=-30,
        max_spl_db=110,
        adc_resolution=12,
    ),
    TargetDevice.FAR_FIELD: DeviceAudioSpec(
        sample_rate=16000,
        bit_depth=24,
        channels=4,
        frequency_response=(40, 16000),
        noise_floor_db=-85,
        sensitivity_dbfs=-26,
        max_spl_db=120,
        adc_resolution=24,
    ),
    TargetDevice.NEAR_FIELD: DeviceAudioSpec(
        sample_rate=16000,
        bit_depth=16,
        channels=1,
        frequency_response=(80, 14000),
        noise_floor_db=-70,
        sensitivity_dbfs=-26,
        max_spl_db=115,
        adc_resolution=16,
    ),
}


# Typical deployment environments
DEVICE_ENVIRONMENTS: Dict[TargetDevice, DeviceEnvironment] = {
    TargetDevice.ESP32_PDM: DeviceEnvironment(
        typical_distances_m=(0.5, 3.0),
        typical_snr_db=(5.0, 25.0),
        reverb_level="medium",
        background_noise_types=["hvac", "appliances", "speech_babble"],
        interference_sources=["wifi", "motor_noise"],
    ),
    TargetDevice.ESP32_I2S: DeviceEnvironment(
        typical_distances_m=(0.5, 4.0),
        typical_snr_db=(10.0, 30.0),
        reverb_level="medium",
        background_noise_types=["hvac", "appliances", "speech_babble"],
        interference_sources=["wifi"],
    ),
    TargetDevice.ESP32_S3: DeviceEnvironment(
        typical_distances_m=(0.5, 5.0),
        typical_snr_db=(10.0, 35.0),
        reverb_level="medium",
        background_noise_types=["hvac", "appliances", "speech_babble", "music"],
        interference_sources=["wifi", "bluetooth"],
    ),
    TargetDevice.RPI_USB: DeviceEnvironment(
        typical_distances_m=(0.3, 2.0),
        typical_snr_db=(15.0, 40.0),
        reverb_level="low",
        background_noise_types=["fan_noise", "hvac", "keyboard"],
        interference_sources=["usb_noise", "power_supply"],
    ),
    TargetDevice.RPI_I2S: DeviceEnvironment(
        typical_distances_m=(0.5, 4.0),
        typical_snr_db=(15.0, 40.0),
        reverb_level="medium",
        background_noise_types=["hvac", "speech_babble", "appliances"],
        interference_sources=["power_supply"],
    ),
    TargetDevice.RPI_HAT: DeviceEnvironment(
        typical_distances_m=(1.0, 6.0),
        typical_snr_db=(10.0, 35.0),
        reverb_level="high",
        background_noise_types=["hvac", "speech_babble", "music", "tv"],
        interference_sources=["power_supply"],
    ),
    TargetDevice.SMART_SPEAKER_BUDGET: DeviceEnvironment(
        typical_distances_m=(1.0, 4.0),
        typical_snr_db=(5.0, 25.0),
        reverb_level="medium",
        background_noise_types=["hvac", "speech_babble", "music", "tv"],
        interference_sources=["speaker_feedback", "wifi"],
    ),
    TargetDevice.SMART_SPEAKER_MID: DeviceEnvironment(
        typical_distances_m=(1.0, 5.0),
        typical_snr_db=(8.0, 30.0),
        reverb_level="medium",
        background_noise_types=["hvac", "speech_babble", "music", "tv", "appliances"],
        interference_sources=["speaker_feedback"],
    ),
    TargetDevice.SMART_SPEAKER_PREMIUM: DeviceEnvironment(
        typical_distances_m=(1.0, 7.0),
        typical_snr_db=(10.0, 35.0),
        reverb_level="high",
        background_noise_types=["hvac", "speech_babble", "music", "tv", "appliances"],
        interference_sources=["speaker_feedback"],
    ),
    TargetDevice.MOBILE_IOS: DeviceEnvironment(
        typical_distances_m=(0.3, 1.5),
        typical_snr_db=(10.0, 40.0),
        reverb_level="low",
        background_noise_types=["traffic", "wind", "speech_babble", "cafeteria"],
        interference_sources=["phone_speaker"],
    ),
    TargetDevice.MOBILE_ANDROID: DeviceEnvironment(
        typical_distances_m=(0.3, 1.5),
        typical_snr_db=(8.0, 35.0),
        reverb_level="low",
        background_noise_types=["traffic", "wind", "speech_babble", "cafeteria"],
        interference_sources=["phone_speaker", "cellular_interference"],
    ),
    TargetDevice.JETSON_NANO: DeviceEnvironment(
        typical_distances_m=(0.5, 4.0),
        typical_snr_db=(15.0, 40.0),
        reverb_level="medium",
        background_noise_types=["fan_noise", "hvac", "speech_babble"],
        interference_sources=["fan_vibration", "power_supply"],
    ),
    TargetDevice.CORAL_DEV: DeviceEnvironment(
        typical_distances_m=(0.5, 3.0),
        typical_snr_db=(15.0, 35.0),
        reverb_level="low",
        background_noise_types=["hvac", "speech_babble"],
        interference_sources=["power_supply"],
    ),
    TargetDevice.LOW_POWER: DeviceEnvironment(
        typical_distances_m=(0.3, 1.5),
        typical_snr_db=(5.0, 20.0),
        reverb_level="medium",
        background_noise_types=["hvac", "appliances"],
        interference_sources=["adc_noise", "power_ripple"],
    ),
    TargetDevice.FAR_FIELD: DeviceEnvironment(
        typical_distances_m=(2.0, 8.0),
        typical_snr_db=(5.0, 25.0),
        reverb_level="high",
        background_noise_types=["hvac", "speech_babble", "music", "tv", "appliances"],
        interference_sources=["speaker_feedback"],
    ),
    TargetDevice.NEAR_FIELD: DeviceEnvironment(
        typical_distances_m=(0.1, 0.5),
        typical_snr_db=(20.0, 50.0),
        reverb_level="low",
        background_noise_types=["breathing", "clothing_rustle"],
        interference_sources=["plosives", "wind_buffeting"],
    ),
}


class DevicePresetManager:
    """
    Manages device-specific augmentation presets.
    
    Provides pre-configured augmentation parameters optimized for different
    target hardware devices used in wake word detection systems.
    """
    
    def __init__(self) -> None:
        """Initialize the preset manager with default presets."""
        self._presets: Dict[TargetDevice, DevicePreset] = {}
        self._build_presets()
    
    def _build_presets(self) -> None:
        """Build all device presets."""
        for device in TargetDevice:
            self._presets[device] = self._create_preset(device)
    
    def _create_preset(self, device: TargetDevice) -> DevicePreset:
        """Create a preset for a specific device."""
        spec = DEVICE_SPECS[device]
        env = DEVICE_ENVIRONMENTS[device]
        
        # Build microphone filter parameters based on device specs
        microphone_filter = self._build_microphone_filter(spec)
        
        # Build room simulation parameters based on environment
        room_simulation = self._build_room_simulation(env)
        
        # Build noise mixing parameters
        noise_mixing = self._build_noise_mixing(env)
        
        # Build telephony effects (for low bandwidth scenarios)
        telephony_effects = self._build_telephony_effects(spec, device)
        
        # Build degradation effects
        degradation_effects = self._build_degradation_effects(spec)
        
        # Determine priority augmentations
        priority_augmentations = self._get_priority_augmentations(device, env)
        
        # Calculate recommended variations
        recommended_variations = self._calculate_variations(env)
        
        return DevicePreset(
            device=device,
            name=self._get_device_name(device),
            description=self._get_device_description(device),
            audio_spec=spec,
            environment=env,
            microphone_filter=microphone_filter,
            room_simulation=room_simulation,
            noise_mixing=noise_mixing,
            telephony_effects=telephony_effects,
            degradation_effects=degradation_effects,
            recommended_variations=recommended_variations,
            priority_augmentations=priority_augmentations,
        )
    
    def _build_microphone_filter(self, spec: DeviceAudioSpec) -> Dict[str, Any]:
        """Build microphone filter parameters from device specs."""
        return {
            "highpass_freq": max(50, spec.frequency_response[0]),
            "lowpass_freq": min(spec.sample_rate // 2 - 100, spec.frequency_response[1]),
            "noise_floor_db": spec.noise_floor_db,
            "sensitivity_dbfs": spec.sensitivity_dbfs,
            "quantization_bits": spec.adc_resolution,
            "add_self_noise": True,
            "self_noise_level_db": spec.noise_floor_db + 10,  # Typical self-noise
        }
    
    def _build_room_simulation(self, env: DeviceEnvironment) -> Dict[str, Any]:
        """Build room simulation parameters from environment."""
        reverb_levels = {
            "low": {"rt60": 0.2, "room_size": 15.0, "wet_level": 0.1},
            "medium": {"rt60": 0.4, "room_size": 35.0, "wet_level": 0.2},
            "high": {"rt60": 0.7, "room_size": 80.0, "wet_level": 0.35},
        }
        
        params = reverb_levels.get(env.reverb_level, reverb_levels["medium"])
        
        return {
            **params,
            "distance_min_m": env.typical_distances_m[0],
            "distance_max_m": env.typical_distances_m[1],
            "apply_distance_attenuation": True,
        }
    
    def _build_noise_mixing(self, env: DeviceEnvironment) -> Dict[str, Any]:
        """Build noise mixing parameters from environment."""
        return {
            "snr_min_db": env.typical_snr_db[0],
            "snr_max_db": env.typical_snr_db[1],
            "noise_types": env.background_noise_types,
            "add_interference": len(env.interference_sources) > 0,
            "interference_types": env.interference_sources,
            "noise_probability": 0.8,  # 80% of samples get noise
        }
    
    def _build_telephony_effects(
        self, 
        spec: DeviceAudioSpec,
        device: TargetDevice
    ) -> Dict[str, Any]:
        """Build telephony effects for bandwidth-limited scenarios."""
        # Only apply telephony effects to low-bandwidth devices
        if spec.sample_rate <= 8000 or device == TargetDevice.LOW_POWER:
            return {
                "enabled": True,
                "phone_type": "voip_low",  # Simulates low-quality VoIP
                "packet_loss_rate": 0.02,
                "jitter_ms": 20.0,
            }
        return {"enabled": False}
    
    def _build_degradation_effects(self, spec: DeviceAudioSpec) -> Dict[str, Any]:
        """Build audio degradation parameters from specs."""
        return {
            "add_quantization_noise": spec.adc_resolution < 16,
            "quantization_bits": spec.adc_resolution,
            "add_clipping": True,
            "clipping_probability": 0.05,
            "add_dc_offset": True,
            "dc_offset_range": (-0.01, 0.01),
        }
    
    def _get_priority_augmentations(
        self, 
        device: TargetDevice,
        env: DeviceEnvironment
    ) -> List[str]:
        """Determine priority augmentations for a device."""
        priorities = ["microphone_simulation", "noise_mixing"]
        
        if env.reverb_level in ("medium", "high"):
            priorities.append("room_simulation")
        
        if env.typical_distances_m[1] > 3.0:
            priorities.append("distance_simulation")
        
        if device in (TargetDevice.MOBILE_IOS, TargetDevice.MOBILE_ANDROID):
            priorities.append("telephony_simulation")
        
        if device in (TargetDevice.ESP32_PDM, TargetDevice.LOW_POWER):
            priorities.append("degradation")
        
        return priorities
    
    def _calculate_variations(self, env: DeviceEnvironment) -> int:
        """Calculate recommended number of variations based on environment complexity."""
        complexity = 5  # Base
        
        # More noise types = more variations needed
        complexity += len(env.background_noise_types)
        
        # Wide distance range needs more variations
        distance_range = env.typical_distances_m[1] - env.typical_distances_m[0]
        if distance_range > 3.0:
            complexity += 5
        
        # Wide SNR range needs more variations
        snr_range = env.typical_snr_db[1] - env.typical_snr_db[0]
        if snr_range > 20:
            complexity += 5
        
        return min(complexity, 25)  # Cap at 25 variations
    
    def _get_device_name(self, device: TargetDevice) -> str:
        """Get human-readable device name."""
        names = {
            TargetDevice.ESP32_PDM: "ESP32 with PDM Microphone",
            TargetDevice.ESP32_I2S: "ESP32 with I2S Microphone (INMP441)",
            TargetDevice.ESP32_S3: "ESP32-S3 Audio Enhanced",
            TargetDevice.RPI_USB: "Raspberry Pi with USB Microphone",
            TargetDevice.RPI_I2S: "Raspberry Pi with I2S Microphone",
            TargetDevice.RPI_HAT: "Raspberry Pi with Audio HAT (ReSpeaker)",
            TargetDevice.SMART_SPEAKER_BUDGET: "Budget Smart Speaker",
            TargetDevice.SMART_SPEAKER_MID: "Mid-Range Smart Speaker (Echo Dot)",
            TargetDevice.SMART_SPEAKER_PREMIUM: "Premium Smart Speaker (Echo Studio)",
            TargetDevice.MOBILE_IOS: "iPhone/iPad",
            TargetDevice.MOBILE_ANDROID: "Android Phone",
            TargetDevice.JETSON_NANO: "NVIDIA Jetson Nano",
            TargetDevice.CORAL_DEV: "Google Coral Dev Board",
            TargetDevice.LOW_POWER: "Low-Power Battery Device",
            TargetDevice.FAR_FIELD: "Far-Field Microphone Array",
            TargetDevice.NEAR_FIELD: "Near-Field/Headset",
        }
        return names.get(device, device.value)
    
    def _get_device_description(self, device: TargetDevice) -> str:
        """Get device description."""
        descriptions = {
            TargetDevice.ESP32_PDM: "ESP32 microcontroller with integrated PDM microphone. Common in DIY smart home projects.",
            TargetDevice.ESP32_I2S: "ESP32 with external I2S MEMS microphone (INMP441). Better audio quality than PDM.",
            TargetDevice.ESP32_S3: "ESP32-S3 with enhanced audio features and better processing power.",
            TargetDevice.RPI_USB: "Raspberry Pi with standard USB microphone. Desktop/development setup.",
            TargetDevice.RPI_I2S: "Raspberry Pi with I2S microphone for better audio quality.",
            TargetDevice.RPI_HAT: "Raspberry Pi with ReSpeaker or similar audio HAT. Multi-microphone array.",
            TargetDevice.SMART_SPEAKER_BUDGET: "Entry-level smart speakers with basic microphone array.",
            TargetDevice.SMART_SPEAKER_MID: "Mid-range smart speakers like Echo Dot with 4-mic array.",
            TargetDevice.SMART_SPEAKER_PREMIUM: "High-end speakers like Echo Studio with 7-mic array.",
            TargetDevice.MOBILE_IOS: "Apple iPhone and iPad devices with high-quality microphones.",
            TargetDevice.MOBILE_ANDROID: "Android smartphones with varying microphone quality.",
            TargetDevice.JETSON_NANO: "NVIDIA Jetson Nano for edge AI applications.",
            TargetDevice.CORAL_DEV: "Google Coral Dev Board for TPU-accelerated inference.",
            TargetDevice.LOW_POWER: "Battery-powered devices with power-saving audio processing.",
            TargetDevice.FAR_FIELD: "Far-field microphone arrays for voice assistants.",
            TargetDevice.NEAR_FIELD: "Near-field scenarios like headsets and earbuds.",
        }
        return descriptions.get(device, f"Preset for {device.value}")
    
    def get_preset(self, device: TargetDevice) -> DevicePreset:
        """
        Get the preset for a specific device.
        
        Args:
            device: Target device enum value.
            
        Returns:
            DevicePreset for the specified device.
        """
        return self._presets[device]
    
    def get_preset_by_name(self, name: str) -> DevicePreset:
        """
        Get a preset by device name string.
        
        Args:
            name: Device name (e.g., "esp32_pdm", "rpi_usb").
            
        Returns:
            DevicePreset for the specified device.
            
        Raises:
            KeyError: If device name is not found.
        """
        device = TargetDevice(name)
        return self.get_preset(device)
    
    def list_presets(self) -> List[Dict[str, Any]]:
        """
        List all available device presets.
        
        Returns:
            List of preset summaries with device info.
        """
        return [
            {
                "device": preset.device.value,
                "name": preset.name,
                "description": preset.description,
                "sample_rate": preset.audio_spec.sample_rate,
                "channels": preset.audio_spec.channels,
                "recommended_variations": preset.recommended_variations,
                "priority_augmentations": preset.priority_augmentations,
            }
            for preset in self._presets.values()
        ]
    
    def get_presets_for_category(self, category: str) -> List[DevicePreset]:
        """
        Get all presets for a device category.
        
        Args:
            category: One of "microcontroller", "sbc", "smart_speaker", 
                     "mobile", "embedded", "generic".
                     
        Returns:
            List of DevicePresets in the category.
        """
        category_devices = {
            "microcontroller": [
                TargetDevice.ESP32_PDM, 
                TargetDevice.ESP32_I2S, 
                TargetDevice.ESP32_S3
            ],
            "sbc": [
                TargetDevice.RPI_USB, 
                TargetDevice.RPI_I2S, 
                TargetDevice.RPI_HAT
            ],
            "smart_speaker": [
                TargetDevice.SMART_SPEAKER_BUDGET,
                TargetDevice.SMART_SPEAKER_MID,
                TargetDevice.SMART_SPEAKER_PREMIUM,
            ],
            "mobile": [
                TargetDevice.MOBILE_IOS, 
                TargetDevice.MOBILE_ANDROID
            ],
            "embedded": [
                TargetDevice.JETSON_NANO, 
                TargetDevice.CORAL_DEV
            ],
            "generic": [
                TargetDevice.LOW_POWER, 
                TargetDevice.FAR_FIELD, 
                TargetDevice.NEAR_FIELD
            ],
        }
        
        devices = category_devices.get(category, [])
        return [self._presets[d] for d in devices]
    
    def get_augmentation_config(
        self, 
        device: TargetDevice,
        include_all: bool = False
    ) -> Dict[str, Any]:
        """
        Get a complete augmentation configuration for a device.
        
        This returns a configuration dict that can be passed directly
        to the augmentation pipeline.
        
        Args:
            device: Target device.
            include_all: If True, include all augmentation types.
                        If False, only include priority augmentations.
                        
        Returns:
            Configuration dict for augmentation pipeline.
        """
        preset = self.get_preset(device)
        
        config = {
            "microphone": preset.microphone_filter,
            "room": preset.room_simulation,
            "noise": preset.noise_mixing,
            "degradation": preset.degradation_effects,
        }
        
        if preset.telephony_effects.get("enabled", False):
            config["telephony"] = preset.telephony_effects
        
        if not include_all:
            # Filter to only priority augmentations
            priority_keys = {
                "microphone_simulation": "microphone",
                "room_simulation": "room",
                "noise_mixing": "noise",
                "degradation": "degradation",
                "telephony_simulation": "telephony",
                "distance_simulation": "room",  # Part of room simulation
            }
            
            active_keys = set()
            for aug in preset.priority_augmentations:
                if aug in priority_keys:
                    active_keys.add(priority_keys[aug])
            
            config = {k: v for k, v in config.items() if k in active_keys}
        
        return config


# Global instance for easy access
device_preset_manager = DevicePresetManager()


def get_device_preset(device: TargetDevice | str) -> DevicePreset:
    """
    Convenience function to get a device preset.
    
    Args:
        device: TargetDevice enum or device name string.
        
    Returns:
        DevicePreset for the specified device.
    """
    if isinstance(device, str):
        return device_preset_manager.get_preset_by_name(device)
    return device_preset_manager.get_preset(device)


def list_device_presets() -> List[Dict[str, Any]]:
    """
    Convenience function to list all device presets.
    
    Returns:
        List of preset summaries.
    """
    return device_preset_manager.list_presets()
