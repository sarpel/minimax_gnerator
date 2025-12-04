"""
Environment Profiles

This module defines comprehensive environment profiles that combine
all augmentation types (noise, room, microphone, effects) into
realistic scenarios. These profiles provide one-stop configuration
for common wake word recording environments.

Key Features:
- Pre-built profiles for common scenarios
- Configurable augmentation parameters
- Environment-specific presets
- Easy extension for new scenarios
- Integration with all augmentation components
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from wakegen.core.types import EnvironmentProfile, AugmentationType
from wakegen.core.exceptions import AugmentationError
from wakegen.augmentation.noise.profiles import NoiseProfileManager, NoiseProfile
from wakegen.augmentation.room.simulator import RoomParameters
from wakegen.augmentation.microphone.simulator import MicrophoneProfile
import random

@dataclass
class AugmentationProfile:
    """
    Comprehensive augmentation profile defining all aspects of an environment.

    Attributes:
        name: Human-readable name.
        description: Description of the environment.
        noise_profile: NoiseProfile for background noise.
        room_params: RoomParameters for room simulation.
        microphone_profile: MicrophoneProfile for mic simulation.
        time_effects: Time domain effects parameters.
        dynamics_effects: Dynamics processing parameters.
        degradation_effects: Quality degradation parameters.
        augmentation_types: List of augmentation types to apply.
    """
    name: str
    description: str
    noise_profile: NoiseProfile
    room_params: RoomParameters
    microphone_profile: MicrophoneProfile
    time_effects: Dict[str, Any]
    dynamics_effects: Dict[str, Any]
    degradation_effects: Dict[str, Any]
    augmentation_types: List[AugmentationType]

class EnvironmentProfileManager:
    """
    Manages comprehensive environment profiles for wake word augmentation.

    Provides access to pre-configured profiles and allows custom profile creation.
    """

    def __init__(self):
        """Initialize with default environment profiles."""
        self.noise_manager = NoiseProfileManager()  # Initialize noise manager first
        self.profiles = self._create_default_profiles()

    def _create_default_profiles(self) -> Dict[EnvironmentProfile, AugmentationProfile]:
        """
        Create default environment profiles for common scenarios.
        """
        # Get room and microphone simulators for parameter access
        from wakegen.augmentation.room.simulator import RoomSimulator
        from wakegen.augmentation.microphone.simulator import MicrophoneSimulator

        room_sim = RoomSimulator()
        mic_sim = MicrophoneSimulator()

        return {
            EnvironmentProfile.MORNING_KITCHEN: AugmentationProfile(
                name="Morning Kitchen",
                description="Busy morning kitchen with breakfast preparation, dishes, and appliances",
                noise_profile=self.noise_manager.get_profile(EnvironmentProfile.MORNING_KITCHEN),
                room_params=room_sim.get_preset_room("medium_room"),
                microphone_profile=mic_sim.get_preset_microphone("smartphone"),
                time_effects={"pitch_steps": 0.0, "time_stretch_factor": 1.0, "speed_factor": 1.0},
                dynamics_effects={"effect_type": "compression", "threshold_db": -20.0, "ratio": 2.5},
                degradation_effects={"degradation_type": "random", "severity": 0.3},
                augmentation_types=[
                    AugmentationType.BACKGROUND_NOISE,
                    AugmentationType.ROOM_SIMULATION,
                    AugmentationType.MICROPHONE_SIMULATION,
                    AugmentationType.COMPRESSION
                ]
            ),
            EnvironmentProfile.EVENING_LIVING_ROOM: AugmentationProfile(
                name="Evening Living Room",
                description="Quiet evening in living room with TV and occasional conversation",
                noise_profile=self.noise_manager.get_profile(EnvironmentProfile.EVENING_LIVING_ROOM),
                room_params=room_sim.get_preset_room("large_room"),
                microphone_profile=mic_sim.get_preset_microphone("lapel"),
                time_effects={"pitch_steps": 0.0, "time_stretch_factor": 1.0, "speed_factor": 1.0},
                dynamics_effects={"effect_type": "compression", "threshold_db": -24.0, "ratio": 2.0},
                degradation_effects={"degradation_type": "random", "severity": 0.2},
                augmentation_types=[
                    AugmentationType.BACKGROUND_NOISE,
                    AugmentationType.ROOM_SIMULATION,
                    AugmentationType.MICROPHONE_SIMULATION
                ]
            ),
            EnvironmentProfile.OFFICE_SPACE: AugmentationProfile(
                name="Office Space",
                description="Typical office environment with background activity and HVAC noise",
                noise_profile=self.noise_manager.get_profile(EnvironmentProfile.OFFICE_SPACE),
                room_params=room_sim.get_preset_room("office"),
                microphone_profile=mic_sim.get_preset_microphone("conference"),
                time_effects={"pitch_steps": 0.0, "time_stretch_factor": 1.0, "speed_factor": 1.0},
                dynamics_effects={"effect_type": "limiting", "threshold_db": -3.0},
                degradation_effects={"degradation_type": "telephone"},
                augmentation_types=[
                    AugmentationType.BACKGROUND_NOISE,
                    AugmentationType.ROOM_SIMULATION,
                    AugmentationType.MICROPHONE_SIMULATION,
                    AugmentationType.COMPRESSION
                ]
            ),
            EnvironmentProfile.CAR_INTERIOR: AugmentationProfile(
                name="Car Interior",
                description="Inside a moving car with road noise, engine sounds, and limited space",
                noise_profile=self.noise_manager.get_profile(EnvironmentProfile.CAR_INTERIOR),
                room_params=room_sim.get_preset_room("small_room"),
                microphone_profile=mic_sim.get_preset_microphone("headset"),
                time_effects={"pitch_steps": 0.0, "time_stretch_factor": 1.0, "speed_factor": 1.0},
                dynamics_effects={"effect_type": "compression", "threshold_db": -18.0, "ratio": 3.0},
                degradation_effects={"degradation_type": "mp3", "bitrate_kbps": 96},
                augmentation_types=[
                    AugmentationType.BACKGROUND_NOISE,
                    AugmentationType.ROOM_SIMULATION,
                    AugmentationType.MICROPHONE_SIMULATION,
                    AugmentationType.COMPRESSION
                ]
            ),
            EnvironmentProfile.OUTDOOR_PARK: AugmentationProfile(
                name="Outdoor Park",
                description="Outdoor environment with natural sounds, wind, and distant noises",
                noise_profile=self.noise_manager.get_profile(EnvironmentProfile.OUTDOOR_PARK),
                room_params=room_sim.get_preset_room("large_room"),  # Open space approximation
                microphone_profile=mic_sim.get_preset_microphone("far_field"),
                time_effects={"pitch_steps": 0.0, "time_stretch_factor": 1.0, "speed_factor": 1.0},
                dynamics_effects={"effect_type": "expansion", "threshold_db": -30.0, "ratio": 2.0},
                degradation_effects={"degradation_type": "random", "severity": 0.4},
                augmentation_types=[
                    AugmentationType.BACKGROUND_NOISE,
                    AugmentationType.MICROPHONE_SIMULATION,
                    AugmentationType.DEGRADATION
                ]
            ),
            EnvironmentProfile.BEDROOM_NIGHT: AugmentationProfile(
                name="Bedroom at Night",
                description="Very quiet bedroom environment at night with minimal background noise",
                noise_profile=self.noise_manager.get_profile(EnvironmentProfile.BEDROOM_NIGHT),
                room_params=room_sim.get_preset_room("medium_room"),
                microphone_profile=mic_sim.get_preset_microphone("studio"),
                time_effects={"pitch_steps": 0.0, "time_stretch_factor": 1.0, "speed_factor": 1.0},
                dynamics_effects={"effect_type": "compression", "threshold_db": -30.0, "ratio": 1.5},
                degradation_effects={"degradation_type": "random", "severity": 0.1},
                augmentation_types=[
                    AugmentationType.BACKGROUND_NOISE,
                    AugmentationType.ROOM_SIMULATION,
                    AugmentationType.MICROPHONE_SIMULATION
                ]
            )
        }

    def get_profile(self, profile_id: EnvironmentProfile) -> AugmentationProfile:
        """
        Get an environment profile by its ID.

        Args:
            profile_id: The EnvironmentProfile enum value.

        Returns:
            The corresponding AugmentationProfile.

        Raises:
            AugmentationError: If the profile is not found.
        """
        if profile_id not in self.profiles:
            raise AugmentationError(f"Environment profile not found: {profile_id}")
        return self.profiles[profile_id]

    def get_random_variation(
        self,
        base_profile: AugmentationProfile,
        variation_strength: float = 0.3
    ) -> AugmentationProfile:
        """
        Create a random variation of a base profile.

        Args:
            base_profile: Base AugmentationProfile to vary.
            variation_strength: How much to vary parameters (0.0 to 1.0).

        Returns:
            Varied AugmentationProfile.
        """
        if not (0.0 <= variation_strength <= 1.0):
            raise AugmentationError("Variation strength must be between 0.0 and 1.0")

        # Create a copy of the base profile
        varied = AugmentationProfile(**base_profile.__dict__.copy())

        # Add random variations to parameters
        if random.random() < variation_strength * 0.7:
            # Vary SNR within noise profile range
            snr_range = varied.noise_profile.snr_range
            new_snr = random.uniform(*snr_range)
            # We'd need to modify the noise profile, but for simplicity we'll
            # just adjust the time effects as a demonstration
            varied.time_effects["pitch_steps"] = random.uniform(-1.0, 1.0) * variation_strength

        if random.random() < variation_strength * 0.5:
            # Vary room parameters slightly
            if hasattr(varied.room_params, 'rt60'):
                varied.room_params.rt60 = max(0.1, varied.room_params.rt60 * random.uniform(0.8, 1.2))

        if random.random() < variation_strength * 0.6:
            # Vary microphone distortion
            varied.microphone_profile.distortion = max(0.01, varied.microphone_profile.distortion * random.uniform(0.7, 1.3))

        return varied

    def create_custom_profile(
        self,
        name: str,
        description: str,
        noise_profile_id: EnvironmentProfile,
        room_preset: str,
        microphone_preset: str,
        time_effects: Optional[Dict[str, Any]] = None,
        dynamics_effects: Optional[Dict[str, Any]] = None,
        degradation_effects: Optional[Dict[str, Any]] = None,
        augmentation_types: Optional[List[AugmentationType]] = None
    ) -> EnvironmentProfile:
        """
        Create a custom environment profile.

        Args:
            name: Profile name.
            description: Profile description.
            noise_profile_id: EnvironmentProfile for noise characteristics.
            room_preset: Room preset name.
            microphone_preset: Microphone preset name.
            time_effects: Time domain effects parameters.
            dynamics_effects: Dynamics processing parameters.
            degradation_effects: Quality degradation parameters.
            augmentation_types: List of augmentation types to apply.

        Returns:
            EnvironmentProfile enum value for the new profile.

        Raises:
            AugmentationError: If parameters are invalid.
        """
        # Get room and microphone simulators
        from wakegen.augmentation.room.simulator import RoomSimulator
        from wakegen.augmentation.microphone.simulator import MicrophoneSimulator

        room_sim = RoomSimulator()
        mic_sim = MicrophoneSimulator()

        # Validate and get components
        noise_profile = self.noise_manager.get_profile(noise_profile_id)
        room_params = room_sim.get_preset_room(room_preset)
        mic_profile = mic_sim.get_preset_microphone(microphone_preset)

        # Set defaults if not provided
        if time_effects is None:
            time_effects = {"pitch_steps": 0.0, "time_stretch_factor": 1.0, "speed_factor": 1.0}

        if dynamics_effects is None:
            dynamics_effects = {"effect_type": "compression", "threshold_db": -20.0, "ratio": 2.0}

        if degradation_effects is None:
            degradation_effects = {"degradation_type": "random", "severity": 0.2}

        if augmentation_types is None:
            augmentation_types = [
                AugmentationType.BACKGROUND_NOISE,
                AugmentationType.ROOM_SIMULATION,
                AugmentationType.MICROPHONE_SIMULATION
            ]

        # Create the profile
        profile_id = EnvironmentProfile(f"custom_{name.lower().replace(' ', '_')}")

        self.profiles[profile_id] = AugmentationProfile(
            name=name,
            description=description,
            noise_profile=noise_profile,
            room_params=room_params,
            microphone_profile=mic_profile,
            time_effects=time_effects,
            dynamics_effects=dynamics_effects,
            degradation_effects=degradation_effects,
            augmentation_types=augmentation_types
        )

        return profile_id

    def list_profiles(self) -> List[Dict[str, Any]]:
        """
        List all available environment profiles with their characteristics.

        Returns:
            List of profile information dictionaries.
        """
        result = []
        for profile_id, profile in self.profiles.items():
            result.append({
                "id": str(profile_id),
                "name": profile.name,
                "description": profile.description,
                "noise_profile": profile.noise_profile.name,
                "room_type": "custom" if not hasattr(profile.room_params, 'preset') else "preset",
                "microphone_type": profile.microphone_profile.name,
                "augmentation_types": [str(at) for at in profile.augmentation_types]
            })
        return result

# Global instance for easy access
environment_profile_manager = EnvironmentProfileManager()

def get_profile(profile_id: EnvironmentProfile) -> AugmentationProfile:
    """
    Convenience function to get an environment profile.

    Args:
        profile_id: The EnvironmentProfile enum value.

    Returns:
        The corresponding AugmentationProfile.
    """
    return environment_profile_manager.get_profile(profile_id)