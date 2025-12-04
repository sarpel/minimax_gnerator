"""
Noise Profiles

This module defines pre-configured noise profiles for different
environmental scenarios. Each profile specifies the type of noise,
SNR levels, and event characteristics that are typical for that environment.

Key Features:
- Pre-built profiles for common scenarios
- Configurable noise types and SNR ranges
- Environment-specific event configurations
- Easy extension for new scenarios
"""

from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass
from wakegen.core.types import EnvironmentProfile
from wakegen.core.exceptions import NoiseError

@dataclass
class NoiseProfile:
    """
    Defines the noise characteristics for a specific environment.

    Attributes:
        name: Human-readable name of the profile.
        description: Description of the environment.
        base_noise_type: Primary noise type ('white', 'pink', 'brown').
        snr_range: Typical SNR range for this environment (min, max).
        event_density: Average events per minute.
        typical_events: List of event types that commonly occur.
        intensity_range: Typical intensity range for events.
    """
    name: str
    description: str
    base_noise_type: str
    snr_range: tuple[float, float]
    event_density: float
    typical_events: List[str]
    intensity_range: tuple[float, float]

class NoiseProfileManager:
    """
    Manages a collection of noise profiles for different environments.

    Provides easy access to pre-configured profiles and allows
    custom profile creation.
    """

    def __init__(self):
        """Initialize with default profiles."""
        self.profiles = self._create_default_profiles()

    def _create_default_profiles(self) -> Dict[EnvironmentProfile, NoiseProfile]:
        """
        Create the default set of noise profiles for common environments.
        """
        return {
            EnvironmentProfile.MORNING_KITCHEN: NoiseProfile(
                name="Morning Kitchen",
                description="Typical morning kitchen with breakfast preparation",
                base_noise_type="pink",
                snr_range=(12.0, 20.0),
                event_density=0.8,
                typical_events=["dish_clink", "water_running", "chair_scrape"],
                intensity_range=(0.4, 0.7)
            ),
            EnvironmentProfile.EVENING_LIVING_ROOM: NoiseProfile(
                name="Evening Living Room",
                description="Quiet evening in living room with occasional activity",
                base_noise_type="brown",
                snr_range=(15.0, 25.0),
                event_density=0.3,
                typical_events=["footstep", "chair_scrape"],
                intensity_range=(0.3, 0.5)
            ),
            EnvironmentProfile.OFFICE_SPACE: NoiseProfile(
                name="Office Space",
                description="Typical office environment with background activity",
                base_noise_type="white",
                snr_range=(10.0, 18.0),
                event_density=0.5,
                typical_events=["chair_scrape", "door_close"],
                intensity_range=(0.3, 0.6)
            ),
            EnvironmentProfile.CAR_INTERIOR: NoiseProfile(
                name="Car Interior",
                description="Inside a moving car with road and engine noise",
                base_noise_type="pink",
                snr_range=(8.0, 15.0),
                event_density=0.2,
                typical_events=["door_close"],
                intensity_range=(0.5, 0.8)
            ),
            EnvironmentProfile.OUTDOOR_PARK: NoiseProfile(
                name="Outdoor Park",
                description="Outdoor environment with natural sounds",
                base_noise_type="brown",
                snr_range=(5.0, 12.0),
                event_density=0.1,
                typical_events=["footstep"],
                intensity_range=(0.2, 0.4)
            ),
            EnvironmentProfile.BEDROOM_NIGHT: NoiseProfile(
                name="Bedroom at Night",
                description="Very quiet bedroom environment at night",
                base_noise_type="white",
                snr_range=(20.0, 30.0),
                event_density=0.1,
                typical_events=[],
                intensity_range=(0.1, 0.3)
            )
        }

    def get_profile(self, profile_id: EnvironmentProfile) -> NoiseProfile:
        """
        Get a noise profile by its environment profile ID.

        Args:
            profile_id: The EnvironmentProfile enum value.

        Returns:
            The corresponding NoiseProfile.

        Raises:
            NoiseError: If the profile is not found.
        """
        if profile_id not in self.profiles:
            raise NoiseError(f"Noise profile not found for environment: {profile_id}")
        return self.profiles[profile_id]

    def get_random_snr(self, profile_id: EnvironmentProfile) -> float:
        """
        Get a random SNR value within the range for a given profile.

        Args:
            profile_id: The EnvironmentProfile enum value.

        Returns:
            Random SNR value in dB.

        Raises:
            NoiseError: If the profile is not found.
        """
        profile = self.get_profile(profile_id)
        return random.uniform(*profile.snr_range)

    def create_custom_profile(
        self,
        name: str,
        description: str,
        base_noise_type: str,
        snr_range: tuple[float, float],
        event_density: float,
        typical_events: List[str],
        intensity_range: tuple[float, float]
    ) -> EnvironmentProfile:
        """
        Create a custom noise profile and add it to the manager.

        Args:
            name: Human-readable name.
            description: Description of the environment.
            base_noise_type: Primary noise type.
            snr_range: SNR range (min, max).
            event_density: Events per minute.
            typical_events: List of event types.
            intensity_range: Intensity range.

        Returns:
            The EnvironmentProfile enum value for the new profile.

        Raises:
            NoiseError: If parameters are invalid.
        """
        # Validate parameters
        if snr_range[0] > snr_range[1]:
            raise NoiseError("SNR range min must be <= max")
        if event_density < 0:
            raise NoiseError("Event density cannot be negative")
        if intensity_range[0] > intensity_range[1]:
            raise NoiseError("Intensity range min must be <= max")

        # Create a new EnvironmentProfile (this is a simplified approach)
        # In a real implementation, we'd need to extend the Enum dynamically
        # For now, we'll use a string-based approach
        profile_id = EnvironmentProfile(f"custom_{name.lower().replace(' ', '_')}")

        # Add to profiles
        self.profiles[profile_id] = NoiseProfile(
            name=name,
            description=description,
            base_noise_type=base_noise_type,
            snr_range=snr_range,
            event_density=event_density,
            typical_events=typical_events,
            intensity_range=intensity_range
        )

        return profile_id

    def list_profiles(self) -> List[Dict[str, Any]]:
        """
        List all available noise profiles with their key characteristics.

        Returns:
            List of profile information dictionaries.
        """
        result = []
        for profile_id, profile in self.profiles.items():
            result.append({
                "id": str(profile_id),
                "name": profile.name,
                "description": profile.description,
                "base_noise": profile.base_noise_type,
                "snr_range": profile.snr_range,
                "event_density": profile.event_density,
                "typical_events": profile.typical_events
            })
        return result

# Global instance for easy access
noise_profile_manager = NoiseProfileManager()