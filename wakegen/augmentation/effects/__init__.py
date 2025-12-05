"""
WakeGen Augmentation Effects Module

This module provides various audio effects for time domain manipulation,
dynamics processing, quality degradation, and environment simulation.
"""

from .time_domain import TimeDomainEffects
from .dynamics import DynamicsProcessor
from .degradation import AudioDegrader
from .telephony import (
    PhoneType,
    TelephonyConfig,
    TelephonySimulator,
    DistanceConfig,
    DistanceSimulator,
)

__all__ = [
    # Core effects
    "TimeDomainEffects",
    "DynamicsProcessor",
    "AudioDegrader",
    # Telephony simulation
    "PhoneType",
    "TelephonyConfig",
    "TelephonySimulator",
    # Distance simulation
    "DistanceConfig",
    "DistanceSimulator",
]