"""
WakeGen Augmentation Effects Module

This module provides various audio effects for time domain manipulation,
dynamics processing, quality degradation, and environment simulation.
"""

from .degradation import AudioDegrader
from .dynamics import DynamicsProcessor
from .telephony import (
    DistanceConfig,
    DistanceSimulator,
    PhoneType,
    TelephonyConfig,
    TelephonySimulator,
)
from .time_domain import TimeDomainEffects

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
