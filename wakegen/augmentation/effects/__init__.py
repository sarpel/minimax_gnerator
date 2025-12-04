"""
WakeGen Augmentation Effects Module

This module provides various audio effects for time domain manipulation,
dynamics processing, and quality degradation.
"""

from .time_domain import TimeDomainEffects
from .dynamics import DynamicsProcessor
from .degradation import AudioDegrader

__all__ = [
    "TimeDomainEffects",
    "DynamicsProcessor",
    "AudioDegrader"
]