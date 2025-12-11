"""
WakeGen Augmentation Room Module

This module provides room simulation capabilities for creating
realistic reverberation and acoustic effects.
"""

from .convolver import RoomConvolver
from .simulator import RoomParameters, RoomSimulator

__all__ = ["RoomConvolver", "RoomParameters", "RoomSimulator"]
