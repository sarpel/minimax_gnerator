"""
WakeGen Augmentation Room Module

This module provides room simulation capabilities for creating
realistic reverberation and acoustic effects.
"""

from .simulator import RoomSimulator, RoomParameters
from .convolver import RoomConvolver

__all__ = [
    "RoomSimulator",
    "RoomParameters",
    "RoomConvolver"
]