"""
WakeGen Augmentation Microphone Module

This module provides microphone simulation capabilities for creating
realistic microphone frequency response and distortion effects.
"""

from .simulator import MicrophoneProfile, MicrophoneSimulator

__all__ = ["MicrophoneProfile", "MicrophoneSimulator"]
