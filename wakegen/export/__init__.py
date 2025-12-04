"""
Export module for wakegen.

This module handles exporting generated datasets to various formats
(like OpenWakeWord) and splitting them into train/val/test sets.
"""

# We expose the main functions here so they can be imported easily.
# For example: from wakegen.export import export_to_openwakeword
# Instead of: from wakegen.export.openwakeword import export_to_openwakeword

from .openwakeword import export_dataset
from .splitter import split_dataset
from .manifest import generate_manifest

__all__ = ["export_dataset", "split_dataset", "generate_manifest"]