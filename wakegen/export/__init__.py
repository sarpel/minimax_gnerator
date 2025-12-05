"""
Export module for wakegen.

This module handles exporting generated datasets to various formats
and splitting them into train/val/test sets.

Supported formats:
- OpenWakeWord
- Mycroft Precise
- Picovoice
- TensorFlow/Keras
- PyTorch
- Hugging Face datasets
"""

# Core exports
from .openwakeword import export_dataset
from .splitter import split_dataset
from .manifest import generate_manifest

# Format exports
from .formats import (
    ExportFormat,
    DatasetMetadata,
    SampleMetadata,
    BaseExporter,
    MycroftPreciseExporter,
    PicovoiceExporter,
    TensorFlowExporter,
    PyTorchExporter,
    HuggingFaceExporter,
    export_to_format,
    list_export_formats,
)

__all__ = [
    # Core
    "export_dataset",
    "split_dataset",
    "generate_manifest",
    # Format types
    "ExportFormat",
    # Metadata
    "DatasetMetadata",
    "SampleMetadata",
    # Exporters
    "BaseExporter",
    "MycroftPreciseExporter",
    "PicovoiceExporter",
    "TensorFlowExporter",
    "PyTorchExporter",
    "HuggingFaceExporter",
    # Functions
    "export_to_format",
    "list_export_formats",
]