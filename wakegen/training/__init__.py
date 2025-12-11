"""
Training module for wakegen.

This module handles generating training scripts and testing trained models.
"""

# We expose the main functions here so they can be imported easily.
# For example: from wakegen.training import generate_training_script

from .ab_comparison import compare_models
from .model_tester import test_model
from .script_generator import generate_training_script

__all__ = ["compare_models", "generate_training_script", "test_model"]
