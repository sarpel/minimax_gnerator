from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import openwakeword
from openwakeword.model import Model
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # type: ignore

# We use 'logging' to print messages to the console in a structured way.
logger = logging.getLogger(__name__)

async def test_model(
    model_path: str,
    test_data_path: str,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Tests a trained OpenWakeWord model against a test dataset.

    We calculate standard metrics to see how well the model performs:
    - Accuracy: Overall correctness.
    - Precision: When it says "wake word", how often is it right?
    - Recall: When the wake word is spoken, how often does it catch it?
    - F1 Score: A balance between Precision and Recall.

    Args:
        model_path: Path to the trained .onnx or .tflite model file.
        test_data_path: Path to the test.json manifest file.
        threshold: The confidence score (0.0 to 1.0) above which we consider it a detection.

    Returns:
        A dictionary containing the calculated metrics.
    """
    model_file = Path(model_path)
    test_manifest = Path(test_data_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not test_manifest.exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    logger.info(f"Loading model from {model_path}...")
    # Load the model using OpenWakeWord's Model class
    # inference_framework="onnx" is usually the default and fastest
    oww_model = Model(wakeword_models=[str(model_file)], inference_framework="onnx")

    logger.info(f"Loading test data from {test_data_path}...")
    with open(test_manifest, "r", encoding="utf-8") as f:
        test_items: List[Dict[str, Any]] = json.load(f)

    true_labels = []
    predicted_labels = []

    logger.info(f"Testing on {len(test_items)} samples...")

    for item in test_items:
        audio_path = item["path"]
        true_label = item["label"]
        
        # OpenWakeWord expects a path to a file or a numpy array
        # predict() returns a dictionary of scores for each model loaded
        # We only loaded one model, so we get that one's score
        
        # Note: predict() processes the whole file in chunks.
        # We want to know if the wake word was detected *at any point* in the file.
        predictions = oww_model.predict(audio_path)
        
        # predictions is a list of dicts (one per chunk)
        # We look for the maximum score across all chunks
        max_score = 0.0
        model_key = list(predictions[0].keys())[0] # Get the model name key
        
        for frame_pred in predictions:
            score = frame_pred[model_key]
            if score > max_score:
                max_score = score
        
        # Reset the model state between files so previous audio doesn't affect the next one
        oww_model.reset()

        # If the max score is above our threshold, we predict "1" (wake word detected)
        predicted_label = 1 if max_score >= threshold else 0
        
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    # Calculate metrics using scikit-learn
    metrics = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision": float(precision_score(true_labels, predicted_labels, zero_division=0)),
        "recall": float(recall_score(true_labels, predicted_labels, zero_division=0)),
        "f1_score": float(f1_score(true_labels, predicted_labels, zero_division=0))
    }

    logger.info("Test Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return metrics