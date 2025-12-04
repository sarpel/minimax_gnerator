from __future__ import annotations
import logging
from typing import Dict, Any
from .model_tester import test_model

# We use 'logging' to print messages to the console in a structured way.
logger = logging.getLogger(__name__)

async def compare_models(
    model_a_path: str,
    model_b_path: str,
    test_data_path: str,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compares two trained models (A and B) on the same test dataset.

    This is useful for A/B testing:
    - Model A might be trained on Dataset Version 1.
    - Model B might be trained on Dataset Version 2 (e.g., with more noise).
    - We want to know which one performs better.

    Args:
        model_a_path: Path to the first model file.
        model_b_path: Path to the second model file.
        test_data_path: Path to the test.json manifest file.
        threshold: The detection threshold to use for both models.

    Returns:
        A dictionary summarizing the comparison results.
    """
    logger.info(f"Comparing Model A ({model_a_path}) vs Model B ({model_b_path})")

    # 1. Test Model A
    logger.info("Testing Model A...")
    metrics_a = await test_model(model_a_path, test_data_path, threshold)

    # 2. Test Model B
    logger.info("Testing Model B...")
    metrics_b = await test_model(model_b_path, test_data_path, threshold)

    # 3. Compare Results
    comparison = {
        "model_a": {
            "path": model_a_path,
            "metrics": metrics_a
        },
        "model_b": {
            "path": model_b_path,
            "metrics": metrics_b
        },
        "diff": {
            "accuracy": metrics_b["accuracy"] - metrics_a["accuracy"],
            "f1_score": metrics_b["f1_score"] - metrics_a["f1_score"]
        },
        "winner": "tie"
    }

    # Determine the winner based on F1 score (usually the best single metric)
    diff_f1 = comparison["diff"]["f1_score"]
    if diff_f1 > 0:
        comparison["winner"] = "model_b"
        logger.info("Model B is better!")
    elif diff_f1 < 0:
        comparison["winner"] = "model_a"
        logger.info("Model A is better!")
    else:
        logger.info("Models performed equally.")

    return comparison