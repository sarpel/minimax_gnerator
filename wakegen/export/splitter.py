from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from sklearn.model_selection import train_test_split # type: ignore

# We use 'logging' to print messages to the console in a structured way.
logger = logging.getLogger(__name__)

async def split_dataset(
    export_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> None:
    """
    Splits the exported dataset into training, validation, and test sets.

    We use 'stratified splitting' to ensure that each set has roughly the same
    proportion of positive (wake word) and negative (background) samples.
    This prevents the model from learning biases (e.g., if the test set had
    only negative samples, the model might just guess "negative" every time).

    Args:
        export_dir: The directory containing the exported dataset (and manifest.json).
        train_ratio: The proportion of data to use for training (default 80%).
        val_ratio: The proportion of data to use for validation (default 10%).
        test_ratio: The proportion of data to use for testing (default 10%).
        seed: A random seed for reproducibility (so we get the same split every time).
    """
    export_path = Path(export_dir)
    manifest_path = export_path / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found at {manifest_path}")

    # 1. Load the manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    logger.info(f"Loaded {len(data)} samples from manifest")

    # 2. Prepare data for splitting
    # We need parallel lists of items and their labels for scikit-learn
    items = data
    labels = [item["label"] for item in data]

    # 3. Perform the split
    # First, we split off the test set
    # The 'stratify=labels' argument is the magic part that keeps the balance correct.
    train_val_items, test_items, train_val_labels, test_labels = train_test_split(
        items, labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels
    )

    # Now we split the remaining data into training and validation
    # We need to adjust the validation ratio because the total size has shrunk
    # Example: If we had 100 items, took 10 for test. Now we have 90.
    # We want 10 items for val (which was 10% of original).
    # So 10 / 90 = 0.111...
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)

    train_items, val_items, train_labels, val_labels = train_test_split(
        train_val_items, train_val_labels,
        test_size=adjusted_val_ratio,
        random_state=seed,
        stratify=train_val_labels
    )

    logger.info(f"Split results: Train={len(train_items)}, Val={len(val_items)}, Test={len(test_items)}")

    # 4. Save the new manifests
    # We create separate JSON files for each set
    _save_manifest(export_path / "train.json", train_items)
    _save_manifest(export_path / "val.json", val_items)
    _save_manifest(export_path / "test.json", test_items)

def _save_manifest(path: Path, data: List[Dict[str, Any]]) -> None:
    """
    Helper function to save a list of items to a JSON file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved manifest to {path}")