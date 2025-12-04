from __future__ import annotations
import os
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# We use 'logging' to print messages to the console in a structured way.
logger = logging.getLogger(__name__)

async def export_dataset(
    source_dir: str,
    output_dir: str,
    wake_word: str,
    negative_samples_dir: str | None = None
) -> None:
    """
    Exports a generated dataset to the OpenWakeWord training format.

    OpenWakeWord expects a specific folder structure or a JSON manifest
    that points to the audio files. We will create a structured folder
    and a JSON manifest.

    Args:
        source_dir: The directory containing the generated wake word audio files.
        output_dir: The directory where the exported dataset will be saved.
        wake_word: The name of the wake word (e.g., "hey_katya").
        negative_samples_dir: Optional directory containing negative samples (not the wake word).
    """
    # Convert strings to Path objects for easier file manipulation
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create the output directory if it doesn't exist
    # parents=True means create parent directories if needed (like mkdir -p)
    # exist_ok=True means don't crash if the directory already exists
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting dataset from {source_path} to {output_path}")

    # 1. Create the folder structure
    # We'll have two main folders: 'positive' (wake words) and 'negative' (other sounds)
    positive_dir = output_path / "positive"
    negative_dir = output_path / "negative"
    
    positive_dir.mkdir(exist_ok=True)
    negative_dir.mkdir(exist_ok=True)

    # 2. Copy positive samples (the wake words)
    # We look for all .wav files in the source directory
    positive_files = list(source_path.glob("*.wav"))
    logger.info(f"Found {len(positive_files)} positive samples")

    manifest_data: List[Dict[str, Any]] = []

    for file_path in positive_files:
        # We copy each file to the new 'positive' folder
        # shutil.copy2 preserves file metadata (like creation time)
        dest_path = positive_dir / file_path.name
        shutil.copy2(file_path, dest_path)
        
        # Add to manifest data
        # OpenWakeWord often uses a JSON format where each entry describes a file
        manifest_data.append({
            "path": str(dest_path.absolute()),
            "label": 1,  # 1 means "this IS the wake word"
            "transcript": wake_word
        })

    # 3. Copy negative samples (if provided)
    if negative_samples_dir:
        neg_source_path = Path(negative_samples_dir)
        negative_files = list(neg_source_path.glob("*.wav"))
        logger.info(f"Found {len(negative_files)} negative samples")

        for file_path in negative_files:
            dest_path = negative_dir / file_path.name
            shutil.copy2(file_path, dest_path)
            
            manifest_data.append({
                "path": str(dest_path.absolute()),
                "label": 0,  # 0 means "this is NOT the wake word"
                "transcript": "unknown"
            })
    else:
        logger.warning("No negative samples directory provided. Training might be unbalanced.")

    # 4. Create the JSON manifest file
    # This file lists all the audio files and their labels
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        # indent=2 makes the JSON file human-readable (pretty-printed)
        json.dump(manifest_data, f, indent=2)

    logger.info(f"Export complete. Manifest saved to {manifest_path}")