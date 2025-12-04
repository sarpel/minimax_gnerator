from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# We use 'logging' to print messages to the console in a structured way.
logger = logging.getLogger(__name__)

async def generate_manifest(
    directory: str,
    output_file: str,
    include_metadata: bool = True
) -> None:
    """
    Generates a generic JSON manifest for a directory of audio files.

    This is useful for keeping track of what files are in a dataset,
    their sizes, and other properties, independent of any specific training format.

    Args:
        directory: The directory to scan for audio files.
        output_file: The path where the JSON manifest will be saved.
        include_metadata: Whether to include file size and other details.
    """
    dir_path = Path(directory)
    output_path = Path(output_file)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    logger.info(f"Scanning {dir_path} for audio files...")

    # We look for common audio extensions
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg"}
    files_data: List[Dict[str, Any]] = []

    # rglob('*') searches recursively (in all subfolders)
    for file_path in dir_path.rglob("*"):
        if file_path.suffix.lower() in audio_extensions:
            file_info = {
                "path": str(file_path.absolute()),
                "filename": file_path.name,
                "relative_path": str(file_path.relative_to(dir_path))
            }

            if include_metadata:
                # stat() gives us file information like size
                stats = file_path.stat()
                file_info["size_bytes"] = stats.st_size
                # We could add duration here if we opened the file with soundfile/librosa,
                # but that would be slow for large datasets.

            files_data.append(file_info)

    logger.info(f"Found {len(files_data)} audio files.")

    # Save the list to a JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(files_data, f, indent=2)

    logger.info(f"Manifest saved to {output_path}")