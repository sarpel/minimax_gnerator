"""
Export Formats Module

This module provides exporters for various wake word training formats:
- OpenWakeWord (existing)
- Mycroft Precise
- Picovoice
- TensorFlow/Keras
- PyTorch
- Hugging Face datasets

Each exporter creates the appropriate directory structure and metadata
files required by the target framework.
"""

from __future__ import annotations
import json
import csv
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats."""
    
    OPENWAKEWORD = "openwakeword"
    MYCROFT_PRECISE = "mycroft_precise"
    PICOVOICE = "picovoice"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    HUGGINGFACE = "huggingface"


@dataclass
class DatasetMetadata:
    """Comprehensive metadata for exported datasets."""
    
    name: str
    wake_word: str
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Sample counts
    total_samples: int = 0
    positive_samples: int = 0
    negative_samples: int = 0
    
    # Split information
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    
    # Audio specifications
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    format: str = "wav"
    
    # Generation info
    providers_used: List[str] = field(default_factory=list)
    voices_used: List[str] = field(default_factory=list)
    augmentations_applied: List[str] = field(default_factory=list)
    
    # Quality metrics
    average_duration_ms: float = 0.0
    duration_std_ms: float = 0.0
    average_snr_db: Optional[float] = None
    
    # Additional info
    description: str = ""
    license: str = "MIT"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, path: Path) -> None:
        """Save metadata to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: Path) -> "DatasetMetadata":
        """Load metadata from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class SampleMetadata:
    """Metadata for individual audio samples."""
    
    filename: str
    label: int  # 1 for positive, 0 for negative
    transcript: str
    
    # Source information
    provider: Optional[str] = None
    voice_id: Optional[str] = None
    
    # Audio properties
    duration_ms: Optional[float] = None
    sample_rate: Optional[int] = None
    
    # Augmentation info
    augmentations: List[str] = field(default_factory=list)
    augmentation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Quality info
    snr_db: Optional[float] = None
    quality_score: Optional[float] = None
    
    # Split assignment
    split: Optional[str] = None  # "train", "val", "test"


class BaseExporter:
    """Base class for dataset exporters."""
    
    format_name: str = "base"
    
    def __init__(
        self,
        source_dir: Path | str,
        output_dir: Path | str,
        wake_word: str,
        metadata: Optional[DatasetMetadata] = None,
    ) -> None:
        """
        Initialize the exporter.
        
        Args:
            source_dir: Directory containing generated audio files.
            output_dir: Directory for exported dataset.
            wake_word: The wake word being trained.
            metadata: Optional dataset metadata.
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.wake_word = wake_word
        self.metadata = metadata or DatasetMetadata(
            name=wake_word.replace(" ", "_"),
            wake_word=wake_word,
        )
    
    async def export(
        self,
        negative_samples_dir: Optional[Path | str] = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> Path:
        """
        Export the dataset.
        
        Args:
            negative_samples_dir: Optional directory with negative samples.
            split_ratios: Train/val/test split ratios.
            
        Returns:
            Path to the exported dataset.
        """
        raise NotImplementedError
    
    def _collect_audio_files(self, directory: Path) -> List[Path]:
        """Collect all audio files from a directory."""
        extensions = {".wav", ".mp3", ".flac", ".ogg"}
        files = []
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"**/*{ext}"))
        return sorted(set(files))
    
    def _create_splits(
        self,
        files: List[Path],
        ratios: Tuple[float, float, float],
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """Split files into train/val/test sets."""
        import random
        
        shuffled = files.copy()
        random.shuffle(shuffled)
        
        total = len(shuffled)
        train_end = int(total * ratios[0])
        val_end = train_end + int(total * ratios[1])
        
        return (
            shuffled[:train_end],
            shuffled[train_end:val_end],
            shuffled[val_end:],
        )


class MycroftPreciseExporter(BaseExporter):
    """
    Export to Mycroft Precise format.
    
    Mycroft Precise expects:
    - wake-word/ directory with positive samples
    - not-wake-word/ directory with negative samples
    - Each in the format: wake-word/category/sample.wav
    
    Reference: https://github.com/MycroftAI/mycroft-precise
    """
    
    format_name = "mycroft_precise"
    
    async def export(
        self,
        negative_samples_dir: Optional[Path | str] = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> Path:
        """Export to Mycroft Precise format."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        wake_word_dir = self.output_dir / "wake-word"
        not_wake_word_dir = self.output_dir / "not-wake-word"
        
        wake_word_dir.mkdir(exist_ok=True)
        not_wake_word_dir.mkdir(exist_ok=True)
        
        # Collect positive samples
        positive_files = self._collect_audio_files(self.source_dir)
        train_pos, val_pos, test_pos = self._create_splits(positive_files, split_ratios)
        
        # Create category subdirectories for positive samples
        categories = {
            "train": (wake_word_dir / "train", train_pos),
            "val": (wake_word_dir / "val", val_pos),
            "test": (wake_word_dir / "test", test_pos),
        }
        
        for category, (dest_dir, files) in categories.items():
            dest_dir.mkdir(exist_ok=True)
            for f in files:
                shutil.copy2(f, dest_dir / f.name)
        
        # Handle negative samples
        if negative_samples_dir:
            neg_path = Path(negative_samples_dir)
            negative_files = self._collect_audio_files(neg_path)
            train_neg, val_neg, test_neg = self._create_splits(negative_files, split_ratios)
            
            neg_categories = {
                "train": (not_wake_word_dir / "train", train_neg),
                "val": (not_wake_word_dir / "val", val_neg),
                "test": (not_wake_word_dir / "test", test_neg),
            }
            
            for category, (dest_dir, files) in neg_categories.items():
                dest_dir.mkdir(exist_ok=True)
                for f in files:
                    shutil.copy2(f, dest_dir / f.name)
        
        # Update metadata
        self.metadata.positive_samples = len(positive_files)
        self.metadata.negative_samples = len(negative_files) if negative_samples_dir else 0
        self.metadata.total_samples = self.metadata.positive_samples + self.metadata.negative_samples
        self.metadata.train_samples = len(train_pos) + (len(train_neg) if negative_samples_dir else 0)
        self.metadata.val_samples = len(val_pos) + (len(val_neg) if negative_samples_dir else 0)
        self.metadata.test_samples = len(test_pos) + (len(test_neg) if negative_samples_dir else 0)
        
        # Save metadata
        self.metadata.to_json(self.output_dir / "metadata.json")
        
        # Create Precise-specific config
        config = {
            "model_name": self.wake_word.replace(" ", "_"),
            "wake_word": self.wake_word,
            "sample_rate": self.metadata.sample_rate,
            "window_size": 0.1,  # 100ms window
            "hop_size": 0.05,    # 50ms hop
            "threshold": 0.5,
        }
        
        with open(self.output_dir / "precise_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Exported to Mycroft Precise format at {self.output_dir}")
        return self.output_dir


class PicovoiceExporter(BaseExporter):
    """
    Export to Picovoice format.
    
    Picovoice Porcupine expects audio files in specific format with metadata.
    Creates a dataset compatible with Picovoice Console training.
    
    Reference: https://picovoice.ai/docs/porcupine/
    """
    
    format_name = "picovoice"
    
    async def export(
        self,
        negative_samples_dir: Optional[Path | str] = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> Path:
        """Export to Picovoice format."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Picovoice structure
        audio_dir = self.output_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # Collect and copy positive samples
        positive_files = self._collect_audio_files(self.source_dir)
        train_pos, val_pos, test_pos = self._create_splits(positive_files, split_ratios)
        
        manifest_entries = []
        
        for idx, f in enumerate(positive_files):
            new_name = f"positive_{idx:05d}.wav"
            shutil.copy2(f, audio_dir / new_name)
            
            split = "train" if f in train_pos else ("val" if f in val_pos else "test")
            manifest_entries.append({
                "audio_filepath": f"audio/{new_name}",
                "text": self.wake_word,
                "label": "positive",
                "split": split,
            })
        
        # Handle negative samples
        if negative_samples_dir:
            neg_path = Path(negative_samples_dir)
            negative_files = self._collect_audio_files(neg_path)
            
            for idx, f in enumerate(negative_files):
                new_name = f"negative_{idx:05d}.wav"
                shutil.copy2(f, audio_dir / new_name)
                
                manifest_entries.append({
                    "audio_filepath": f"audio/{new_name}",
                    "text": "",
                    "label": "negative",
                    "split": "train",  # Usually all negatives go to train
                })
        
        # Create manifest CSV (Picovoice Console format)
        manifest_path = self.output_dir / "manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_filepath", "text", "label", "split"])
            writer.writeheader()
            writer.writerows(manifest_entries)
        
        # Create Picovoice config
        config = {
            "wake_word": self.wake_word,
            "language": "en",  # Can be customized
            "sensitivity": 0.5,
            "audio_format": {
                "sample_rate": 16000,
                "channels": 1,
                "sample_width": 2,  # 16-bit
            },
            "dataset_stats": {
                "positive_samples": len(positive_files),
                "negative_samples": len(negative_files) if negative_samples_dir else 0,
            },
        }
        
        with open(self.output_dir / "picovoice_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Save full metadata
        self.metadata.positive_samples = len(positive_files)
        self.metadata.negative_samples = len(negative_files) if negative_samples_dir else 0
        self.metadata.total_samples = self.metadata.positive_samples + self.metadata.negative_samples
        self.metadata.to_json(self.output_dir / "metadata.json")
        
        logger.info(f"Exported to Picovoice format at {self.output_dir}")
        return self.output_dir


class TensorFlowExporter(BaseExporter):
    """
    Export to TensorFlow/Keras format.
    
    Creates:
    - Directory structure compatible with tf.keras.utils.image_dataset_from_directory
    - TFRecord files for efficient data loading
    - tf.data pipeline configuration
    """
    
    format_name = "tensorflow"
    
    async def export(
        self,
        negative_samples_dir: Optional[Path | str] = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        create_tfrecords: bool = True,
    ) -> Path:
        """Export to TensorFlow format."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure for each split
        for split in ["train", "val", "test"]:
            (self.output_dir / split / "positive").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "negative").mkdir(parents=True, exist_ok=True)
        
        # Collect and split positive samples
        positive_files = self._collect_audio_files(self.source_dir)
        train_pos, val_pos, test_pos = self._create_splits(positive_files, split_ratios)
        
        # Copy positive samples
        for f in train_pos:
            shutil.copy2(f, self.output_dir / "train" / "positive" / f.name)
        for f in val_pos:
            shutil.copy2(f, self.output_dir / "val" / "positive" / f.name)
        for f in test_pos:
            shutil.copy2(f, self.output_dir / "test" / "positive" / f.name)
        
        # Handle negative samples
        negative_files = []
        if negative_samples_dir:
            neg_path = Path(negative_samples_dir)
            negative_files = self._collect_audio_files(neg_path)
            train_neg, val_neg, test_neg = self._create_splits(negative_files, split_ratios)
            
            for f in train_neg:
                shutil.copy2(f, self.output_dir / "train" / "negative" / f.name)
            for f in val_neg:
                shutil.copy2(f, self.output_dir / "val" / "negative" / f.name)
            for f in test_neg:
                shutil.copy2(f, self.output_dir / "test" / "negative" / f.name)
        
        # Create TFRecords if requested
        if create_tfrecords:
            await self._create_tfrecords()
        
        # Create tf.data pipeline configuration
        pipeline_config = self._create_pipeline_config()
        with open(self.output_dir / "tf_data_config.json", "w") as f:
            json.dump(pipeline_config, f, indent=2)
        
        # Create example loading script
        self._create_example_script()
        
        # Update and save metadata
        self.metadata.positive_samples = len(positive_files)
        self.metadata.negative_samples = len(negative_files)
        self.metadata.total_samples = self.metadata.positive_samples + self.metadata.negative_samples
        self.metadata.train_samples = len(train_pos) + len(train_neg) if negative_samples_dir else len(train_pos)
        self.metadata.val_samples = len(val_pos) + len(val_neg) if negative_samples_dir else len(val_pos)
        self.metadata.test_samples = len(test_pos) + len(test_neg) if negative_samples_dir else len(test_pos)
        self.metadata.to_json(self.output_dir / "metadata.json")
        
        logger.info(f"Exported to TensorFlow format at {self.output_dir}")
        return self.output_dir
    
    async def _create_tfrecords(self) -> None:
        """Create TFRecord files for each split."""
        try:
            import tensorflow as tf
        except ImportError:
            logger.warning("TensorFlow not installed, skipping TFRecord creation")
            return
        
        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / split
            tfrecord_path = self.output_dir / f"{split}.tfrecord"
            
            with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
                for label_name, label_value in [("positive", 1), ("negative", 0)]:
                    label_dir = split_dir / label_name
                    if not label_dir.exists():
                        continue
                    
                    for audio_file in label_dir.glob("*.wav"):
                        audio_bytes = audio_file.read_bytes()
                        
                        feature = {
                            "audio": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[audio_bytes])
                            ),
                            "label": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[label_value])
                            ),
                            "filename": tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[audio_file.name.encode()]
                                )
                            ),
                        }
                        
                        example = tf.train.Example(
                            features=tf.train.Features(feature=feature)
                        )
                        writer.write(example.SerializeToString())
            
            logger.info(f"Created TFRecord: {tfrecord_path}")
    
    def _create_pipeline_config(self) -> Dict[str, Any]:
        """Create tf.data pipeline configuration."""
        return {
            "batch_size": 32,
            "shuffle_buffer_size": 1000,
            "prefetch_buffer_size": "AUTOTUNE",
            "cache": True,
            "audio_config": {
                "sample_rate": self.metadata.sample_rate,
                "duration_ms": 1000,
                "n_mels": 80,
                "n_fft": 512,
                "hop_length": 160,
            },
            "augmentation": {
                "time_shift_ms": 100,
                "spec_augment": {
                    "freq_mask_param": 10,
                    "time_mask_param": 20,
                },
            },
            "class_names": ["negative", "positive"],
            "num_classes": 2,
        }
    
    def _create_example_script(self) -> None:
        """Create an example TensorFlow data loading script."""
        script = '''"""
Example TensorFlow data loading script for wake word dataset.

Usage:
    python load_tf_dataset.py --data_dir ./exported_dataset
"""

import tensorflow as tf
import json
from pathlib import Path


def load_audio(file_path: str, label: int):
    """Load and preprocess audio file."""
    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    return audio, label


def create_dataset(data_dir: str, split: str, config: dict) -> tf.data.Dataset:
    """Create a tf.data.Dataset for the specified split."""
    data_path = Path(data_dir) / split
    
    # Collect all files
    positive_files = list((data_path / "positive").glob("*.wav"))
    negative_files = list((data_path / "negative").glob("*.wav"))
    
    all_files = [(str(f), 1) for f in positive_files] + [(str(f), 0) for f in negative_files]
    
    # Create dataset
    files, labels = zip(*all_files) if all_files else ([], [])
    ds = tf.data.Dataset.from_tensor_slices((list(files), list(labels)))
    
    # Map loading function
    ds = ds.map(load_audio, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply configuration
    if split == "train":
        ds = ds.shuffle(config.get("shuffle_buffer_size", 1000))
    
    ds = ds.batch(config.get("batch_size", 32))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds


def load_from_tfrecord(tfrecord_path: str, config: dict) -> tf.data.Dataset:
    """Load dataset from TFRecord file."""
    feature_description = {
        "audio": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "filename": tf.io.FixedLenFeature([], tf.string),
    }
    
    def parse_example(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        audio, _ = tf.audio.decode_wav(example["audio"], desired_channels=1)
        audio = tf.squeeze(audio, axis=-1)
        return audio, example["label"]
    
    ds = tf.data.TFRecordDataset(tfrecord_path)
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config.get("batch_size", 32))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to exported dataset")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.data_dir) / "tf_data_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # Create datasets
    train_ds = create_dataset(args.data_dir, "train", config)
    val_ds = create_dataset(args.data_dir, "val", config)
    test_ds = create_dataset(args.data_dir, "test", config)
    
    print(f"Train batches: {len(list(train_ds))}")
    print(f"Val batches: {len(list(val_ds))}")
    print(f"Test batches: {len(list(test_ds))}")
'''
        
        with open(self.output_dir / "load_tf_dataset.py", "w") as f:
            f.write(script)


class PyTorchExporter(BaseExporter):
    """
    Export to PyTorch format.
    
    Creates:
    - Directory structure for torch.utils.data.DataLoader
    - DataLoader configuration
    - Example Dataset class
    """
    
    format_name = "pytorch"
    
    async def export(
        self,
        negative_samples_dir: Optional[Path | str] = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> Path:
        """Export to PyTorch format."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        for split in ["train", "val", "test"]:
            (self.output_dir / split / "positive").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "negative").mkdir(parents=True, exist_ok=True)
        
        # Collect and split positive samples
        positive_files = self._collect_audio_files(self.source_dir)
        train_pos, val_pos, test_pos = self._create_splits(positive_files, split_ratios)
        
        # Copy positive samples
        for f in train_pos:
            shutil.copy2(f, self.output_dir / "train" / "positive" / f.name)
        for f in val_pos:
            shutil.copy2(f, self.output_dir / "val" / "positive" / f.name)
        for f in test_pos:
            shutil.copy2(f, self.output_dir / "test" / "positive" / f.name)
        
        # Handle negative samples
        negative_files = []
        if negative_samples_dir:
            neg_path = Path(negative_samples_dir)
            negative_files = self._collect_audio_files(neg_path)
            train_neg, val_neg, test_neg = self._create_splits(negative_files, split_ratios)
            
            for f in train_neg:
                shutil.copy2(f, self.output_dir / "train" / "negative" / f.name)
            for f in val_neg:
                shutil.copy2(f, self.output_dir / "val" / "negative" / f.name)
            for f in test_neg:
                shutil.copy2(f, self.output_dir / "test" / "negative" / f.name)
        
        # Create labels file for each split
        await self._create_labels_files()
        
        # Create DataLoader configuration
        dataloader_config = self._create_dataloader_config()
        with open(self.output_dir / "dataloader_config.json", "w") as f:
            json.dump(dataloader_config, f, indent=2)
        
        # Create example Dataset class
        self._create_dataset_class()
        
        # Update and save metadata
        self.metadata.positive_samples = len(positive_files)
        self.metadata.negative_samples = len(negative_files)
        self.metadata.total_samples = self.metadata.positive_samples + self.metadata.negative_samples
        self.metadata.train_samples = len(train_pos) + (len(train_neg) if negative_samples_dir else 0)
        self.metadata.val_samples = len(val_pos) + (len(val_neg) if negative_samples_dir else 0)
        self.metadata.test_samples = len(test_pos) + (len(test_neg) if negative_samples_dir else 0)
        self.metadata.to_json(self.output_dir / "metadata.json")
        
        logger.info(f"Exported to PyTorch format at {self.output_dir}")
        return self.output_dir
    
    async def _create_labels_files(self) -> None:
        """Create CSV labels file for each split."""
        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / split
            labels = []
            
            for label_name, label_value in [("positive", 1), ("negative", 0)]:
                label_dir = split_dir / label_name
                if label_dir.exists():
                    for audio_file in label_dir.glob("*.wav"):
                        labels.append({
                            "filename": f"{label_name}/{audio_file.name}",
                            "label": label_value,
                            "label_name": label_name,
                        })
            
            labels_path = split_dir / "labels.csv"
            with open(labels_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "label", "label_name"])
                writer.writeheader()
                writer.writerows(labels)
    
    def _create_dataloader_config(self) -> Dict[str, Any]:
        """Create DataLoader configuration."""
        return {
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": False,
            "audio_config": {
                "sample_rate": self.metadata.sample_rate,
                "duration_ms": 1000,
                "n_mels": 80,
                "n_fft": 512,
                "hop_length": 160,
            },
            "class_names": ["negative", "positive"],
            "num_classes": 2,
        }
    
    def _create_dataset_class(self) -> None:
        """Create an example PyTorch Dataset class."""
        script = '''"""
Example PyTorch Dataset class for wake word dataset.

Usage:
    from wake_word_dataset import WakeWordDataset
    from torch.utils.data import DataLoader
    
    train_ds = WakeWordDataset("./exported_dataset", split="train")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
"""

import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
import pandas as pd
import json


class WakeWordDataset(Dataset):
    """PyTorch Dataset for wake word detection."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        target_sample_rate: int = 16000,
        target_duration_ms: int = 1000,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to the exported dataset.
            split: One of "train", "val", "test".
            transform: Optional transform to apply to audio.
            target_sample_rate: Resample audio to this rate.
            target_duration_ms: Pad/trim audio to this duration.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.target_duration_ms = target_duration_ms
        self.target_length = int(target_sample_rate * target_duration_ms / 1000)
        
        # Load labels
        labels_path = self.data_dir / split / "labels.csv"
        self.labels_df = pd.read_csv(labels_path)
        
        # Load config if available
        config_path = self.data_dir / "dataloader_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}
    
    def __len__(self) -> int:
        return len(self.labels_df)
    
    def __getitem__(self, idx: int) -> tuple:
        row = self.labels_df.iloc[idx]
        audio_path = self.data_dir / self.split / row["filename"]
        label = row["label"]
        
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Pad or trim to target length
        waveform = self._pad_or_trim(waveform)
        
        # Apply transform if provided
        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform, label
    
    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad or trim waveform to target length."""
        current_length = waveform.shape[-1]
        
        if current_length < self.target_length:
            # Pad with zeros
            padding = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_length > self.target_length:
            # Trim
            waveform = waveform[..., :self.target_length]
        
        return waveform


class MelSpectrogramTransform:
    """Transform audio to mel spectrogram."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
    ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_transform(waveform)
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        return mel_spec


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to exported dataset")
    args = parser.parse_args()
    
    # Create dataset with mel spectrogram transform
    transform = MelSpectrogramTransform()
    train_ds = WakeWordDataset(args.data_dir, split="train", transform=transform)
    
    # Create DataLoader
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    
    # Test iteration
    for batch_idx, (audio, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: audio shape {audio.shape}, labels shape {labels.shape}")
        if batch_idx >= 2:
            break
'''
        
        with open(self.output_dir / "wake_word_dataset.py", "w") as f:
            f.write(script)


class HuggingFaceExporter(BaseExporter):
    """
    Export to Hugging Face datasets format.
    
    Creates a dataset compatible with the Hugging Face datasets library,
    including a dataset card and loading script.
    """
    
    format_name = "huggingface"
    
    async def export(
        self,
        negative_samples_dir: Optional[Path | str] = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> Path:
        """Export to Hugging Face datasets format."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data directory
        data_dir = self.output_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Collect positive samples
        positive_files = self._collect_audio_files(self.source_dir)
        train_pos, val_pos, test_pos = self._create_splits(positive_files, split_ratios)
        
        # Collect negative samples
        negative_files = []
        train_neg, val_neg, test_neg = [], [], []
        if negative_samples_dir:
            neg_path = Path(negative_samples_dir)
            negative_files = self._collect_audio_files(neg_path)
            train_neg, val_neg, test_neg = self._create_splits(negative_files, split_ratios)
        
        # Create JSON Lines files for each split
        for split_name, pos_files, neg_files in [
            ("train", train_pos, train_neg),
            ("validation", val_pos, val_neg),
            ("test", test_pos, test_neg),
        ]:
            await self._create_split_files(
                split_name, pos_files, neg_files, data_dir
            )
        
        # Create dataset loading script
        self._create_loading_script()
        
        # Create dataset card (README.md)
        self._create_dataset_card()
        
        # Update and save metadata
        self.metadata.positive_samples = len(positive_files)
        self.metadata.negative_samples = len(negative_files)
        self.metadata.total_samples = self.metadata.positive_samples + self.metadata.negative_samples
        self.metadata.train_samples = len(train_pos) + len(train_neg)
        self.metadata.val_samples = len(val_pos) + len(val_neg)
        self.metadata.test_samples = len(test_pos) + len(test_neg)
        self.metadata.to_json(self.output_dir / "metadata.json")
        
        logger.info(f"Exported to Hugging Face format at {self.output_dir}")
        return self.output_dir
    
    async def _create_split_files(
        self,
        split_name: str,
        pos_files: List[Path],
        neg_files: List[Path],
        data_dir: Path,
    ) -> None:
        """Create audio files and metadata for a split."""
        split_audio_dir = data_dir / split_name
        split_audio_dir.mkdir(exist_ok=True)
        
        metadata_entries = []
        
        # Copy positive samples
        for idx, f in enumerate(pos_files):
            new_name = f"positive_{idx:05d}.wav"
            dest_path = split_audio_dir / new_name
            shutil.copy2(f, dest_path)
            
            metadata_entries.append({
                "file_name": f"{split_name}/{new_name}",
                "label": 1,
                "label_name": "positive",
                "transcript": self.wake_word,
            })
        
        # Copy negative samples
        for idx, f in enumerate(neg_files):
            new_name = f"negative_{idx:05d}.wav"
            dest_path = split_audio_dir / new_name
            shutil.copy2(f, dest_path)
            
            metadata_entries.append({
                "file_name": f"{split_name}/{new_name}",
                "label": 0,
                "label_name": "negative",
                "transcript": "",
            })
        
        # Write metadata JSON Lines file
        metadata_path = data_dir / f"{split_name}.jsonl"
        with open(metadata_path, "w", encoding="utf-8") as f:
            for entry in metadata_entries:
                f.write(json.dumps(entry) + "\n")
    
    def _create_loading_script(self) -> None:
        """Create Hugging Face dataset loading script."""
        script = f'''"""
Hugging Face dataset loading script for {self.wake_word} wake word dataset.

Usage:
    from datasets import load_dataset
    
    dataset = load_dataset("path/to/dataset")
    # Or if uploaded to Hub:
    # dataset = load_dataset("username/{self.metadata.name}")
"""

import datasets
from datasets import DatasetDict, Dataset, Audio, ClassLabel, Value, Features
from pathlib import Path
import json


_DESCRIPTION = """
Wake word detection dataset for "{self.wake_word}".
Contains positive samples (wake word utterances) and negative samples (non-wake word audio).
"""

_HOMEPAGE = ""
_LICENSE = "{self.metadata.license}"

_LABELS = ["negative", "positive"]


class WakeWordDataset(datasets.GeneratorBasedBuilder):
    """Wake word detection dataset."""
    
    VERSION = datasets.Version("{self.metadata.version}")
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=Features({{
                "audio": Audio(sampling_rate={self.metadata.sample_rate}),
                "label": ClassLabel(names=_LABELS),
                "transcript": Value("string"),
            }}),
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )
    
    def _split_generators(self, dl_manager):
        data_dir = Path(self.config.data_dir) / "data"
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={{"data_dir": data_dir, "split": "train"}},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={{"data_dir": data_dir, "split": "validation"}},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={{"data_dir": data_dir, "split": "test"}},
            ),
        ]
    
    def _generate_examples(self, data_dir, split):
        metadata_path = data_dir / f"{{split}}.jsonl"
        
        with open(metadata_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                entry = json.loads(line)
                audio_path = data_dir / entry["file_name"]
                
                yield idx, {{
                    "audio": str(audio_path),
                    "label": entry["label"],
                    "transcript": entry["transcript"],
                }}
'''
        
        with open(self.output_dir / f"{self.metadata.name}.py", "w") as f:
            f.write(script)
    
    def _create_dataset_card(self) -> None:
        """Create dataset card (README.md)."""
        card = f'''---
language:
- en
license: {self.metadata.license}
task_categories:
- audio-classification
tags:
- wake-word
- keyword-spotting
- speech
size_categories:
- 1K<n<10K
---

# {self.metadata.name}

## Dataset Description

Wake word detection dataset for **"{self.wake_word}"**.

### Dataset Summary

This dataset contains audio samples for training wake word detection models.
It includes both positive samples (containing the wake word) and negative 
samples (random speech/audio that does not contain the wake word).

### Supported Tasks

- **Audio Classification**: Classify audio as containing the wake word or not.
- **Keyword Spotting**: Detect the wake word in continuous audio.

### Languages

English (en)

## Dataset Structure

### Data Instances

```python
{{
    "audio": {{"path": "data/train/positive_00000.wav", "array": [...], "sampling_rate": {self.metadata.sample_rate}}},
    "label": 1,
    "transcript": "{self.wake_word}"
}}
```

### Data Fields

- `audio`: Audio file containing the sample
- `label`: 0 for negative (no wake word), 1 for positive (contains wake word)
- `transcript`: Text transcript of the audio (wake word for positive, empty for negative)

### Data Splits

| Split | Positive | Negative | Total |
|-------|----------|----------|-------|
| train | {self.metadata.train_samples // 2} | {self.metadata.train_samples // 2} | {self.metadata.train_samples} |
| validation | {self.metadata.val_samples // 2} | {self.metadata.val_samples // 2} | {self.metadata.val_samples} |
| test | {self.metadata.test_samples // 2} | {self.metadata.test_samples // 2} | {self.metadata.test_samples} |

## Dataset Creation

### Source Data

Generated using WakeGen - Wake Word Dataset Generator.

- **Wake Word**: {self.wake_word}
- **Sample Rate**: {self.metadata.sample_rate} Hz
- **Format**: WAV, 16-bit, mono
- **Created**: {self.metadata.created_at}

### Generation Process

Audio samples were generated using text-to-speech synthesis with various
voices and then augmented with realistic environmental conditions.

## Usage

```python
from datasets import load_dataset

# Load from local directory
dataset = load_dataset("path/to/this/directory")

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# Iterate through examples
for example in train_data:
    audio = example["audio"]["array"]
    label = example["label"]
    print(f"Label: {{'positive' if label == 1 else 'negative'}}, Shape: {{len(audio)}}")
```

## License

{self.metadata.license}
'''
        
        with open(self.output_dir / "README.md", "w") as f:
            f.write(card)


# Export format registry
EXPORTERS: Dict[ExportFormat, type] = {
    ExportFormat.MYCROFT_PRECISE: MycroftPreciseExporter,
    ExportFormat.PICOVOICE: PicovoiceExporter,
    ExportFormat.TENSORFLOW: TensorFlowExporter,
    ExportFormat.PYTORCH: PyTorchExporter,
    ExportFormat.HUGGINGFACE: HuggingFaceExporter,
}


async def export_to_format(
    format_type: ExportFormat | str,
    source_dir: Path | str,
    output_dir: Path | str,
    wake_word: str,
    negative_samples_dir: Optional[Path | str] = None,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    metadata: Optional[DatasetMetadata] = None,
    **kwargs: Any,
) -> Path:
    """
    Export a dataset to the specified format.
    
    Args:
        format_type: The target export format.
        source_dir: Directory containing generated audio files.
        output_dir: Directory for exported dataset.
        wake_word: The wake word being trained.
        negative_samples_dir: Optional directory with negative samples.
        split_ratios: Train/val/test split ratios.
        metadata: Optional dataset metadata.
        **kwargs: Additional format-specific arguments.
        
    Returns:
        Path to the exported dataset.
    """
    if isinstance(format_type, str):
        format_type = ExportFormat(format_type)
    
    exporter_class = EXPORTERS.get(format_type)
    if not exporter_class:
        raise ValueError(f"Unsupported export format: {format_type}")
    
    exporter = exporter_class(
        source_dir=source_dir,
        output_dir=output_dir,
        wake_word=wake_word,
        metadata=metadata,
    )
    
    return await exporter.export(
        negative_samples_dir=negative_samples_dir,
        split_ratios=split_ratios,
        **kwargs,
    )


def list_export_formats() -> List[Dict[str, str]]:
    """List all available export formats."""
    return [
        {
            "format": fmt.value,
            "name": fmt.name,
            "description": EXPORTERS[fmt].__doc__.strip().split("\n")[0] if fmt in EXPORTERS else "",
        }
        for fmt in ExportFormat
    ]
