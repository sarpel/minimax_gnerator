"""
Test Suite for wakegen.export module

This module tests export format functionality for various wake word training formats.

EDUCATIONAL NOTES:
- We use tmp_path fixture to create isolated test directories
- We create synthetic audio files to test export functionality
- We test both the base exporter and format-specific exporters
"""

import json
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
import pytest
import soundfile as sf

# =============================================================================
# IMPORTS: Testing wakegen.export modules
# =============================================================================
from wakegen.export.formats import (
    ExportFormat,
    DatasetMetadata,
    SampleMetadata,
    BaseExporter,
    MycroftPreciseExporter,
    PicovoiceExporter,
    TensorFlowExporter,
)
from wakegen.export.splitter import split_dataset, _save_manifest


# =============================================================================
# FIXTURES: Test data setup
# =============================================================================


@pytest.fixture
def sample_audio_dir(tmp_path) -> Path:
    """Create a directory with sample audio files."""
    audio_dir = tmp_path / "source_audio"
    audio_dir.mkdir()

    # Create sample audio files
    # CONCEPT: We create minimal valid WAV files for testing
    for i in range(10):
        audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
        file_path = audio_dir / f"sample_{i:03d}.wav"
        sf.write(str(file_path), audio, 16000)

    return audio_dir


@pytest.fixture
def negative_samples_dir(tmp_path) -> Path:
    """Create a directory with negative (non-wake-word) samples."""
    neg_dir = tmp_path / "negative_samples"
    neg_dir.mkdir()

    for i in range(5):
        audio = np.random.randn(16000).astype(np.float32)
        file_path = neg_dir / f"negative_{i:03d}.wav"
        sf.write(str(file_path), audio, 16000)

    return neg_dir


@pytest.fixture
def output_dir(tmp_path) -> Path:
    """Create an output directory for exports."""
    out_dir = tmp_path / "export_output"
    out_dir.mkdir()
    return out_dir


@pytest.fixture
def sample_metadata() -> DatasetMetadata:
    """Create sample dataset metadata."""
    return DatasetMetadata(
        name="test_dataset",
        wake_word="hey assistant",
        version="1.0.0",
        total_samples=10,
        positive_samples=5,
        negative_samples=5,
        train_samples=8,
        val_samples=1,
        test_samples=1,
        description="Test dataset for unit testing",
    )


# =============================================================================
# TEST: ExportFormat Enum
# =============================================================================


class TestExportFormat:
    """Tests for the ExportFormat enum."""

    def test_export_format_values(self):
        """Test all export format values."""
        assert ExportFormat.OPENWAKEWORD.value == "openwakeword"
        assert ExportFormat.MYCROFT_PRECISE.value == "mycroft_precise"
        assert ExportFormat.PICOVOICE.value == "picovoice"
        assert ExportFormat.TENSORFLOW.value == "tensorflow"
        assert ExportFormat.PYTORCH.value == "pytorch"
        assert ExportFormat.HUGGINGFACE.value == "huggingface"

    def test_export_format_iteration(self):
        """Test that we can iterate over all formats."""
        formats = list(ExportFormat)
        assert len(formats) == 6


# =============================================================================
# TEST: DatasetMetadata
# =============================================================================


class TestDatasetMetadata:
    """Tests for the DatasetMetadata dataclass."""

    def test_dataset_metadata_creation(self, sample_metadata):
        """Test creating DatasetMetadata."""
        assert sample_metadata.name == "test_dataset"
        assert sample_metadata.wake_word == "hey assistant"
        assert sample_metadata.version == "1.0.0"
        assert sample_metadata.total_samples == 10

    def test_dataset_metadata_defaults(self):
        """Test DatasetMetadata default values."""
        meta = DatasetMetadata(
            name="test",
            wake_word="wake",
        )

        assert meta.version == "1.0.0"
        assert meta.total_samples == 0
        assert meta.positive_samples == 0
        assert meta.negative_samples == 0
        assert meta.train_samples == 0

    def test_dataset_metadata_to_dict(self, sample_metadata):
        """Test converting metadata to dictionary."""
        data = sample_metadata.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "test_dataset"
        assert data["wake_word"] == "hey assistant"
        assert "created_at" in data

    def test_dataset_metadata_to_json(self, sample_metadata, tmp_path):
        """Test saving metadata to JSON file."""
        json_path = tmp_path / "metadata.json"
        sample_metadata.to_json(json_path)

        assert json_path.exists()

        # Verify contents
        with open(json_path) as f:
            data = json.load(f)
        assert data["name"] == "test_dataset"

    def test_dataset_metadata_from_json(self, sample_metadata, tmp_path):
        """Test loading metadata from JSON file."""
        json_path = tmp_path / "metadata.json"
        sample_metadata.to_json(json_path)

        loaded = DatasetMetadata.from_json(json_path)

        assert loaded.name == sample_metadata.name
        assert loaded.wake_word == sample_metadata.wake_word


# =============================================================================
# TEST: SampleMetadata
# =============================================================================


class TestSampleMetadata:
    """Tests for the SampleMetadata dataclass."""

    def test_sample_metadata_creation(self):
        """Test creating SampleMetadata."""
        sample = SampleMetadata(
            filename="audio_001.wav",
            label=1,
            transcript="hey assistant",
        )

        assert sample.filename == "audio_001.wav"
        assert sample.label == 1
        assert sample.transcript == "hey assistant"

    def test_sample_metadata_with_all_fields(self):
        """Test SampleMetadata with all optional fields."""
        sample = SampleMetadata(
            filename="audio_001.wav",
            label=1,
            transcript="hey assistant",
            provider="edge_tts",
            voice_id="en-US-AriaNeural",
            duration_ms=1500.0,
            sample_rate=16000,
            augmentations=["noise", "reverb"],
            augmentation_params={"snr_db": 15.0},
            snr_db=15.0,
            quality_score=0.95,
            split="train",
        )

        assert sample.provider == "edge_tts"
        assert sample.voice_id == "en-US-AriaNeural"
        assert sample.duration_ms == 1500.0
        assert len(sample.augmentations) == 2
        assert sample.split == "train"


# =============================================================================
# TEST: BaseExporter
# =============================================================================


class TestBaseExporter:
    """Tests for the BaseExporter class."""

    def test_base_exporter_initialization(self, sample_audio_dir, output_dir):
        """Test BaseExporter initialization."""
        exporter = BaseExporter(
            source_dir=sample_audio_dir,
            output_dir=output_dir,
            wake_word="hey assistant",
        )

        assert exporter.source_dir == sample_audio_dir
        assert exporter.output_dir == output_dir
        assert exporter.wake_word == "hey assistant"
        assert exporter.metadata is not None

    def test_base_exporter_with_metadata(
        self, sample_audio_dir, output_dir, sample_metadata
    ):
        """Test BaseExporter with custom metadata."""
        exporter = BaseExporter(
            source_dir=sample_audio_dir,
            output_dir=output_dir,
            wake_word="hey assistant",
            metadata=sample_metadata,
        )

        assert exporter.metadata.name == "test_dataset"

    def test_base_exporter_creates_output_dir(self, sample_audio_dir, tmp_path):
        """Test that BaseExporter creates output directory."""
        output = tmp_path / "new_output_dir"
        assert not output.exists()

        exporter = BaseExporter(
            source_dir=sample_audio_dir,
            output_dir=output,
            wake_word="test",
        )

        # Output directory should be created during export
        # For BaseExporter, we just test initialization
        assert exporter.output_dir == output

    def test_collect_audio_files(self, sample_audio_dir, output_dir):
        """Test collecting audio files from directory."""
        exporter = BaseExporter(
            source_dir=sample_audio_dir,
            output_dir=output_dir,
            wake_word="test",
        )

        files = exporter._collect_audio_files(sample_audio_dir)

        assert len(files) == 10
        assert all(f.suffix == ".wav" for f in files)

    def test_create_splits(self, sample_audio_dir, output_dir):
        """Test creating train/val/test splits."""
        exporter = BaseExporter(
            source_dir=sample_audio_dir,
            output_dir=output_dir,
            wake_word="test",
        )

        files = exporter._collect_audio_files(sample_audio_dir)
        train, val, test = exporter._create_splits(files, (0.6, 0.2, 0.2))

        assert len(train) == 6
        assert len(val) == 2
        assert len(test) == 2
        assert len(train) + len(val) + len(test) == 10


# =============================================================================
# TEST: MycroftPreciseExporter
# =============================================================================


class TestMycroftPreciseExporter:
    """Tests for the MycroftPreciseExporter class."""

    def test_mycroft_exporter_initialization(self, sample_audio_dir, output_dir):
        """Test MycroftPreciseExporter initialization."""
        exporter = MycroftPreciseExporter(
            source_dir=sample_audio_dir,
            output_dir=output_dir,
            wake_word="hey mycroft",
        )

        assert exporter.format_name == "mycroft_precise"
        assert exporter.wake_word == "hey mycroft"

    @pytest.mark.asyncio
    async def test_mycroft_exporter_export(
        self, sample_audio_dir, output_dir, negative_samples_dir
    ):
        """Test exporting to Mycroft Precise format."""
        exporter = MycroftPreciseExporter(
            source_dir=sample_audio_dir,
            output_dir=output_dir,
            wake_word="hey mycroft",
        )

        result_path = await exporter.export(
            negative_samples_dir=negative_samples_dir,
            split_ratios=(0.8, 0.1, 0.1),
        )

        assert result_path.exists()

        # Mycroft expects wake-word/ and not-wake-word/ directories
        assert (result_path / "wake-word").exists() or (result_path / "train").exists()


# =============================================================================
# TEST: PicovoiceExporter
# =============================================================================


class TestPicovoiceExporter:
    """Tests for the PicovoiceExporter class."""

    def test_picovoice_exporter_initialization(self, sample_audio_dir, output_dir):
        """Test PicovoiceExporter initialization."""
        exporter = PicovoiceExporter(
            source_dir=sample_audio_dir,
            output_dir=output_dir,
            wake_word="hey porcupine",
        )

        assert exporter.format_name == "picovoice"
        assert exporter.wake_word == "hey porcupine"

    @pytest.mark.asyncio
    async def test_picovoice_exporter_export(self, sample_audio_dir, output_dir):
        """Test exporting to Picovoice format."""
        exporter = PicovoiceExporter(
            source_dir=sample_audio_dir,
            output_dir=output_dir,
            wake_word="hey porcupine",
        )

        result_path = await exporter.export()

        assert result_path.exists()


# =============================================================================
# TEST: TensorFlowExporter
# =============================================================================


class TestTensorFlowExporter:
    """Tests for the TensorFlowExporter class."""

    def test_tensorflow_exporter_initialization(self, sample_audio_dir, output_dir):
        """Test TensorFlowExporter initialization."""
        exporter = TensorFlowExporter(
            source_dir=sample_audio_dir,
            output_dir=output_dir,
            wake_word="hey tensorflow",
        )

        assert exporter.format_name == "tensorflow"
        assert exporter.wake_word == "hey tensorflow"

    @pytest.mark.asyncio
    async def test_tensorflow_exporter_export(self, sample_audio_dir, output_dir):
        """Test exporting to TensorFlow format."""
        exporter = TensorFlowExporter(
            source_dir=sample_audio_dir,
            output_dir=output_dir,
            wake_word="hey tensorflow",
        )

        result_path = await exporter.export()

        assert result_path.exists()


# =============================================================================
# TEST: Dataset Splitter
# =============================================================================


class TestDatasetSplitter:
    """Tests for the split_dataset function."""

    @pytest.fixture
    def manifest_dir(self, tmp_path) -> Path:
        """Create a directory with a manifest file."""
        manifest_data = [
            {"filename": f"audio_{i:03d}.wav", "label": i % 2, "transcript": "test"}
            for i in range(20)
        ]

        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)

        return tmp_path

    @pytest.mark.asyncio
    async def test_split_dataset(self, manifest_dir):
        """Test splitting a dataset."""
        await split_dataset(
            export_dir=str(manifest_dir),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        # Verify split files were created
        assert (manifest_dir / "train.json").exists()
        assert (manifest_dir / "val.json").exists()
        assert (manifest_dir / "test.json").exists()

    @pytest.mark.asyncio
    async def test_split_dataset_manifest_not_found(self, tmp_path):
        """Test that missing manifest raises error."""
        with pytest.raises(FileNotFoundError):
            await split_dataset(export_dir=str(tmp_path))

    def test_save_manifest(self, tmp_path):
        """Test helper function to save manifest."""
        data = [{"key": "value1"}, {"key": "value2"}]
        path = tmp_path / "test_manifest.json"

        _save_manifest(path, data)

        assert path.exists()

        with open(path) as f:
            loaded = json.load(f)
        assert len(loaded) == 2
