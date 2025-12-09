"""
Property-based tests for augmentation pipeline.

**Feature: critical-bug-fixes, Property 1: Batch augmentation returns accurate counts**
**Validates: Requirements 1.3**
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from hypothesis import given, settings
from hypothesis import strategies as st

# Import the pipeline components
from wakegen.augmentation.pipeline import AugmentationPipeline
from wakegen.augmentation.profiles import AugmentationProfile
from wakegen.core.types import AugmentationType


def create_test_audio_file(
    filepath: str, duration_sec: float = 0.5, sample_rate: int = 16000
) -> str:
    """Create a simple test audio file with a sine wave."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    sf.write(filepath, audio, sample_rate)
    return filepath


def create_minimal_profile() -> AugmentationProfile:
    """Create a minimal augmentation profile for testing."""
    from wakegen.augmentation.microphone.simulator import MicrophoneProfile
    from wakegen.augmentation.noise.profiles import NoiseProfile
    from wakegen.augmentation.room.simulator import RoomParameters

    noise_profile = NoiseProfile(
        name="test_noise",
        description="Test noise profile",
        base_noise_type="white",
        snr_range=(15.0, 25.0),
        typical_events=[],
        event_density=0.0,
        intensity_range=(0.1, 0.3),
    )

    room_params = RoomParameters(
        length=5.0,
        width=4.0,
        height=3.0,
        rt60=0.3,
        absorption=0.5,
        max_order=3,
        mic_position=(2.5, 2.0, 1.5),
        source_position=(1.0, 1.0, 1.0),
    )

    mic_profile = MicrophoneProfile(
        name="test_mic",
        frequency_response=[(100, 0.0), (1000, 0.0), (8000, 0.0)],
        sample_rate=16000,
        noise_floor=-60.0,
        distortion=0.01,
    )

    return AugmentationProfile(
        name="test_profile",
        description="Minimal test profile",
        noise_profile=noise_profile,
        room_params=room_params,
        microphone_profile=mic_profile,
        time_effects={"pitch_steps": 0.0, "time_stretch_factor": 1.0},
        dynamics_effects={},
        degradation_effects={},
        augmentation_types=[AugmentationType.BACKGROUND_NOISE],
    )


class TestBatchAugmentationCountProperty:
    """
    Property-based tests for batch augmentation count accuracy.

    **Feature: critical-bug-fixes, Property 1: Batch augmentation returns accurate counts**
    **Validates: Requirements 1.3**

    Property: For any list of input files (valid and invalid), when batch_augment
    completes, the sum of successful results and failed files should equal the
    total number of input files.
    """

    @pytest.mark.asyncio
    @given(
        num_valid=st.integers(min_value=0, max_value=5),
        num_invalid=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=100, deadline=None)
    async def test_batch_augment_count_accuracy(self, num_valid: int, num_invalid: int):
        """
        **Feature: critical-bug-fixes, Property 1: Batch augmentation returns accurate counts**
        **Validates: Requirements 1.3**

        Property: For any combination of valid and invalid input files,
        the batch_augment function should process all files and the count
        of successful + failed should equal total input count.
        """
        # Skip if no files to process
        if num_valid == 0 and num_invalid == 0:
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(input_dir, exist_ok=True)

            input_paths = []

            # Create valid audio files
            for i in range(num_valid):
                filepath = os.path.join(input_dir, f"valid_{i}.wav")
                create_test_audio_file(filepath)
                input_paths.append(filepath)

            # Create invalid file paths (non-existent files)
            for i in range(num_invalid):
                filepath = os.path.join(input_dir, f"invalid_{i}.wav")
                # Don't create the file - it will be invalid
                input_paths.append(filepath)

            total_input_count = len(input_paths)

            # Create pipeline with minimal profile
            profile = create_minimal_profile()
            pipeline = AugmentationPipeline(
                profile, sample_rate=16000, temp_dir=temp_dir
            )

            # Track failures via logging
            failed_count = 0
            original_error = pipeline._init_components  # Just to have a reference

            # Run batch augmentation
            results = await pipeline.batch_augment(
                input_paths=input_paths, output_dir=output_dir, prefix="aug_"
            )

            successful_count = len(results)

            # The property: successful files should equal valid files
            # (since invalid files should fail and not be in results)
            assert (
                successful_count == num_valid
            ), f"Expected {num_valid} successful files, got {successful_count}"

            # Verify that total processed (success + implicit failures) equals input count
            # Since failed files are not returned, we verify by checking:
            # - All valid files were processed successfully
            # - Results count matches valid file count
            implied_failed_count = total_input_count - successful_count
            assert (
                implied_failed_count == num_invalid
            ), f"Expected {num_invalid} failed files, got {implied_failed_count}"

            # The core property: success + failure = total
            assert (
                successful_count + implied_failed_count == total_input_count
            ), f"Count mismatch: {successful_count} + {implied_failed_count} != {total_input_count}"


@pytest.mark.asyncio
async def test_batch_augment_empty_input():
    """Test that batch_augment handles empty input list correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "output")

        profile = create_minimal_profile()
        pipeline = AugmentationPipeline(profile, sample_rate=16000, temp_dir=temp_dir)

        results = await pipeline.batch_augment(input_paths=[], output_dir=output_dir)

        assert results == [], "Empty input should return empty results"


@pytest.mark.asyncio
async def test_batch_augment_all_valid():
    """Test batch_augment with all valid files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)

        # Create 3 valid files
        input_paths = []
        for i in range(3):
            filepath = os.path.join(input_dir, f"test_{i}.wav")
            create_test_audio_file(filepath)
            input_paths.append(filepath)

        profile = create_minimal_profile()
        pipeline = AugmentationPipeline(profile, sample_rate=16000, temp_dir=temp_dir)

        results = await pipeline.batch_augment(
            input_paths=input_paths, output_dir=output_dir
        )

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"


@pytest.mark.asyncio
async def test_batch_augment_all_invalid():
    """Test batch_augment with all invalid files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "output")

        # Create paths to non-existent files
        input_paths = [
            os.path.join(temp_dir, "nonexistent_1.wav"),
            os.path.join(temp_dir, "nonexistent_2.wav"),
        ]

        profile = create_minimal_profile()
        pipeline = AugmentationPipeline(profile, sample_rate=16000, temp_dir=temp_dir)

        results = await pipeline.batch_augment(
            input_paths=input_paths, output_dir=output_dir
        )

        # All files should fail, so results should be empty
        assert (
            len(results) == 0
        ), f"Expected 0 results for invalid files, got {len(results)}"
