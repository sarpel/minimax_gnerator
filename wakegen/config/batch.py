"""Shared Configuration Classes

AH-001 Fix: Consolidates duplicate BatchConfig classes from different modules
into a single shared configuration module.

ELI5: Instead of having two different "recipe cards" with the same name in
different cookbooks, we create one master recipe card that everyone can use.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BatchConfig:
    """
    Unified configuration for batch processing operations.

    AH-001 Fix: Consolidates BatchConfig from async_helpers.py and batch_processor.py
    into a single shared class with all necessary fields.

    This configuration supports both:
    - General batch processing (batch_size, concurrent batches)
    - Audio generation batching (concurrent tasks, retries, rate limits)
    """

    # General batch processing settings
    batch_size: int = 10
    """Number of items per batch"""

    max_concurrent_batches: int = 2
    """Maximum number of batches to process concurrently"""

    delay_between_batches: float = 0.0
    """Delay in seconds between processing batches"""

    # Audio generation specific settings
    max_concurrent_tasks: int = 5
    """Maximum number of concurrent generation tasks"""

    retry_attempts: int = 3
    """Number of retry attempts for failed tasks"""

    timeout_seconds: int = 300
    """Timeout for individual generation tasks"""

    rate_limits: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"commercial": (10, 60), "free": (5, 60)}
    )
    """Rate limits per provider type (provider_type -> (max_requests, period_seconds))"""

    sample_rate: int = 16000
    """Target sample rate for generated audio in Hz (default: 16000)"""

    caching_enabled: bool = True
    """Whether to enable caching of generated audio (default: True)"""

    def __post_init__(self) -> None:
        pass
