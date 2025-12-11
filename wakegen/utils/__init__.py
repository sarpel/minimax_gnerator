# This file initializes the 'utils' module.
# This module contains helper functions that are used across the application,
# such as audio processing, logging configuration, and async helpers.

from wakegen.utils.async_helpers import (
    BatchConfig,
    ParallelExecutor,
    ParallelExecutorStats,
    RateLimiter,
    TaskResult,
    first_successful,
    gather_with_limit,
    process_in_batches,
    retry_async,
)
from wakegen.utils.caching import CacheEntry, CacheManager, CacheStats, GenerationCache
from wakegen.utils.gpu import (
    GPUBackend,
    GPUInfo,
    GPUManager,
    GPUStatus,
    detect_gpu_status,
    get_best_device,
    is_gpu_available,
)

__all__ = [
    # Async helpers
    "retry_async",
    "TaskResult",
    "ParallelExecutor",
    "ParallelExecutorStats",
    "RateLimiter",
    "BatchConfig",
    "process_in_batches",
    "gather_with_limit",
    "first_successful",
    # Caching
    "CacheManager",
    "GenerationCache",
    "CacheStats",
    "CacheEntry",
    # GPU management
    "GPUBackend",
    "GPUInfo",
    "GPUStatus",
    "GPUManager",
    "detect_gpu_status",
    "is_gpu_available",
    "get_best_device",
]
