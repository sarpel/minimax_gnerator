"""
System API Router

This module provides endpoints for system status, GPU information, and cache management.

    ENDPOINTS:
    ==========
    GET  /gpu/summary      - Get GPU availability and usage
    GET  /cache/summary    - Get cache usage statistics
    POST /cache/clear      - Clear application caches
    GET  /env              - List relevant environment variables

"""

import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class GPUInfo(BaseModel):
    available: bool
    name: str | None = None
    memory_total: float | None = None  # In GB
    memory_allocated: float | None = None  # In GB
    memory_reserved: float | None = None  # In GB
    utilization: float | None = None  # Percentage


class CacheInfo(BaseModel):
    path: str
    size_bytes: int
    size_human: str
    file_count: int


class SystemEnv(BaseModel):
    python_version: str
    platform: str
    variables: dict[str, str]


# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter()


@router.get("/gpu/summary", response_model=GPUInfo)
async def get_gpu_summary() -> GPUInfo:
    """Check GPU status and memory usage (PyTorch)."""
    try:
        import torch

        available = torch.cuda.is_available()
        if not available:
            return GPUInfo(available=False)

        # Get current device info
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)

        # Memory stats (convert bytes to GB)
        gb = 1024**3
        total = properties.total_memory / gb
        allocated = torch.cuda.memory_allocated(device) / gb
        reserved = torch.cuda.memory_reserved(device) / gb

        return GPUInfo(
            available=True,
            name=properties.name,
            memory_total=round(total, 2),
            memory_allocated=round(allocated, 2),
            memory_reserved=round(reserved, 2),
            utilization=round((allocated / total) * 100, 1),
        )

    except ImportError:
        return GPUInfo(available=False)
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
        return GPUInfo(available=False)


@router.get("/cache/summary", response_model=CacheInfo)
async def get_cache_summary() -> CacheInfo:
    """Get information about the file cache."""
    # This should match your project's cache or temp directory
    # For now, we'll check the output directory as a proxy or specific cache temp
    cache_dir = Path("./.cache") if Path("./.cache").exists() else Path("./output")

    total_size = 0
    file_count = 0

    if cache_dir.exists():
        for p in cache_dir.rglob("*"):
            if p.is_file():
                total_size += p.stat().st_size
                file_count += 1

    # Human readable size
    total_size_float = float(total_size)
    for unit in ["B", "KB", "MB", "GB"]:
        if total_size_float < 1024:
            size_human = f"{total_size_float:.1f} {unit}"
            break
        total_size_float /= 1024
    else:
        size_human = f"{total_size_float:.1f} TB"

    return CacheInfo(
        path=str(cache_dir),
        size_bytes=int(
            total_size * (1024 if unit != "B" else 1)
        ),  # Approximate rebuild for model
        size_human=size_human,
        file_count=file_count,
    )


@router.post("/cache/clear")
async def clear_cache() -> dict[str, str]:
    """Clear temporary cache files."""
    cache_dir = Path("./.cache")
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            cache_dir.mkdir()
            return {"status": "success", "message": "Cache cleared"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "success", "message": "Cache was empty"}


@router.get("/env", response_model=SystemEnv)
async def get_env_info() -> SystemEnv:
    """Get system environment information."""
    import platform
    import sys

    # List of relevant environment variable keys to check
    # For security, we only report if they're SET, not their actual values
    relevant_keys = [
        # CUDA/GPU
        "CUDA_VISIBLE_DEVICES",
        # WakeGen specific
        "WAKEGEN_ENV",
        # API keys (check presence only - value will be "SET" or empty)
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "MINIMAX_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
    ]

    # For API keys, only report if set (not actual value for security)
    api_key_prefixes = ("_API_KEY", "_TOKEN", "CREDENTIALS")
    env_vars = {}
    for key in relevant_keys:
        if key in os.environ:
            # For sensitive keys, just report "SET" instead of actual value
            if any(key.endswith(suffix) for suffix in api_key_prefixes):
                env_vars[key] = "SET"
            else:
                env_vars[key] = os.environ.get(key, "")

    return SystemEnv(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        variables=env_vars,
    )
