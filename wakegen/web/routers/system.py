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
from typing import Dict, Any, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class GPUInfo(BaseModel):
    available: bool
    name: Optional[str] = None
    memory_total: Optional[float] = None  # In GB
    memory_allocated: Optional[float] = None  # In GB
    memory_reserved: Optional[float] = None  # In GB
    utilization: Optional[float] = None  # Percentage

class CacheInfo(BaseModel):
    path: str
    size_bytes: int
    size_human: str
    file_count: int

class SystemEnv(BaseModel):
    python_version: str
    platform: str
    variables: Dict[str, str]

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
        gb = 1024 ** 3
        total = properties.total_memory / gb
        allocated = torch.cuda.memory_allocated(device) / gb
        reserved = torch.cuda.memory_reserved(device) / gb
        
        return GPUInfo(
            available=True,
            name=properties.name,
            memory_total=round(total, 2),
            memory_allocated=round(allocated, 2),
            memory_reserved=round(reserved, 2),
            utilization=round((allocated / total) * 100, 1)
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
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024:
            size_human = f"{total_size:.1f} {unit}"
            break
        total_size /= 1024
    else:
        size_human = f"{total_size:.1f} TB"
        
    return CacheInfo(
        path=str(cache_dir),
        size_bytes=int(total_size * (1024 if unit != 'B' else 1)), # Approximate rebuild for model
        size_human=size_human,
        file_count=file_count
    )

@router.post("/cache/clear")
async def clear_cache() -> Dict[str, str]:
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
    import sys
    import platform
    
    # Filter for relevant env vars (don't show everything for security)
    relevant_keys = ["CUDA_VISIBLE_DEVICES", "WAKEGEN_ENV", "PYTHONPATH"]
    env_vars = {k: os.environ.get(k, "") for k in relevant_keys if k in os.environ}
    
    return SystemEnv(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        variables=env_vars
    )
