"""
GPU Management Module

This module provides utilities for detecting and managing GPU resources
for TTS models. It helps optimize batch sizes and distribute models
across available GPUs.

Key Features:
- Automatic GPU detection (NVIDIA via PyTorch/CUDA)
- Memory monitoring and management
- Optimal batch size calculation
- Multi-GPU model assignment

Think of it like a parking lot attendant for your GPUs:
- It knows how many parking spots (GPUs) are available
- It tracks how much space is used in each spot
- It directs new "cars" (models) to available spots
- It prevents overcrowding (OOM errors)
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from wakegen.utils.logging import get_logger

logger = get_logger("wakegen.gpu")


# =============================================================================
# GPU INFORMATION
# =============================================================================


class GPUBackend(Enum):
    """Supported GPU backends."""
    CUDA = "cuda"        # NVIDIA GPUs via PyTorch
    MPS = "mps"          # Apple Silicon GPUs
    ROCM = "rocm"        # AMD GPUs via ROCm
    CPU = "cpu"          # CPU fallback (no GPU)


@dataclass
class GPUInfo:
    """
    Information about a single GPU.
    
    Attributes:
        id: GPU index (0, 1, 2, etc.)
        name: GPU model name (e.g., "NVIDIA GeForce RTX 3080")
        total_memory_mb: Total VRAM in megabytes
        free_memory_mb: Currently available VRAM
        used_memory_mb: Currently used VRAM
        compute_capability: CUDA compute capability (e.g., "8.6")
        backend: The backend used to access this GPU
    """
    id: int
    name: str
    total_memory_mb: float
    free_memory_mb: float
    used_memory_mb: float
    compute_capability: Optional[str] = None
    backend: GPUBackend = GPUBackend.CUDA
    
    @property
    def utilization(self) -> float:
        """Memory utilization as a percentage (0.0 to 1.0)."""
        if self.total_memory_mb == 0:
            return 0.0
        return self.used_memory_mb / self.total_memory_mb
    
    @property
    def is_available(self) -> bool:
        """Whether this GPU has sufficient free memory for typical models."""
        # Consider GPU available if it has at least 500MB free
        return self.free_memory_mb >= 500


@dataclass
class GPUStatus:
    """
    Overall GPU status summary.
    
    Attributes:
        backend: The GPU backend in use
        gpus: List of detected GPUs
        total_memory_mb: Combined VRAM across all GPUs
        total_free_mb: Combined free VRAM
    """
    backend: GPUBackend
    gpus: List[GPUInfo] = field(default_factory=list)
    
    @property
    def num_gpus(self) -> int:
        return len(self.gpus)
    
    @property
    def total_memory_mb(self) -> float:
        return sum(g.total_memory_mb for g in self.gpus)
    
    @property
    def total_free_mb(self) -> float:
        return sum(g.free_memory_mb for g in self.gpus)
    
    @property
    def has_gpu(self) -> bool:
        return self.backend != GPUBackend.CPU and self.num_gpus > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "backend": self.backend.value,
            "num_gpus": self.num_gpus,
            "gpus": [
                {
                    "id": g.id,
                    "name": g.name,
                    "total_memory_mb": g.total_memory_mb,
                    "free_memory_mb": g.free_memory_mb,
                    "utilization": f"{g.utilization:.1%}",
                }
                for g in self.gpus
            ],
            "total_memory_mb": self.total_memory_mb,
            "total_free_mb": self.total_free_mb,
        }


# =============================================================================
# GPU DETECTION
# =============================================================================


def detect_gpu_status() -> GPUStatus:
    """
    Detect available GPUs and their status.
    
    This function tries multiple backends in order:
    1. CUDA (NVIDIA GPUs via PyTorch)
    2. MPS (Apple Silicon)
    3. CPU fallback
    
    Returns:
        GPUStatus with information about available GPUs.
    """
    # Try CUDA first (most common for ML)
    try:
        import torch
        if torch.cuda.is_available():
            gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                
                # Get memory info
                # Note: torch.cuda.mem_get_info returns (free, total) in bytes
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                
                gpus.append(GPUInfo(
                    id=i,
                    name=props.name,
                    total_memory_mb=total_bytes / (1024 * 1024),
                    free_memory_mb=free_bytes / (1024 * 1024),
                    used_memory_mb=(total_bytes - free_bytes) / (1024 * 1024),
                    compute_capability=f"{props.major}.{props.minor}",
                    backend=GPUBackend.CUDA,
                ))
            
            logger.info(f"Detected {len(gpus)} CUDA GPU(s)")
            return GPUStatus(backend=GPUBackend.CUDA, gpus=gpus)
    except ImportError:
        logger.debug("PyTorch not installed, skipping CUDA detection")
    except Exception as e:
        logger.warning(f"CUDA detection failed: {e}")
    
    # Try MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't provide detailed memory info like CUDA
            # We estimate based on system memory
            gpus = [GPUInfo(
                id=0,
                name="Apple Silicon GPU",
                total_memory_mb=0,  # Unknown for MPS
                free_memory_mb=0,
                used_memory_mb=0,
                backend=GPUBackend.MPS,
            )]
            logger.info("Detected Apple Silicon GPU (MPS)")
            return GPUStatus(backend=GPUBackend.MPS, gpus=gpus)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"MPS detection failed: {e}")
    
    # Fallback to CPU
    logger.info("No GPU detected, using CPU")
    return GPUStatus(backend=GPUBackend.CPU, gpus=[])


def is_gpu_available() -> bool:
    """Quick check if any GPU is available."""
    status = detect_gpu_status()
    return status.has_gpu


def get_best_device() -> str:
    """
    Get the best available device string for PyTorch.
    
    Returns:
        Device string like "cuda:0", "mps", or "cpu".
    """
    status = detect_gpu_status()
    
    if status.backend == GPUBackend.CUDA and status.gpus:
        # Return the GPU with most free memory
        best_gpu = max(status.gpus, key=lambda g: g.free_memory_mb)
        return f"cuda:{best_gpu.id}"
    elif status.backend == GPUBackend.MPS:
        return "mps"
    else:
        return "cpu"


# =============================================================================
# GPU MANAGER
# =============================================================================


class GPUManager:
    """
    Manages GPU resources for TTS models.
    
    This class helps coordinate GPU usage across multiple models,
    preventing out-of-memory errors and optimizing batch sizes.
    
    Usage:
        manager = GPUManager()
        
        # Check available resources
        print(f"GPUs: {manager.num_gpus}")
        print(f"Free memory: {manager.total_free_mb:.0f} MB")
        
        # Assign a model to a GPU
        device = manager.assign_model("kokoro", required_memory_mb=500)
        
        # Get optimal batch size
        batch_size = manager.get_optimal_batch_size("kokoro", sample_memory_mb=50)
    """
    
    # Estimated memory requirements for different TTS models (in MB)
    MODEL_MEMORY_ESTIMATES: Dict[str, int] = {
        "piper": 100,           # Piper is very lightweight
        "edge_tts": 0,          # Cloud-based, no local GPU
        "kokoro": 300,          # 82M params, efficient
        "mimic3": 200,          # Lightweight
        "coqui_xtts": 2000,     # XTTS is large
        "f5_tts": 1500,         # Medium-large
        "styletts2": 1000,      # Medium
        "bark": 4000,           # Large model
        "default": 500,         # Conservative default
    }
    
    def __init__(self):
        """Initialize the GPU manager."""
        self._status: Optional[GPUStatus] = None
        self._model_assignments: Dict[str, int] = {}  # model_id -> gpu_id
        self._reserved_memory: Dict[int, float] = {}  # gpu_id -> reserved_mb
        
    @property
    def status(self) -> GPUStatus:
        """Get current GPU status (cached)."""
        if self._status is None:
            self._status = detect_gpu_status()
        return self._status
    
    def refresh_status(self) -> GPUStatus:
        """Force refresh of GPU status."""
        self._status = detect_gpu_status()
        return self._status
    
    @property
    def num_gpus(self) -> int:
        """Number of available GPUs."""
        return self.status.num_gpus
    
    @property
    def total_free_mb(self) -> float:
        """Total free memory across all GPUs."""
        return self.status.total_free_mb
    
    @property
    def has_gpu(self) -> bool:
        """Whether any GPU is available."""
        return self.status.has_gpu
    
    def get_best_device(self) -> str:
        """Get the best available device string."""
        return get_best_device()
    
    def get_available_memory(self, gpu_id: int = 0) -> float:
        """
        Get available memory on a specific GPU (accounting for reservations).
        
        Args:
            gpu_id: GPU index.
            
        Returns:
            Available memory in MB.
        """
        if gpu_id >= len(self.status.gpus):
            return 0.0
        
        gpu = self.status.gpus[gpu_id]
        reserved = self._reserved_memory.get(gpu_id, 0.0)
        return max(0.0, gpu.free_memory_mb - reserved)
    
    def assign_model(
        self,
        model_id: str,
        required_memory_mb: Optional[float] = None,
        preferred_gpu: Optional[int] = None
    ) -> str:
        """
        Assign a model to a GPU.
        
        Args:
            model_id: Identifier for the model (e.g., "kokoro", "piper").
            required_memory_mb: Memory needed by the model. If None, uses estimate.
            preferred_gpu: Preferred GPU index. If None, auto-selects.
            
        Returns:
            Device string (e.g., "cuda:0", "cpu").
        """
        if not self.has_gpu:
            logger.debug(f"No GPU available, assigning {model_id} to CPU")
            return "cpu"
        
        # Get memory requirement
        if required_memory_mb is None:
            required_memory_mb = self.MODEL_MEMORY_ESTIMATES.get(
                model_id.lower(),
                self.MODEL_MEMORY_ESTIMATES["default"]
            )
        
        # Find best GPU
        if preferred_gpu is not None and preferred_gpu < self.num_gpus:
            # Check if preferred GPU has enough memory
            if self.get_available_memory(preferred_gpu) >= required_memory_mb:
                gpu_id = preferred_gpu
            else:
                logger.warning(
                    f"Preferred GPU {preferred_gpu} doesn't have enough memory, "
                    f"auto-selecting"
                )
                gpu_id = self._find_best_gpu(required_memory_mb)
        else:
            gpu_id = self._find_best_gpu(required_memory_mb)
        
        if gpu_id is None:
            logger.warning(f"No GPU has enough memory for {model_id}, using CPU")
            return "cpu"
        
        # Record assignment
        self._model_assignments[model_id] = gpu_id
        self._reserved_memory[gpu_id] = self._reserved_memory.get(gpu_id, 0) + required_memory_mb
        
        device = f"cuda:{gpu_id}" if self.status.backend == GPUBackend.CUDA else "mps"
        logger.info(f"Assigned {model_id} to {device} ({required_memory_mb:.0f} MB)")
        return device
    
    def release_model(self, model_id: str) -> None:
        """
        Release a model's GPU assignment.
        
        Args:
            model_id: The model identifier to release.
        """
        if model_id in self._model_assignments:
            gpu_id = self._model_assignments.pop(model_id)
            # Estimate the memory that was reserved
            memory = self.MODEL_MEMORY_ESTIMATES.get(
                model_id.lower(),
                self.MODEL_MEMORY_ESTIMATES["default"]
            )
            self._reserved_memory[gpu_id] = max(
                0,
                self._reserved_memory.get(gpu_id, 0) - memory
            )
            logger.debug(f"Released {model_id} from GPU {gpu_id}")
    
    def _find_best_gpu(self, required_memory_mb: float) -> Optional[int]:
        """Find the GPU with most available memory that fits requirements."""
        candidates = [
            (gpu.id, self.get_available_memory(gpu.id))
            for gpu in self.status.gpus
            if self.get_available_memory(gpu.id) >= required_memory_mb
        ]
        
        if not candidates:
            return None
        
        # Return GPU with most free memory
        return max(candidates, key=lambda x: x[1])[0]
    
    def get_optimal_batch_size(
        self,
        model_id: str,
        sample_memory_mb: float = 50.0,
        max_batch_size: int = 32,
        memory_safety_margin: float = 0.8
    ) -> int:
        """
        Calculate optimal batch size based on available GPU memory.
        
        This helps prevent OOM errors by sizing batches to fit available memory.
        
        Args:
            model_id: The model identifier.
            sample_memory_mb: Estimated memory per sample during inference.
            max_batch_size: Maximum batch size to return.
            memory_safety_margin: Leave this fraction of memory as headroom.
            
        Returns:
            Recommended batch size (at least 1).
        """
        if model_id not in self._model_assignments:
            logger.debug(f"Model {model_id} not assigned, returning batch size 1")
            return 1
        
        gpu_id = self._model_assignments[model_id]
        available = self.get_available_memory(gpu_id) * memory_safety_margin
        
        # Subtract estimated model memory (already reserved, but double-check)
        model_memory = self.MODEL_MEMORY_ESTIMATES.get(
            model_id.lower(),
            self.MODEL_MEMORY_ESTIMATES["default"]
        )
        available -= model_memory
        
        if available <= 0:
            return 1
        
        # Calculate batch size
        batch_size = int(available / sample_memory_mb)
        batch_size = max(1, min(batch_size, max_batch_size))
        
        logger.debug(
            f"Optimal batch size for {model_id}: {batch_size} "
            f"(available: {available:.0f} MB, per sample: {sample_memory_mb:.0f} MB)"
        )
        return batch_size
    
    def clear_gpu_cache(self) -> None:
        """
        Clear GPU memory cache (PyTorch CUDA cache).
        
        Call this after releasing models to actually free up memory.
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of GPU status and assignments."""
        return {
            "status": self.status.to_dict(),
            "model_assignments": dict(self._model_assignments),
            "reserved_memory": dict(self._reserved_memory),
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    "GPUBackend",
    "GPUInfo",
    "GPUStatus",
    "GPUManager",
    "detect_gpu_status",
    "is_gpu_available",
    "get_best_device",
]
