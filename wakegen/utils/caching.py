"""
Caching Module

This module provides intelligent caching for generated audio files to avoid
redundant TTS API calls and speed up dataset generation.

Key Features:
- Content-based hashing (same text + voice + provider = same cache key)
- Automatic cache size management (LRU eviction)
- Cache statistics and hit rate tracking
- Thread-safe operations
- Async-friendly interface

Think of it like a library's book return system:
- Each generated audio is a "book" with a unique ID
- Before generating, we check if the "book" is already in the library
- If yes, we just grab it from the shelf (cache hit)
- If no, we generate it and add it to the library for next time
"""

import hashlib
import os
import json
import shutil
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from wakegen.utils.logging import get_logger

logger = get_logger("wakegen.caching")


# =============================================================================
# CACHE STATISTICS
# =============================================================================


@dataclass
class CacheStats:
    """
    Statistics about cache usage.
    
    Track these to understand how effective the cache is and whether
    you need to adjust the cache size.
    """
    hits: int = 0           # Number of times we found what we needed
    misses: int = 0         # Number of times we had to generate new
    total_size_bytes: int = 0  # Total size of cached files
    file_count: int = 0     # Number of files in cache
    evictions: int = 0      # Number of files removed to make space
    
    @property
    def hit_rate(self) -> float:
        """Calculate the cache hit rate (0.0 to 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def total_size_mb(self) -> float:
        """Total cache size in megabytes."""
        return self.total_size_bytes / (1024 * 1024)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "total_size_mb": f"{self.total_size_mb:.2f}",
            "file_count": self.file_count,
            "evictions": self.evictions,
        }


# =============================================================================
# CACHE ENTRY
# =============================================================================


@dataclass
class CacheEntry:
    """
    Metadata about a cached file.
    
    We store this alongside the actual audio file to track access patterns
    and enable LRU (Least Recently Used) eviction.
    """
    cache_key: str
    file_path: str
    text: str
    voice_id: str
    provider: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    file_size: int = 0
    
    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


# =============================================================================
# GENERATION CACHE
# =============================================================================


class GenerationCache:
    """
    Advanced caching system for generated audio files.
    
    This cache:
    - Stores generated audio to avoid regenerating the same content
    - Uses LRU eviction when the cache gets too large
    - Tracks statistics to help optimize cache size
    - Is thread-safe for concurrent access
    
    Usage:
        cache = GenerationCache(cache_dir=".wakegen_cache", max_size_mb=500)
        
        # Check if we have a cached version
        cached_path = cache.get(text, voice_id, provider)
        if cached_path:
            return cached_path  # Use cached version
        
        # Generate new audio
        audio_path = generate_audio(text, voice_id)
        
        # Add to cache
        cache.put(text, voice_id, provider, audio_path)
    """
    
    def __init__(
        self,
        cache_dir: str = ".wakegen_cache",
        max_size_mb: float = 500.0,
        enabled: bool = True
    ):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cached files.
            max_size_mb: Maximum cache size in megabytes. When exceeded,
                         oldest files are removed (LRU eviction).
            enabled: Whether caching is enabled. If False, all operations
                     are no-ops for easy toggling.
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.enabled = enabled
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory index of cache entries
        self._entries: Dict[str, CacheEntry] = {}
        
        # Statistics
        self._stats = CacheStats()
        
        # Initialize
        if self.enabled:
            self._init_cache()
    
    def _init_cache(self) -> None:
        """Initialize cache directory and load existing entries."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata if exists
        metadata_file = self.cache_dir / "cache_metadata.json"
        if metadata_file.exists():
            try:
                self._load_metadata(metadata_file)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        
        # Scan for any files not in metadata
        self._scan_cache_dir()
        
        # Update stats
        self._update_stats()
    
    def _get_cache_key(self, text: str, voice_id: str, provider: str) -> str:
        """
        Generate a unique cache key for the given parameters.
        
        The key is a hash of the text, voice, and provider combined.
        Same inputs always produce the same key.
        """
        # Normalize inputs
        normalized = f"{text.strip().lower()}|{voice_id}|{provider}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    
    def get(
        self,
        text: str,
        voice_id: str,
        provider: str
    ) -> Optional[str]:
        """
        Try to retrieve a cached audio file.
        
        Args:
            text: The text that was converted to speech.
            voice_id: The voice ID used.
            provider: The TTS provider used.
        
        Returns:
            Path to cached file if found, None otherwise.
        """
        if not self.enabled:
            return None
        
        with self._lock:
            cache_key = self._get_cache_key(text, voice_id, provider)
            
            if cache_key in self._entries:
                entry = self._entries[cache_key]
                
                # Verify file still exists
                if Path(entry.file_path).exists():
                    entry.touch()
                    self._stats.hits += 1
                    logger.debug(f"Cache hit for '{text[:30]}...' ({provider})")
                    return entry.file_path
                else:
                    # File was deleted externally
                    del self._entries[cache_key]
            
            self._stats.misses += 1
            return None
    
    def put(
        self,
        text: str,
        voice_id: str,
        provider: str,
        source_path: str,
        copy: bool = True
    ) -> str:
        """
        Add a generated audio file to the cache.
        
        Args:
            text: The text that was converted to speech.
            voice_id: The voice ID used.
            provider: The TTS provider used.
            source_path: Path to the generated audio file.
            copy: If True, copy the file to cache. If False, assume it's
                  already in the cache directory.
        
        Returns:
            Path to the cached file.
        """
        if not self.enabled:
            return source_path
        
        with self._lock:
            cache_key = self._get_cache_key(text, voice_id, provider)
            
            # Determine cache file path
            source = Path(source_path)
            cache_file = self.cache_dir / f"{cache_key}{source.suffix}"
            
            # Copy file to cache if needed
            if copy and str(source) != str(cache_file):
                shutil.copy2(source, cache_file)
            
            # Create entry
            entry = CacheEntry(
                cache_key=cache_key,
                file_path=str(cache_file),
                text=text,
                voice_id=voice_id,
                provider=provider,
                file_size=cache_file.stat().st_size,
            )
            
            self._entries[cache_key] = entry
            
            # Check if we need to evict
            self._evict_if_needed()
            
            # Update stats
            self._update_stats()
            
            logger.debug(f"Cached '{text[:30]}...' ({provider})")
            return str(cache_file)
    
    def get_cache_path(
        self,
        text: str,
        voice_id: str,
        provider: str,
        extension: str = ".wav"
    ) -> str:
        """
        Get the path where a new cache file should be saved.
        
        Use this when you want to generate directly to the cache location.
        
        Args:
            text: The text to be converted.
            voice_id: The voice ID.
            provider: The provider name.
            extension: File extension (default: .wav).
        
        Returns:
            Path where the cache file should be saved.
        """
        cache_key = self._get_cache_key(text, voice_id, provider)
        return str(self.cache_dir / f"{cache_key}{extension}")
    
    def _evict_if_needed(self) -> None:
        """
        Remove oldest entries if cache exceeds max size.
        
        Uses LRU (Least Recently Used) strategy.
        """
        current_size = sum(e.file_size for e in self._entries.values())
        
        if current_size <= self.max_size_bytes:
            return
        
        # Sort by last access time (oldest first)
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.last_accessed
        )
        
        # Evict until under limit
        for entry in sorted_entries:
            if current_size <= self.max_size_bytes * 0.8:  # Leave 20% headroom
                break
            
            try:
                Path(entry.file_path).unlink(missing_ok=True)
                current_size -= entry.file_size
                del self._entries[entry.cache_key]
                self._stats.evictions += 1
                logger.debug(f"Evicted cache entry: {entry.cache_key}")
            except Exception as e:
                logger.warning(f"Failed to evict cache entry: {e}")
    
    def _update_stats(self) -> None:
        """Update cache statistics."""
        self._stats.file_count = len(self._entries)
        self._stats.total_size_bytes = sum(e.file_size for e in self._entries.values())
    
    def _scan_cache_dir(self) -> None:
        """Scan cache directory for files not in metadata."""
        for file_path in self.cache_dir.glob("*"):
            if file_path.name == "cache_metadata.json":
                continue
            if file_path.suffix not in (".wav", ".mp3", ".flac"):
                continue
            
            cache_key = file_path.stem
            if cache_key not in self._entries:
                # Found orphaned cache file
                entry = CacheEntry(
                    cache_key=cache_key,
                    file_path=str(file_path),
                    text="[unknown]",
                    voice_id="[unknown]",
                    provider="[unknown]",
                    file_size=file_path.stat().st_size,
                )
                self._entries[cache_key] = entry
    
    def _load_metadata(self, metadata_file: Path) -> None:
        """Load cache metadata from file."""
        with open(metadata_file, "r") as f:
            data = json.load(f)
        
        for entry_data in data.get("entries", []):
            entry = CacheEntry(**entry_data)
            if Path(entry.file_path).exists():
                self._entries[entry.cache_key] = entry
    
    def save_metadata(self) -> None:
        """Save cache metadata to file."""
        if not self.enabled:
            return
        
        with self._lock:
            metadata_file = self.cache_dir / "cache_metadata.json"
            data = {
                "entries": [
                    {
                        "cache_key": e.cache_key,
                        "file_path": e.file_path,
                        "text": e.text,
                        "voice_id": e.voice_id,
                        "provider": e.provider,
                        "created_at": e.created_at,
                        "last_accessed": e.last_accessed,
                        "access_count": e.access_count,
                        "file_size": e.file_size,
                    }
                    for e in self._entries.values()
                ]
            }
            with open(metadata_file, "w") as f:
                json.dump(data, f, indent=2)
    
    def clear(self) -> None:
        """Clear all cached files."""
        with self._lock:
            for entry in self._entries.values():
                try:
                    Path(entry.file_path).unlink(missing_ok=True)
                except Exception:
                    pass
            self._entries.clear()
            self._stats = CacheStats()
            logger.info("Cache cleared")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._update_stats()
            return self._stats
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_metadata()


# =============================================================================
# LEGACY CACHE MANAGER (for backward compatibility)
# =============================================================================


class CacheManager:
    """
    Manages a simple file-based cache to avoid re-generating the same audio.
    
    It works like this:
    1. We create a unique ID (hash) based on the text and voice.
    2. We check if a file with that ID exists in the cache folder.
    3. If yes, we use it. If no, we generate it and save it.
    
    Note: This is the legacy interface. For new code, use GenerationCache.
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_hash(self, text: str, voice_id: str, params: Dict[str, Any]) -> str:
        """
        Creates a unique fingerprint for a request.
        """
        # Combine all inputs into a single string
        data = f"{text}|{voice_id}|{json.dumps(params, sort_keys=True)}"
        # Calculate MD5 hash
        return hashlib.md5(data.encode("utf-8")).hexdigest()
        
    def get(self, text: str, voice_id: str, params: Dict[str, Any] = {}) -> Optional[str]:
        """
        Tries to retrieve a file path from the cache.
        """
        file_hash = self._get_hash(text, voice_id, params)
        # We assume cached files are wavs for now, but this could be improved
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.wav")
        
        if os.path.exists(cache_path):
            logger.debug(f"Cache hit for '{text}'")
            return cache_path
            
        return None
        
    def get_path(self, text: str, voice_id: str, params: Dict[str, Any] = {}) -> str:
        """
        Returns the path where a new cache file should be saved.
        """
        file_hash = self._get_hash(text, voice_id, params)
        return os.path.join(self.cache_dir, f"{file_hash}.wav")


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    "CacheManager",
    "GenerationCache",
    "CacheStats",
    "CacheEntry",
]