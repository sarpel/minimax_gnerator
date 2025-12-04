import hashlib
import os
import json
from typing import Optional, Dict, Any
from wakegen.utils.logging import get_logger

logger = get_logger("wakegen.caching")

class CacheManager:
    """
    Manages a simple file-based cache to avoid re-generating the same audio.
    
    It works like this:
    1. We create a unique ID (hash) based on the text and voice.
    2. We check if a file with that ID exists in the cache folder.
    3. If yes, we use it. If no, we generate it and save it.
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