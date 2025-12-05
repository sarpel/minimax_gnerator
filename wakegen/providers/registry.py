"""
Provider Registry Module

This module manages the registration and discovery of TTS providers.
Think of it like a phone book for TTS engines - you can look up any provider
by its name and get an instance ready to use.

Key Features:
- Register new providers dynamically
- Get provider instances by type
- List available providers
- Auto-discover which providers are actually usable (dependencies installed)
"""

from dataclasses import dataclass
from typing import Dict, Type, List, Optional
import importlib.util
import sys

from wakegen.core.types import ProviderType
from wakegen.core.protocols import TTSProvider
from wakegen.core.exceptions import ConfigError
from wakegen.models.config import ProviderConfig


# =============================================================================
# PROVIDER INFO
# =============================================================================


@dataclass
class ProviderInfo:
    """
    Information about a TTS provider's availability and requirements.
    
    This is like a product spec sheet - tells you what you need to use it.
    """
    type: ProviderType
    name: str
    description: str
    requires_gpu: bool = False
    requires_api_key: bool = False
    is_available: bool = False
    missing_dependencies: List[str] = None
    install_hint: Optional[str] = None
    
    def __post_init__(self):
        if self.missing_dependencies is None:
            self.missing_dependencies = []


# =============================================================================
# PROVIDER REGISTRY
# =============================================================================


# The registry is a dictionary that maps ProviderType (e.g., "edge_tts")
# to the class that implements it (e.g., EdgeTTSProvider).
_PROVIDER_REGISTRY: Dict[ProviderType, Type[TTSProvider]] = {}

# Cache for provider dependency checks (so we don't re-check every time)
_PROVIDER_AVAILABILITY_CACHE: Dict[ProviderType, ProviderInfo] = {}


def register_provider(provider_type: ProviderType, provider_class: Type[TTSProvider]) -> None:
    """
    Registers a provider class for a given type.
    
    This is called when provider modules are imported. It adds the provider
    to our "phone book" so we can look it up later.
    
    Args:
        provider_type: The enum value identifying this provider (e.g., ProviderType.EDGE_TTS)
        provider_class: The class that implements the TTSProvider protocol
    """
    _PROVIDER_REGISTRY[provider_type] = provider_class
    # Invalidate cache when new provider is registered
    if provider_type in _PROVIDER_AVAILABILITY_CACHE:
        del _PROVIDER_AVAILABILITY_CACHE[provider_type]


def get_provider(provider_type: ProviderType, config: ProviderConfig) -> TTSProvider:
    """
    Creates and returns an instance of the requested provider.

    This is the main way to get a usable provider. It looks up the class
    in our registry and creates a new instance with your config.

    Args:
        provider_type: The type of provider to create.
        config: The configuration to pass to the provider.

    Returns:
        A ready-to-use TTSProvider instance.

    Raises:
        ConfigError: If the provider type is unknown or not registered.
    """
    if provider_type not in _PROVIDER_REGISTRY:
        raise ConfigError(f"Unknown provider type: {provider_type}")
    
    provider_class = _PROVIDER_REGISTRY[provider_type]
    # We assume the provider class accepts 'config' in its constructor
    return provider_class(config)  # type: ignore


def list_available_providers() -> List[ProviderType]:
    """
    Returns a list of all registered provider types.
    
    This allows the CLI to dynamically show which providers are supported
    without hardcoding the list.
    
    Note: This returns ALL registered providers, even if their dependencies
    aren't installed. Use discover_available_providers() to get only the
    ones that are actually usable.
    
    Returns:
        List of ProviderType enum values for all registered providers.
    """
    return list(_PROVIDER_REGISTRY.keys())


# =============================================================================
# PROVIDER AUTO-DISCOVERY
# =============================================================================


def _check_module_available(module_name: str) -> bool:
    """
    Check if a Python module is available for import.
    
    This doesn't actually import the module (which could be slow or have side effects),
    it just checks if the module exists in the system.
    
    Args:
        module_name: The name of the module to check (e.g., "kokoro_onnx")
    
    Returns:
        True if the module can be imported, False otherwise.
    """
    # Check if it's already imported
    if module_name in sys.modules:
        return True
    
    # Check if it can be found
    spec = importlib.util.find_spec(module_name)
    return spec is not None


# Define what each provider needs to work
# Format: {ProviderType: (required_modules, optional_modules, api_key_env_var, gpu_required, install_hint)}
_PROVIDER_REQUIREMENTS: Dict[ProviderType, dict] = {
    ProviderType.EDGE_TTS: {
        "required": ["edge_tts"],
        "optional": [],
        "api_key_env": None,
        "gpu_required": False,
        "description": "Microsoft Edge TTS - free, online, high quality",
        "install_hint": "pip install edge-tts",
    },
    ProviderType.PIPER: {
        "required": ["piper"],
        "optional": [],
        "api_key_env": None,
        "gpu_required": False,
        "description": "Piper TTS - local, fast, lightweight",
        "install_hint": "pip install piper-tts",
    },
    ProviderType.COQUI_XTTS: {
        "required": ["TTS"],
        "optional": ["torch"],
        "api_key_env": None,
        "gpu_required": True,  # Works on CPU but very slow
        "description": "Coqui XTTS - voice cloning, high quality",
        "install_hint": "pip install TTS",
    },
    ProviderType.KOKORO: {
        "required": ["kokoro_onnx"],
        "optional": [],
        "api_key_env": None,
        "gpu_required": False,
        "description": "Kokoro TTS - lightweight 82M model, fast on CPU",
        "install_hint": "pip install kokoro-onnx",
    },
    ProviderType.MIMIC3: {
        "required": ["mimic3_tts"],
        "optional": [],
        "api_key_env": None,
        "gpu_required": False,
        "description": "Mimic 3 - Mycroft's privacy-friendly TTS",
        "install_hint": "pip install mycroft-mimic3-tts",
    },
    ProviderType.MINIMAX: {
        "required": ["httpx"],
        "optional": [],
        "api_key_env": "MINIMAX_API_KEY",
        "gpu_required": False,
        "description": "MiniMax TTS - commercial API service",
        "install_hint": "Set MINIMAX_API_KEY environment variable",
    },
    ProviderType.F5_TTS: {
        "required": ["f5_tts"],
        "optional": ["torch"],
        "api_key_env": None,
        "gpu_required": True,  # GPU recommended for good performance
        "description": "F5-TTS - high quality with voice cloning",
        "install_hint": "pip install f5-tts",
    },
    ProviderType.STYLETTS2: {
        "required": ["styletts2"],
        "optional": ["torch"],
        "api_key_env": None,
        "gpu_required": True,  # GPU recommended
        "description": "StyleTTS 2 - state-of-the-art expressive TTS",
        "install_hint": "pip install styletts2",
    },
    ProviderType.ORPHEUS: {
        "required": ["orpheus_tts"],  # or orpheus_speech
        "optional": ["torch"],
        "api_key_env": None,
        "gpu_required": False,  # Small models work on CPU
        "description": "Orpheus TTS - scalable 150M to 3B models",
        "install_hint": "pip install orpheus-tts",
    },
    ProviderType.BARK: {
        "required": ["bark"],
        "optional": ["torch"],
        "api_key_env": None,
        "gpu_required": True,  # GPU strongly recommended
        "description": "Bark - expressive TTS with emotions and non-speech sounds",
        "install_hint": "pip install git+https://github.com/suno-ai/bark.git",
    },
    ProviderType.CHATTTS: {
        "required": ["ChatTTS"],
        "optional": ["torch"],
        "api_key_env": None,
        "gpu_required": False,  # Works on CPU but slower
        "description": "ChatTTS - conversational speech synthesis",
        "install_hint": "pip install chattts",
    },
}


def check_provider_availability(provider_type: ProviderType) -> ProviderInfo:
    """
    Check if a specific provider is available and ready to use.
    
    This checks:
    1. Is the provider registered?
    2. Are the required Python packages installed?
    3. If it needs an API key, is the environment variable set?
    
    Results are cached for performance.
    
    Args:
        provider_type: The provider to check.
    
    Returns:
        ProviderInfo with availability status and any missing dependencies.
    """
    # Return cached result if available
    if provider_type in _PROVIDER_AVAILABILITY_CACHE:
        return _PROVIDER_AVAILABILITY_CACHE[provider_type]
    
    # Get requirements for this provider
    requirements = _PROVIDER_REQUIREMENTS.get(provider_type, {
        "required": [],
        "optional": [],
        "api_key_env": None,
        "gpu_required": False,
        "description": "Unknown provider",
        "install_hint": None,
    })
    
    missing_deps = []
    is_available = True
    
    # Check if provider is registered
    if provider_type not in _PROVIDER_REGISTRY:
        is_available = False
        missing_deps.append(f"Provider '{provider_type.value}' not registered")
    
    # Check required modules
    for module in requirements.get("required", []):
        if not _check_module_available(module):
            is_available = False
            missing_deps.append(module)
    
    # Check API key if required
    api_key_env = requirements.get("api_key_env")
    if api_key_env and not os.environ.get(api_key_env):
        is_available = False
        missing_deps.append(f"Missing {api_key_env} environment variable")
    
    # Create and cache the result
    info = ProviderInfo(
        type=provider_type,
        name=provider_type.value,
        description=requirements.get("description", ""),
        requires_gpu=requirements.get("gpu_required", False),
        requires_api_key=api_key_env is not None,
        is_available=is_available,
        missing_dependencies=missing_deps,
        install_hint=requirements.get("install_hint"),
    )
    
    _PROVIDER_AVAILABILITY_CACHE[provider_type] = info
    return info


def discover_available_providers() -> List[ProviderInfo]:
    """
    Automatically discover which providers are available and ready to use.
    
    This is the main function for provider auto-discovery. It checks each
    registered provider to see if its dependencies are installed and
    any required credentials are configured.
    
    Use this when you want to know what providers the user can actually use
    right now, not just what's theoretically supported.
    
    Returns:
        List of ProviderInfo objects for all registered providers,
        with availability status for each.
    
    Example:
        providers = discover_available_providers()
        for p in providers:
            if p.is_available:
                print(f"✓ {p.name}: {p.description}")
            else:
                print(f"✗ {p.name}: Missing {p.missing_dependencies}")
    """
    all_providers = []
    
    # Check all known provider types (from the enum)
    for provider_type in ProviderType:
        info = check_provider_availability(provider_type)
        all_providers.append(info)
    
    return all_providers


def get_available_provider_types() -> List[ProviderType]:
    """
    Get a list of provider types that are currently available for use.
    
    This is a convenience function that filters discover_available_providers()
    to return only the types that can actually be used.
    
    Returns:
        List of ProviderType values for providers that are available.
    """
    return [
        info.type 
        for info in discover_available_providers() 
        if info.is_available
    ]


def clear_availability_cache() -> None:
    """
    Clear the provider availability cache.
    
    Call this if the environment changes (e.g., after installing a package
    or setting an environment variable) to force re-checking availability.
    """
    _PROVIDER_AVAILABILITY_CACHE.clear()


# Need to import os for API key checking
import os