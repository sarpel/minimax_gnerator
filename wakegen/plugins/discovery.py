"""
Plugin Discovery Module

This module handles discovering, loading, and managing TTS plugins.
It uses Python's entry points mechanism (from importlib.metadata) to find
plugins installed via pip.

How Plugin Discovery Works:
1. When wakegen starts, we scan for packages that register the "wakegen.plugins" entry point
2. Each entry point points to a plugin class implementing TTSPlugin
3. We instantiate the plugin and register it with the provider registry
4. The plugin is now available just like built-in providers!

For plugin developers:
In your pyproject.toml, add:
```toml
[project.entry-points."wakegen.plugins"]
my-plugin = "my_package.provider:MyTTSPlugin"
```

Then users can install your plugin and it will be auto-discovered:
```bash
pip install my-wakegen-plugin
wakegen list-plugins  # Shows your plugin!
wakegen generate --provider my-plugin --text "hello"
```
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type
from importlib.metadata import entry_points, EntryPoint

from wakegen.plugins.base import (
    TTSPlugin,
    PluginMetadata,
    LoadedPlugin,
    PluginLoadError,
    PluginValidationError,
)
from wakegen.core.types import ProviderType
from wakegen.core.exceptions import ConfigError


# =============================================================================
# MODULE STATE
# =============================================================================


# Logger for plugin-related messages
logger = logging.getLogger(__name__)

# Entry point group name - plugins register under this name
PLUGIN_ENTRY_POINT_GROUP = "wakegen.plugins"

# Cache of loaded plugins: {plugin_name: LoadedPlugin}
_loaded_plugins: Dict[str, LoadedPlugin] = {}

# Flag to track if initial discovery has been done
_discovery_done: bool = False


# =============================================================================
# PLUGIN DISCOVERY
# =============================================================================


def discover_plugins(force_reload: bool = False) -> List[LoadedPlugin]:
    """
    Discover and load all installed wakegen plugins.
    
    This scans the Python environment for packages that registered entry points
    under the "wakegen.plugins" group. Each discovered plugin is loaded and
    made available for use.
    
    Results are cached - subsequent calls return the cached list unless
    force_reload=True is passed.
    
    Args:
        force_reload: If True, re-scan for plugins even if already done.
    
    Returns:
        List of LoadedPlugin objects for all discovered plugins.
    
    Example:
        plugins = discover_plugins()
        for plugin in plugins:
            if plugin.is_enabled:
                print(f"Found: {plugin.name} v{plugin.metadata.version}")
    """
    global _discovery_done, _loaded_plugins
    
    # Return cached results if already discovered
    if _discovery_done and not force_reload:
        return list(_loaded_plugins.values())
    
    # Clear cache if reloading
    if force_reload:
        _loaded_plugins.clear()
    
    logger.info(f"Discovering plugins from entry point group: {PLUGIN_ENTRY_POINT_GROUP}")
    
    # Get all entry points in our group
    # Python 3.10+ uses select(), older versions use different API
    try:
        # Python 3.10+ style
        eps = entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
    except TypeError:
        # Python 3.9 style (entry_points() returns a dict-like object)
        all_eps = entry_points()
        eps = all_eps.get(PLUGIN_ENTRY_POINT_GROUP, [])
    
    # Load each discovered plugin
    for ep in eps:
        try:
            loaded = load_plugin(ep)
            if loaded:
                _loaded_plugins[loaded.name] = loaded
                logger.info(f"Loaded plugin: {loaded.name} v{loaded.metadata.version}")
        except Exception as e:
            logger.warning(f"Failed to load plugin from entry point '{ep.name}': {e}")
    
    _discovery_done = True
    logger.info(f"Plugin discovery complete. Found {len(_loaded_plugins)} plugins.")
    
    return list(_loaded_plugins.values())


def load_plugin(entry_point: EntryPoint) -> Optional[LoadedPlugin]:
    """
    Load a single plugin from an entry point.
    
    This:
    1. Loads the plugin class from the entry point
    2. Instantiates the plugin
    3. Validates it implements the TTSPlugin protocol
    4. Wraps it in a LoadedPlugin object
    
    Args:
        entry_point: The entry point to load from.
    
    Returns:
        LoadedPlugin if successful, None if loading failed.
    
    Raises:
        PluginLoadError: If the plugin fails to load or validate.
    """
    logger.debug(f"Loading plugin from entry point: {entry_point.name}")
    
    try:
        # Load the plugin class
        plugin_class = entry_point.load()
        
        # Instantiate the plugin
        # Some plugins might need config, so we try with no args first
        try:
            plugin_instance = plugin_class()
        except TypeError:
            # Plugin might need config argument
            plugin_instance = plugin_class(config=None)
        
        # Verify it implements TTSPlugin protocol
        if not isinstance(plugin_instance, TTSPlugin):
            raise PluginLoadError(
                f"Plugin '{entry_point.name}' does not implement TTSPlugin protocol. "
                f"Make sure it has: metadata property, generate(), list_voices(), validate_config()"
            )
        
        # Get metadata
        try:
            metadata = plugin_instance.metadata
        except Exception as e:
            raise PluginLoadError(f"Failed to get metadata from plugin '{entry_point.name}': {e}")
        
        # Validate metadata
        if not isinstance(metadata, PluginMetadata):
            raise PluginLoadError(
                f"Plugin '{entry_point.name}' metadata must be a PluginMetadata instance"
            )
        
        # Create LoadedPlugin wrapper
        loaded = LoadedPlugin(
            instance=plugin_instance,
            metadata=metadata,
            entry_point=entry_point.name,
            is_enabled=True,
        )
        
        return loaded
        
    except PluginLoadError:
        raise
    except Exception as e:
        raise PluginLoadError(f"Failed to load plugin '{entry_point.name}': {e}") from e


# =============================================================================
# PLUGIN MANAGEMENT
# =============================================================================


def get_loaded_plugins() -> Dict[str, LoadedPlugin]:
    """
    Get all currently loaded plugins.
    
    If plugins haven't been discovered yet, this will trigger discovery.
    
    Returns:
        Dictionary mapping plugin names to LoadedPlugin objects.
    """
    if not _discovery_done:
        discover_plugins()
    return _loaded_plugins.copy()


def get_plugin(name: str) -> Optional[LoadedPlugin]:
    """
    Get a specific loaded plugin by name.
    
    Args:
        name: The plugin name (from metadata.name).
    
    Returns:
        LoadedPlugin if found, None otherwise.
    """
    if not _discovery_done:
        discover_plugins()
    return _loaded_plugins.get(name)


def reload_plugins() -> List[LoadedPlugin]:
    """
    Force reload all plugins.
    
    This clears the cache and re-discovers all plugins. Useful after
    installing new plugins without restarting the application.
    
    Returns:
        List of newly loaded plugins.
    """
    return discover_plugins(force_reload=True)


# =============================================================================
# PROVIDER REGISTRATION
# =============================================================================


def register_plugin_provider(plugin: LoadedPlugin) -> bool:
    """
    Register a loaded plugin as a provider in the wakegen registry.
    
    This creates a wrapper that adapts the plugin to the TTSProvider interface
    used by the main wakegen system, then registers it so it can be used
    just like built-in providers.
    
    Args:
        plugin: The loaded plugin to register.
    
    Returns:
        True if registration succeeded, False otherwise.
    """
    from wakegen.providers.registry import register_provider, _PROVIDER_REGISTRY
    
    try:
        # Create a dynamic ProviderType for this plugin
        # We'll use a string-based type for plugins
        plugin_type_name = f"plugin:{plugin.name}"
        
        # Create a wrapper class that adapts TTSPlugin to TTSProvider
        wrapper_class = _create_plugin_wrapper_class(plugin)
        
        # For plugins, we need to handle them differently since they're not
        # in the ProviderType enum. We'll store them in a separate dict.
        # Register in a plugins-specific registry
        _register_plugin_to_registry(plugin.name, wrapper_class)
        
        logger.info(f"Registered plugin provider: {plugin.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register plugin '{plugin.name}' as provider: {e}")
        return False


def _create_plugin_wrapper_class(plugin: LoadedPlugin) -> type:
    """
    Create a wrapper class that adapts a TTSPlugin to the TTSProvider interface.
    
    This is needed because plugins use a slightly different interface than
    built-in providers, and we need to bridge the gap.
    """
    from wakegen.core.types import ProviderType
    from wakegen.models.config import ProviderConfig
    
    class PluginProviderWrapper:
        """
        Wrapper that adapts a TTSPlugin to work as a TTSProvider.
        """
        def __init__(self, config: ProviderConfig):
            self._plugin = plugin.instance
            self._config = config
            self._metadata = plugin.metadata
        
        @property
        def provider_type(self) -> str:
            """Return a string type since plugins aren't in the enum."""
            return f"plugin:{self._metadata.name}"
        
        async def generate(self, text: str, voice_id: str, output_path: str) -> None:
            """Delegate to plugin's generate method."""
            await self._plugin.generate(text, voice_id, output_path)
        
        async def list_voices(self):
            """Delegate to plugin's list_voices method."""
            return await self._plugin.list_voices()
        
        async def validate_config(self) -> None:
            """Delegate to plugin's validate_config method."""
            await self._plugin.validate_config()
    
    # Name the class after the plugin for better debugging
    PluginProviderWrapper.__name__ = f"{plugin.name.replace('-', '_')}_wrapper"
    PluginProviderWrapper.__qualname__ = PluginProviderWrapper.__name__
    
    return PluginProviderWrapper


# Plugin provider registry (separate from main registry since they're not enum-based)
_plugin_providers: Dict[str, type] = {}


def _register_plugin_to_registry(name: str, wrapper_class: type) -> None:
    """Register a plugin wrapper class to the plugin providers registry."""
    _plugin_providers[name] = wrapper_class


def get_plugin_provider(name: str, config=None):
    """
    Get an instance of a plugin provider by name.
    
    Args:
        name: The plugin name.
        config: Optional provider config.
    
    Returns:
        Instance of the plugin provider wrapper.
    
    Raises:
        ConfigError: If the plugin is not found.
    """
    from wakegen.models.config import ProviderConfig
    
    if name not in _plugin_providers:
        raise ConfigError(f"Plugin provider not found: {name}")
    
    wrapper_class = _plugin_providers[name]
    return wrapper_class(config or ProviderConfig())


def list_plugin_providers() -> List[str]:
    """Get list of registered plugin provider names."""
    return list(_plugin_providers.keys())


# =============================================================================
# AUTO-REGISTRATION
# =============================================================================


def auto_register_plugins() -> int:
    """
    Discover and register all plugins as providers.
    
    This is a convenience function that:
    1. Discovers all installed plugins
    2. Registers each as a provider
    
    Call this once at startup to make all plugins available.
    
    Returns:
        Number of successfully registered plugins.
    """
    plugins = discover_plugins()
    registered = 0
    
    for plugin in plugins:
        if plugin.is_enabled:
            if register_plugin_provider(plugin):
                registered += 1
    
    return registered


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Discovery
    "discover_plugins",
    "load_plugin",
    "reload_plugins",
    # Management
    "get_loaded_plugins",
    "get_plugin",
    # Provider registration
    "register_plugin_provider",
    "get_plugin_provider",
    "list_plugin_providers",
    "auto_register_plugins",
    # Constants
    "PLUGIN_ENTRY_POINT_GROUP",
]
