# =============================================================================
# Wakegen Plugin System
# =============================================================================
# This module enables third-party TTS provider plugins to be discovered and
# registered automatically using Python's entry points mechanism.
#
# Plugin developers can create their own TTS providers and publish them as
# separate packages. When installed, they'll be automatically discovered
# and available for use in wakegen.
#
# Example plugin package structure:
#   my-wakegen-plugin/
#   ├── pyproject.toml        # With entry point: wakegen.plugins
#   ├── my_provider/
#   │   ├── __init__.py
#   │   └── provider.py       # Implements TTSPlugin protocol
#
# To use: pip install my-wakegen-plugin
# Then the provider will be automatically available!

from wakegen.plugins.base import (
    TTSPlugin,
    PluginMetadata,
    PluginLoadError,
)
from wakegen.plugins.discovery import (
    discover_plugins,
    load_plugin,
    get_loaded_plugins,
    register_plugin_provider,
    reload_plugins,
)

__all__ = [
    # Base classes and protocols
    "TTSPlugin",
    "PluginMetadata",
    "PluginLoadError",
    # Discovery functions
    "discover_plugins",
    "load_plugin",
    "get_loaded_plugins",
    "register_plugin_provider",
    "reload_plugins",
]
