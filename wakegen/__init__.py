# This file marks the 'wakegen' directory as a Python package.
# It allows us to import code from this directory using 'import wakegen'.

# The version of our package.
# We use Semantic Versioning (Major.Minor.Patch).
__version__ = "0.1.0"

# Auto-register plugins when wakegen is imported
# This makes plugins available immediately without explicit initialization
def _init_plugins():
    """Initialize the plugin system on first import."""
    try:
        from wakegen.plugins import auto_register_plugins
        auto_register_plugins()
    except ImportError:
        # Plugins module not available (shouldn't happen but be safe)
        pass
    except Exception:
        # Don't crash on plugin errors during import
        pass

# Call on import (lazy - only when plugins are needed)
# _init_plugins()  # Uncomment to auto-load plugins at import time