"""
Web Server Configuration

This module defines configuration settings for the WakeGen web server using Pydantic.
Pydantic is a data validation library that allows us to define settings with type hints
and automatic validation.

    WHAT THIS DOES:
    ===============
    - Defines all configurable options for the web server
    - Loads settings from environment variables (with WAKEGEN_WEB_ prefix)
    - Provides sensible defaults for development
    - Validates that settings are correct types

    CONFIGURATION OPTIONS:
    ======================
    - host: The IP address to bind to (default: 127.0.0.1 = localhost only)
    - port: The port number to listen on (default: 8080)
    - debug: Enable debug mode with extra logging (default: False)
    - reload: Auto-reload when code changes (default: False, use for development)

    ENVIRONMENT VARIABLES:
    =====================
    You can override any setting with environment variables:
        WAKEGEN_WEB_HOST=0.0.0.0       # Listen on all interfaces
        WAKEGEN_WEB_PORT=9000           # Use port 9000
        WAKEGEN_WEB_DEBUG=true          # Enable debug mode
        WAKEGEN_WEB_RELOAD=true         # Enable auto-reload
"""

from pydantic_settings import BaseSettings
from typing import Optional


class WebConfig(BaseSettings):
    """
    Configuration settings for the WakeGen web server.

    These settings control how the web server runs. You can change them via
    environment variables or by creating a WebConfig instance with custom values.

        ATTRIBUTES EXPLAINED:
        =====================
        host (str):
            The network interface to bind to.
            - "127.0.0.1" = Only accept connections from this computer (more secure)
            - "0.0.0.0" = Accept connections from any computer on the network

        port (int):
            The port number the server listens on.
            - Ports below 1024 require admin/root privileges
            - Common choices: 8080, 8000, 3000, 5000

        debug (bool):
            When True, provides more detailed error messages.
            - NEVER enable this in production! It can leak sensitive info.

        reload (bool):
            When True, the server automatically restarts when you change code.
            - Great for development, but causes a brief interruption
            - Don't use in production

        log_level (str):
            How verbose the logging should be.
            - DEBUG: Everything (very noisy)
            - INFO: Normal operation messages
            - WARNING: Only problems
            - ERROR: Only serious problems
    """

    # =========================================================================
    # SERVER BINDING SETTINGS
    # =========================================================================
    # These control WHERE the server listens for connections

    host: str = "127.0.0.1"
    """
    IP address to bind to. Use 127.0.0.1 for local-only access (safer for development),
    or 0.0.0.0 to allow connections from other computers on your network.
    """

    port: int = 8080
    """
    Port number to listen on. Must be between 1 and 65535.
    Ports below 1024 require administrator privileges.
    """

    # =========================================================================
    # DEVELOPMENT SETTINGS
    # =========================================================================
    # These help during development but should be off in production

    debug: bool = False
    """
    Enable debug mode. When True:
    - More detailed error pages are shown
    - Extra logging is enabled
    WARNING: Never enable in production!
    """

    reload: bool = False
    """
    Enable auto-reload. When True:
    - Server restarts automatically when Python files change
    - Useful during development to see changes immediately
    """

    # =========================================================================
    # LOGGING SETTINGS
    # =========================================================================

    log_level: str = "INFO"
    """
    Logging verbosity level. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    DEBUG is most verbose, CRITICAL only shows fatal errors.
    """

    # =========================================================================
    # PYDANTIC CONFIGURATION
    # =========================================================================
    # This inner class tells Pydantic how to behave

    class Config:
        """
        Pydantic configuration for this settings class.

        env_prefix: All environment variables must start with "WAKEGEN_WEB_"
                   For example, WAKEGEN_WEB_PORT=9000 sets the port to 9000.

        case_sensitive: Environment variable names are case-insensitive
                       So WAKEGEN_WEB_PORT and wakegen_web_port both work.
        """
        env_prefix = "WAKEGEN_WEB_"
        case_sensitive = False


# Create a global instance with default settings
# This can be imported and used directly: from wakegen.web.config import settings
# Or you can create a new WebConfig() with custom values
_settings: Optional[WebConfig] = None


def get_settings() -> WebConfig:
    """
    Get the global settings instance, creating it if needed.

    This uses a "singleton" pattern - there's only one settings instance
    shared across the entire application. This ensures consistency.

        RETURNS:
        ========
        WebConfig: The global configuration settings instance.

        EXAMPLE:
        ========
        >>> settings = get_settings()
        >>> print(f"Server will run on port {settings.port}")
        Server will run on port 8080
    """
    global _settings
    if _settings is None:
        _settings = WebConfig()
    return _settings


# Convenience alias - you can do: from wakegen.web.config import settings
settings = get_settings()
