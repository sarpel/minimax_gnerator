"""
WakeGen Web UI Module

This module provides a beautiful web interface for the WakeGen wake word generator.
It uses FastAPI for the backend API and Jinja2 templates with TailwindCSS, Alpine.js,
and HTMX for a modern, reactive frontend.

    WHAT THIS MODULE DOES:
    =======================
    - Provides a web server that exposes all WakeGen functionality via a browser
    - Creates REST API endpoints for providers, generation, configuration, etc.
    - Uses WebSockets for real-time progress updates during generation
    - Serves static files (CSS, JS) and HTML templates

    HOW TO USE:
    ===========
    From the command line:
        wakegen serve                    # Start with defaults
        wakegen serve --port 9000        # Custom port
        wakegen serve --reload           # Auto-reload for development

    From Python:
        from wakegen.web import create_app
        app = create_app()
        # Use with uvicorn: uvicorn.run(app, host="127.0.0.1", port=8080)

    ARCHITECTURE:
    =============
    wakegen/web/
    ├── __init__.py         # This file - module entry point
    ├── app.py              # FastAPI application factory
    ├── config.py           # Web server configuration
    ├── websocket.py        # WebSocket connection manager
    ├── routers/            # API endpoint definitions
    │   ├── providers.py    # TTS provider endpoints
    │   ├── config.py       # Configuration endpoints
    │   ├── generation.py   # Generation endpoints
    │   └── ...             # Other routers
    ├── services/           # Business logic layer
    │   └── generation_service.py  # Background job management
    ├── templates/          # Jinja2 HTML templates
    │   ├── base.html       # Base layout template
    │   ├── pages/          # Full page templates
    │   └── components/     # Reusable UI components
    └── static/             # Static assets (CSS, JS, images)
        ├── css/
        └── js/
"""

# Import the main application factory so users can do:
# from wakegen.web import create_app
from wakegen.web.app import create_app

# Export the app factory for easy access
__all__ = ["create_app"]
