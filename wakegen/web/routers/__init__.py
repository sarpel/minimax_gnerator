"""
WakeGen Web Routers Package

This package contains all API route definitions organized by feature area.
Each router module groups related endpoints together.

    ROUTER MODULES:
    ===============
    - providers.py   : TTS provider management (list, status, test)
    - config_router.py : Configuration file management
    - generation.py  : Audio sample generation
    - augmentation.py: Audio augmentation settings
    - quality.py     : Quality assurance and validation
    - export.py      : Dataset export functionality
    - system.py      : System status and utilities
    - audio.py       : Audio file serving and playback

    HOW ROUTERS WORK:
    =================
    Each router is an APIRouter instance that defines endpoints.
    The main app.py includes each router with a URL prefix.

    Example:
        # In providers.py
        router = APIRouter()

        @router.get("/")
        async def list_providers():
            return [...]

        # In app.py
        app.include_router(providers.router, prefix="/api/providers")

        # Result: GET /api/providers/ calls list_providers()
"""

# We'll import routers here as they're created
# This allows: from wakegen.web.routers import providers

# Note: Import statements will be added as router modules are created
# For now, we leave this minimal to avoid import errors
