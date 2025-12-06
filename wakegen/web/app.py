"""
FastAPI Application Factory

This module creates and configures the main FastAPI application for the WakeGen Web UI.
It follows the "application factory" pattern - a function that builds and returns
a configured application instance.

    WHAT IS AN APPLICATION FACTORY?
    ================================
    Instead of creating the app at module level (app = FastAPI()), we use a function
    that creates it. This allows:
    - Different configurations for testing vs production
    - Multiple instances if needed
    - Cleaner startup logic

    KEY COMPONENTS:
    ===============
    1. FastAPI app: The main web framework that handles HTTP requests
    2. CORS middleware: Allows the browser to make requests from different origins
    3. Static files: Serves CSS, JavaScript, and images
    4. Jinja2 templates: Renders HTML pages with dynamic content
    5. Routers: Groups of related API endpoints (providers, config, etc.)

    HOW FASTAPI WORKS:
    ==================
    FastAPI is a modern Python web framework that:
    - Uses type hints to validate request/response data automatically
    - Generates OpenAPI documentation automatically
    - Supports async/await for high performance
    - Is based on Starlette (for web handling) and Pydantic (for validation)

    REQUEST FLOW:
    =============
    Browser Request → CORS Check → Router Match → Endpoint Function → Response
                                      ↓
                              Template Render (for HTML)
                                   or
                              JSON Response (for API)
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Import our web configuration
from wakegen.web.config import WebConfig, get_settings

# Set up logging for this module
# Logging helps us track what's happening in the application
logger = logging.getLogger(__name__)


def create_app(config: Optional[WebConfig] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    This is the "application factory" - it builds a complete, configured
    FastAPI instance ready to handle web requests.

        PARAMETERS:
        ===========
        config (WebConfig, optional):
            Custom configuration settings. If not provided, uses defaults
            from environment variables or fallback values.

        RETURNS:
        ========
        FastAPI: A fully configured application instance.

        WHAT THIS FUNCTION DOES:
        ========================
        1. Creates a new FastAPI instance with metadata (title, version, etc.)
        2. Adds CORS middleware to allow browser requests
        3. Mounts static file directory for CSS/JS/images
        4. Sets up Jinja2 template engine for HTML rendering
        5. Adds the health check endpoint
        6. Registers all API routers (providers, config, generation, etc.)

        EXAMPLE USAGE:
        ==============
        # Create app with default settings
        >>> app = create_app()

        # Create app with custom settings
        >>> custom_config = WebConfig(port=9000, debug=True)
        >>> app = create_app(custom_config)

        # Run with uvicorn
        >>> import uvicorn
        >>> uvicorn.run(app, host="127.0.0.1", port=8080)
    """
    # Use provided config or get global settings
    # The "or" here means: if config is None, use get_settings() instead
    settings = config or get_settings()

    # =========================================================================
    # CREATE THE FASTAPI INSTANCE
    # =========================================================================
    # FastAPI() creates our web application. The parameters here provide
    # metadata that appears in the auto-generated API documentation.

    app = FastAPI(
        # Title appears at the top of the API docs page
        title="WakeGen Web UI",

        # Description shown in the API docs - supports Markdown!
        description="""
## Wake Word Dataset Generator

A comprehensive web interface for generating, augmenting, and exporting
wake word audio datasets using multiple TTS providers.

### Features
- 🎙️ **11 TTS Providers** - Edge TTS, Piper, Coqui, and more
- 🎛️ **Augmentation** - Add noise, reverb, room simulation
- 📦 **Export** - OpenWakeWord, Mycroft, Picovoice formats
- 📊 **Quality Assurance** - Validate and analyze your dataset
        """,

        # Version of our API - should match pyproject.toml
        version="1.0.0",

        # URL where API docs are served (default is /docs)
        docs_url="/api/docs",

        # URL for alternative ReDoc documentation
        redoc_url="/api/redoc",

        # OpenAPI JSON schema URL
        openapi_url="/api/openapi.json",

        # Debug mode enables more detailed error responses
        debug=settings.debug,
    )

    # =========================================================================
    # ADD CORS MIDDLEWARE
    # =========================================================================
    # CORS = Cross-Origin Resource Sharing
    #
    # By default, browsers block requests from one website to another.
    # For example, if you open http://localhost:3000 and it tries to fetch
    # from http://localhost:8080/api/..., the browser blocks it.
    #
    # CORS middleware tells the browser: "It's okay, allow these requests."
    #
    # For development, we allow everything (*). In production, you'd want
    # to restrict this to only your frontend's domain.

    app.add_middleware(
        CORSMiddleware,
        # Which origins (websites) can make requests
        # ["*"] means "any website" - okay for local development
        allow_origins=["*"],

        # Allow cookies and authentication headers
        allow_credentials=True,

        # Which HTTP methods are allowed (GET, POST, PUT, DELETE, etc.)
        allow_methods=["*"],

        # Which request headers are allowed
        allow_headers=["*"],
    )

    # =========================================================================
    # SET UP STATIC FILES
    # =========================================================================
    # Static files are things that don't change: CSS, JavaScript, images.
    # Instead of processing them, we just serve them directly to the browser.
    #
    # StaticFiles mounts a directory and serves its contents at a URL path.
    # For example: /static/css/custom.css → templates/../static/css/custom.css

    # Get the directory where this Python file lives
    web_dir = Path(__file__).parent

    # Path to the static files directory
    static_dir = web_dir / "static"

    # Create the directory if it doesn't exist
    # (parents=True creates parent directories too, exist_ok=True doesn't error if exists)
    static_dir.mkdir(parents=True, exist_ok=True)

    # Mount the static files directory at the /static URL path
    # name="static" allows us to reference it in templates with url_for("static", ...)
    app.mount(
        "/static",
        StaticFiles(directory=str(static_dir)),
        name="static"
    )

    # =========================================================================
    # SET UP JINJA2 TEMPLATES
    # =========================================================================
    # Jinja2 is a template engine that lets us create HTML pages with
    # dynamic content. It looks like HTML but with special {{ }} and {% %} tags.
    #
    # Example template syntax:
    #   <h1>Hello, {{ username }}!</h1>
    #   {% for item in items %}
    #     <li>{{ item.name }}</li>
    #   {% endfor %}

    # Path to the templates directory
    templates_dir = web_dir / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    # Create the Jinja2 templates engine
    # This is stored on `app.state` so we can access it from route handlers
    app.state.templates = Jinja2Templates(directory=str(templates_dir))

    # =========================================================================
    # REGISTER ROUTES (Endpoints)
    # =========================================================================
    # Routes define what happens when someone visits a URL.
    # We start with basic routes here and add routers for more complex endpoints.

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request) -> HTMLResponse:
        """
        Serve the main dashboard page.

        This is what users see when they visit http://localhost:8080/

            PARAMETERS:
            ===========
            request (Request):
                The incoming HTTP request. FastAPI automatically provides this.
                We need it to render templates (templates need the request object).

            RETURNS:
            ========
            HTMLResponse: The rendered HTML page.

            HOW TEMPLATE RENDERING WORKS:
            =============================
            1. We call templates.TemplateResponse() with the template filename
            2. Jinja2 reads the template file
            3. It replaces {{ variables }} with actual values from our context
            4. The rendered HTML is returned to the browser
        """
        # Get the templates engine from app state
        templates: Jinja2Templates = request.app.state.templates

        # Render the dashboard template
        # "request" MUST be included in the context for Jinja2 to work properly
        return templates.TemplateResponse(
            "pages/dashboard.html",
            {
                "request": request,
                "page_title": "Dashboard",
            }
        )

    @app.get("/providers", response_class=HTMLResponse)
    async def providers_page(request: Request) -> HTMLResponse:
        """Serve the providers management page."""
        templates: Jinja2Templates = request.app.state.templates
        return templates.TemplateResponse(
            "pages/providers.html",
            {"request": request, "page_title": "Providers"}
        )

    @app.get("/generate", response_class=HTMLResponse)
    async def generate_page(request: Request) -> HTMLResponse:
        """Serve the generation page."""
        templates: Jinja2Templates = request.app.state.templates
        return templates.TemplateResponse(
            "pages/generate.html",
            {"request": request, "page_title": "Generate"}
        )

    @app.get("/config", response_class=HTMLResponse)
    async def config_page(request: Request) -> HTMLResponse:
        """Serve the configuration editor page."""
        templates: Jinja2Templates = request.app.state.templates
        return templates.TemplateResponse(
            "pages/config.html",
            {"request": request, "page_title": "Configuration"}
        )

    @app.get("/augmentation", response_class=HTMLResponse)
    async def augmentation_page(request: Request) -> HTMLResponse:
        """Serve the augmentation settings page."""
        templates: Jinja2Templates = request.app.state.templates
        return templates.TemplateResponse(
            "pages/augmentation.html",
            {"request": request, "page_title": "Augmentation"}
        )

    @app.get("/quality", response_class=HTMLResponse)
    async def quality_page(request: Request) -> HTMLResponse:
        """Serve the quality dashboard page."""
        templates: Jinja2Templates = request.app.state.templates
        return templates.TemplateResponse(
            "pages/quality.html",
            {"request": request, "page_title": "Quality"}
        )

    @app.get("/export", response_class=HTMLResponse)
    async def export_page(request: Request) -> HTMLResponse:
        """Serve the export page."""
        templates: Jinja2Templates = request.app.state.templates
        return templates.TemplateResponse(
            "pages/export.html",
            {"request": request, "page_title": "Export"}
        )

    @app.get("/system", response_class=HTMLResponse)
    async def system_page(request: Request) -> HTMLResponse:
        """Serve the system status page."""
        templates: Jinja2Templates = request.app.state.templates
        return templates.TemplateResponse(
            "pages/system.html",
            {"request": request, "page_title": "System"}
        )

    @app.get("/api/health")
    async def health_check() -> dict:
        """
        Health check endpoint.

        This is a simple endpoint that returns OK if the server is running.
        It's commonly used by:
        - Load balancers to check if the server is healthy
        - Monitoring systems to track uptime
        - Developers to verify the server is responding

            RETURNS:
            ========
            dict: A simple status object like {"status": "ok"}

            WHY NO TYPE ANNOTATION ON RETURN?
            ==================================
            FastAPI automatically converts Python dicts to JSON responses.
            The return type `dict` is implicit here, but we could use
            Pydantic models for more complex responses.
        """
        return {
            "status": "ok",
            "service": "wakegen-web",
            "version": "1.0.0"
        }

    # =========================================================================
    # REGISTER API ROUTERS
    # =========================================================================
    # Routers group related endpoints together. Instead of defining all
    # endpoints in this file, we organize them into separate modules.
    #
    # Each router is "included" into the main app with a URL prefix.
    # For example, the providers router with prefix="/api/providers" means:
    #   - Router's "/" becomes "/api/providers/"
    #   - Router's "/{id}" becomes "/api/providers/{id}"

    # Import and register routers (we'll create these files next)
    try:
        from wakegen.web.routers import providers, config_router, generation

        # Provider management endpoints
        app.include_router(
            providers.router,
            prefix="/api/providers",
            tags=["Providers"]  # Groups endpoints in API docs
        )

        # Configuration endpoints
        app.include_router(
            config_router.router,
            prefix="/api/config",
            tags=["Configuration"]
        )

        # Generation endpoints
        app.include_router(
            generation.router,
            prefix="/api/generate",
            tags=["Generation"]
        )

        logger.info("All API routers registered successfully")

    except ImportError as e:
        # If routers aren't created yet, log a warning but don't crash
        logger.warning(f"Some routers not yet available: {e}")

    # Register audio router separately (optional feature)
    try:
        from wakegen.web.routers import audio

        app.include_router(
            audio.router,
            prefix="/api/audio",
            tags=["Audio"]
        )
        logger.info("Audio router registered")
    except ImportError as e:
        logger.warning(f"Audio router not available: {e}")

    # Register augmentation router
    try:
        from wakegen.web.routers import augmentation

        app.include_router(
            augmentation.router,
            prefix="/api/augmentation",
            tags=["Augmentation"]
        )
        logger.info("Augmentation router registered")
    except ImportError as e:
        logger.warning(f"Augmentation router not available: {e}")

    # Register quality router
    try:
        from wakegen.web.routers import quality

        app.include_router(
            quality.router,
            prefix="/api/quality",
            tags=["Quality"]
        )
        logger.info("Quality router registered")
    except ImportError as e:
        logger.warning(f"Quality router not available: {e}")

    # Register export router
    try:
        from wakegen.web.routers import export

        app.include_router(
            export.router,
            prefix="/api/export",
            tags=["Export"]
        )
        logger.info("Export router registered")
    except ImportError as e:
        logger.warning(f"Export router not available: {e}")

    # Register system router
    try:
        from wakegen.web.routers import system

        app.include_router(
            system.router,
            prefix="/api/system",
            tags=["System"]
        )
        logger.info("System router registered")
    except ImportError as e:
        logger.warning(f"System router not available: {e}")

    # =========================================================================
    # WEBSOCKET ROUTES
    # =========================================================================
    # WebSocket endpoints for real-time updates (progress, stats, etc.)

    try:
        from wakegen.web import websocket

        app.include_router(
            websocket.router,
            prefix="/ws",
            tags=["WebSocket"]
        )
        logger.info("WebSocket router registered")
    except ImportError as e:
        logger.warning(f"WebSocket router not available: {e}")

    # =========================================================================
    # STARTUP AND SHUTDOWN EVENTS
    # =========================================================================
    # These functions run when the server starts and stops.
    # Useful for initializing connections, loading data, etc.

    @app.on_event("startup")
    async def on_startup() -> None:
        """
        Called when the server starts up.

        Use this for:
        - Connecting to databases
        - Loading configuration
        - Initializing caches
        - Logging startup messages
        """
        logger.info("=" * 60)
        logger.info("🎤 WakeGen Web UI starting up...")
        logger.info(f"   Host: {settings.host}")
        logger.info(f"   Port: {settings.port}")
        logger.info(f"   Debug: {settings.debug}")
        logger.info("=" * 60)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        """
        Called when the server shuts down.

        Use this for:
        - Closing database connections
        - Flushing caches to disk
        - Cleanup operations
        """
        logger.info("🎤 WakeGen Web UI shutting down...")

    return app
