"""
TTS Providers API Router

This module defines all API endpoints for managing TTS providers.
It exposes functionality to list providers, check their status, list voices,
and test audio generation.

    ENDPOINTS:
    ==========
    GET  /                    - List all providers with availability status
    GET  /{provider_id}/status - Check if a specific provider is available
    GET  /{provider_id}/voices - List available voices for a provider
    POST /{provider_id}/test   - Generate a test audio sample
    GET  /summary             - Get summary stats for dashboard

    HOW THIS INTEGRATES WITH EXISTING CODE:
    =======================================
    This router uses the existing provider registry from:
        wakegen.providers.registry

    It wraps the existing functions like:
        - discover_available_providers() -> list of ProviderInfo
        - get_provider() -> TTSProvider instance
        - check_provider_availability() -> availability status

    FASTAPI ROUTER PATTERNS:
    ========================
    - @router.get("/path") decorates an async function to handle GET requests
    - Path parameters like {provider_id} are passed as function arguments
    - Pydantic models define request/response schemas (auto-validated)
    - HTTPException is raised for error responses
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Import existing provider functionality from the codebase
from wakegen.providers.registry import (
    discover_available_providers,
    check_provider_availability,
    get_provider,
    ProviderInfo,
)
from wakegen.core.types import ProviderType
from wakegen.models.audio import Voice
from wakegen.config.settings import get_provider_config

# Set up logging for this module
logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS (Response Schemas)
# =============================================================================
# Pydantic models define the shape of our API responses.
# FastAPI uses these to:
#   1. Validate response data automatically
#   2. Generate OpenAPI documentation
#   3. Serialize Python objects to JSON


class ProviderResponse(BaseModel):
    """
    Response schema for a single TTS provider.

    This mirrors the ProviderInfo dataclass from the registry but as a
    Pydantic model for API serialization.

        FIELDS EXPLAINED:
        =================
        id: The provider's unique identifier (e.g., "edge_tts", "piper")
        name: Human-readable name (e.g., "Edge TTS")
        description: What this provider does
        is_available: Whether it can be used right now
        requires_gpu: Does it need a GPU?
        requires_api_key: Does it need an API key?
        missing_dependencies: What's missing to make it work
        install_hint: How to install missing dependencies
    """

    id: str = Field(..., description="Unique provider identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Provider description")
    is_available: bool = Field(..., description="Whether provider is usable")
    requires_gpu: bool = Field(False, description="Requires GPU for inference")
    requires_api_key: bool = Field(False, description="Requires API key")
    missing_dependencies: List[str] = Field(
        default_factory=list,
        description="List of missing dependencies"
    )
    install_hint: Optional[str] = Field(None, description="Installation instructions")

    class Config:
        """Pydantic config for this model."""
        # Allow creating from ORM objects or dataclasses
        from_attributes = True


class VoiceResponse(BaseModel):
    """
    Response schema for a single voice.

    Represents a voice that can be used for TTS generation.
    """

    id: str = Field(..., description="Unique voice identifier")
    name: str = Field(..., description="Human-readable voice name")
    language: str = Field(..., description="Language code (e.g., en-US, tr-TR)")
    gender: str = Field(..., description="Voice gender (male, female, neutral)")


class ProviderSummaryResponse(BaseModel):
    """
    Summary statistics for the dashboard card.
    """

    available_count: int = Field(..., description="Number of available providers")
    total_count: int = Field(..., description="Total number of providers")


class TestGenerationRequest(BaseModel):
    """
    Request schema for testing TTS generation.
    """

    text: str = Field("Hello, this is a test.", description="Text to synthesize")
    voice_id: Optional[str] = Field(None, description="Voice ID to use (optional)")


class TestGenerationResponse(BaseModel):
    """
    Response schema for test generation.
    """

    success: bool = Field(..., description="Whether generation succeeded")
    message: str = Field(..., description="Result message")
    audio_url: Optional[str] = Field(None, description="URL to play the audio")


# =============================================================================
# API ROUTER
# =============================================================================
# APIRouter is like a mini-FastAPI app that groups related endpoints.
# We'll mount this in the main app with a prefix like "/api/providers".

router = APIRouter()


@router.get(
    "/",
    response_model=List[ProviderResponse],
    summary="List all TTS providers",
    description="Returns information about all supported TTS providers and their availability status."
)
async def list_providers(
    available_only: bool = Query(
        False,
        description="If true, only return providers that are ready to use"
    )
) -> List[ProviderResponse]:
    """
    List all TTS providers with their availability status.

    This endpoint returns information about every TTS provider that WakeGen
    supports, including whether it's currently available for use.

        PARAMETERS:
        ===========
        available_only (bool):
            When true, filters out providers that aren't ready to use.
            Useful for populating dropdowns on the UI.

        RETURNS:
        ========
        List[ProviderResponse]: A list of provider information objects.

        HOW IT WORKS:
        =============
        1. Calls discover_available_providers() from the registry
        2. Optionally filters to only available ones
        3. Converts ProviderInfo dataclasses to ProviderResponse models
        4. FastAPI automatically serializes to JSON

        EXAMPLE RESPONSE:
        =================
        [
            {
                "id": "edge_tts",
                "name": "Edge TTS",
                "description": "Microsoft Edge Text-to-Speech",
                "is_available": true,
                "requires_gpu": false,
                ...
            },
            ...
        ]
    """
    # Get all providers from the existing registry
    # discover_available_providers() checks each provider's dependencies
    providers = discover_available_providers()

    # Filter if requested
    if available_only:
        providers = [p for p in providers if p.is_available]

    # Convert to response models
    # We need to extract the provider type ID from the ProviderInfo
    result = []
    for p in providers:
        # Get the provider type enum value as the ID
        # ProviderInfo has a 'type' field that's a ProviderType enum
        provider_id = p.type.value if hasattr(p, 'type') else p.name.lower().replace(' ', '_')

        result.append(ProviderResponse(
            id=provider_id,
            name=p.name,
            description=p.description,
            is_available=p.is_available,
            requires_gpu=p.requires_gpu,
            requires_api_key=p.requires_api_key,
            missing_dependencies=p.missing_dependencies or [],
            install_hint=p.install_hint
        ))

    return result


@router.get(
    "/summary",
    response_model=ProviderSummaryResponse,
    summary="Get provider summary for dashboard"
)
async def get_provider_summary() -> ProviderSummaryResponse:
    """
    Get summary statistics about providers for the dashboard card.

    Returns a quick count of available vs total providers, used to display
    on the dashboard without needing all the details.
    """
    providers = discover_available_providers()
    available = sum(1 for p in providers if p.is_available)

    return ProviderSummaryResponse(
        available_count=available,
        total_count=len(providers)
    )


@router.get(
    "/{provider_id}/status",
    response_model=ProviderResponse,
    summary="Check provider availability"
)
async def get_provider_status(provider_id: str) -> ProviderResponse:
    """
    Get detailed status information for a specific provider.

    This checks whether it's available and what dependencies are missing.

        PATH PARAMETERS:
        ================
        provider_id: The provider's ID (e.g., "edge_tts", "piper")

        RETURNS:
        ========
        ProviderResponse: Detailed status information

        RAISES:
        =======
        404: If the provider ID is not recognized
    """
    # Try to convert the ID to a ProviderType enum
    try:
        provider_type = ProviderType(provider_id.lower())
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown provider: {provider_id}"
        )

    # Check availability using existing function
    info = check_provider_availability(provider_type)

    return ProviderResponse(
        id=provider_id,
        name=info.name,
        description=info.description,
        is_available=info.is_available,
        requires_gpu=info.requires_gpu,
        requires_api_key=info.requires_api_key,
        missing_dependencies=info.missing_dependencies or [],
        install_hint=info.install_hint
    )


@router.get(
    "/{provider_id}/voices",
    response_model=List[VoiceResponse],
    summary="List voices for a provider"
)
async def list_provider_voices(
    provider_id: str,
    language: Optional[str] = Query(
        None,
        description="Filter by language code (e.g., tr-TR, en-US)"
    ),
    limit: int = Query(
        100,
        ge=1,
        le=500,
        description="Maximum number of voices to return"
    )
) -> List[VoiceResponse]:
    """
    List available voices for a specific TTS provider.

    Some providers have hundreds of voices, so use the limit and language
    parameters to filter the results.

        PATH PARAMETERS:
        ================
        provider_id: The provider's ID

        QUERY PARAMETERS:
        =================
        language: Optional language filter (e.g., "tr-TR" for Turkish)
        limit: Maximum voices to return (default 100, max 500)

        RETURNS:
        ========
        List[VoiceResponse]: Available voices

        RAISES:
        =======
        404: Unknown provider
        503: Provider not available
    """
    # Validate provider type
    try:
        provider_type = ProviderType(provider_id.lower())
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown provider: {provider_id}"
        )

    # Check if provider is available
    info = check_provider_availability(provider_type)
    if not info.is_available:
        raise HTTPException(
            status_code=503,
            detail=f"Provider {provider_id} is not available. "
                   f"Missing: {', '.join(info.missing_dependencies or ['unknown'])}"
        )

    # Get provider instance
    try:
        provider_config = get_provider_config()
        provider = get_provider(provider_type, provider_config)

        # Fetch voices
        voices = await provider.list_voices()

        # Filter by language if requested
        if language:
            voices = [
                v for v in voices
                if language.lower() in v.language.lower()
            ]

        # Apply limit
        voices = voices[:limit]

        # Convert to response models
        return [
            VoiceResponse(
                id=v.id,
                name=v.name,
                language=v.language,
                gender=v.gender.value
            )
            for v in voices
        ]

    except Exception as e:
        logger.error(f"Error listing voices for {provider_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching voices: {str(e)}"
        )


@router.post(
    "/{provider_id}/test",
    response_model=TestGenerationResponse,
    summary="Test TTS generation"
)
async def test_provider(
    provider_id: str,
    request: TestGenerationRequest
) -> TestGenerationResponse:
    """
    Generate a test audio sample to verify the provider works.

    This creates a small audio file and returns a URL to play it.
    Useful for testing provider configuration before running a full
    generation job.

        PATH PARAMETERS:
        ================
        provider_id: The provider's ID

        REQUEST BODY:
        =============
        text: Text to synthesize (default: "Hello, this is a test.")
        voice_id: Optional specific voice to use

        RETURNS:
        ========
        TestGenerationResponse: Success status and audio URL

        RAISES:
        =======
        404: Unknown provider
        503: Provider not available
    """
    # Validate provider type
    try:
        provider_type = ProviderType(provider_id.lower())
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown provider: {provider_id}"
        )

    # Check availability
    info = check_provider_availability(provider_type)
    if not info.is_available:
        raise HTTPException(
            status_code=503,
            detail=f"Provider {provider_id} is not available"
        )

    try:
        # Get provider and generate
        provider_config = get_provider_config()
        provider = get_provider(provider_type, provider_config)

        # Determine voice to use
        voice_id = request.voice_id
        if not voice_id:
            # Auto-select first voice
            voices = await provider.list_voices()
            if not voices:
                return TestGenerationResponse(
                    success=False,
                    message="No voices available for this provider"
                )
            voice_id = voices[0].id

        # Generate to a test directory
        from pathlib import Path
        import time
        
        # Use output/test_samples directory
        output_dir = Path("./output/test_samples")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"test_{provider_id}_{int(time.time())}.wav"
        file_path = output_dir / filename
        
        await provider.generate(request.text, voice_id, str(file_path))

        # Return a URL that the audio router can handle
        # The audio router expects a file path encoded in the URL
        # We'll use the absolute path to be safe, or relative to cwd
        abs_path = file_path.resolve()
        
        # URL encode the path for the API call
        from urllib.parse import quote
        encoded_path = quote(str(abs_path))
        audio_url = f"/api/audio/play/{encoded_path}"

        return TestGenerationResponse(
            success=True,
            message=f"Successfully generated test audio with voice {voice_id}",
            audio_url=audio_url
        )

    except Exception as e:
        logger.error(f"Test generation failed for {provider_id}: {e}")
        return TestGenerationResponse(
            success=False,
            message=f"Generation failed: {str(e)}"
        )
