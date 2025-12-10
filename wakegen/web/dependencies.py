"""
Web API Dependencies

This module provides FastAPI dependencies for authentication and security.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from wakegen.web.config import get_settings

# Define the API Key header scheme
# "X-API-Key" is a standard header name for API keys
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key_header_val: Annotated[str | None, Security(api_key_header)],
) -> str | None:
    """
    Verify the API key provided in the request header.

    If an API key is configured in settings:
    - Rejects requests without the key (401 Unauthorized)
    - Rejects requests with an invalid key (401 Unauthorized)

    If no API key is configured (default):
    - Allows all requests (returns None)

    Args:
        api_key_header_val: The value of the X-API-Key header

    Returns:
        The API key if valid, or None if auth is disabled.

    Raises:
        HTTPException(401): If authentication fails
    """
    settings = get_settings()

    # If no API key is configured, authentication is disabled
    if not settings.api_key:
        return None

    # Get the configured key's secret value
    configured_key = settings.api_key.get_secret_value()

    # Check if header is missing
    if not api_key_header_val:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key header (X-API-Key)",
        )

    # Check if key matches (constant-time comparison would be better for high security,
    # but for this use case direct comparison is acceptable)
    if api_key_header_val != configured_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )

    return api_key_header_val
