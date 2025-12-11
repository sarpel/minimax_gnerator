"""API Authentication Module

Provides API key authentication for FastAPI endpoints.

SEC-004 Fix: Implements simple API key authentication to protect API endpoints.
This prevents unauthorized access to the wake word generation API.

ELI5: Think of API keys like passwords for programs. Before a program can use
our API, it needs to provide a valid API key, just like you need a password
to log into a website.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# SEC-004: API key authentication
# The API key is checked in the X-API-Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_valid_api_keys() -> set[str]:
    """
    Get the set of valid API keys from environment variables.

    Returns:
        Set of valid API keys

    ELI5: This function reads the list of allowed "passwords" (API keys)
    from the server's configuration. Each key is like a different password
    that can be used to access the API.
    """
    # Read API keys from environment variable
    # Format: WAKEGEN_API_KEYS="key1,key2,key3"
    api_keys_str = os.getenv("WAKEGEN_API_KEYS", "")

    if not api_keys_str:
        logger.warning(
            "No API keys configured! Set WAKEGEN_API_KEYS environment variable. "
            "API will be accessible without authentication (NOT RECOMMENDED for production)."
        )
        return set()

    # Split by comma and strip whitespace
    keys = {key.strip() for key in api_keys_str.split(",") if key.strip()}

    logger.info(f"Loaded {len(keys)} API key(s) from environment")
    return keys


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    """
    Verify that the provided API key is valid.

    Args:
        api_key: API key from request header

    Returns:
        The validated API key

    Raises:
        HTTPException: If API key is missing or invalid

    ELI5: This function checks if the "password" (API key) provided by
    the program is in our list of allowed passwords. If it's not valid,
    we reject the request.
    """
    valid_keys = get_valid_api_keys()

    # If no keys are configured, allow all requests (development mode)
    if not valid_keys:
        logger.debug("No API keys configured - allowing request without authentication")
        return "development"

    # Check if API key was provided
    if not api_key:
        logger.warning("API request rejected: missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Check if API key is valid
    if api_key not in valid_keys:
        logger.warning(
            f"API request rejected: invalid API key (starts with {api_key[:8]}...)"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    logger.debug("API request authenticated successfully")
    return api_key


# Type alias for dependency injection
# Use this in route parameters: api_key: APIKeyDependency
APIKeyDependency = Annotated[str, Depends(verify_api_key)]
