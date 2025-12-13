"""
Circuit Breaker Pattern for TTS Providers

AR-004 Implementation: Circuit breakers protect against cascading failures
when external TTS services become unavailable or slow.

    WHAT IS A CIRCUIT BREAKER?
    ==========================
    ELI5: Imagine a light switch that automatically turns off when it detects
    too many failures. It stays off for a while to let things cool down,
    then "tests" if things are working again before fully turning back on.

    STATES:
    =======
    1. CLOSED (normal): Requests flow through normally
    2. OPEN (tripped): All requests fail immediately (fast-fail)
    3. HALF-OPEN (testing): Allow one request to test if service is back

    WHY USE THIS?
    =============
    - Prevents hammering a struggling service with retries
    - Fails fast instead of waiting for timeouts
    - Allows services time to recover
    - Improves overall system resilience

    USAGE:
    ======
    from wakegen.core.circuit_breaker import get_provider_breaker

    async def generate_audio(provider, text, voice_id, output_path):
        breaker = get_provider_breaker(provider.provider_type.value)
        try:
            await breaker.call_async(provider.generate, text, voice_id, output_path)
        except CircuitBreakerError:
            # Service is currently unavailable, use fallback
            pass
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

import pybreaker

from wakegen.core.exceptions import ProviderError
from wakegen.core.types import ProviderType

# CONCEPT: Module-level logger
# Each module gets its own logger so we can filter logs by component.
logger = logging.getLogger(__name__)


# =============================================================================
# CIRCUIT BREAKER LISTENER
# =============================================================================
# CONCEPT: Listeners let us react to state changes (logging, metrics, alerts)


class ProviderBreakerListener(pybreaker.CircuitBreakerListener):
    """
    Listener to log circuit breaker state changes.

    This helps us monitor when breakers trip (service problems) and
    when they recover (service back online).
    """

    def state_change(
        self, cb: pybreaker.CircuitBreaker, old_state: str, new_state: str
    ) -> None:
        """
        Called when the circuit breaker changes state.

        Args:
            cb: The circuit breaker instance
            old_state: Previous state (closed, open, half-open)
            new_state: New state
        """
        # SYNTAX: f-strings with embedded expressions
        # cb.name is the provider name we set when creating the breaker
        logger.warning(
            f"🔌 Circuit breaker '{cb.name}' state change: {old_state} → {new_state}"
        )

    def failure(self, cb: pybreaker.CircuitBreaker, exc: Exception) -> None:
        """
        Called when a call fails and is recorded as a failure.

        Args:
            cb: The circuit breaker instance
            exc: The exception that caused the failure
        """
        logger.debug(f"Circuit breaker '{cb.name}' recorded failure: {exc!s}")

    def success(self, cb: pybreaker.CircuitBreaker) -> None:
        """
        Called when a call succeeds.

        Args:
            cb: The circuit breaker instance
        """
        logger.debug(f"Circuit breaker '{cb.name}' recorded success")


# =============================================================================
# CIRCUIT BREAKER REGISTRY
# =============================================================================
# CONCEPT: We keep one breaker per provider to track failures independently


# PATTERN: Module-level registry (similar to provider registry)
# This dict maps provider names to their circuit breaker instances.
_provider_breakers: dict[str, pybreaker.CircuitBreaker] = {}

# CONCEPT: Shared listener instance
# We use one listener for all breakers to centralize logging.
_listener = ProviderBreakerListener()


def get_provider_breaker(
    provider_name: str,
    fail_max: int = 5,
    reset_timeout: int = 60,
    success_threshold: int = 2,
) -> pybreaker.CircuitBreaker:
    """
    Get or create a circuit breaker for a TTS provider.

    Each provider gets its own independent circuit breaker, so failures
    in one provider don't affect others.

    Args:
        provider_name: Unique name for this provider (e.g., "minimax", "edge_tts")
        fail_max: Number of consecutive failures before tripping open
        reset_timeout: Seconds to wait before testing if service is back
        success_threshold: Successful calls needed to close after half-open

    Returns:
        pybreaker.CircuitBreaker: The circuit breaker instance for this provider

    Example:
        >>> breaker = get_provider_breaker("minimax")
        >>> try:
        ...     await breaker.call_async(provider.generate, text, voice, path)
        ... except pybreaker.CircuitBreakerError:
        ...     print("Service unavailable, using fallback")
    """
    # PATTERN: Lazy initialization with caching
    # We only create a breaker the first time it's requested.
    if provider_name not in _provider_breakers:
        # Create new circuit breaker with our configuration
        breaker = pybreaker.CircuitBreaker(
            # CONCEPT: Each breaker has a unique name for logging/debugging
            name=f"tts_{provider_name}",
            # How many failures before we trip open?
            fail_max=fail_max,
            # How long to wait (in seconds) before trying again?
            reset_timeout=reset_timeout,
            # How many successes needed in half-open to fully close?
            # ELI5: We need 2 successful calls to trust the service again.
            success_threshold=success_threshold,
            # CONCEPT: Excluded exceptions
            # These should NOT count as failures (they're expected errors)
            exclude=[
                # Don't trip on validation errors - those are user's fault
                ProviderError,
                # Don't trip on invalid arguments
                ValueError,
            ],
            # Attach our listener for logging
            listeners=[_listener],
        )
        _provider_breakers[provider_name] = breaker
        logger.debug(f"Created circuit breaker for provider: {provider_name}")

    return _provider_breakers[provider_name]


def reset_provider_breaker(provider_name: str) -> None:
    """
    Reset a provider's circuit breaker to closed state.

    Useful when we know a provider is back online and want to
    immediately allow requests without waiting for reset_timeout.

    Args:
        provider_name: Name of the provider to reset
    """
    if provider_name in _provider_breakers:
        _provider_breakers[provider_name].close()
        logger.info(f"Circuit breaker for '{provider_name}' manually reset")


def get_breaker_status(provider_name: str) -> dict[str, Any]:
    """
    Get current status of a provider's circuit breaker.

    Args:
        provider_name: Name of the provider

    Returns:
        dict with state, fail_counter, success_counter, etc.
    """
    if provider_name not in _provider_breakers:
        return {"state": "no_breaker", "message": "No circuit breaker exists yet"}

    breaker = _provider_breakers[provider_name]
    return {
        "state": breaker.current_state,
        "name": breaker.name,
        "fail_counter": breaker.fail_counter,
        "success_counter": breaker.success_counter,
        "fail_max": breaker.fail_max,
        "reset_timeout": breaker.reset_timeout,
    }


def get_all_breaker_statuses() -> dict[str, dict[str, Any]]:
    """
    Get status of all registered circuit breakers.

    Returns:
        dict mapping provider names to their breaker status
    """
    return {name: get_breaker_status(name) for name in _provider_breakers}


# =============================================================================
# ASYNC WRAPPER
# =============================================================================
# CONCEPT: pybreaker doesn't natively support async, so we wrap it


async def call_with_breaker(
    provider_name: str,
    async_func: Callable[..., Coroutine[Any, Any, Any]],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Call an async function with circuit breaker protection.

    This is the main way to protect TTS provider calls. It wraps the
    async function so that failures are tracked by the circuit breaker.

    Args:
        provider_name: Name of the provider (for breaker lookup)
        async_func: The async function to call (e.g., provider.generate)
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Whatever the async_func returns

    Raises:
        pybreaker.CircuitBreakerError: If the circuit is open
        Any exception from async_func if it fails

    Example:
        >>> await call_with_breaker(
        ...     "minimax",
        ...     provider.generate,
        ...     "hello world",
        ...     "voice_123",
        ...     "/output/audio.wav"
        ... )
    """
    breaker = get_provider_breaker(provider_name)

    # CONCEPT: Wrapping async in sync for pybreaker
    # pybreaker's call() method expects a synchronous function.
    # We use asyncio.to_thread to run a sync wrapper that internally
    # awaits the async function in a new event loop.

    def sync_wrapper() -> Any:
        """
        Sync wrapper that runs the async function.

        This creates a new event loop and runs the async function in it.
        ELI5: It's like creating a mini async sandbox for pybreaker to use.
        """
        # Create a new event loop for this thread
        return asyncio.run(async_func(*args, **kwargs))

    # Run the sync wrapper in a thread pool with circuit breaker protection
    # SYNTAX: asyncio.to_thread runs a sync function in a thread pool
    # The breaker.call() wraps it with circuit breaker logic
    return await asyncio.to_thread(breaker.call, sync_wrapper)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "get_provider_breaker",
    "reset_provider_breaker",
    "get_breaker_status",
    "get_all_breaker_statuses",
    "call_with_breaker",
    "ProviderBreakerListener",
]
