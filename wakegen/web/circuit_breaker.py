"""Circuit Breaker Pattern for TTS API Calls

AR-004 Fix: Implements circuit breaker pattern to prevent cascading failures
when TTS APIs are down or experiencing issues.

ELI5: Imagine you have a light switch that's broken. If you keep flipping it,
you might damage the electrical system. A circuit breaker is like a smart switch
that says "this isn't working, let's stop trying for a bit and give it time to fix itself."
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """States of the circuit breaker."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Circuit is open, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open and blocks a request."""

    pass


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    AR-004 Fix: Prevents overwhelming a failing TTS API with requests.
    When failures exceed a threshold, the circuit "opens" and blocks
    requests for a cooldown period.

    ELI5: This is like a protective fuse in your home's electrical system.
    If something goes wrong (too many failures), it "trips" and stops all
    electricity (requests) for a while to prevent damage. After some time,
    it tries again to see if things are working.

    States:
    - CLOSED: Everything working, requests go through
    - OPEN: Too many failures, blocking all requests
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type[Exception] = Exception,
        name: str = "circuit_breaker",
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that counts as failure
            name: Name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name

        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.last_success_time: datetime | None = None

        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"threshold={failure_threshold}, timeout={recovery_timeout}s"
        )

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If func raises an exception
        """
        # Check if circuit should transition from OPEN to HALF_OPEN
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
                self.state = CircuitState.HALF_OPEN
            else:
                # Circuit still open, block request
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service unavailable, try again in {self._time_until_reset()}s"
                )

        try:
            # Execute the function
            result = await func(*args, **kwargs)

            # Success! Reset failure count
            self._on_success()
            return result

        except self.expected_exception as e:
            # Expected failure, record it
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if not self.last_failure_time:
            return True

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def _time_until_reset(self) -> int:
        """Calculate seconds until circuit can attempt reset."""
        if not self.last_failure_time:
            return 0

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        remaining = max(0, self.recovery_timeout - elapsed)
        return int(remaining)

    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.last_success_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            # Recovery successful, close circuit
            logger.info(f"Circuit breaker '{self.name}' recovered, closing circuit")
            self.state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            # Too many failures, open circuit
            if self.state != CircuitState.OPEN:
                logger.warning(
                    f"Circuit breaker '{self.name}' OPENED after {self.failure_count} failures. "
                    f"Blocking requests for {self.recovery_timeout}s"
                )
                self.state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        logger.info(f"Circuit breaker '{self.name}' manually reset")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self.state == CircuitState.OPEN

    def get_status(self) -> dict[str, Any]:
        """
        Get current circuit breaker status.

        Returns:
            Status dictionary with state, failure count, etc.
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_success": (
                self.last_success_time.isoformat() if self.last_success_time else None
            ),
            "time_until_reset": (
                self._time_until_reset() if self.state == CircuitState.OPEN else 0
            ),
        }


# Global circuit breakers for each TTS provider
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(provider_name: str) -> CircuitBreaker:
    """
    Get or create a circuit breaker for a TTS provider.

    Args:
        provider_name: Name of the TTS provider

    Returns:
        CircuitBreaker instance for this provider

    ELI5: Each TTS service (like MiniMax, Edge TTS, etc.) gets its own
    circuit breaker. If one service is down, it doesn't affect the others.
    """
    if provider_name not in _circuit_breakers:
        _circuit_breakers[provider_name] = CircuitBreaker(
            failure_threshold=5,  # Open after 5 failures
            recovery_timeout=60,  # Wait 60 seconds before retry
            name=provider_name,
        )

    return _circuit_breakers[provider_name]


def get_all_circuit_breakers() -> dict[str, dict[str, Any]]:
    """
    Get status of all circuit breakers.

    Returns:
        Dictionary mapping provider names to their status
    """
    return {name: breaker.get_status() for name, breaker in _circuit_breakers.items()}
