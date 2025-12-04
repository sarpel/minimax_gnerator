"""Rate Limiter

Token bucket-based rate limiter for API calls.
Ensures providers don't exceed their rate limits.

Features:
- Token bucket algorithm for rate limiting
- Async-compatible waiting
- Multiple independent limiters
- Configurable burst capacity
- Thread-safe operations
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

class RateLimiter:
    """Token bucket rate limiter for API calls.

    This implementation uses the token bucket algorithm:
    - Tokens are added to the bucket at a fixed rate
    - Each API call consumes one token
    - If no tokens are available, calls are delayed until tokens become available

    Attributes:
        max_requests: Maximum number of requests allowed in the period
        period_seconds: Time period in seconds for the rate limit
        _tokens: Current number of tokens in the bucket
        _last_refill: Timestamp of last token refill
        _lock: Async lock for thread safety
    """

    def __init__(self, max_requests: int, period_seconds: int):
        """Initialize the rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in the period
            period_seconds: Time period in seconds for the rate limit
        """
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if period_seconds <= 0:
            raise ValueError("period_seconds must be positive")

        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self._tokens = max_requests  # Start with full bucket
        self._last_refill = time.time()
        self._lock = asyncio.Lock()

    async def wait_for_token(self) -> None:
        """Wait until a token is available for an API call.

        This method implements the token bucket algorithm:
        1. Calculate how many tokens should have been added since last refill
        2. Refill the bucket up to maximum capacity
        3. If tokens are available, consume one and return immediately
        4. If no tokens are available, wait until tokens become available
        """
        async with self._lock:
            # Calculate time since last refill
            now = time.time()
            time_since_refill = now - self._last_refill

            # Calculate how many tokens to add (fractional tokens based on time)
            tokens_to_add = time_since_refill * (self.max_requests / self.period_seconds)

            # Refill the bucket
            self._tokens = min(self.max_requests, self._tokens + tokens_to_add)
            self._last_refill = now

            if self._tokens >= 1:
                # Token available, consume it
                self._tokens -= 1
                return

            # No tokens available, calculate when next token will be available
            # This is when we'll have at least 1 token
            tokens_needed = 1 - self._tokens
            seconds_needed = tokens_needed * (self.period_seconds / self.max_requests)

            # Wait for the required time
            await asyncio.sleep(seconds_needed)

            # After waiting, we should have at least 1 token
            self._tokens = max(0, self._tokens - 1)

    def get_current_rate(self) -> float:
        """Get the current rate in requests per second.

        Returns:
            Current rate in requests per second
        """
        return self.max_requests / self.period_seconds

    def get_available_tokens(self) -> int:
        """Get the number of currently available tokens.

        Returns:
            Number of available tokens
        """
        return self._tokens

    async def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        async with self._lock:
            self._tokens = self.max_requests
            self._last_refill = time.time()

    def __repr__(self) -> str:
        """String representation of the rate limiter."""
        return f"RateLimiter(max_requests={self.max_requests}, period_seconds={self.period_seconds})"