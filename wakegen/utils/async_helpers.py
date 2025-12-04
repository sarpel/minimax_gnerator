import asyncio
import logging
from functools import wraps
from typing import Callable, Any, Type

logger = logging.getLogger("wakegen")

# This is a "decorator". It's a function that wraps another function to add extra behavior.
# In this case, we are adding "retry" logic. If the wrapped function fails, we try again!

def retry_async(
    retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    A decorator to retry an async function if it raises an exception.

    Args:
        retries: Number of times to retry.
        delay: Seconds to wait between retries.
        exceptions: Tuple of exceptions to catch and retry on.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            
            # Try the function 'retries + 1' times (1 initial try + retries)
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    remaining = retries - attempt
                    
                    if remaining > 0:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {str(e)}. "
                            f"Retrying in {delay}s... ({remaining} attempts left)"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {retries + 1} attempts failed.")
            
            # If we get here, all attempts failed. Raise the last exception.
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator