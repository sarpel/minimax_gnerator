"""
Async Helpers Module

This module provides utilities for asynchronous programming in wakegen,
including retry logic, parallel execution, and rate limiting.

Key Features:
- Retry decorator for resilient API calls
- Parallel task execution with concurrency control
- Rate limiting for API providers
- Progress tracking for batch operations
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import (
    Callable, Any, Type, List, Optional, TypeVar, Generic,
    Coroutine, AsyncIterator, Dict
)

logger = logging.getLogger("wakegen")

T = TypeVar("T")


# =============================================================================
# RETRY DECORATOR
# =============================================================================


def retry_async(
    retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    exponential_backoff: bool = False
) -> Callable:
    """
    A decorator to retry an async function if it raises an exception.

    This is super useful for API calls that might fail temporarily due to
    network issues, rate limits, or server hiccups.

    Args:
        retries: Number of times to retry after initial failure.
        delay: Seconds to wait between retries.
        exceptions: Tuple of exceptions to catch and retry on.
        exponential_backoff: If True, double the delay after each failure.

    Example:
        @retry_async(retries=3, delay=1.0)
        async def fetch_data():
            # This will be retried up to 3 times if it fails
            return await api.get_data()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            current_delay = delay
            
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
                            f"Retrying in {current_delay:.1f}s... ({remaining} attempts left)"
                        )
                        await asyncio.sleep(current_delay)
                        
                        if exponential_backoff:
                            current_delay *= 2
                    else:
                        logger.error(f"All {retries + 1} attempts failed.")
            
            # If we get here, all attempts failed. Raise the last exception.
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


# =============================================================================
# TASK RESULT
# =============================================================================


@dataclass
class TaskResult(Generic[T]):
    """
    Result of a parallel task execution.
    
    Wraps the result or error from a task, making it easy to handle
    failures in batch operations without stopping other tasks.
    """
    task_id: Any
    success: bool
    result: Optional[T] = None
    error: Optional[Exception] = None
    duration_ms: float = 0.0
    
    @property
    def failed(self) -> bool:
        return not self.success


# =============================================================================
# PARALLEL EXECUTOR
# =============================================================================


@dataclass
class ParallelExecutorStats:
    """Statistics from parallel execution."""
    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    total_duration_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return self.completed / self.total_tasks if self.total_tasks > 0 else 0.0


class ParallelExecutor:
    """
    Execute multiple async tasks in parallel with concurrency control.
    
    This is like a factory assembly line - you can have multiple workers
    (max_concurrent) processing tasks at the same time, but not too many
    to overwhelm the system.
    
    Features:
    - Configurable concurrency limit
    - Automatic error handling (failures don't stop other tasks)
    - Progress tracking
    - Rate limiting per provider
    
    Example:
        async def generate_sample(task):
            await provider.generate(task.text, task.voice, task.output)
            return task.output
        
        executor = ParallelExecutor(max_concurrent=4)
        results = await executor.execute(tasks, generate_sample)
        
        for result in results:
            if result.success:
                print(f"Generated: {result.result}")
            else:
                print(f"Failed: {result.error}")
    """
    
    def __init__(
        self,
        max_concurrent: int = 4,
        rate_limit: Optional[float] = None
    ):
        """
        Initialize the executor.
        
        Args:
            max_concurrent: Maximum number of tasks running at once.
            rate_limit: Optional minimum seconds between task starts.
        """
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._last_task_time: float = 0
        self._stats = ParallelExecutorStats()
    
    async def execute(
        self,
        tasks: List[T],
        worker: Callable[[T], Coroutine[Any, Any, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[TaskResult]:
        """
        Execute tasks in parallel.
        
        Args:
            tasks: List of task objects to process.
            worker: Async function that processes a single task.
            progress_callback: Optional callback(completed, total) for progress.
        
        Returns:
            List of TaskResult objects with results or errors.
        """
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._stats = ParallelExecutorStats(total_tasks=len(tasks))
        
        results: List[TaskResult] = []
        completed = 0
        
        async def run_task(task_id: int, task: T) -> TaskResult:
            nonlocal completed
            
            async with self._semaphore:
                # Rate limiting
                if self.rate_limit:
                    now = time.time()
                    elapsed = now - self._last_task_time
                    if elapsed < self.rate_limit:
                        await asyncio.sleep(self.rate_limit - elapsed)
                    self._last_task_time = time.time()
                
                start_time = time.time()
                try:
                    result = await worker(task)
                    duration = (time.time() - start_time) * 1000
                    
                    completed += 1
                    self._stats.completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(tasks))
                    
                    return TaskResult(
                        task_id=task_id,
                        success=True,
                        result=result,
                        duration_ms=duration
                    )
                except Exception as e:
                    duration = (time.time() - start_time) * 1000
                    
                    completed += 1
                    self._stats.failed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(tasks))
                    
                    logger.warning(f"Task {task_id} failed: {e}")
                    return TaskResult(
                        task_id=task_id,
                        success=False,
                        error=e,
                        duration_ms=duration
                    )
        
        # Run all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*[
            run_task(i, task) for i, task in enumerate(tasks)
        ])
        self._stats.total_duration_ms = (time.time() - start_time) * 1000
        
        return results
    
    async def execute_streaming(
        self,
        tasks: List[T],
        worker: Callable[[T], Coroutine[Any, Any, Any]]
    ) -> AsyncIterator[TaskResult]:
        """
        Execute tasks and yield results as they complete.
        
        Unlike execute(), this yields results as soon as each task finishes,
        which is useful for showing real-time progress.
        
        Args:
            tasks: List of tasks to process.
            worker: Async function to process each task.
        
        Yields:
            TaskResult objects as tasks complete.
        """
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def run_task(task_id: int, task: T) -> TaskResult:
            async with self._semaphore:
                if self.rate_limit:
                    await asyncio.sleep(self.rate_limit)
                
                start_time = time.time()
                try:
                    result = await worker(task)
                    return TaskResult(
                        task_id=task_id,
                        success=True,
                        result=result,
                        duration_ms=(time.time() - start_time) * 1000
                    )
                except Exception as e:
                    return TaskResult(
                        task_id=task_id,
                        success=False,
                        error=e,
                        duration_ms=(time.time() - start_time) * 1000
                    )
        
        # Create all task coroutines
        pending = [
            asyncio.create_task(run_task(i, task))
            for i, task in enumerate(tasks)
        ]
        
        # Yield results as they complete
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                yield task.result()
    
    def get_stats(self) -> ParallelExecutorStats:
        """Get execution statistics."""
        return self._stats


# =============================================================================
# RATE LIMITER
# =============================================================================


class RateLimiter:
    """
    Rate limiter for API calls.
    
    Use this to prevent overwhelming APIs with too many requests.
    Implements a token bucket algorithm.
    
    Example:
        limiter = RateLimiter(calls_per_second=10)
        
        for item in items:
            await limiter.acquire()
            await api.call(item)
    """
    
    def __init__(
        self,
        calls_per_second: float = 10.0,
        burst_size: int = 1
    ):
        """
        Initialize the rate limiter.
        
        Args:
            calls_per_second: Maximum sustained rate.
            burst_size: Maximum burst of calls allowed.
        """
        self.rate = calls_per_second
        self.burst_size = burst_size
        self._tokens = float(burst_size)
        self._last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """
        Wait until a request is allowed.
        
        Call this before each API request to enforce the rate limit.
        """
        async with self._lock:
            now = time.time()
            
            # Add tokens for elapsed time
            elapsed = now - self._last_update
            self._tokens = min(
                self.burst_size,
                self._tokens + elapsed * self.rate
            )
            self._last_update = now
            
            # Wait if no tokens available
            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self.rate
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


# =============================================================================
# BATCH PROCESSOR
# =============================================================================


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 10
    max_concurrent_batches: int = 2
    delay_between_batches: float = 0.0


async def process_in_batches(
    items: List[T],
    processor: Callable[[List[T]], Coroutine[Any, Any, List[Any]]],
    config: BatchConfig = BatchConfig()
) -> List[Any]:
    """
    Process items in batches for memory efficiency.
    
    This is useful when you have thousands of items and don't want
    to load them all into memory at once.
    
    Args:
        items: All items to process.
        processor: Async function that processes a batch of items.
        config: Batch processing configuration.
    
    Returns:
        Combined results from all batches.
    """
    results = []
    batches = [
        items[i:i + config.batch_size]
        for i in range(0, len(items), config.batch_size)
    ]
    
    semaphore = asyncio.Semaphore(config.max_concurrent_batches)
    
    async def process_batch(batch: List[T]) -> List[Any]:
        async with semaphore:
            result = await processor(batch)
            if config.delay_between_batches > 0:
                await asyncio.sleep(config.delay_between_batches)
            return result
    
    batch_results = await asyncio.gather(*[
        process_batch(batch) for batch in batches
    ])
    
    # Flatten results
    for batch_result in batch_results:
        results.extend(batch_result)
    
    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


async def gather_with_limit(
    coros: List[Coroutine],
    limit: int
) -> List[Any]:
    """
    Like asyncio.gather but with a concurrency limit.
    
    Args:
        coros: List of coroutines to execute.
        limit: Maximum concurrent coroutines.
    
    Returns:
        List of results in order.
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def limited_coro(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*[limited_coro(c) for c in coros])


async def first_successful(
    coros: List[Coroutine],
    timeout: Optional[float] = None
) -> Any:
    """
    Return the result of the first coroutine that succeeds.
    
    Useful for trying multiple providers and using whichever works first.
    
    Args:
        coros: List of coroutines to try.
        timeout: Optional timeout in seconds.
    
    Returns:
        Result from the first successful coroutine.
    
    Raises:
        Exception: If all coroutines fail.
    """
    tasks = [asyncio.create_task(c) for c in coros]
    
    try:
        done, pending = await asyncio.wait(
            tasks,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        # Get the first successful result
        for task in done:
            if not task.exception():
                return task.result()
        
        # All completed tasks failed
        raise Exception("All coroutines failed")
        
    except asyncio.TimeoutError:
        for task in tasks:
            task.cancel()
        raise


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Decorators
    "retry_async",
    # Classes
    "TaskResult",
    "ParallelExecutor",
    "ParallelExecutorStats",
    "RateLimiter",
    "BatchConfig",
    # Functions
    "process_in_batches",
    "gather_with_limit",
    "first_successful",
]