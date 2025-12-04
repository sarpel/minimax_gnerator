import time
import functools
import logging
import psutil
import os
from typing import Any, Callable, TypeVar, cast

# Set up a logger for performance metrics
logger = logging.getLogger("wakegen.performance")

# Define a generic type for functions (so type hints work with decorators)
F = TypeVar("F", bound=Callable[..., Any])

def measure_time(func: F) -> F:
    """
    A decorator that measures how long a function takes to run.
    
    Usage:
        @measure_time
        def my_slow_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.debug(f"Function '{func.__name__}' took {duration:.4f} seconds")
    return cast(F, wrapper)

def log_memory_usage(tag: str = "") -> None:
    """
    Logs the current memory usage of the process.
    
    Useful for checking if we are using too much RAM on a Raspberry Pi.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # Convert bytes to Megabytes (MB)
    rss_mb = mem_info.rss / 1024 / 1024
    
    logger.info(f"[Memory {tag}] RSS: {rss_mb:.2f} MB")

class PerformanceMonitor:
    """
    A context manager to monitor a block of code.
    
    Usage:
        with PerformanceMonitor("Generating Audio"):
            # do heavy work
    """
    def __init__(self, name: str):
        self.name = name
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        log_memory_usage(f"Start {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        logger.info(f"Block '{self.name}' finished in {duration:.4f} seconds")
        log_memory_usage(f"End {self.name}")