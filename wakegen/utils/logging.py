import logging
from typing import Optional
from rich.logging import RichHandler

# We use 'rich' to make our logs look nice in the terminal (colors, timestamps).
# This makes it much easier to spot errors and warnings.


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: The name of the logger. If None, returns the root wakegen logger.
        
    Returns:
        A configured logger instance.
    """
    if name is None:
        return logging.getLogger("wakegen")
    return logging.getLogger(f"wakegen.{name}")


def setup_logging(level: str = "INFO") -> None:
    """
    Configures the logging system.

    Args:
        level: The logging level (DEBUG, INFO, WARNING, ERROR).
               DEBUG shows everything. ERROR shows only critical problems.
    """
    # Create a configuration for the logging system
    logging.basicConfig(
        level=level,
        format="%(message)s", # We only want the message, Rich adds the timestamp/level
        datefmt="[%X]",       # Time format (e.g., [14:30:05])
        handlers=[
            # RichHandler is the magic part that adds colors and formatting
            RichHandler(rich_tracebacks=True, markup=True)
        ]
    )

    # Get the logger for our application
    logger = logging.getLogger("wakegen")
    
    # Set the level for our logger specifically
    logger.setLevel(level)
    
    # Suppress overly verbose logs from third-party libraries if needed
    # logging.getLogger("httpx").setLevel(logging.WARNING)