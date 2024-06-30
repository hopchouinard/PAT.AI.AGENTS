import asyncio
import logging
from functools import wraps
from typing import Type, Tuple, Callable, Any
import aiohttp

logger = logging.getLogger(__name__)

class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""
    pass

def async_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (aiohttp.ClientError, asyncio.TimeoutError),
    logger: logging.Logger = logger
) -> Callable:
    """
    A decorator for asynchronous retry logic with exponential backoff.
    
    Args:
        max_retries (int): Maximum number of retry attempts.
        base_delay (float): Initial delay between retries in seconds.
        max_delay (float): Maximum delay between retries in seconds.
        backoff_factor (float): Multiplicative factor for exponential backoff.
        exceptions (Tuple[Type[Exception], ...]): Exception types to catch and retry on.
        logger (logging.Logger): Logger to use for logging retry attempts.
    
    Returns:
        Callable: Decorated function with retry logic.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed. "
                        f"Retrying in {delay:.2f} seconds. Error: {str(e)}"
                    )
                    await asyncio.sleep(delay)
            
            logger.error(f"All {max_retries} retry attempts exhausted.")
            raise RetryExhaustedError(f"Operation failed after {max_retries} attempts") from last_exception
        
        return wrapper
    return decorator

async def with_semaphore(semaphore: asyncio.Semaphore, func: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Execute a function with a semaphore to limit concurrent operations.
    
    Args:
        semaphore (asyncio.Semaphore): Semaphore to use for limiting concurrency.
        func (Callable): Function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
    
    Returns:
        Any: Result of the function execution.
    """
    async with semaphore:
        return await func(*args, **kwargs)