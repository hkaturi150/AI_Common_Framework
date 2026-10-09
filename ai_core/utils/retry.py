"""
Exponential-backoff retry decorator for async and sync callables.

Usage:
    result = await retry_async(my_async_fn, arg1, arg2, max_retries=3)

    @retry_sync(max_retries=5, base_delay=0.5)
    def fetch_data():
        ...
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from typing import Any, Callable, Optional, Tuple, Type

from ai_core.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_RETRYABLE = (
    ConnectionError,
    TimeoutError,
    OSError,
)


async def retry_async(
    fn: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = _DEFAULT_RETRYABLE,
    **kwargs,
) -> Any:
    """Call fn(*args, **kwargs) with exponential backoff on failure."""
    attempt = 0
    last_exc = None
    while attempt <= max_retries:
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            return fn(*args, **kwargs)
        except retryable_exceptions as e:
            last_exc = e
            if attempt == max_retries:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay *= random.uniform(0.5, 1.5)
            logger.warning("retry.attempt", fn=fn.__name__, attempt=attempt + 1,
                           delay_s=round(delay, 2), error=str(e))
            await asyncio.sleep(delay)
            attempt += 1
        except Exception:
            raise   # non-retryable — re-raise immediately

    raise last_exc


def retry_sync(
    fn: Optional[Callable] = None,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = _DEFAULT_RETRYABLE,
):
    """Decorator for sync functions with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            last_exc = None
            while attempt <= max_retries:
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exc = e
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if jitter:
                        delay *= random.uniform(0.5, 1.5)
                    time.sleep(delay)
                    attempt += 1
                except Exception:
                    raise
            raise last_exc
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
