"""
Context manager and decorator for timing code blocks.

Usage:
    with timer("embedding 1000 docs"):
        ...
    # → prints: [timer] embedding 1000 docs: 1.23s

    @timer("model inference")
    async def infer(x): ...
"""

from __future__ import annotations

import asyncio
import functools
import time
from contextlib import contextmanager
from typing import Callable, Optional

from ai_core.utils.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def timer(label: str = "block", log: bool = True):
    """Context manager that times a with-block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if log:
            logger.info("timer", label=label, elapsed_s=round(elapsed, 4))


def timed(_fn: Optional[Callable] = None, *, label: Optional[str] = None):
    """Decorator that logs how long a function takes."""
    def decorator(fn: Callable) -> Callable:
        lbl = label or fn.__qualname__

        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                with timer(lbl):
                    return await fn(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                with timer(lbl):
                    return fn(*args, **kwargs)
            return sync_wrapper

    if _fn is not None:
        return decorator(_fn)
    return decorator
