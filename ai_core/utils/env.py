"""
Environment variable helper — typed reads with defaults and validation.

Usage:
    from ai_core.utils.env import env

    api_key  = env("OPENAI_API_KEY")               # required
    debug    = env("DEBUG", default=False, cast=bool)
    timeout  = env("TIMEOUT", default=30, cast=int)
    origins  = env("CORS_ORIGINS", default="*", cast=lambda s: s.split(","))
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

_MISSING = object()


def env(
    key: str,
    default: Any = _MISSING,
    cast: Optional[Callable] = None,
    required: Optional[bool] = None,
) -> Any:
    """
    Read an environment variable with optional type casting.

    Args:
        key:      Environment variable name.
        default:  Value if the variable is not set. If not given, variable is required.
        cast:     Callable to convert the string value (e.g. int, bool, float, json.loads).
        required: Force required=True even when a default is given.

    Raises:
        EnvironmentError: If the variable is required but not set.
    """
    raw = os.getenv(key)
    is_required = required if required is not None else (default is _MISSING)

    if raw is None:
        if is_required:
            raise EnvironmentError(
                f"Required environment variable {key!r} is not set."
            )
        return default

    if cast is None:
        return raw

    # special-case bool
    if cast is bool:
        return raw.lower() in ("1", "true", "yes", "on")

    try:
        return cast(raw)
    except (ValueError, TypeError) as e:
        raise EnvironmentError(f"Cannot cast {key}={raw!r} using {cast}: {e}") from e


class Settings:
    """
    Base class for typed settings from environment variables.

    Subclass it in each job:

        class MyJobSettings(Settings):
            openai_key: str = env("OPENAI_API_KEY")
            debug: bool     = env("DEBUG", default=False, cast=bool)
            workers: int    = env("WORKERS", default=4, cast=int)

        cfg = MyJobSettings()
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__class__.__dict__.items()
                if not k.startswith("_") and not callable(v)}
