"""
Structured JSON logging for AI services.

Usage:
    logger = get_logger(__name__)
    logger.info("llm.call", model="gpt-4o", tokens=512)
    logger.error("pipeline.fail", step="embed", error=str(e))
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # pull extra key=value pairs from record
        for key, val in record.__dict__.items():
            if key not in {
                "args", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "message",
                "module", "msecs", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread",
                "threadName",
            }:
                log[key] = val

        if record.exc_info:
            log["exc"] = self.formatException(record.exc_info)

        return json.dumps(log, default=str)


class AILogger(logging.LoggerAdapter):
    """Logger that accepts keyword args as structured fields."""

    def process(self, msg, kwargs):
        # pop extra fields that aren't standard logging kwargs
        extra_fields = {k: v for k, v in kwargs.pop("extra_fields", {}).items()}
        return msg, kwargs

    # Reserved logging attributes that clash with extra= dict keys
    _RESERVED = frozenset({
        "args", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "message", "module", "msecs",
        "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
    })

    def _log_with_fields(self, level: int, event: str, **fields: Any):
        # Prefix reserved keys to avoid collisions with LogRecord attributes
        safe = {(f"f_{k}" if k in self._RESERVED else k): v for k, v in fields.items()}
        self.logger.log(level, event, extra=safe, stacklevel=3)

    def debug(self, event: str, **fields):   self._log_with_fields(logging.DEBUG, event, **fields)
    def info(self, event: str, **fields):    self._log_with_fields(logging.INFO, event, **fields)
    def warning(self, event: str, **fields): self._log_with_fields(logging.WARNING, event, **fields)
    def error(self, event: str, **fields):   self._log_with_fields(logging.ERROR, event, **fields)
    def critical(self, event: str, **fields):self._log_with_fields(logging.CRITICAL, event, **fields)


_configured = False

def _configure_root(level: str = "INFO", json_output: bool = True):
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter() if json_output else logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    _configured = True


def get_logger(name: str, level: str = "INFO", json_output: bool = True) -> AILogger:
    """Get a structured logger for the given module name."""
    _configure_root(level, json_output)
    return AILogger(logging.getLogger(name), {})
