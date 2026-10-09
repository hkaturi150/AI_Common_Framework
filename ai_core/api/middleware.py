"""
Common FastAPI middleware: logging, timing, CORS, error handling.
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ai_core.utils.logging import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, and latency."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.perf_counter()

        response = await call_next(request)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "http.request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=round(elapsed, 1),
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed:.1f}ms"
        return response


def add_middleware(
    app: FastAPI,
    *,
    cors_origins: list[str] = ["*"],
    log_requests: bool = True,
) -> FastAPI:
    """
    Add standard middleware to a FastAPI app.
    Call this right after create_app().
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if log_requests:
        app.add_middleware(RequestLoggingMiddleware)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    return app
