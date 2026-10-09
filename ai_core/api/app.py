"""
create_app — factory that wires together LLM, embedder, and RAG pipeline
into a ready-to-run FastAPI application.

Usage (minimal):
    from ai_core import create_app, LLMClient, LLMConfig
    app = create_app(llm=LLMClient(LLMConfig()))
    # uvicorn main:app --reload

Usage (full):
    app = create_app(
        title="My AI Service",
        llm=LLMClient(LLMConfig(provider="openai", model="gpt-4o")),
        embedder=Embedder(EmbedderConfig()),
        rag=RAGPipeline(llm, embedder, store),
        cors_origins=["https://myapp.com"],
    )
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI

from ai_core.api.middleware import add_middleware
from ai_core.api.routers import ai_router, health_router, _get_llm, _get_embedder, _get_rag


def create_app(
    *,
    title: str = "AI Service",
    version: str = "1.0.0",
    llm=None,
    embedder=None,
    rag=None,
    cors_origins: list[str] = ["*"],
    prefix: str = "/api/v1",
    log_requests: bool = True,
    extra_routers: list = None,
) -> FastAPI:
    """
    Build and return a production-ready FastAPI app with AI endpoints.

    Args:
        title:          OpenAPI title.
        version:        API version string.
        llm:            LLMClient instance (enables /chat and /chat/stream).
        embedder:       Embedder instance (enables /embed).
        rag:            RAGPipeline instance (enables /rag).
        cors_origins:   Allowed CORS origins.
        prefix:         URL prefix for AI routes.
        log_requests:   Enable request logging middleware.
        extra_routers:  Additional APIRouter instances to mount.
    """
    app = FastAPI(title=title, version=version, docs_url="/docs", redoc_url="/redoc")

    # Middleware
    add_middleware(app, cors_origins=cors_origins, log_requests=log_requests)

    # Wire dependencies
    if llm is not None:
        app.dependency_overrides[_get_llm] = lambda: llm
    if embedder is not None:
        app.dependency_overrides[_get_embedder] = lambda: embedder
    if rag is not None:
        app.dependency_overrides[_get_rag] = lambda: rag

    # Routers
    app.include_router(health_router)
    app.include_router(ai_router, prefix=prefix)
    for router in (extra_routers or []):
        app.include_router(router)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        from ai_core.utils.logging import get_logger
        get_logger("ai_core").info("app.startup", title=title, version=version)
        yield

    app.router.lifespan_context = lifespan
    return app
