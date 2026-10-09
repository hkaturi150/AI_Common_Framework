"""
Pre-built FastAPI routers for common AI endpoints.

Include them in your app:
    app.include_router(health_router)
    app.include_router(ai_router, prefix="/api/v1")
"""

from __future__ import annotations

import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ai_core.api.schemas import (
    ChatRequest, ChatResponse,
    EmbedRequest, EmbedResponse,
    RAGRequest, RAGResponse, RAGSourceSchema,
)
from ai_core.llm.types import Message, Role
from ai_core.utils.logging import get_logger

logger = get_logger(__name__)

# ── Dependency stubs (defined first so decorators can reference them) ─────────

def _get_llm():
    raise RuntimeError("LLM not configured. Pass llm= to create_app().")

def _get_embedder():
    raise RuntimeError("Embedder not configured. Pass embedder= to create_app().")

def _get_rag():
    raise RuntimeError("RAGPipeline not configured. Pass rag= to create_app().")


# ── Health ───────────────────────────────────────────────────────────────────

health_router = APIRouter(tags=["health"])

@health_router.get("/health")
async def health():
    return {"status": "ok", "ts": time.time()}

@health_router.get("/ready")
async def ready():
    return {"ready": True}


# ── AI (mounted at /api/v1 by default) ────────────────────────────────────

ai_router = APIRouter(tags=["ai"])


@ai_router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, llm=Depends(_get_llm)):
    """Unified chat endpoint — model/provider set in app config."""
    msgs = [Message(role=Role(m.role), content=m.content) for m in req.messages]
    try:
        resp = await llm.complete(
            msgs,
            system=req.system,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return ChatResponse(
        content=resp.content,
        model=resp.model,
        provider=resp.provider,
        usage=resp.usage,
        finish_reason=resp.finish_reason,
    )


@ai_router.post("/chat/stream")
async def chat_stream(req: ChatRequest, llm=Depends(_get_llm)):
    """Streaming SSE endpoint."""
    msgs = [Message(role=Role(m.role), content=m.content) for m in req.messages]

    async def _gen() -> AsyncIterator[str]:
        async for token in llm.stream(msgs, system=req.system, temperature=req.temperature):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@ai_router.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest, embedder=Depends(_get_embedder)):
    """Generate embeddings for a list of texts."""
    try:
        vectors = await embedder.embed(req.texts)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return EmbedResponse(
        embeddings=vectors,
        model=embedder.config.model,
        dimensions=len(vectors[0]) if vectors else 0,
    )


@ai_router.post("/rag", response_model=RAGResponse)
async def rag_query(req: RAGRequest, rag=Depends(_get_rag)):
    """Answer a question using retrieval-augmented generation."""
    try:
        result = await rag.query(req.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return RAGResponse(
        answer=result.response,
        question=result.query,
        sources=[
            RAGSourceSchema(text=s.document.text, score=s.score, metadata=s.document.metadata)
            for s in result.sources
        ],
    )


# (dependency stubs defined at top of file)
