"""
example_app/main.py — drop-in FastAPI app using ai_core.

Run:
    pip install -e ..
    OPENAI_API_KEY=sk-... uvicorn example_app.main:app --reload

Endpoints:
    POST /api/v1/chat       → LLM chat completion
    POST /api/v1/chat/stream → SSE streaming
    POST /api/v1/embed      → text embeddings
    POST /api/v1/rag        → RAG question-answering
    GET  /health            → liveness check
    GET  /docs              → Swagger UI
"""

import os

from ai_core import (
    LLMClient, LLMConfig,
    RAGPipeline, VectorStore,
    Embedder,
    create_app,
)
from ai_core.rag import RAGConfig
from ai_core.rag.embedder import EmbedderConfig

# ── Wire up components ─────────────────────────────────────────────────────

llm = LLMClient(LLMConfig(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.3,
))

embedder = Embedder(EmbedderConfig(
    provider="openai",
    model="text-embedding-3-small",
))

store = VectorStore.create("memory")  # swap with "chroma" or "pinecone" in prod

rag = RAGPipeline(
    llm=llm,
    embedder=embedder,
    store=store,
    config=RAGConfig(top_k=5, chunk_size=512),
)

# ── Build app ──────────────────────────────────────────────────────────────

app = create_app(
    title="My AI Service",
    version="1.0.0",
    llm=llm,
    embedder=embedder,
    rag=rag,
    cors_origins=["*"],
)
