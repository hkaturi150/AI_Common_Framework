"""Tests for the FastAPI app (no LLM calls — uses mock dependencies)."""

import pytest
from fastapi.testclient import TestClient
from ai_core.api.app import create_app
from ai_core.api.routers import _get_llm, _get_embedder, _get_rag
from ai_core.llm.types import CompletionResponse


# ── Stubs ──────────────────────────────────────────────────────────────────

class MockLLM:
    config = type("cfg", (), {"provider": "mock", "temperature": 0.7, "max_tokens": 512})()

    async def complete(self, messages, **kwargs) -> CompletionResponse:
        return CompletionResponse(
            content="This is a mock response.",
            model="mock-model",
            provider="mock",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )

    async def stream(self, messages, **kwargs):
        for token in ["Hello", " ", "world"]:
            yield token


class MockEmbedder:
    config = type("cfg", (), {"model": "mock-embed"})()

    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class MockRAG:
    async def query(self, question, **kwargs):
        from ai_core.rag.vectorstore import Document, SearchResult
        from ai_core.rag.pipeline import RAGResult
        return RAGResult(
            response="Mock RAG answer.",
            sources=[SearchResult(document=Document(text="source text"), score=0.9, rank=0)],
            query=question,
            context_used="source text",
        )


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    app = create_app(
        llm=MockLLM(),
        embedder=MockEmbedder(),
        rag=MockRAG(),
    )
    return TestClient(app)


# ── Tests ──────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200


def test_chat(client):
    r = client.post("/api/v1/chat", json={
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert r.status_code == 200
    body = r.json()
    assert "content" in body
    assert body["provider"] == "mock"


def test_embed(client):
    r = client.post("/api/v1/embed", json={"texts": ["hello", "world"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["embeddings"]) == 2
    assert body["dimensions"] == 3


def test_rag(client):
    r = client.post("/api/v1/rag", json={"question": "What is AI?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert len(body["sources"]) == 1


def test_chat_no_llm_raises():
    app = create_app()  # no llm wired
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/api/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code in (500, 502)


def test_docs_available(client):
    r = client.get("/docs")
    assert r.status_code == 200
