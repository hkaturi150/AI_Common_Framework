"""Tests for RAG components (no live API — uses in-memory store and stub embedder)."""

import pytest
from ai_core.rag.vectorstore import Document, InMemoryVectorStore, SearchResult


@pytest.fixture
def store():
    return InMemoryVectorStore()


def test_document_chunk():
    doc = Document(text=" ".join([f"word{i}" for i in range(200)]))
    chunks = doc.chunk(size=50, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["parent_id"] == doc.id
    assert all(len(c.text.split()) <= 50 for c in chunks)


def test_document_chunk_short():
    doc = Document(text="short text")
    chunks = doc.chunk(size=512)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


@pytest.mark.asyncio
async def test_store_upsert_and_count(store):
    docs = [Document(text=f"doc {i}", embedding=[float(i)] * 4) for i in range(5)]
    n = await store.upsert(docs)
    assert n == 5
    assert await store.count() == 5


@pytest.mark.asyncio
async def test_store_delete(store):
    docs = [Document(id="a", text="hello", embedding=[1.0, 0.0]),
            Document(id="b", text="world", embedding=[0.0, 1.0])]
    await store.upsert(docs)
    deleted = await store.delete(["a"])
    assert deleted == 1
    assert await store.count() == 1


class StubEmbedder:
    async def embed_one(self, text: str):
        # simple stub: embedding is [1,0] for everything
        return [1.0, 0.0]


@pytest.mark.asyncio
async def test_store_search(store):
    docs = [
        Document(id="x", text="AI is cool", embedding=[1.0, 0.0]),
        Document(id="y", text="Python rocks", embedding=[0.0, 1.0]),
    ]
    await store.upsert(docs)
    results = await store.search("AI", StubEmbedder(), top_k=2)
    assert len(results) > 0
    # doc "x" has embedding [1,0] which matches query [1,0] perfectly
    assert results[0].document.id == "x"
    assert results[0].score > results[1].score


def test_vector_store_create_unknown():
    from ai_core.rag.vectorstore import VectorStore
    with pytest.raises(ValueError, match="Unknown backend"):
        VectorStore.create("faiss")
