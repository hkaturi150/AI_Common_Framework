"""
VectorStore — unified adapter for Pinecone, Chroma, pgvector, in-memory.

Usage:
    store = VectorStore.create("chroma", collection="my-docs")
    await store.upsert([Document(id="1", text="hello", metadata={"src": "wiki"})])
    results = await store.search("what is hello?", embedder=embedder, top_k=5)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai_core.utils.logging import get_logger

logger = get_logger(__name__)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-10)


@dataclass
class Document:
    text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = field(default=None, repr=False)

    def chunk(self, size: int = 512, overlap: int = 64) -> List["Document"]:
        """Split document into overlapping text chunks."""
        words = self.text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i : i + size]
            chunk_text = " ".join(chunk_words)
            chunks.append(
                Document(
                    text=chunk_text,
                    id=f"{self.id}_chunk_{len(chunks)}",
                    metadata={**self.metadata, "chunk_index": len(chunks), "parent_id": self.id},
                )
            )
            i += size - overlap
        return chunks or [self]


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int = 0


class VectorStore:
    """
    Abstract base for all vector backends.
    Use VectorStore.create(backend, ...) to get a concrete instance.
    """

    @staticmethod
    def create(backend: str, **kwargs) -> "VectorStore":
        backends = {
            "memory": InMemoryVectorStore,
            "chroma": ChromaVectorStore,
            "pinecone": PineconeVectorStore,
            "pgvector": PgVectorStore,
        }
        if backend not in backends:
            raise ValueError(f"Unknown backend: {backend!r}. Choose from {list(backends)}")
        return backends[backend](**kwargs)

    async def upsert(self, docs: List[Document]) -> int:
        raise NotImplementedError

    async def search(
        self,
        query: str,
        embedder,
        top_k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[SearchResult]:
        raise NotImplementedError

    async def delete(self, ids: List[str]) -> int:
        raise NotImplementedError

    async def count(self) -> int:
        raise NotImplementedError


# -----------------------------------------------------------------------
# In-memory backend (no dependencies, great for testing)
# -----------------------------------------------------------------------

class InMemoryVectorStore(VectorStore):
    def __init__(self, **_):
        self._docs: Dict[str, Document] = {}

    async def upsert(self, docs: List[Document]) -> int:
        for doc in docs:
            self._docs[doc.id] = doc
        return len(docs)

    async def search(self, query, embedder, top_k=5, filter=None) -> List[SearchResult]:
        q_vec = await embedder.embed_one(query)
        scored = []
        for doc in self._docs.values():
            if doc.embedding is None:
                continue
            cos = _cosine(q_vec, doc.embedding)
            scored.append((doc, cos))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [SearchResult(document=d, score=s, rank=i) for i, (d, s) in enumerate(scored[:top_k])]

    async def delete(self, ids: List[str]) -> int:
        count = 0
        for id_ in ids:
            if id_ in self._docs:
                del self._docs[id_]
                count += 1
        return count

    async def count(self) -> int:
        return len(self._docs)


# -----------------------------------------------------------------------
# Chroma backend
# -----------------------------------------------------------------------

class ChromaVectorStore(VectorStore):
    def __init__(self, collection: str = "default", persist_path: Optional[str] = None, **_):
        import chromadb
        client = chromadb.PersistentClient(path=persist_path) if persist_path else chromadb.EphemeralClient()
        self._col = client.get_or_create_collection(collection)

    async def upsert(self, docs: List[Document]) -> int:
        ids = [d.id for d in docs]
        texts = [d.text for d in docs]
        metadatas = [d.metadata or {} for d in docs]
        embeddings = [d.embedding for d in docs if d.embedding] or None
        self._col.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        return len(docs)

    async def search(self, query, embedder, top_k=5, filter=None) -> List[SearchResult]:
        q_vec = await embedder.embed_one(query)
        r = self._col.query(query_embeddings=[q_vec], n_results=top_k, where=filter)
        results = []
        for i, (id_, text, meta, dist) in enumerate(zip(
            r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]
        )):
            doc = Document(id=id_, text=text, metadata=meta)
            results.append(SearchResult(document=doc, score=1 - dist, rank=i))
        return results

    async def delete(self, ids: List[str]) -> int:
        self._col.delete(ids=ids)
        return len(ids)

    async def count(self) -> int:
        return self._col.count()


# -----------------------------------------------------------------------
# Pinecone backend
# -----------------------------------------------------------------------

class PineconeVectorStore(VectorStore):
    def __init__(self, index_name: str, api_key: Optional[str] = None, namespace: str = "", **_):
        import pinecone, os
        pc = pinecone.Pinecone(api_key=api_key or os.getenv("PINECONE_API_KEY"))
        self._index = pc.Index(index_name)
        self._ns = namespace

    async def upsert(self, docs: List[Document]) -> int:
        vectors = [(d.id, d.embedding, d.metadata) for d in docs if d.embedding]
        self._index.upsert(vectors=vectors, namespace=self._ns)
        return len(vectors)

    async def search(self, query, embedder, top_k=5, filter=None) -> List[SearchResult]:
        q_vec = await embedder.embed_one(query)
        r = self._index.query(vector=q_vec, top_k=top_k, filter=filter,
                               include_metadata=True, namespace=self._ns)
        results = []
        for i, match in enumerate(r.matches):
            doc = Document(id=match.id, text=match.metadata.get("text", ""), metadata=match.metadata)
            results.append(SearchResult(document=doc, score=match.score, rank=i))
        return results

    async def delete(self, ids: List[str]) -> int:
        self._index.delete(ids=ids, namespace=self._ns)
        return len(ids)

    async def count(self) -> int:
        return self._index.describe_index_stats().total_vector_count


# -----------------------------------------------------------------------
# pgvector backend
# -----------------------------------------------------------------------

class PgVectorStore(VectorStore):
    def __init__(self, dsn: str, table: str = "embeddings", dimensions: int = 1536, **_):
        self._dsn = dsn
        self._table = table
        self._dims = dimensions

    async def _get_conn(self):
        import asyncpg
        return await asyncpg.connect(self._dsn)

    async def upsert(self, docs: List[Document]) -> int:
        conn = await self._get_conn()
        try:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    metadata JSONB,
                    embedding vector({self._dims})
                )
            """)
            for doc in docs:
                import json
                await conn.execute(
                    f"INSERT INTO {self._table}(id,text,metadata,embedding) VALUES($1,$2,$3,$4) "
                    f"ON CONFLICT(id) DO UPDATE SET text=EXCLUDED.text, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding",
                    doc.id, doc.text, json.dumps(doc.metadata), str(doc.embedding),
                )
            return len(docs)
        finally:
            await conn.close()

    async def search(self, query, embedder, top_k=5, filter=None) -> List[SearchResult]:
        q_vec = await embedder.embed_one(query)
        conn = await self._get_conn()
        try:
            rows = await conn.fetch(
                f"SELECT id, text, metadata, 1-(embedding<=>$1::vector) AS score "
                f"FROM {self._table} ORDER BY embedding<=>$1::vector LIMIT $2",
                str(q_vec), top_k,
            )
            return [
                SearchResult(
                    document=Document(id=r["id"], text=r["text"], metadata=r["metadata"] or {}),
                    score=r["score"],
                    rank=i,
                )
                for i, r in enumerate(rows)
            ]
        finally:
            await conn.close()

    async def delete(self, ids: List[str]) -> int:
        conn = await self._get_conn()
        try:
            result = await conn.execute(f"DELETE FROM {self._table} WHERE id=ANY($1)", ids)
            return int(result.split()[-1])
        finally:
            await conn.close()

    async def count(self) -> int:
        conn = await self._get_conn()
        try:
            return await conn.fetchval(f"SELECT COUNT(*) FROM {self._table}")
        finally:
            await conn.close()
