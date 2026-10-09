"""
RAGPipeline — end-to-end retrieval-augmented generation.

Usage:
    rag = RAGPipeline(
        llm=LLMClient(LLMConfig(provider="openai")),
        embedder=Embedder(),
        store=VectorStore.create("chroma", collection="kb"),
    )
    # index documents
    await rag.index([Document(text="..."), ...])
    # answer a question
    answer = await rag.query("What is X?")
    print(answer.response)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ai_core.llm.client import LLMClient
from ai_core.llm.types import Message
from ai_core.rag.embedder import Embedder
from ai_core.rag.vectorstore import Document, SearchResult, VectorStore
from ai_core.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Use ONLY the context provided to answer the question. "
    "If the context doesn't contain the answer, say so clearly."
)

_DEFAULT_PROMPT = """\
Context:
{context}

Question: {question}

Answer based only on the context above:"""


@dataclass
class RAGConfig:
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64
    system_prompt: str = _DEFAULT_SYSTEM
    qa_prompt: str = _DEFAULT_PROMPT
    score_threshold: float = 0.0   # filter results below this cosine score
    embed_on_index: bool = True


@dataclass
class RAGResult:
    response: str
    sources: List[SearchResult]
    query: str
    context_used: str


class RAGPipeline:
    """Full RAG pipeline: chunk → embed → store → retrieve → generate."""

    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder,
        store: VectorStore,
        config: RAGConfig | None = None,
    ):
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.config = config or RAGConfig()

    async def index(self, docs: List[Document]) -> int:
        """Chunk, embed, and upsert documents. Returns total chunks indexed."""
        cfg = self.config
        chunks: List[Document] = []
        for doc in docs:
            chunks.extend(doc.chunk(size=cfg.chunk_size, overlap=cfg.chunk_overlap))

        if cfg.embed_on_index:
            texts = [c.text for c in chunks]
            vectors = await self.embedder.embed(texts)
            for chunk, vec in zip(chunks, vectors):
                chunk.embedding = vec

        count = await self.store.upsert(chunks)
        logger.info("rag.index", docs=len(docs), chunks=count)
        return count

    async def query(self, question: str, **llm_kwargs) -> RAGResult:
        """Retrieve relevant context and generate an answer."""
        cfg = self.config

        results = await self.store.search(
            question, self.embedder, top_k=cfg.top_k
        )
        # filter by score threshold
        results = [r for r in results if r.score >= cfg.score_threshold]

        context = "\n\n".join(
            f"[Source {r.rank + 1}] {r.document.text}" for r in results
        )
        prompt = cfg.qa_prompt.format(context=context or "No context found.", question=question)

        resp = await self.llm.complete(
            [Message.user(prompt)],
            system=cfg.system_prompt,
            **llm_kwargs,
        )

        logger.info("rag.query", question=question[:80], sources=len(results))
        return RAGResult(
            response=resp.content,
            sources=results,
            query=question,
            context_used=context,
        )
