"""
Embedder — unified embedding interface for OpenAI, Cohere, HuggingFace.

Usage:
    embedder = Embedder(EmbedderConfig(provider="openai"))
    vectors = await embedder.embed(["doc 1", "doc 2"])
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Optional

from ai_core.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbedderConfig:
    provider: str = "openai"              # openai | cohere | huggingface | local
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    batch_size: int = 512
    dimensions: Optional[int] = None     # truncate if supported


class Embedder:
    """
    Converts text into float vectors.
    Supports batching and async operation.
    """

    def __init__(self, config: EmbedderConfig | None = None):
        self.config = config or EmbedderConfig()
        self._client = self._build_client()

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts. Returns list of float vectors."""
        if not texts:
            return []
        # batch to avoid token limits
        all_vectors = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            vectors = await self._embed_batch(batch)
            all_vectors.extend(vectors)
        return all_vectors

    async def embed_one(self, text: str) -> List[float]:
        """Embed a single string."""
        result = await self.embed([text])
        return result[0]

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        return asyncio.run(self.embed(texts))

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    def _build_client(self):
        p = self.config.provider
        if p == "openai":
            import openai, os
            key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            return openai.AsyncOpenAI(api_key=key)
        elif p == "cohere":
            import cohere, os
            key = self.config.api_key or os.getenv("COHERE_API_KEY")
            return cohere.AsyncClient(key)
        elif p in ("huggingface", "local"):
            return None  # loaded lazily
        else:
            raise ValueError(f"Unknown embedder provider: {p!r}")

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        p = self.config.provider
        if p == "openai":
            return await self._embed_openai(texts)
        elif p == "cohere":
            return await self._embed_cohere(texts)
        elif p in ("huggingface", "local"):
            return await self._embed_hf(texts)
        raise ValueError(f"Unknown provider: {p}")

    async def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        kwargs = {"input": texts, "model": self.config.model}
        if self.config.dimensions:
            kwargs["dimensions"] = self.config.dimensions
        r = await self._client.embeddings.create(**kwargs)
        return [item.embedding for item in r.data]

    async def _embed_cohere(self, texts: List[str]) -> List[List[float]]:
        r = await self._client.embed(
            texts=texts,
            model=self.config.model,
            input_type="search_document",
        )
        return [list(v) for v in r.embeddings]

    async def _embed_hf(self, texts: List[str]) -> List[List[float]]:
        from sentence_transformers import SentenceTransformer
        import asyncio
        loop = asyncio.get_event_loop()
        model = SentenceTransformer(self.config.model)
        vecs = await loop.run_in_executor(None, model.encode, texts)
        return [v.tolist() for v in vecs]
