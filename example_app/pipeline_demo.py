"""
pipeline_demo.py — example ETL pipeline for ingesting docs into RAG.

Run:
    OPENAI_API_KEY=sk-... python -m example_app.pipeline_demo
"""

import asyncio
from pathlib import Path
from typing import List

from ai_core import LLMClient, LLMConfig, RAGPipeline, VectorStore, Embedder
from ai_core.pipeline import Pipeline
from ai_core.rag import Document
from ai_core.rag.embedder import EmbedderConfig


# ── Pipeline step functions ────────────────────────────────────────────────

def load_files(config: dict) -> List[str]:
    """Load raw text from a directory."""
    path = Path(config["path"])
    texts = []
    for f in path.glob("**/*.txt"):
        texts.append(f.read_text())
    print(f"  Loaded {len(texts)} files from {path}")
    return texts


def clean_text(texts: List[str]) -> List[str]:
    """Strip extra whitespace and short lines."""
    cleaned = []
    for t in texts:
        lines = [l.strip() for l in t.splitlines() if len(l.strip()) > 20]
        cleaned.append("\n".join(lines))
    print(f"  Cleaned {len(cleaned)} documents")
    return cleaned


def to_documents(texts: List[str]) -> List[Document]:
    """Wrap strings in Document objects."""
    return [Document(text=t, metadata={"source": f"doc_{i}"}) for i, t in enumerate(texts)]


async def embed_and_store(docs: List[Document]) -> dict:
    """Embed all documents and upsert into vector store."""
    embedder = Embedder(EmbedderConfig(provider="openai"))
    store = VectorStore.create("memory")
    rag = RAGPipeline(
        llm=LLMClient(LLMConfig()),
        embedder=embedder,
        store=store,
    )
    count = await rag.index(docs)
    return {"indexed": count, "rag": rag}


# ── Build and run ──────────────────────────────────────────────────────────

async def main():
    pipeline = (
        Pipeline("doc-ingestion")
        .step("load",      load_files,      description="Read files from disk")
        .step("clean",     clean_text,      description="Normalize text")
        .step("wrap",      to_documents,    description="Create Document objects")
        .step("index",     embed_and_store, description="Embed + upsert to vector store")
    )

    result = await pipeline.run({"path": "./docs"})
    print("\n" + result.summary())

    if result.success:
        rag = result.outputs["index"]["rag"]
        answer = await rag.query("What is in these documents?")
        print("\nSample RAG answer:", answer.response[:200])


if __name__ == "__main__":
    asyncio.run(main())
