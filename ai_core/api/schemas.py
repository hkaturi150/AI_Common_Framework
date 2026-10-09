"""Pydantic request/response schemas for AI API endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Chat ────────────────────────────────────────────────────────────────────

class MessageSchema(BaseModel):
    role: str = Field(..., examples=["user"])
    content: str

class ChatRequest(BaseModel):
    messages: List[MessageSchema]
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    stream: bool = False
    system: Optional[str] = None

class ChatResponse(BaseModel):
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = {}
    finish_reason: str = "stop"


# ── Embeddings ───────────────────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    model: Optional[str] = None

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int


# ── RAG ──────────────────────────────────────────────────────────────────────

class RAGRequest(BaseModel):
    question: str
    collection: Optional[str] = "default"
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

class RAGSourceSchema(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any] = {}

class RAGResponse(BaseModel):
    answer: str
    sources: List[RAGSourceSchema]
    question: str


# ── Errors ───────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
