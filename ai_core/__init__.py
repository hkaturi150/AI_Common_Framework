"""
ai_core — Common AI Engineering Framework
==========================================
Reusable building blocks for LLM calls, RAG, agents, data pipelines,
and FastAPI utilities. Drop this package into any project.
"""

from ai_core.llm import LLMClient, LLMConfig, Message, Role
from ai_core.rag import RAGPipeline, VectorStore, Embedder
from ai_core.agents import Agent, Tool, AgentResult
from ai_core.pipeline import Pipeline, Step, PipelineResult
from ai_core.api import create_app, ai_router, health_router
from ai_core.utils import get_logger, retry, timer, env

__version__ = "1.0.0"
__all__ = [
    # LLM
    "LLMClient", "LLMConfig", "Message", "Role",
    # RAG
    "RAGPipeline", "VectorStore", "Embedder",
    # Agents
    "Agent", "Tool", "AgentResult",
    # Pipeline
    "Pipeline", "Step", "PipelineResult",
    # API
    "create_app", "ai_router", "health_router",
    # Utils
    "get_logger", "retry", "timer", "env",
]
