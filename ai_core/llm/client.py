"""
Unified LLM client — one interface for OpenAI, Anthropic, Azure, Ollama.

Usage:
    from ai_core.llm import LLMClient, LLMConfig, Message

    client = LLMClient(LLMConfig(provider="openai", model="gpt-4o"))
    resp = await client.complete([Message.user("Hello")])
    print(resp.content)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

from ai_core.llm.types import CompletionResponse, Message, Role
from ai_core.utils.logging import get_logger
from ai_core.utils.retry import retry_async

logger = get_logger(__name__)


@dataclass
class LLMConfig:
    provider: str = "openai"                # openai | anthropic | azure | ollama
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None           # falls back to env var
    base_url: Optional[str] = None          # for Azure / Ollama / proxies
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60
    max_retries: int = 3
    system_prompt: Optional[str] = None
    extra_kwargs: dict = field(default_factory=dict)


class LLMClient:
    """
    Provider-agnostic LLM client.

    Handles:
    - OpenAI  (openai>=1.0)
    - Anthropic (anthropic>=0.20)
    - Azure OpenAI
    - Ollama (local)

    All calls are async; use `complete_sync` for blocking contexts.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: List[Message],
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> CompletionResponse:
        """Send messages and return a single completion."""
        cfg = self.config
        sys_prompt = system or cfg.system_prompt
        temp = temperature if temperature is not None else cfg.temperature
        mtok = max_tokens or cfg.max_tokens

        start = time.perf_counter()
        try:
            resp = await retry_async(
                self._call,
                messages=messages,
                system=sys_prompt,
                temperature=temp,
                max_tokens=mtok,
                extra=kwargs,
                max_retries=cfg.max_retries,
            )
        finally:
            elapsed = time.perf_counter() - start
            logger.debug(
                "llm.complete",
                provider=cfg.provider,
                model=cfg.model,
                elapsed_s=round(elapsed, 3),
            )
        return resp

    async def stream(
        self,
        messages: List[Message],
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Yield completion tokens as they arrive."""
        cfg = self.config
        sys_prompt = system or cfg.system_prompt
        async for token in self._stream_call(
            messages,
            system=sys_prompt,
            temperature=temperature or cfg.temperature,
            max_tokens=max_tokens or cfg.max_tokens,
        ):
            yield token

    def complete_sync(self, messages: List[Message], **kwargs) -> CompletionResponse:
        """Blocking wrapper for use outside async contexts."""
        return asyncio.run(self.complete(messages, **kwargs))

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    def _build_client(self):
        p = self.config.provider
        if p == "openai":
            return self._build_openai()
        elif p == "anthropic":
            return self._build_anthropic()
        elif p == "azure":
            return self._build_azure()
        elif p == "ollama":
            return self._build_ollama()
        else:
            raise ValueError(f"Unknown provider: {p!r}. Use openai|anthropic|azure|ollama")

    def _build_openai(self):
        import openai, os
        key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        kwargs = {"api_key": key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return openai.AsyncOpenAI(**kwargs)

    def _build_anthropic(self):
        import anthropic, os
        key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        return anthropic.AsyncAnthropic(api_key=key)

    def _build_azure(self):
        import openai, os
        return openai.AsyncAzureOpenAI(
            api_key=self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=self.config.base_url or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version=self.config.extra_kwargs.get("api_version", "2024-02-01"),
        )

    def _build_ollama(self):
        import openai
        return openai.AsyncOpenAI(
            api_key="ollama",
            base_url=self.config.base_url or "http://localhost:11434/v1",
        )

    async def _call(
        self,
        messages: List[Message],
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        extra: dict,
    ) -> CompletionResponse:
        p = self.config.provider
        if p in ("openai", "azure", "ollama"):
            return await self._call_openai(messages, system, temperature, max_tokens, extra)
        elif p == "anthropic":
            return await self._call_anthropic(messages, system, temperature, max_tokens, extra)
        raise ValueError(f"Unknown provider: {p}")

    async def _call_openai(self, messages, system, temperature, max_tokens, extra):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend([m.to_dict() for m in messages])

        r = await self._client.chat.completions.create(
            model=self.config.model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            **{**self.config.extra_kwargs, **extra},
        )
        choice = r.choices[0]
        usage = r.usage or {}
        return CompletionResponse(
            content=choice.message.content or "",
            model=r.model,
            provider=self.config.provider,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
            finish_reason=choice.finish_reason or "stop",
            raw=r.model_dump() if hasattr(r, "model_dump") else {},
        )

    async def _call_anthropic(self, messages, system, temperature, max_tokens, extra):
        msgs = [m.to_dict() for m in messages if m.role != Role.SYSTEM]
        kwargs = dict(
            model=self.config.model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            **{**self.config.extra_kwargs, **extra},
        )
        if system:
            kwargs["system"] = system

        r = await self._client.messages.create(**kwargs)
        usage = r.usage
        return CompletionResponse(
            content=r.content[0].text if r.content else "",
            model=r.model,
            provider="anthropic",
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            finish_reason=r.stop_reason or "stop",
            raw=r.model_dump() if hasattr(r, "model_dump") else {},
        )

    async def _stream_call(self, messages, system, temperature, max_tokens):
        p = self.config.provider
        if p in ("openai", "azure", "ollama"):
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.extend([m.to_dict() for m in messages])
            async with self._client.chat.completions.stream(
                model=self.config.model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            ) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
        elif p == "anthropic":
            msgs = [m.to_dict() for m in messages if m.role != Role.SYSTEM]
            kwargs = dict(model=self.config.model, messages=msgs,
                          temperature=temperature, max_tokens=max_tokens)
            if system:
                kwargs["system"] = system
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
