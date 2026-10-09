"""
Agent — ReAct-style tool-calling loop.

Usage:
    agent = Agent(
        llm=LLMClient(LLMConfig(provider="openai", model="gpt-4o")),
        tools=[web_search, get_weather],
        system="You are a helpful research assistant.",
    )
    result = await agent.run("What is the weather in NYC?")
    print(result.output)
    print(result.steps)      # list of (thought, tool_call, tool_result)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ai_core.agents.tool import Tool, ToolCall, ToolResult
from ai_core.llm.client import LLMClient
from ai_core.llm.types import Message, Role
from ai_core.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentConfig:
    max_iterations: int = 10
    system_prompt: str = "You are a helpful assistant with access to tools."
    verbose: bool = False


@dataclass
class AgentStep:
    iteration: int
    tool_call: Optional[ToolCall]
    tool_result: Optional[ToolResult]
    assistant_message: str


@dataclass
class AgentResult:
    output: str
    steps: List[AgentStep]
    iterations: int
    success: bool = True
    error: Optional[str] = None


class Agent:
    """
    A tool-calling agent that runs in a loop until the model
    stops calling tools (i.e. produces a final answer).

    Supports OpenAI function calling and Anthropic tool use.
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: List[Union[Tool, Any]] = None,
        system: Optional[str] = None,
        config: AgentConfig | None = None,
    ):
        self.llm = llm
        self.tools: Dict[str, Tool] = {}
        for t in (tools or []):
            if isinstance(t, Tool):
                self.tools[t.name] = t
            else:
                raise TypeError(f"Expected Tool, got {type(t)}. Use @tool decorator.")
        self.system = system or (config or AgentConfig()).system_prompt
        self.config = config or AgentConfig()

    async def run(self, user_message: str, context: Optional[str] = None) -> AgentResult:
        """Run the agent on a user message and return the final result."""
        cfg = self.config
        messages: List[Message] = []

        if context:
            messages.append(Message.user(f"Context:\n{context}\n\nTask: {user_message}"))
        else:
            messages.append(Message.user(user_message))

        steps: List[AgentStep] = []
        provider = self.llm.config.provider

        for iteration in range(cfg.max_iterations):
            # Build tool schemas for the provider
            tool_schemas = self._get_tool_schemas(provider)

            # Call LLM
            raw = await self._call_with_tools(messages, tool_schemas, provider)

            # Parse response
            tool_calls = self._extract_tool_calls(raw, provider)
            assistant_text = self._extract_text(raw, provider)

            if cfg.verbose:
                logger.debug("agent.iteration", i=iteration, text=assistant_text[:100],
                             tool_calls=[tc.name for tc in tool_calls])

            # Add assistant turn to history
            messages.append(Message(role=Role.ASSISTANT, content=assistant_text or ""))

            if not tool_calls:
                # Model is done — return final answer
                return AgentResult(output=assistant_text, steps=steps, iterations=iteration + 1)

            # Execute tools
            for tc in tool_calls:
                result = await self._execute_tool(tc)
                # add tool result to history
                messages.append(Message(
                    role=Role.TOOL,
                    content=result.to_message_content(),
                    name=result.name,
                    tool_call_id=result.tool_call_id,
                ))
                steps.append(AgentStep(
                    iteration=iteration,
                    tool_call=tc,
                    tool_result=result,
                    assistant_message=assistant_text or "",
                ))

        return AgentResult(
            output="Max iterations reached without a final answer.",
            steps=steps,
            iterations=cfg.max_iterations,
            success=False,
            error="max_iterations_exceeded",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_tool_schemas(self, provider: str) -> List[Dict]:
        if provider in ("openai", "azure", "ollama"):
            return [t.to_openai_schema() for t in self.tools.values()]
        elif provider == "anthropic":
            return [t.to_anthropic_schema() for t in self.tools.values()]
        return []

    async def _call_with_tools(self, messages, tool_schemas, provider):
        if provider in ("openai", "azure", "ollama"):
            msgs = [{"role": "system", "content": self.system}]
            msgs += [m.to_dict() for m in messages]
            kwargs = {"tools": tool_schemas, "tool_choice": "auto"} if tool_schemas else {}
            return await self.llm._client.chat.completions.create(
                model=self.llm.config.model,
                messages=msgs,
                **kwargs,
            )
        elif provider == "anthropic":
            msgs = [m.to_dict() for m in messages if m.role != Role.SYSTEM]
            kwargs = {"tools": tool_schemas} if tool_schemas else {}
            return await self.llm._client.messages.create(
                model=self.llm.config.model,
                system=self.system,
                messages=msgs,
                max_tokens=self.llm.config.max_tokens,
                **kwargs,
            )

    def _extract_tool_calls(self, raw, provider) -> List[ToolCall]:
        calls = []
        if provider in ("openai", "azure", "ollama"):
            choice = raw.choices[0]
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments or "{}"),
                    ))
        elif provider == "anthropic":
            for block in (raw.content or []):
                if block.type == "tool_use":
                    calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input or {}))
        return calls

    def _extract_text(self, raw, provider) -> str:
        if provider in ("openai", "azure", "ollama"):
            return raw.choices[0].message.content or ""
        elif provider == "anthropic":
            for block in (raw.content or []):
                if block.type == "text":
                    return block.text
        return ""

    async def _execute_tool(self, tc: ToolCall) -> ToolResult:
        t = self.tools.get(tc.name)
        if not t:
            return ToolResult(tool_call_id=tc.id, name=tc.name, result=None,
                              error=f"Unknown tool: {tc.name}")
        try:
            result = await t(**tc.arguments)
            return ToolResult(tool_call_id=tc.id, name=tc.name, result=result)
        except Exception as e:
            logger.warning("agent.tool_error", tool=tc.name, error=str(e))
            return ToolResult(tool_call_id=tc.id, name=tc.name, result=None, error=str(e))
