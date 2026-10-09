"""
Tool definition and decorator for AI agents.

Usage:
    @tool(description="Search the web for a query")
    async def web_search(query: str) -> str:
        ...

    @tool
    async def get_weather(city: str, units: str = "celsius") -> dict:
        ...
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, get_type_hints


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    result: Any
    error: Optional[str] = None

    def to_message_content(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        if isinstance(self.result, str):
            return self.result
        return json.dumps(self.result, default=str)


class Tool:
    """Wraps a callable with its JSON schema for function calling."""

    def __init__(self, fn: Callable, description: str = "", name: Optional[str] = None):
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description or (inspect.getdoc(fn) or "")
        self._schema = self._build_schema()

    async def __call__(self, **kwargs) -> Any:
        if inspect.iscoroutinefunction(self.fn):
            return await self.fn(**kwargs)
        return self.fn(**kwargs)

    def to_openai_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._schema,
            },
        }

    def to_anthropic_schema(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._schema,
        }

    def _build_schema(self) -> Dict:
        sig = inspect.signature(self.fn)
        hints = get_type_hints(self.fn)
        properties: Dict[str, Any] = {}
        required: List[str] = []

        _py_to_json = {
            str: "string", int: "integer", float: "number",
            bool: "boolean", list: "array", dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            hint = hints.get(param_name, str)
            json_type = _py_to_json.get(hint, "string")
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {"type": "object", "properties": properties, "required": required}


def tool(_fn: Optional[Callable] = None, *, description: str = "", name: Optional[str] = None):
    """
    Decorator to create a Tool from a function.

    Can be used with or without arguments:
        @tool
        async def my_tool(x: str) -> str: ...

        @tool(description="Custom description")
        async def my_tool(x: str) -> str: ...
    """
    def decorator(fn: Callable) -> Tool:
        return Tool(fn, description=description, name=name)

    if _fn is not None:
        return decorator(_fn)
    return decorator
