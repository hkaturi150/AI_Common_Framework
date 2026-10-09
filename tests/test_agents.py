"""Tests for the Tool and Agent building blocks."""

import pytest
from ai_core.agents.tool import Tool, ToolCall, ToolResult, tool


# ── Tool definition ────────────────────────────────────────────────────────

def test_tool_from_decorator():
    @tool(description="Add two numbers")
    async def add(a: int, b: int) -> int:
        return a + b

    assert isinstance(add, Tool)
    assert add.name == "add"
    assert add.description == "Add two numbers"


def test_tool_schema_required():
    @tool
    async def greet(name: str) -> str:
        return f"hello {name}"

    schema = greet.to_openai_schema()
    assert schema["type"] == "function"
    assert "name" in schema["function"]["parameters"]["required"]


def test_tool_schema_optional():
    @tool
    async def greet(name: str, greeting: str = "hello") -> str:
        return f"{greeting} {name}"

    schema = greet.to_openai_schema()
    required = schema["function"]["parameters"]["required"]
    assert "name" in required
    assert "greeting" not in required


@pytest.mark.asyncio
async def test_tool_call():
    @tool
    async def multiply(x: int, y: int) -> int:
        return x * y

    result = await multiply(x=3, y=4)
    assert result == 12


def test_tool_result_error():
    r = ToolResult(tool_call_id="1", name="t", result=None, error="oops")
    assert "ERROR" in r.to_message_content()


def test_tool_result_json():
    r = ToolResult(tool_call_id="1", name="t", result={"key": "val"})
    import json
    assert json.loads(r.to_message_content()) == {"key": "val"}


def test_tool_anthropic_schema():
    @tool(description="Fetch data")
    async def fetch(url: str) -> str:
        return ""

    schema = fetch.to_anthropic_schema()
    assert schema["name"] == "fetch"
    assert "input_schema" in schema


# ── Pipeline ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_basic():
    from ai_core.pipeline import Pipeline

    pipeline = (
        Pipeline("test")
        .step("double", lambda x: x * 2)
        .step("add_ten", lambda x: x + 10)
    )
    result = await pipeline.run(5)
    assert result.success
    assert result.outputs["add_ten"] == 20


@pytest.mark.asyncio
async def test_pipeline_stops_on_error():
    from ai_core.pipeline import Pipeline

    def fail(x):
        raise ValueError("intentional")

    pipeline = (
        Pipeline("test", on_error="stop")
        .step("ok", lambda x: x + 1)
        .step("fail", fail)
        .step("never_reached", lambda x: x)
    )
    result = await pipeline.run(1)
    assert not result.success
    assert "fail" in [s.name for s in result.steps if not s.success]
    assert "never_reached" not in [s.name for s in result.steps]


@pytest.mark.asyncio
async def test_pipeline_continue_on_error():
    from ai_core.pipeline import Pipeline

    def fail(x):
        raise ValueError("intentional")

    pipeline = (
        Pipeline("test", on_error="continue")
        .step("ok", lambda x: x + 1)
        .step("fail", fail, skip_on_error=True)
        .step("after", lambda x: 99)
    )
    result = await pipeline.run(1)
    # Pipeline continues despite the error
    assert "after" in [s.name for s in result.steps]
