"""Tests for LLM types, prompt templates, and client logic (no live API calls)."""

import pytest
from ai_core.llm.types import Message, Role, CompletionResponse
from ai_core.llm.prompt import PromptTemplate, PromptLibrary


def test_message_roles():
    assert Message.system("hi").role == Role.SYSTEM
    assert Message.user("hi").role == Role.USER
    assert Message.assistant("hi").role == Role.ASSISTANT


def test_message_to_dict():
    m = Message.user("hello")
    d = m.to_dict()
    assert d == {"role": "user", "content": "hello"}


def test_completion_response_usage():
    r = CompletionResponse(
        content="ok", model="gpt-4o", provider="openai",
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
    )
    assert r.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_prompt_template_render():
    t = PromptTemplate(name="greet", template="Hello {name}, you are {age} years old.")
    assert t.render(name="Alice", age=30) == "Hello Alice, you are 30 years old."


def test_prompt_template_variables():
    t = PromptTemplate(name="t", template="Dear {name}, your order {order_id} is ready.")
    assert t.variables == {"name", "order_id"}


def test_prompt_template_missing_variable():
    t = PromptTemplate(name="t", template="Hello {name}!")
    with pytest.raises(ValueError, match="missing variables"):
        t.render()


def test_prompt_template_to_message():
    t = PromptTemplate(name="t", template="Summarise: {text}")
    msg = t.to_message(text="long article")
    assert msg.role == Role.USER
    assert "long article" in msg.content


def test_prompt_library():
    lib = PromptLibrary()
    t = PromptTemplate(name="qa", template="Q: {q}\nA:")
    lib.register(t)
    assert "qa" in lib.names
    assert lib.render("qa", q="What is AI?").startswith("Q: What is AI?")


def test_prompt_library_missing():
    lib = PromptLibrary()
    with pytest.raises(KeyError):
        lib.get("nonexistent")


def test_prompt_template_save_load(tmp_path):
    t = PromptTemplate(name="test", template="Say {word}", description="test template")
    p = t.save(tmp_path)
    loaded = PromptTemplate.load(p)
    assert loaded.name == "test"
    assert loaded.variables == {"word"}
    assert loaded.render(word="hello") == "Say hello"
