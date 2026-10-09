"""Tests for utilities: logging, retry, timer, env."""

import os
import pytest
from ai_core.utils.env import env
from ai_core.utils.timer import timer, timed
from ai_core.utils.retry import retry_async, retry_sync


# ── env ────────────────────────────────────────────────────────────────────

def test_env_reads_variable(monkeypatch):
    monkeypatch.setenv("TEST_VAR", "hello")
    assert env("TEST_VAR") == "hello"


def test_env_default(monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    assert env("MISSING_VAR", default="fallback") == "fallback"


def test_env_required_raises(monkeypatch):
    monkeypatch.delenv("REQUIRED_VAR", raising=False)
    with pytest.raises(EnvironmentError, match="REQUIRED_VAR"):
        env("REQUIRED_VAR")


def test_env_cast_int(monkeypatch):
    monkeypatch.setenv("PORT", "8080")
    assert env("PORT", cast=int) == 8080


def test_env_cast_bool_true(monkeypatch):
    for val in ("1", "true", "True", "yes", "on"):
        monkeypatch.setenv("FLAG", val)
        assert env("FLAG", cast=bool) is True


def test_env_cast_bool_false(monkeypatch):
    monkeypatch.setenv("FLAG", "false")
    assert env("FLAG", cast=bool) is False


# ── timer ──────────────────────────────────────────────────────────────────

def test_timer_context_manager():
    with timer("test block", log=False):
        x = sum(range(1000))
    assert x == 499500


@pytest.mark.asyncio
async def test_timed_decorator_async():
    @timed(label="test")
    async def slow():
        return 42

    result = await slow()
    assert result == 42


# ── retry ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retry_succeeds_eventually():
    calls = [0]

    async def flaky():
        calls[0] += 1
        if calls[0] < 3:
            raise ConnectionError("not yet")
        return "ok"

    result = await retry_async(
        flaky,
        max_retries=5,
        base_delay=0.01,
        retryable_exceptions=(ConnectionError,),
    )
    assert result == "ok"
    assert calls[0] == 3


@pytest.mark.asyncio
async def test_retry_exhausted():
    async def always_fails():
        raise ConnectionError("nope")

    with pytest.raises(ConnectionError):
        await retry_async(
            always_fails,
            max_retries=2,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )


@pytest.mark.asyncio
async def test_retry_non_retryable():
    async def bad():
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        await retry_async(bad, max_retries=3, base_delay=0.01)
