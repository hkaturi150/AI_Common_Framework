"""
agent_demo.py — example of the Agent + tool system.

Run:
    OPENAI_API_KEY=sk-... python -m example_app.agent_demo
"""

import asyncio
from ai_core import LLMClient, LLMConfig, Agent
from ai_core.agents import tool


# ── Define tools ──────────────────────────────────────────────────────────

@tool(description="Search the web and return a short summary of results")
async def web_search(query: str) -> str:
    # plug in your real search client (serpapi, brave, tavily, etc.)
    return f"[stub] Top results for: {query}"


@tool(description="Get the current weather for a city")
async def get_weather(city: str, units: str = "celsius") -> dict:
    return {"city": city, "temp": 22, "condition": "sunny", "units": units}


@tool(description="Save a note to memory with a given key")
async def save_note(key: str, content: str) -> str:
    # In real usage, write to DB / Redis / file
    return f"Saved note '{key}'"


# ── Run agent ─────────────────────────────────────────────────────────────

async def main():
    llm = LLMClient(LLMConfig(provider="openai", model="gpt-4o", temperature=0))
    agent = Agent(
        llm=llm,
        tools=[web_search, get_weather, save_note],
        system="You are a helpful research assistant. Use tools to answer questions.",
        config={"max_iterations": 6, "verbose": True},
    )

    result = await agent.run("What's the weather in NYC and find me 3 recent AI papers?")
    print("\n=== FINAL ANSWER ===")
    print(result.output)
    print(f"\nCompleted in {result.iterations} iterations, {len(result.steps)} tool calls")


if __name__ == "__main__":
    asyncio.run(main())
