"""Agent loop using OpenAI Agents SDK + LiteLLM for model routing."""

import json
import os
from pathlib import Path
from agents import Agent, Runner
from agents.extensions.models.litellm import LitellmModel

AGENT_DIR = Path(__file__).parent
SESSIONS_DIR = AGENT_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

# In-memory session cache
_sessions: dict[str, list] = {}


def _get_model(config: dict) -> LitellmModel:
    """Build a LitellmModel from agent config."""
    model_name = config.get("model", "openrouter/google/gemini-2.0-flash-001")
    api_key = config.get("api_key", "")
    api_base = config.get("api_base", "https://api.termo.ai/v1")
    return LitellmModel(model=model_name, api_key=api_key, base_url=api_base)


def _build_agent(config: dict, soul_md: str) -> Agent:
    """Build an Agent instance from config and system prompt."""
    model = _get_model(config)
    return Agent(
        name=config.get("persona_name", "Assistant"),
        instructions=soul_md,
        model=model,
    )


def _load_session(session_key: str) -> list:
    """Load session messages from memory or disk."""
    if session_key in _sessions:
        return _sessions[session_key]
    safe_key = session_key.replace(":", "_")
    session_file = SESSIONS_DIR / f"{safe_key}.json"
    if session_file.exists():
        messages = json.loads(session_file.read_text())
        _sessions[session_key] = messages
        return messages
    _sessions[session_key] = []
    return _sessions[session_key]


def _save_session(session_key: str, messages: list) -> None:
    """Persist session messages to disk."""
    _sessions[session_key] = messages
    safe_key = session_key.replace(":", "_")
    session_file = SESSIONS_DIR / f"{safe_key}.json"
    session_file.write_text(json.dumps(messages, default=str))


def get_available_tools() -> list[dict]:
    """Return list of available tool descriptions."""
    return []


def clear_sessions():
    """Clear all in-memory sessions."""
    _sessions.clear()


async def run_agent_stream(message: str, session_key: str, config: dict, soul_md: str):
    """Run the agent and yield SSE events."""
    agent = _build_agent(config, soul_md)
    messages = _load_session(session_key)

    # Add user message to session
    messages.append({"role": "user", "content": message})

    try:
        result = Runner.run_streamed(agent, input=messages)

        assistant_content = ""
        tokens_in = 0
        tokens_out = 0

        async for event in result.stream_events():
            event_type = getattr(event, "type", str(type(event).__name__))

            if event_type == "raw_response_event":
                raw = event.data
                # Delta text from streaming
                delta = getattr(raw, "delta", None)
                if delta:
                    assistant_content += delta
                    yield {"type": "token", "content": delta}

            elif event_type == "run_item_stream_event":
                item = event.item
                item_type = getattr(item, "type", "")
                if item_type == "tool_call_item":
                    tool_name = getattr(item, "name", "unknown")
                    yield {"type": "tool_start", "name": tool_name}
                elif item_type == "tool_call_output_item":
                    tool_name = getattr(item, "name", "unknown")
                    yield {"type": "tool_end", "name": tool_name}

        # Get final output
        final_output = result.final_output
        if final_output and not assistant_content:
            assistant_content = str(final_output)

        # Extract usage from result
        if hasattr(result, "raw_responses") and result.raw_responses:
            for resp in result.raw_responses:
                usage = getattr(resp, "usage", None)
                if usage:
                    tokens_in += getattr(usage, "prompt_tokens", 0) or 0
                    tokens_out += getattr(usage, "completion_tokens", 0) or 0

        # Save assistant message to session
        messages.append({"role": "assistant", "content": assistant_content})
        _save_session(session_key, messages)

        yield {
            "type": "done",
            "content": assistant_content,
            "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
        }

    except Exception as e:
        yield {"type": "error", "content": str(e)}


async def run_agent(message: str, session_key: str, config: dict, soul_md: str) -> dict:
    """Run the agent synchronously (non-streaming) and return the result."""
    agent = _build_agent(config, soul_md)
    messages = _load_session(session_key)
    messages.append({"role": "user", "content": message})

    result = await Runner.run(agent, input=messages)
    assistant_content = str(result.final_output) if result.final_output else ""

    messages.append({"role": "assistant", "content": assistant_content})
    _save_session(session_key, messages)

    tokens_in = tokens_out = 0
    if hasattr(result, "raw_responses") and result.raw_responses:
        for resp in result.raw_responses:
            usage = getattr(resp, "usage", None)
            if usage:
                tokens_in += getattr(usage, "prompt_tokens", 0) or 0
                tokens_out += getattr(usage, "completion_tokens", 0) or 0

    return {
        "content": assistant_content,
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    }
