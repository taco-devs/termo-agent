"""Agent loop using OpenAI Agents SDK + LiteLLM for model routing."""

import json
import os
import re
import subprocess
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path
from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel

AGENT_DIR = Path(__file__).parent
SESSIONS_DIR = AGENT_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)
WORKSPACE = Path("/home/sprite/workspace")
WORKSPACE.mkdir(exist_ok=True)

# Max messages to keep in context (sliding window)
MAX_CONTEXT_MESSAGES = 50

# In-memory session cache
_sessions: dict[str, list] = {}

# Dangerous command patterns
_DENY_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/\b",
    r"\bmkfs\b", r"\bdd\s+.*of=/dev/", r"\b:(){ :\|:& };:\b",
    r"\bshutdown\b", r"\breboot\b", r"\bhalt\b",
    r"\bchmod\s+(-R\s+)?777\s+/\b",
    r"\bchown\s+.*\s+/\b",
    r"\b>\s*/dev/sd[a-z]", r"\bwget\s.*\|\s*sh\b", r"\bcurl\s.*\|\s*sh\b",
]


# --- Helper: simple HTML to text ---

class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._text = []
        self._skip = False
        self._skip_tags = {"script", "style", "noscript", "svg", "path"}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True
        if tag in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._text.append("\n")

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._text.append(data)

    def get_text(self) -> str:
        text = "".join(self._text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


# --- Tools ---

@function_tool
def execute_command(command: str) -> str:
    """Execute a shell command on the agent's machine. Returns stdout + stderr.
    Use this to run scripts, install packages, check system info, etc.
    The working directory is /home/sprite/workspace."""
    for pattern in _DENY_PATTERNS:
        if re.search(pattern, command):
            return f"[blocked: dangerous command pattern detected]"
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=120, cwd=str(WORKSPACE),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output[:100000] or "[no output]"
    except subprocess.TimeoutExpired:
        return "[error: command timed out after 120s]"
    except Exception as e:
        return f"[error: {e}]"


@function_tool
def read_file(path: str) -> str:
    """Read the contents of a file. Path can be absolute or relative to /home/sprite/workspace."""
    try:
        p = Path(path)
        if not p.is_absolute():
            p = WORKSPACE / p
        if not p.exists():
            return f"[error: file not found: {p}]"
        content = p.read_text(errors="replace")
        if len(content) > 50000:
            return content[:50000] + "\n... [truncated]"
        return content
    except Exception as e:
        return f"[error: {e}]"


@function_tool
def write_file(path: str, content: str) -> str:
    """Write content to a file. Path can be absolute or relative to /home/sprite/workspace.
    Creates parent directories if needed."""
    try:
        p = Path(path)
        if not p.is_absolute():
            p = WORKSPACE / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"[wrote {len(content)} bytes to {p}]"
    except Exception as e:
        return f"[error: {e}]"


@function_tool
def edit_file(path: str, find: str, replace: str) -> str:
    """Edit a file by replacing the first occurrence of 'find' with 'replace'.
    More precise than rewriting the whole file. Path can be absolute or relative."""
    try:
        p = Path(path)
        if not p.is_absolute():
            p = WORKSPACE / p
        if not p.exists():
            return f"[error: file not found: {p}]"
        content = p.read_text()
        if find not in content:
            return f"[error: search string not found in {p}]"
        new_content = content.replace(find, replace, 1)
        p.write_text(new_content)
        return f"[edited {p} — replaced 1 occurrence ({len(find)} chars -> {len(replace)} chars)]"
    except Exception as e:
        return f"[error: {e}]"


@function_tool
def list_files(path: str = ".") -> str:
    """List files and directories at the given path. Defaults to workspace root."""
    try:
        p = Path(path)
        if not p.is_absolute():
            p = WORKSPACE / p
        if not p.exists():
            return f"[error: path not found: {p}]"
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        lines = []
        for e in entries[:200]:
            prefix = "d " if e.is_dir() else "f "
            size = e.stat().st_size if e.is_file() else 0
            lines.append(f"{prefix}{e.name:40s} {size:>10d}")
        return "\n".join(lines) or "[empty directory]"
    except Exception as e:
        return f"[error: {e}]"


@function_tool
def web_search(query: str) -> str:
    """Search the web using Brave Search API. Returns top results with titles, URLs, and snippets."""
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        return "[error: BRAVE_API_KEY not configured — web search unavailable]"
    try:
        q = urllib.parse.urlencode({"q": query, "count": 5})
        req = urllib.request.Request(
            f"https://api.search.brave.com/res/v1/web/search?{q}",
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        results = data.get("web", {}).get("results", [])
        if not results:
            return "[no results found]"
        lines = []
        for r in results[:5]:
            lines.append(f"**{r.get('title', '')}**")
            lines.append(r.get("url", ""))
            desc = r.get("description", "")
            if desc:
                lines.append(desc)
            lines.append("")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"[error: {e}]"


@function_tool
def web_fetch(url: str) -> str:
    """Fetch a URL and return its text content. HTML is converted to readable text.
    Use this to read documentation, articles, API responses, etc."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; TermoAgent/1.0)",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            body = resp.read(500_000).decode("utf-8", errors="replace")
        if "html" in content_type.lower() or body.strip().startswith("<"):
            body = _html_to_text(body)
        if len(body) > 50000:
            body = body[:50000] + "\n... [truncated]"
        return body or "[empty response]"
    except Exception as e:
        return f"[error: {e}]"


@function_tool
def send_message(content: str, conversation_id: str) -> str:
    """Send a proactive message to the user in a conversation. Use this when you
    want to notify the user about something — e.g., a completed background task
    or an important update."""
    api_url = os.environ.get("TERMO_API_URL", "")
    token = os.environ.get("TERMO_TOKEN", "")
    if not api_url or not token:
        return "[error: TERMO_API_URL or TERMO_TOKEN not configured]"
    if not conversation_id:
        return "[error: conversation_id is required]"
    try:
        payload = json.dumps({
            "token": token,
            "conversation_id": conversation_id,
            "content": content,
        }).encode()
        req = urllib.request.Request(
            f"{api_url}/agents/deliver",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
        return f"[message delivered: {result.get('status', 'ok')}]"
    except Exception as e:
        return f"[error: {e}]"


@function_tool
def remember(content: str, category: str = "fact") -> str:
    """Save information to semantic long-term memory. This persists across conversations.
    Use this to remember user preferences, key facts, project details, identity, skills,
    or anything the user asks you to remember.
    Categories: identity, preference, fact, project, skill, user_profile."""
    try:
        from memory import remember as mem_remember
        result = mem_remember(content, category)
        return f"[memory {result['status']}: {result['id']}] {content[:80]}"
    except Exception as e:
        return f"[error: {e}]"


@function_tool
def recall(query: str, limit: int = 5, category: str = "") -> str:
    """Search memories semantically. Use before answering questions about past context,
    user preferences, or previously discussed topics. Returns the most relevant memories."""
    try:
        from memory import recall as mem_recall
        cat = category if category else None
        results = mem_recall(query, limit=limit, category=cat)
        if not results:
            return "[no matching memories found]"
        lines = []
        for m in results:
            lines.append(f"[{m['category']}] (similarity: {m['similarity']}) {m['content']}")
        return "\n".join(lines)
    except Exception as e:
        return f"[error: {e}]"


@function_tool
def forget(query: str) -> str:
    """Delete the closest matching memory. Use to remove outdated or incorrect memories."""
    try:
        from memory import forget as mem_forget
        result = mem_forget(query)
        if result.get("deleted"):
            return f"[deleted memory {result['id']}]: {result['content'][:80]}"
        return f"[not deleted: {result.get('error', 'unknown')}]"
    except Exception as e:
        return f"[error: {e}]"


@function_tool
def update_memory(query: str, new_content: str) -> str:
    """Update an existing memory in place. Finds the closest match to 'query'
    and replaces its content with 'new_content'."""
    try:
        from memory import update_memory_entry
        result = update_memory_entry(query, new_content)
        if result.get("updated"):
            return f"[updated memory {result['id']}]: {new_content[:80]}"
        return f"[not updated: {result.get('error', 'unknown')}]"
    except Exception as e:
        return f"[error: {e}]"


@function_tool
def search_messages(query: str, limit: int = 10) -> str:
    """Search past conversation messages across all conversations.
    Use this to recall what was discussed in previous conversations."""
    api_url = os.environ.get("TERMO_API_URL", "")
    token = os.environ.get("TERMO_TOKEN", "")
    if not api_url or not token:
        return "[error: TERMO_API_URL or TERMO_TOKEN not configured]"
    try:
        payload = json.dumps({"token": token, "query": query, "limit": limit}).encode()
        req = urllib.request.Request(
            f"{api_url}/agents/search-messages",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        messages = data.get("messages", [])
        if not messages:
            return "[no matching messages found]"
        lines = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")[:200]
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)
    except Exception as e:
        return f"[error: {e}]"


TOOLS = [
    execute_command, read_file, write_file, edit_file, list_files,
    web_search, web_fetch, send_message,
    remember, recall, forget, update_memory, search_messages,
]


def _needs_bootstrap() -> bool:
    """Check if this agent needs first-run bootstrap."""
    flag = AGENT_DIR / ".bootstrapped"
    bootstrap = AGENT_DIR / "BOOTSTRAP.md"
    return bootstrap.exists() and not flag.exists()


def _get_bootstrap() -> str:
    """Load and consume BOOTSTRAP.md. Marks bootstrap as done."""
    bootstrap = AGENT_DIR / "BOOTSTRAP.md"
    if not bootstrap.exists():
        return ""
    content = bootstrap.read_text().strip()
    flag = AGENT_DIR / ".bootstrapped"
    flag.write_text("done")
    return content


def _get_model(config: dict) -> LitellmModel:
    """Build a LitellmModel from agent config."""
    model_name = config.get("model", "openrouter/google/gemini-2.0-flash-001")
    api_key = config.get("api_key", "")
    api_base = config.get("api_base", "https://api.termo.ai/v1")
    return LitellmModel(
        model=model_name,
        api_key=api_key,
        base_url=api_base,
    )


def _build_agent(config: dict, soul_md: str, message: str = "") -> Agent:
    """Build an Agent instance with smart memory context injection."""
    model = _get_model(config)
    instructions = soul_md

    # 1. Always inject identity + preference memories (core personality)
    try:
        from memory import get_identity_and_preference_memories, recall as mem_recall
        core = get_identity_and_preference_memories()
        if core:
            core_text = "\n".join(f"- [{m['category']}] {m['content']}" for m in core)
            instructions += f"\n\n## Your Core Identity & Preferences\n{core_text}"

        # 2. Add query-relevant memories (not identity/preference, similarity > 0.3)
        if message:
            relevant = mem_recall(message, limit=5)
            relevant = [m for m in relevant if m["category"] not in ("identity", "preference") and m["similarity"] > 0.3]
            if relevant:
                rel_text = "\n".join(f"- [{m['category']}] {m['content']}" for m in relevant)
                instructions += f"\n\n## Relevant Memories\n{rel_text}"
    except Exception:
        # Fallback: try legacy memory.md
        mem_file = AGENT_DIR / "memory" / "memory.md"
        if mem_file.exists():
            content = mem_file.read_text().strip()
            if content:
                instructions += f"\n\n## Your Memory\n{content}"

    # Inject bootstrap on first-ever message
    if _needs_bootstrap():
        bootstrap = _get_bootstrap()
        if bootstrap:
            instructions += f"\n\n## FIRST RUN — Bootstrap\n{bootstrap}"

    return Agent(
        name=config.get("persona_name", "Assistant"),
        instructions=instructions,
        model=model,
        tools=TOOLS,
    )


def _trim_messages(messages: list) -> list:
    """Keep messages within MAX_CONTEXT_MESSAGES to prevent context overflow.
    Always preserves the first message if it is a system message."""
    if len(messages) <= MAX_CONTEXT_MESSAGES:
        return messages
    if messages and messages[0].get("role") == "system":
        return [messages[0]] + messages[-(MAX_CONTEXT_MESSAGES - 1):]
    return messages[-MAX_CONTEXT_MESSAGES:]


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
    return [
        {"name": "execute_command", "description": "Execute a shell command"},
        {"name": "read_file", "description": "Read a file's contents"},
        {"name": "write_file", "description": "Write content to a file"},
        {"name": "edit_file", "description": "Search-and-replace edit a file"},
        {"name": "list_files", "description": "List files in a directory"},
        {"name": "web_search", "description": "Search the web (Brave)"},
        {"name": "web_fetch", "description": "Fetch and read a URL"},
        {"name": "send_message", "description": "Send a proactive message to the user"},
        {"name": "remember", "description": "Save to semantic long-term memory"},
        {"name": "recall", "description": "Search memories semantically"},
        {"name": "forget", "description": "Delete a memory"},
        {"name": "update_memory", "description": "Update an existing memory"},
        {"name": "search_messages", "description": "Search past conversation messages"},
    ]


def clear_sessions():
    """Clear all in-memory sessions."""
    _sessions.clear()


async def run_agent_stream(message: str, session_key: str, config: dict, soul_md: str):
    """Run the agent and yield SSE events."""
    agent = _build_agent(config, soul_md, message=message)
    messages = _load_session(session_key)

    # Add user message to session
    messages.append({"role": "user", "content": message})

    # Trim for context window but keep full history on disk
    trimmed = _trim_messages(messages)

    try:
        result = Runner.run_streamed(agent, input=trimmed)

        streamed_content = ""
        tokens_in = 0
        tokens_out = 0
        prev_delta = None
        last_tool_name = "unknown"

        async for event in result.stream_events():
            event_type = getattr(event, "type", str(type(event).__name__))

            if event_type == "raw_response_event":
                raw = event.data
                delta = None
                if hasattr(raw, "delta") and isinstance(raw.delta, str):
                    delta = raw.delta
                elif hasattr(raw, "choices") and raw.choices:
                    d = getattr(raw.choices[0], "delta", None)
                    if d:
                        delta = getattr(d, "content", None)
                if delta and delta == prev_delta:
                    prev_delta = None
                    continue
                if delta:
                    prev_delta = delta
                    streamed_content += delta
                    yield {"type": "token", "content": delta}

            elif event_type == "run_item_stream_event":
                item = event.item
                item_type = getattr(item, "type", "")
                if item_type == "tool_call_item":
                    raw = getattr(item, "raw_item", None)
                    tool_name = getattr(raw, "name", "unknown") if raw else "unknown"
                    tool_args = getattr(raw, "arguments", "") if raw else ""
                    last_tool_name = tool_name
                    # Parse arguments JSON for structured input
                    tool_input = None
                    if tool_args:
                        try:
                            tool_input = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                        except (json.JSONDecodeError, TypeError):
                            tool_input = {"raw": tool_args}
                    yield {"type": "tool_start", "name": tool_name, "input": tool_input}
                elif item_type == "tool_call_output_item":
                    output = getattr(item, "output", "") or ""
                    yield {"type": "tool_end", "name": last_tool_name, "output": str(output)[:5000]}

        assistant_content = str(result.final_output) if result.final_output else streamed_content

        if hasattr(result, "raw_responses") and result.raw_responses:
            for resp in result.raw_responses:
                usage = getattr(resp, "usage", None)
                if usage:
                    tokens_in += getattr(usage, "prompt_tokens", 0) or 0
                    tokens_out += getattr(usage, "completion_tokens", 0) or 0

        # Save to full session (not trimmed)
        messages.append({"role": "assistant", "content": assistant_content})
        _save_session(session_key, messages)

        yield {
            "type": "done",
            "content": assistant_content,
            "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
        }

    except Exception as e:
        import traceback as tb
        yield {"type": "error", "content": str(e), "trace": tb.format_exc()}


async def run_agent(message: str, session_key: str, config: dict, soul_md: str) -> dict:
    """Run the agent synchronously (non-streaming) and return the result."""
    agent = _build_agent(config, soul_md, message=message)
    messages = _load_session(session_key)
    messages.append({"role": "user", "content": message})

    trimmed = _trim_messages(messages)
    result = await Runner.run(agent, input=trimmed)
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
