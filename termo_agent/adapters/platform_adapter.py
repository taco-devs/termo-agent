"""Termo platform adapter — full-featured AgentAdapter for Sprites deployment.

Refactored from the embedded _AGENT_LOOP_PY + _SERVER_PY strings in provision_agent.py.
Runs inside a Sprite via `termo-agent --adapter platform_adapter`.
"""

import asyncio
import json
import mimetypes
import os
import re
import subprocess
import time as _time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import AsyncIterator

from aiohttp import web
from dotenv import load_dotenv

from termo_agent.adapter import AgentAdapter, StreamEvent

# Resolve agent dir: when installed as a pip package, __file__ points to
# the package directory, not the sprite's agent dir.  Use TERMO_AGENT_DIR
# env var or fall back to the well-known sprite path.
AGENT_DIR = Path(os.environ.get("TERMO_AGENT_DIR", "/home/sprite/agent"))
load_dotenv(AGENT_DIR / ".env")

SESSIONS_DIR = AGENT_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)
WORKSPACE = Path("/home/sprite/workspace")
WORKSPACE.mkdir(exist_ok=True)

# Protected files — prevent accidental deletion of critical config
_PROTECTED_FILES = {"/home/sprite/agent/.env"}


def _is_protected(p: Path) -> bool:
    """Return True if path is a protected file (e.g. .env with tokens)."""
    try:
        return str(p.resolve()) in _PROTECTED_FILES
    except Exception:
        return False


# Max messages to keep in context (sliding window)
MAX_CONTEXT_MESSAGES = 50

# In-memory session cache
_sessions: dict[str, list] = {}

# Current conversation context — set before each agent run so tools can read it
_current_conversation_id: str | None = None

# Dangerous command patterns
_DENY_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/\b",
    r"\bmkfs\b", r"\bdd\s+.*of=/dev/", r"\b:(){ :\|:& };:\b",
    r"\bshutdown\b", r"\breboot\b", r"\bhalt\b",
    r"\bchmod\s+(-R\s+)?777\s+/\b",
    r"\bchown\s+.*\s+/\b",
    r"\b>\s*/dev/sd[a-z]", r"\bwget\s.*\|\s*sh\b", r"\bcurl\s.*\|\s*sh\b",
    # Block binding to port 8080 (reserved for agent runtime)
    r"\b8080\b.*\b(server|listen|bind|http\.server)\b",
    r"\b(server|listen|bind|http\.server)\b.*\b8080\b",
]

# --- Smart memory: extraction prompt ---

_MEMORY_EXTRACTION_PROMPT = """You are a memory extraction system. Given a conversation exchange, extract 0-3 durable facts worth remembering long-term. Return ONLY a JSON array.

Rules:
- Skip greetings, acknowledgments, and transient requests ("ok", "thanks", "hello")
- Skip information that is only relevant to the current task and won't matter later
- Focus on: user preferences, identity facts, project details, technical choices, personal details
- Each fact must be a standalone sentence that makes sense without context
- Categories: identity, preference, fact, project, user_profile
- When a fact relates to a concept that could be its own memory, wrap it in [[double brackets]]. Examples:
  - "User is building a [[React]] app with [[Supabase]]"
  - "Prefers [[dark mode]] in all editors"
  - "Working on [[Project Atlas]], a [[Python]] CLI tool"
- Only link nouns/concepts, not verbs or adjectives. Max 3 links per fact.

Return format: [{{"content": "...", "category": "..."}}]
Return [] if nothing is worth remembering.

Conversation:
User: {user_message}
Assistant: {assistant_response}

Recent context:
{session_context}

Extract durable facts (JSON only):"""

# --- Smart memory: vague/short message detection ---

_VAGUE_PATTERNS = [
    re.compile(r"^(ok|okay|sure|yes|no|yep|nope|thanks|thank you|thx|ty|cool|nice|great|good|fine|alright|got it|hmm|hm|ah|oh|wow|lol|haha)[\.\!\?]?$", re.IGNORECASE),
    re.compile(r"^(keep going|continue|go on|go ahead|proceed|next|more|do it|fix that|fix it|try again|same|again|what\?|why\?|how\?)[\.\!\?]?$", re.IGNORECASE),
]

_CODING_SIGNALS = re.compile(
    r"\b(code|debug|deploy|react|python|javascript|typescript|api|database|sql|css|html|git|docker|server|backend|frontend|component|function|class|error|bug|test|build|compile|npm|pip|package)\b",
    re.IGNORECASE,
)

_ACTION_SIGNALS = re.compile(
    r"\b(create|build|write|install|deploy|setup|configure|make|add|implement|develop|design|generate|run|execute|start|launch|send|fetch|update|delete|remove)\b",
    re.IGNORECASE,
)


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


# ---------------------------------------------------------------------------
# Memory extraction helpers (sync, called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _extract_and_save_memories(user_message: str, assistant_response: str, session_tail: list, config: dict) -> None:
    """Background extraction of durable facts from a conversation turn."""
    try:
        if len(user_message) < 20 and len(assistant_response) < 50:
            return

        context_lines = []
        for msg in session_tail[-5:]:
            role = msg.get("role", "?")
            content = str(msg.get("content", ""))[:300]
            context_lines.append(f"{role}: {content}")
        session_context = "\n".join(context_lines) if context_lines else "(no prior context)"

        prompt = _MEMORY_EXTRACTION_PROMPT.format(
            user_message=user_message[:500],
            assistant_response=assistant_response[:500],
            session_context=session_context,
        )

        api_base = config.get("api_base", "https://api.termo.ai/v1")
        api_key = config.get("api_key", "")

        payload = json.dumps({
            "model": "openrouter/google/gemini-2.0-flash-001",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.0,
        }).encode()

        req = urllib.request.Request(
            f"{api_base}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
        raw_text = re.sub(r"\s*```$", "", raw_text.strip())

        extractions = json.loads(raw_text)
        if not isinstance(extractions, list):
            return

        from termo_agent.adapters.memory_engine import remember as mem_remember
        for item in extractions[:3]:
            content = item.get("content", "").strip()
            category = item.get("category", "fact")
            if content and len(content) > 5:
                mem_remember(content, category)
    except Exception:
        pass  # Never crash the agent


def _build_recall_query(message: str, session_messages: list | None = None) -> str:
    """Build an enriched recall query. For vague/short messages, include session context."""
    if not session_messages:
        return message

    is_vague = len(message) < 15 or any(p.match(message.strip()) for p in _VAGUE_PATTERNS)

    if not is_vague and len(message) > 30:
        return message

    context_parts = []
    for msg in session_messages[-5:]:
        content = str(msg.get("content", ""))[:200]
        if content:
            context_parts.append(content)
    context_parts.append(message)
    return " ".join(context_parts)[:500]


def _detect_categories(message: str, session_messages: list | None = None) -> list[str]:
    """Detect which memory categories to do targeted recall for."""
    combined = message
    if session_messages:
        for msg in session_messages[-3:]:
            combined += " " + str(msg.get("content", ""))[:200]

    categories = []
    if _CODING_SIGNALS.search(combined):
        categories.append("project")
    return categories


# ---------------------------------------------------------------------------
# @function_tool definitions (imported lazily from agents SDK)
# ---------------------------------------------------------------------------


def _define_tools():
    """Define all function_tool tools. Called once during initialize()."""
    from agents import function_tool

    @function_tool
    def execute_command(command: str) -> str:
        """Execute a shell command on the agent's machine. Returns stdout + stderr.
        Use this to run scripts, install packages, check system info, etc.
        The working directory is /home/sprite/workspace."""
        for pattern in _DENY_PATTERNS:
            if re.search(pattern, command):
                return "[blocked: dangerous command pattern detected]"
        if re.search(r"(rm|mv|truncate|>)\s.*\.env\b", command):
            return "[blocked: .env is a protected file]"
        # Block killing the agent runtime process (by PID or name)
        _agent_pid = str(os.getpid())
        _parent_pid = str(os.getppid())
        kill_match = re.findall(r"\bkill\s+(?:-\d+\s+)?(\d+)", command)
        pkill_match = re.search(r"\bpkill\s.*(?:termo|agent|platform)", command)
        if pkill_match:
            return "[blocked: cannot kill the agent runtime process]"
        for pid in kill_match:
            if pid in (_agent_pid, _parent_pid, "1"):
                return "[blocked: cannot kill the agent runtime process]"
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
            if _is_protected(p):
                return "[blocked: .env is a protected file]"
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
            if _is_protected(p):
                return "[blocked: .env is a protected file]"
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
        """Search the web. Returns top results with titles, URLs, and snippets."""
        api_url = os.environ.get("TERMO_API_URL", "")
        token = os.environ.get("TERMO_TOKEN", "")
        if not api_url or not token:
            return "[error: TERMO_API_URL or TERMO_TOKEN not configured]"
        try:
            payload = json.dumps({"token": token, "query": query}).encode()
            req = urllib.request.Request(
                f"{api_url}/agents/tools/search",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
            if "error" in data:
                return f"[error: {data['error']}]"
            results = data.get("results", [])
            if not results:
                return "[no results found]"
            lines = []
            for r in results[:5]:
                lines.append(f"**{r.get('title', '')}**")
                lines.append(r.get("url", ""))
                highlights = r.get("highlights", [])
                if highlights:
                    lines.append(highlights[0])
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
            from termo_agent.adapters.memory_engine import remember as mem_remember
            result = mem_remember(content, category)
            return f"[memory {result['status']}: {result['id']}] {content[:80]}"
        except Exception as e:
            return f"[error: {e}]"

    @function_tool
    def recall(query: str, limit: int = 5, category: str = "") -> str:
        """Search memories semantically. Use before answering questions about past context,
        user preferences, or previously discussed topics. Returns the most relevant memories."""
        try:
            from termo_agent.adapters.memory_engine import recall as mem_recall
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
            from termo_agent.adapters.memory_engine import forget as mem_forget
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
            from termo_agent.adapters.memory_engine import update_memory_entry
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

    @function_tool
    def create_schedule(schedule: str, prompt: str, name: str = "") -> str:
        """Create a recurring scheduled task. Each time the schedule fires, the
        agent will be prompted with the given instruction and will generate a fresh response.
        IMPORTANT: 'prompt' must be the user's original request or instruction — NOT
        your own generated content. For example if the user says "send me a joke every
        hour", the prompt should be "Tell me an original joke" (the instruction), not
        the joke itself.
        schedule: cron expression (e.g. "0 9 * * *" for daily at 9am, "*/30 * * * *" for every 30 min)
        prompt: the instruction/request that will be sent to the agent on each trigger (must be short and actionable)
        name: optional short label for the schedule"""
        api_url = os.environ.get("TERMO_API_URL", "")
        agent_id = os.environ.get("TERMO_AGENT_ID", "")
        if not api_url or not agent_id:
            return "[error: TERMO_API_URL or TERMO_AGENT_ID not configured]"
        try:
            label = name or prompt[:60]
            cron_payload = {"message": prompt}
            if _current_conversation_id:
                cron_payload["conversation_id"] = _current_conversation_id
            payload = json.dumps({
                "name": label,
                "schedule": {"cron": schedule},
                "payload": cron_payload,
            }).encode()
            req = urllib.request.Request(
                f"{api_url}/agents/{agent_id}/crons",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            return f"[schedule created: '{label}' at {schedule}] id={data.get('id', '?')}"
        except Exception as e:
            return f"[error: {e}]"

    @function_tool
    def list_schedules() -> str:
        """List all active scheduled tasks for this agent."""
        api_url = os.environ.get("TERMO_API_URL", "")
        agent_id = os.environ.get("TERMO_AGENT_ID", "")
        if not api_url or not agent_id:
            return "[error: TERMO_API_URL or TERMO_AGENT_ID not configured]"
        try:
            req = urllib.request.Request(
                f"{api_url}/agents/{agent_id}/crons",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            crons = data.get("crons", [])
            if not crons:
                return "[no schedules found]"
            lines = []
            for c in crons:
                sched = c.get("schedule", {})
                cron_expr = sched.get("cron", str(sched))
                msg = c.get("payload", {}).get("message", c.get("name", ""))
                status = "enabled" if c.get("enabled") else "disabled"
                lines.append(f"- {c['id']}: [{cron_expr}] {msg} ({status}, {c.get('run_count', 0)} runs)")
            return "\n".join(lines)
        except Exception as e:
            return f"[error: {e}]"

    @function_tool
    def delete_schedule(schedule_id: str) -> str:
        """Delete a scheduled task by its ID. Use list_schedules to find IDs."""
        api_url = os.environ.get("TERMO_API_URL", "")
        agent_id = os.environ.get("TERMO_AGENT_ID", "")
        if not api_url or not agent_id:
            return "[error: TERMO_API_URL or TERMO_AGENT_ID not configured]"
        try:
            req = urllib.request.Request(
                f"{api_url}/agents/{agent_id}/crons/{schedule_id}",
                headers={"Content-Type": "application/json"},
                method="DELETE",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                json.loads(resp.read())
            return f"[schedule {schedule_id} deleted]"
        except Exception as e:
            return f"[error: {e}]"

    @function_tool
    def search_skills(query: str, limit: int = 10) -> str:
        """Search the skill marketplace for installable skills. Returns matching skills
        with names, slugs, summaries, and popularity stats. Use install_skill to install one."""
        api_url = os.environ.get("TERMO_API_URL", "")
        token = os.environ.get("TERMO_TOKEN", "")
        if not api_url or not token:
            return "[error: TERMO_API_URL or TERMO_TOKEN not configured]"
        try:
            payload = json.dumps({"token": token, "query": query, "limit": min(limit, 20)}).encode()
            req = urllib.request.Request(
                f"{api_url}/agents/search-skills",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            skills = data.get("skills", [])
            if not skills:
                return "[no matching skills found]"
            lines = []
            for s in skills:
                name = s.get("display_name", s.get("slug", "unknown"))
                slug = s.get("slug", "")
                summary = s.get("summary", "")[:120]
                downloads = s.get("downloads", 0)
                stars = s.get("stars", 0)
                lines.append(f"- **{name}** (`{slug}`) — {summary} [{downloads} downloads, {stars} stars]")
            return "\n".join(lines)
        except Exception as e:
            return f"[error: {e}]"

    @function_tool
    def install_skill(slug: str) -> str:
        """Install a skill from the marketplace. After installing,
        use load_skill(slug) to load the full instructions into context."""
        api_url = os.environ.get("TERMO_API_URL", "")
        token = os.environ.get("TERMO_TOKEN", "")
        if not api_url or not token:
            return "[error: TERMO_API_URL or TERMO_TOKEN not configured]"
        try:
            payload = json.dumps({"token": token, "slug": slug}).encode()
            req = urllib.request.Request(
                f"{api_url}/agents/install-skill",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 409:
                return f"[skill '{slug}' is already installed — use load_skill('{slug}') to access it]"
            if e.code == 404:
                return f"[skill '{slug}' not found in the marketplace]"
            return f"[error: {e}]"
        except Exception as e:
            return f"[error: {e}]"

        name = data.get("name", slug)
        version = data.get("version", "1.0.0")

        return f"[installed '{name}' v{version} — use load_skill('{slug}') to access instructions]"

    @function_tool
    def load_skill(slug: str) -> str:
        """Load the full instructions of an installed skill into context.
        Use this when you need to follow a skill's instructions."""
        api_url = os.environ.get("TERMO_API_URL", "")
        token = os.environ.get("TERMO_TOKEN", "")
        if not api_url or not token:
            return "[error: TERMO_API_URL or TERMO_TOKEN not configured]"
        try:
            payload = json.dumps({"token": token, "slug": slug}).encode()
            req = urllib.request.Request(
                f"{api_url}/agents/load-skill",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return f"[skill '{slug}' not found or not installed]"
            return f"[error: {e}]"
        except Exception as e:
            return f"[error: {e}]"

        name = data.get("name", slug)
        content = data.get("content", "")
        if not content:
            return f"[skill '{name}' has no content]"
        return f"[SKILL: {name}]\n\n{content}"

    @function_tool
    def uninstall_skill(slug: str) -> str:
        """Remove an installed skill."""
        api_url = os.environ.get("TERMO_API_URL", "")
        token = os.environ.get("TERMO_TOKEN", "")
        if not api_url or not token:
            return "[error: TERMO_API_URL or TERMO_TOKEN not configured]"
        try:
            payload = json.dumps({"token": token, "slug": slug}).encode()
            req = urllib.request.Request(
                f"{api_url}/agents/uninstall-skill",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return f"[skill '{slug}' not found or not installed]"
            return f"[error: {e}]"
        except Exception as e:
            return f"[error: {e}]"

        return f"[uninstalled '{slug}']"

    # Return all tools as a dict
    return {
        "execute_command": execute_command,
        "read_file": read_file,
        "write_file": write_file,
        "edit_file": edit_file,
        "list_files": list_files,
        "web_search": web_search,
        "web_fetch": web_fetch,
        "send_message": send_message,
        "remember": remember,
        "recall": recall,
        "forget": forget,
        "update_memory": update_memory,
        "search_messages": search_messages,
        "create_schedule": create_schedule,
        "list_schedules": list_schedules,
        "delete_schedule": delete_schedule,
        "search_skills": search_skills,
        "install_skill": install_skill,
        "load_skill": load_skill,
        "uninstall_skill": uninstall_skill,
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _api_call(url: str, payload: dict, timeout: int = 15) -> dict:
    """Make a JSON POST to the Termo API."""
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def _needs_bootstrap() -> bool:
    flag = AGENT_DIR / ".bootstrapped"
    bootstrap = AGENT_DIR / "BOOTSTRAP.md"
    return bootstrap.exists() and not flag.exists()


def _get_bootstrap() -> str:
    bootstrap = AGENT_DIR / "BOOTSTRAP.md"
    if not bootstrap.exists():
        return ""
    content = bootstrap.read_text().strip()
    flag = AGENT_DIR / ".bootstrapped"
    flag.write_text("done")
    return content


def _trim_messages(messages: list) -> list:
    if len(messages) <= MAX_CONTEXT_MESSAGES:
        return messages
    if messages and messages[0].get("role") == "system":
        return [messages[0]] + messages[-(MAX_CONTEXT_MESSAGES - 1):]
    return messages[-MAX_CONTEXT_MESSAGES:]


def _load_session(session_key: str) -> list:
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
    _sessions[session_key] = messages
    safe_key = session_key.replace(":", "_")
    session_file = SESSIONS_DIR / f"{safe_key}.json"
    session_file.write_text(json.dumps(messages, default=str))


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------


class Adapter(AgentAdapter):
    """Full-featured platform adapter for Termo agents running on Sprites."""

    def __init__(self):
        self.config: dict = {}
        self.soul_md: str = "You are a helpful assistant."
        self._tools: dict = {}
        self._tool_list: list = []
        self._subtask_model = None
        self._skills_cache: list | None = None
        self._skills_cache_ts: float = 0

    async def initialize(self, config_path: str | None = None) -> None:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        (AGENT_DIR / "memory").mkdir(parents=True, exist_ok=True)
        (AGENT_DIR / "memory" / "chromadb").mkdir(parents=True, exist_ok=True)

        config_file = Path(config_path) if config_path else AGENT_DIR / "config.json"
        if config_file.exists():
            self.config = json.loads(config_file.read_text())

        soul_file = AGENT_DIR / "SOUL.md"
        if soul_file.exists():
            self.soul_md = soul_file.read_text()

        # Define all tools
        self._tools = _define_tools()
        self._tool_list = list(self._tools.values())

        # Add launch_task (needs tools + model ref)
        self._build_launch_task()

    def _get_model(self):
        from openai import AsyncOpenAI
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

        model_name = self.config.get("model", "openrouter/google/gemini-2.0-flash-001")
        api_key = self.config.get("api_key", "")
        api_base = self.config.get("api_base", "")

        if not api_base:
            raise ValueError("api_base is required in config — agent must route through the Termo proxy")

        # Strip openrouter/ prefix — proxy expects bare model names
        if model_name.startswith("openrouter/"):
            model_name = model_name[len("openrouter/"):]

        client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        return OpenAIChatCompletionsModel(model=model_name, openai_client=client)

    def _build_launch_task(self):
        from agents import Agent, Runner, function_tool

        subtask_tools = self._tool_list[:]
        self._subtask_model = self._get_model()
        subtask_model = self._subtask_model

        @function_tool
        async def launch_task(instructions: str) -> str:
            """Launch a focused subtask with its own context. Use for complex multi-step
            work that benefits from focused execution (e.g. research + summarize, build +
            test). The subtask has access to all tools (shell, files, web, memory) but
            cannot launch further subtasks. Returns the subtask result.

            instructions: detailed description of what the subtask should accomplish
            """
            api_url = os.environ.get("TERMO_API_URL", "")
            token = os.environ.get("TERMO_TOKEN", "")
            conversation_id = _current_conversation_id

            if not api_url or not token or not conversation_id:
                return "[error: missing TERMO_API_URL, TERMO_TOKEN, or conversation_id]"

            register_resp = _api_call(f"{api_url}/agents/subtask/register", {
                "token": token,
                "conversation_id": conversation_id,
                "task_name": instructions[:100],
                "task_instructions": instructions,
            })
            parent_message_id = register_resp.get("parent_message_id")
            if not parent_message_id:
                return f"[error: subtask register failed: {register_resp}]"

            from agents import ModelSettings as _MS
            from openai.types.shared import Reasoning as _R

            sub_agent = Agent(
                name="Subtask executor",
                instructions="You are executing a focused subtask. Complete the following task "
                             "and return a clear, concise summary of results.\n\n" + instructions,
                model=subtask_model,
                model_settings=_MS(reasoning=_R(effort="medium")),
                tools=subtask_tools,
            )

            call_id_to_msg: dict[str, dict] = {}
            tokens_in = 0
            tokens_out = 0

            try:
                result = Runner.run_streamed(sub_agent, input=instructions, max_turns=25)

                async for event in result.stream_events():
                    event_type = getattr(event, "type", "")

                    if event_type == "run_item_stream_event":
                        item = event.item
                        item_type = getattr(item, "type", "")

                        if item_type == "tool_call_item":
                            raw = getattr(item, "raw_item", None)
                            call_id = getattr(raw, "call_id", "") if raw else ""
                            tool_name = getattr(raw, "name", "unknown") if raw else "unknown"
                            tool_args = getattr(raw, "arguments", "") if raw else ""
                            tool_input = None
                            if tool_args:
                                try:
                                    tool_input = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                                except (json.JSONDecodeError, TypeError):
                                    tool_input = {"raw": tool_args}

                            resp = _api_call(f"{api_url}/agents/subtask/tool-start", {
                                "token": token,
                                "conversation_id": conversation_id,
                                "parent_message_id": parent_message_id,
                                "tool_name": tool_name,
                                "tool_input": tool_input,
                            })
                            msg_id = resp.get("message_id")
                            if call_id and msg_id:
                                call_id_to_msg[call_id] = {"message_id": msg_id, "started_at": _time.time()}

                        elif item_type == "tool_call_output_item":
                            output = getattr(item, "output", "") or ""
                            raw = getattr(item, "raw_item", None)
                            if isinstance(raw, dict):
                                call_id = raw.get("call_id", "")
                            else:
                                call_id = getattr(raw, "call_id", "") if raw else ""
                            info = call_id_to_msg.get(call_id)
                            if info:
                                duration_ms = int((_time.time() - info["started_at"]) * 1000)
                                _api_call(f"{api_url}/agents/subtask/tool-end", {
                                    "token": token,
                                    "message_id": info["message_id"],
                                    "tool_output": str(output)[:5000],
                                    "tool_duration_ms": duration_ms,
                                })

                    elif event_type == "raw_response_event":
                        raw = event.data
                        if hasattr(raw, "usage") and raw.usage:
                            tokens_in += getattr(raw.usage, "input_tokens", 0) or getattr(raw.usage, "prompt_tokens", 0) or 0
                            tokens_out += getattr(raw.usage, "output_tokens", 0) or getattr(raw.usage, "completion_tokens", 0) or 0

                final_output = str(result.final_output) if result.final_output else "[subtask completed with no output]"

            except Exception as e:
                final_output = f"[subtask error: {e}]"

            _api_call(f"{api_url}/agents/subtask/complete", {
                "token": token,
                "conversation_id": conversation_id,
                "parent_message_id": parent_message_id,
                "content": final_output,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "model": str(subtask_model.model) if subtask_model else "",
                "duration_ms": None,
            })

            return final_output

        self._tools["launch_task"] = launch_task
        self._tool_list = list(self._tools.values())

    def _get_installed_skills(self) -> list[dict]:
        """Fetch installed skills from the API, cached for 60s."""
        now = _time.time()
        if self._skills_cache is not None and (now - self._skills_cache_ts) < 60:
            return self._skills_cache

        api_url = os.environ.get("TERMO_API_URL", "")
        token = os.environ.get("TERMO_TOKEN", "")
        if not api_url or not token:
            return []

        try:
            payload = json.dumps({"token": token}).encode()
            req = urllib.request.Request(
                f"{api_url}/agents/list-skills",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            self._skills_cache = data.get("skills", [])
            self._skills_cache_ts = now
            return self._skills_cache
        except Exception:
            return self._skills_cache or []

    def _build_agent(self, message: str = "", session_messages: list | None = None):
        from agents import Agent

        model = self._get_model()
        now = datetime.now(timezone.utc)
        instructions = self.soul_md + f"\n\n## Current Date & Time\n{now.strftime('%A, %B %d, %Y at %H:%M UTC')}"

        public_url = self.config.get("public_url") or self.config.get("sprite_url", "")
        if public_url:
            instructions += f"\n\n## Your Public URL\n{public_url}\nFiles you save in /home/sprite/workspace are publicly accessible at {public_url}/workspace/<filename>. Use this to share files, images, and outputs in markdown links."

        # 1. Always inject identity + preference memories (core personality)
        try:
            from termo_agent.adapters.memory_engine import get_identity_and_preference_memories, recall as mem_recall
            core = get_identity_and_preference_memories()
            if core:
                core_text = "\n".join(f"- [{m['category']}] {m['content']}" for m in core)
                instructions += f"\n\n## Your Core Identity & Preferences\n{core_text}"

            # 2. Context-aware recall
            injected_ids = set()
            recall_query = message
            if message:
                recall_query = _build_recall_query(message, session_messages)
                relevant = mem_recall(recall_query, limit=5)
                relevant = [m for m in relevant if m["category"] not in ("identity", "preference") and m["similarity"] > 0.3]
                if relevant:
                    for m in relevant:
                        injected_ids.add(m["id"])
                    rel_text = "\n".join(f"- [{m['category']}] {m['content']}" for m in relevant)
                    instructions += f"\n\n## Relevant Memories\n{rel_text}"

            # 3. Category-targeted recall
            if message:
                target_cats = _detect_categories(message, session_messages)
                contextual = []
                for cat in target_cats:
                    try:
                        cat_results = mem_recall(recall_query, limit=3, category=cat)
                        for m in cat_results:
                            if m["id"] not in injected_ids and m["similarity"] > 0.25:
                                injected_ids.add(m["id"])
                                contextual.append(m)
                    except Exception:
                        pass
                if contextual:
                    ctx_text = "\n".join(f"- [{m['category']}] {m['content']}" for m in contextual)
                    instructions += f"\n\n## Contextual Knowledge\n{ctx_text}"
        except Exception:
            mem_file = AGENT_DIR / "memory" / "memory.md"
            if mem_file.exists():
                content = mem_file.read_text().strip()
                if content:
                    instructions += f"\n\n## Your Memory\n{content}"

        # 4. Inject installed skill summaries
        try:
            skills = self._get_installed_skills()
            if skills:
                skill_lines = []
                for s in skills:
                    skill_lines.append(f"- **{s.get('name', s.get('slug'))}** (`{s['slug']}`) — {s.get('description', '')}")
                skills_text = "\n".join(skill_lines)
                instructions += (
                    f"\n\n## Installed Skills\n"
                    f"These are already installed. Do NOT reinstall them. "
                    f"Use `load_skill(slug)` to load full instructions.\n"
                    f"{skills_text}"
                )
        except Exception:
            pass

        # Inject bootstrap on first-ever message
        if _needs_bootstrap():
            bootstrap = _get_bootstrap()
            if bootstrap:
                instructions += f"\n\n## FIRST RUN — Bootstrap\n{bootstrap}"

        from agents import ModelSettings
        from openai.types.shared import Reasoning

        reasoning_effort = self.config.get("reasoning_effort", "medium")
        settings = ModelSettings(
            reasoning=Reasoning(effort=reasoning_effort),
        )

        return Agent(
            name=self.config.get("persona_name", "Assistant"),
            instructions=instructions,
            model=model,
            model_settings=settings,
            tools=self._tool_list,
        )

    async def shutdown(self) -> None:
        for key, messages in _sessions.items():
            _save_session(key, messages)

    # --- Core messaging ---

    async def send_message(self, message: str, session_key: str) -> str:
        global _current_conversation_id
        from agents import Runner

        parts = session_key.split(":")
        _current_conversation_id = parts[1] if len(parts) >= 2 and parts[0] != "cron" else None

        messages = _load_session(session_key)
        agent = self._build_agent(message=message, session_messages=messages)
        messages.append({"role": "user", "content": message})

        trimmed = _trim_messages(messages)
        result = await Runner.run(agent, input=trimmed, max_turns=25)
        assistant_content = str(result.final_output) if result.final_output else ""

        messages.append({"role": "assistant", "content": assistant_content})
        _save_session(session_key, messages)

        session_tail = messages[:-2] if len(messages) > 2 else []
        asyncio.create_task(asyncio.to_thread(
            _extract_and_save_memories, message, assistant_content, session_tail, self.config
        ))

        return assistant_content

    async def send_message_stream(self, message: str, session_key: str) -> AsyncIterator[StreamEvent]:
        global _current_conversation_id
        from agents import Runner

        parts = session_key.split(":")
        _current_conversation_id = parts[1] if len(parts) >= 2 and parts[0] != "cron" else None

        messages = _load_session(session_key)
        agent = self._build_agent(message=message, session_messages=messages)
        messages.append({"role": "user", "content": message})

        trimmed = _trim_messages(messages)
        streamed_content = ""

        try:
            result = Runner.run_streamed(agent, input=trimmed, max_turns=25)

            tokens_in = 0
            tokens_out = 0
            prev_delta = None
            call_id_to_name: dict[str, str] = {}

            async for event in result.stream_events():
                event_type = getattr(event, "type", str(type(event).__name__))

                if event_type == "raw_response_event":
                    raw = event.data
                    raw_type = getattr(raw, "type", "") or ""

                    # Reasoning / thinking tokens
                    if "reasoning" in raw_type and hasattr(raw, "delta") and isinstance(raw.delta, str):
                        yield StreamEvent(type="thinking", content=raw.delta)
                    # Text content tokens
                    elif raw_type == "response.output_text.delta" and hasattr(raw, "delta") and isinstance(raw.delta, str):
                        delta = raw.delta
                        if delta and delta == prev_delta:
                            prev_delta = None
                            continue
                        if delta:
                            prev_delta = delta
                            streamed_content += delta
                            yield StreamEvent(type="token", content=delta)
                    # Usage info
                    elif hasattr(raw, "usage") and raw.usage:
                        tokens_in += getattr(raw.usage, "input_tokens", 0) or getattr(raw.usage, "prompt_tokens", 0) or 0
                        tokens_out += getattr(raw.usage, "output_tokens", 0) or getattr(raw.usage, "completion_tokens", 0) or 0

                elif event_type == "run_item_stream_event":
                    item = event.item
                    item_type = getattr(item, "type", "")
                    if item_type == "tool_call_item":
                        raw = getattr(item, "raw_item", None)
                        call_id = getattr(raw, "call_id", "") if raw else ""
                        tool_name = getattr(raw, "name", "unknown") if raw else "unknown"
                        tool_args = getattr(raw, "arguments", "") if raw else ""
                        if call_id:
                            call_id_to_name[call_id] = tool_name
                        tool_input = None
                        if tool_args:
                            try:
                                tool_input = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                            except (json.JSONDecodeError, TypeError):
                                tool_input = {"raw": tool_args}
                        yield StreamEvent(
                            type="tool_start",
                            name=tool_name,
                            metadata={"input": tool_input, "call_id": call_id},
                        )
                    elif item_type == "tool_call_output_item":
                        output = getattr(item, "output", "") or ""
                        raw = getattr(item, "raw_item", None)
                        if isinstance(raw, dict):
                            call_id = raw.get("call_id", "")
                        else:
                            call_id = getattr(raw, "call_id", "") if raw else ""
                        tool_name = call_id_to_name.get(call_id, "unknown")
                        yield StreamEvent(
                            type="tool_end",
                            name=tool_name,
                            metadata={"output": str(output)[:5000], "call_id": call_id},
                        )

            assistant_content = str(result.final_output) if result.final_output else streamed_content

            if hasattr(result, "raw_responses") and result.raw_responses:
                for resp in result.raw_responses:
                    usage = getattr(resp, "usage", None)
                    if usage:
                        tokens_in += getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0
                        tokens_out += getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0

            messages.append({"role": "assistant", "content": assistant_content})
            _save_session(session_key, messages)

            session_tail = messages[:-2] if len(messages) > 2 else []
            asyncio.create_task(asyncio.to_thread(
                _extract_and_save_memories, message, assistant_content, session_tail, self.config
            ))

            yield StreamEvent(
                type="done",
                content=assistant_content,
                usage={"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
            )

        except Exception as e:
            messages.append({"role": "assistant", "content": streamed_content or "(response interrupted)"})
            _save_session(session_key, messages)
            yield StreamEvent(type="error", content=str(e), metadata={"trace": traceback.format_exc()})

    # --- Sessions ---

    async def get_history(self, session_key: str) -> list[dict]:
        return _load_session(session_key)

    async def list_sessions(self) -> list[dict]:
        sessions = []
        for sf in SESSIONS_DIR.glob("*.json"):
            key = sf.stem.replace("_", ":", 1)
            sessions.append({"session_key": key})
        return sessions

    # --- Config ---

    async def get_config(self) -> dict:
        return self.config

    async def update_config(self, updates: dict) -> None:
        self.config.update(updates)
        (AGENT_DIR / "config.json").write_text(json.dumps(self.config, indent=2))

    # --- Memory ---

    async def get_memory(self) -> dict | None:
        try:
            from termo_agent.adapters.memory_engine import get_all_memories
            memories = get_all_memories()
            return {"memories": memories}
        except Exception:
            return None

    async def update_memory(self, content: str) -> None:
        try:
            body = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            body = {"operation": "legacy", "content": content}

        operation = body.get("operation")
        if operation == "remember":
            from termo_agent.adapters.memory_engine import remember
            remember(body["content"], body.get("category", "fact"), body.get("relationships"))
        elif operation == "forget":
            from termo_agent.adapters.memory_engine import forget
            forget(body["query"])
        elif operation == "update":
            from termo_agent.adapters.memory_engine import update_memory_entry
            update_memory_entry(body["query"], body["content"])
        else:
            from termo_agent.adapters.memory_engine import save_memory_legacy
            save_memory_legacy(body.get("content", content))

    # --- Heartbeat ---

    async def get_heartbeat(self) -> dict:
        hb_path = AGENT_DIR / "HEARTBEAT.md"
        content = hb_path.read_text() if hb_path.exists() else ""
        return {"content": content, "enabled": bool(content.strip()), "interval_s": 1800}

    async def update_heartbeat(self, content: str) -> None:
        (AGENT_DIR / "HEARTBEAT.md").write_text(content)

    # --- Tools ---

    async def list_tools(self) -> list[dict]:
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
            {"name": "create_schedule", "description": "Create a recurring scheduled task (cron)"},
            {"name": "list_schedules", "description": "List all active schedules"},
            {"name": "delete_schedule", "description": "Delete a scheduled task"},
            {"name": "launch_task", "description": "Launch a focused subtask with its own context"},
            {"name": "search_skills", "description": "Search the skill marketplace"},
            {"name": "install_skill", "description": "Install a skill into long-term memory"},
        ]

    # --- System ---

    async def health(self) -> dict:
        return {"status": "ok", "adapter": "platform"}

    async def update(self) -> dict:
        return {"status": "not supported"}

    async def restart(self) -> None:
        _sessions.clear()

    # --- Extra routes ---

    def extra_routes(self) -> list:
        return [
            ("POST", "/api/memory/search", self._handle_search_memory),
            ("GET", "/workspace/{path:.*}", self._handle_workspace_file),
        ]

    def public_route_prefixes(self) -> list[str]:
        return ["/health", "/workspace/"]

    # --- Extra route handlers ---

    async def _handle_search_memory(self, request: web.Request) -> web.Response:
        body = await request.json()
        query = body.get("query", "")
        limit = body.get("limit", 5)
        category = body.get("category")
        if not query:
            return web.json_response({"error": "query is required"}, status=400)
        from termo_agent.adapters.memory_engine import recall
        results = recall(query, limit=limit, category=category)
        return web.json_response({"memories": results})

    async def _handle_workspace_file(self, request: web.Request) -> web.Response:
        rel_path = request.match_info["path"]
        file_path = (Path("/home/sprite/workspace") / rel_path).resolve()
        if not str(file_path).startswith("/home/sprite/workspace"):
            raise web.HTTPForbidden(text="path traversal blocked")
        if not file_path.is_file():
            raise web.HTTPNotFound(text="file not found")
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        return web.FileResponse(file_path, headers={"Content-Type": content_type})
