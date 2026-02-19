"""NanobotAdapter — connects vanilla nanobot-ai (from PyPI) to termo-agent."""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from dataclasses import asdict
from typing import AsyncIterator

from termo_agent.adapter import AgentAdapter, StreamEvent

logger = logging.getLogger("termo_agent.nanobot")

# Fields in config.json that belong to termo-agent, NOT to nanobot.
# Stripped before nanobot parses the file so upstream schema doesn't reject them.
_EXTRA_CONFIG_KEYS = {"api"}


class Adapter(AgentAdapter):
    """Adapter for the nanobot-ai agent framework (vanilla PyPI version)."""

    def __init__(self):
        self.config = None
        self.agent = None
        self.cron = None
        self.bus = None
        self.session_mgr = None
        self.heartbeat = None
        self.channels = None
        self._has_stream = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, config_path: str | None = None) -> None:
        from nanobot.config.loader import load_config, get_data_dir
        from nanobot.bus.queue import MessageBus
        from nanobot.agent.loop import AgentLoop
        from nanobot.session.manager import SessionManager
        from nanobot.cron.service import CronService
        from nanobot.cron.types import CronJob

        # Strip non-nanobot fields so vanilla nanobot can load the config
        _sanitize_config(config_path)

        self.config = load_config(config_path)
        self.bus = MessageBus()
        provider = _make_provider(self.config)
        self.session_mgr = SessionManager(self.config.workspace_path)

        cron_store = get_data_dir() / "cron" / "jobs.json"
        self.cron = CronService(cron_store)

        self.agent = AgentLoop(
            bus=self.bus,
            provider=provider,
            workspace=self.config.workspace_path,
            model=self.config.agents.defaults.model,
            temperature=self.config.agents.defaults.temperature,
            max_tokens=self.config.agents.defaults.max_tokens,
            max_iterations=self.config.agents.defaults.max_tool_iterations,
            memory_window=self.config.agents.defaults.memory_window,
            brave_api_key=self.config.tools.web.search.api_key or None,
            exec_config=self.config.tools.exec,
            cron_service=self.cron,
            restrict_to_workspace=self.config.tools.restrict_to_workspace,
            session_manager=self.session_mgr,
            mcp_servers=self.config.tools.mcp_servers,
        )

        # Detect streaming support
        self._has_stream = hasattr(self.agent, "process_direct_stream")

        # Wire cron callback
        async def _on_cron_job(job: CronJob) -> str | None:
            response = await self.agent.process_direct(
                job.payload.message,
                session_key=f"cron:{job.id}",
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to or "direct",
            )
            if job.payload.deliver and job.payload.to and response:
                await _deliver_cron_response(job, response)
            return response

        self.cron.on_job = _on_cron_job

        # Heartbeat (optional — may not exist in all versions)
        try:
            from nanobot.heartbeat.service import HeartbeatService

            async def _on_heartbeat(prompt: str) -> str:
                return await self.agent.process_direct(prompt, session_key="heartbeat")

            self.heartbeat = HeartbeatService(
                workspace=self.config.workspace_path,
                on_heartbeat=_on_heartbeat,
                interval_s=30 * 60,
                enabled=True,
            )
        except ImportError:
            logger.info("HeartbeatService not available, skipping")

        # Channels (optional)
        try:
            from nanobot.channels.manager import ChannelManager
            self.channels = ChannelManager(self.config, self.bus)
        except ImportError:
            logger.info("ChannelManager not available, skipping")

        # Intercept outbound bus messages for the "api" channel.
        # The nanobot `message` tool publishes to bus.publish_outbound().
        # ChannelManager handles Telegram/Discord/etc. but has no "api" handler,
        # so API-channel messages would be silently dropped.  This intercept
        # catches them BEFORE they hit the queue and writes to Supabase.
        _original_publish = self.bus.publish_outbound

        async def _intercepted_publish(msg) -> None:
            if msg.channel == "api" and msg.chat_id:
                parts = msg.chat_id.split(":", 1)
                if len(parts) == 2:
                    await _deliver_to_supabase(parts[1], msg.content)
                else:
                    logger.warning(f"Outbound API message dropped: bad chat_id format: {msg.chat_id}")
            await _original_publish(msg)

        self.bus.publish_outbound = _intercepted_publish

        # Start background services
        await self.cron.start()
        if self.heartbeat:
            await self.heartbeat.start()
        asyncio.create_task(self.agent.run())
        if self.channels:
            asyncio.create_task(self.channels.start_all())

        logger.info(
            "Nanobot initialized  model=%s  workspace=%s  stream=%s",
            self.config.agents.defaults.model,
            self.config.workspace_path,
            self._has_stream,
        )

    async def shutdown(self) -> None:
        if self.agent:
            await self.agent.close_mcp()
            self.agent.stop()
        if self.heartbeat:
            self.heartbeat.stop()
        if self.cron:
            self.cron.stop()
        if self.channels:
            await self.channels.stop_all()
        logger.info("Nanobot shut down")

    # ------------------------------------------------------------------
    # Core messaging
    # ------------------------------------------------------------------

    async def send_message(self, message: str, session_key: str) -> str:
        return await self.agent.process_direct(
            content=message,
            session_key=session_key,
            channel="api",
            chat_id=session_key,
        )

    async def send_message_stream(
        self, message: str, session_key: str
    ) -> AsyncIterator[StreamEvent]:
        if self._has_stream:
            # Native streaming support
            async for raw in self.agent.process_direct_stream(
                content=message,
                session_key=session_key,
                channel="api",
                chat_id=session_key,
            ):
                try:
                    data = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                yield StreamEvent(
                    type=data.get("type", "token"),
                    content=data.get("content", ""),
                    name=data.get("name", ""),
                    usage=data.get("usage", {}),
                )
        else:
            # Fallback: run sync and emit done event
            result = await self.send_message(message, session_key)
            yield StreamEvent(type="token", content=result)
            yield StreamEvent(type="done", content=result)

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def get_history(self, session_key: str) -> list[dict]:
        session = self.agent.sessions.get_or_create(session_key)
        return session.messages

    async def list_sessions(self) -> list[dict]:
        return self.agent.sessions.list_sessions()

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[dict]:
        tools = []
        for name, tool in self.agent.tools._tools.items():
            tools.append({
                "name": name,
                "description": getattr(tool, "description", ""),
            })
        return tools

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    async def get_config(self) -> dict:
        return {
            "model": self.config.agents.defaults.model,
            "temperature": self.config.agents.defaults.temperature,
            "max_tokens": self.config.agents.defaults.max_tokens,
            "max_tool_iterations": self.config.agents.defaults.max_tool_iterations,
        }

    async def update_config(self, updates: dict) -> None:
        defaults = self.config.agents.defaults
        if "model" in updates:
            defaults.model = updates["model"]
        if "temperature" in updates:
            defaults.temperature = float(updates["temperature"])
        if "maxTokens" in updates or "max_tokens" in updates:
            defaults.max_tokens = int(updates.get("maxTokens") or updates.get("max_tokens"))

        if "channels" in updates:
            channels_data = updates["channels"]
            channels_cfg = self.config.channels
            for ch_name, ch_update in channels_data.items():
                if not isinstance(ch_update, dict):
                    continue
                ch_obj = getattr(channels_cfg, ch_name, None)
                if ch_obj is None:
                    continue
                for key, value in ch_update.items():
                    snake_key = key
                    for attr in (
                        "allow_from", "bridge_url", "bridge_token",
                        "app_id", "app_secret", "encrypt_key",
                        "verification_token",
                    ):
                        if key == attr.replace("_", "") or key == attr:
                            snake_key = attr
                            break
                    if hasattr(ch_obj, snake_key):
                        setattr(ch_obj, snake_key, value)

        from nanobot.config.loader import save_config
        save_config(self.config)

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    async def get_memory(self) -> str | None:
        from nanobot.agent.memory import MemoryStore
        memory = MemoryStore(self.agent.workspace)
        return memory.read_long_term()

    async def update_memory(self, content: str) -> None:
        from nanobot.agent.memory import MemoryStore
        memory = MemoryStore(self.agent.workspace)
        memory.write_long_term(content)

    # ------------------------------------------------------------------
    # Cron
    # ------------------------------------------------------------------

    async def list_crons(self) -> list[dict]:
        jobs = self.cron.list_jobs(include_disabled=True)
        result = []
        for j in jobs:
            # Flatten schedule to a human-readable string for the frontend
            sched = j.schedule
            if sched.kind == "cron" and sched.expr:
                schedule_str = sched.expr
            elif sched.kind == "every" and sched.every_ms:
                mins = sched.every_ms / 60_000
                schedule_str = f"every {int(mins)}m" if mins == int(mins) else f"every {mins:.1f}m"
            elif sched.kind == "at" and sched.at_ms:
                schedule_str = f"once at {sched.at_ms}"
            else:
                schedule_str = str(asdict(sched))

            result.append({
                "id": j.id,
                "name": j.name,
                "enabled": j.enabled,
                "schedule": schedule_str,
                "message": j.payload.message,
                "deliver": j.payload.deliver,
                "channel": j.payload.channel,
                "to": j.payload.to,
                "last_status": j.state.last_status,
                "next_run_at_ms": j.state.next_run_at_ms,
            })
        return result

    async def add_cron(self, spec: dict) -> dict:
        from nanobot.cron.types import CronSchedule
        sched_raw = spec.get("schedule", {})

        # Frontend sends schedule as a plain string (cron expr or "every Xm")
        if isinstance(sched_raw, str):
            sched_raw = sched_raw.strip()
            schedule = _parse_schedule_string(sched_raw)
        else:
            # Structured dict from API clients
            schedule = CronSchedule(
                kind=sched_raw.get("kind", "every"),
                at_ms=sched_raw.get("at_ms") or sched_raw.get("atMs"),
                every_ms=sched_raw.get("every_ms") or sched_raw.get("everyMs"),
                expr=sched_raw.get("expr"),
                tz=sched_raw.get("tz"),
            )

        job = self.cron.add_job(
            name=spec.get("name", "unnamed"),
            schedule=schedule,
            message=spec.get("message", ""),
            deliver=spec.get("deliver", False),
            channel=spec.get("channel"),
            to=spec.get("to"),
        )
        return {"id": job.id, "name": job.name, "schedule": sched_raw}

    async def delete_cron(self, job_id: str) -> bool:
        return self.cron.remove_job(job_id)

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------

    async def health(self) -> dict:
        return {"status": "ok"}

    async def update(self) -> dict:
        """Pull latest Docker image and restart.

        When running in Docker (systemd restarts the container), SIGTERM
        causes the container to exit; systemd then does ``docker run``
        again which uses the freshly-pulled image.  Falls back to pip
        upgrade when not running in Docker.
        """
        logger.info("Update requested")
        try:
            # Try docker pull first (works when host has Docker)
            pull = await asyncio.to_thread(
                subprocess.run,
                ["docker", "pull", "registry.digitalocean.com/termo/termo-agent:latest"],
                capture_output=True, text=True, timeout=120,
            )
            if pull.returncode == 0:
                method = "docker"
                output = pull.stdout.strip()
            else:
                # Fallback: pip upgrade (bare-metal / dev installs)
                pip = await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, "-m", "pip", "install", "--upgrade", "nanobot-ai", "termo-agent"],
                    capture_output=True, text=True, timeout=120,
                )
                method = "pip"
                output = "ok" if pip.returncode == 0 else (pip.stdout.strip() or pip.stderr.strip())

            # Schedule restart after response is sent
            async def _delayed_restart():
                await asyncio.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

            asyncio.get_event_loop().create_task(_delayed_restart())

            return {"status": "updating", "method": method, "output": output}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def restart(self) -> None:
        logger.info("Restart requested")
        os.kill(os.getpid(), signal.SIGTERM)


# ------------------------------------------------------------------
# Helpers (outside the class)
# ------------------------------------------------------------------

def _sanitize_config(config_path: str | None) -> None:
    """Strip non-nanobot fields from config.json before nanobot loads it.

    termo-agent owns the API layer (port, token, etc.) — those don't belong
    in the framework config.  If a config contains them, remove them so
    vanilla nanobot's strict schema doesn't reject the whole file.
    """
    from pathlib import Path

    if config_path:
        p = Path(config_path)
    else:
        p = Path.home() / ".nanobot" / "config.json"

    if not p.exists():
        return

    try:
        raw = json.loads(p.read_text())
    except Exception:
        return

    removed = {k for k in _EXTRA_CONFIG_KEYS if k in raw}
    if not removed:
        return

    for k in removed:
        del raw[k]
    p.write_text(json.dumps(raw, indent=2))
    logger.info("Stripped non-nanobot fields from config: %s", removed)


def _parse_schedule_string(s: str):
    """Parse a human-friendly schedule string into a CronSchedule.

    Accepts:
      - Cron expressions: "0 9 * * *", "*/5 * * * *"
      - Interval shorthand: "every 5m", "every 1h", "every 30s"
      - Raw milliseconds: "every 60000ms"
    Falls back to treating the string as a cron expression.
    """
    from nanobot.cron.types import CronSchedule
    import re

    low = s.lower()

    # "every Xm", "every Xh", "every Xs", "every Xms"
    m = re.match(r"every\s+([\d.]+)\s*(ms|s|m|h)", low)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        multipliers = {"ms": 1, "s": 1000, "m": 60_000, "h": 3_600_000}
        every_ms = int(val * multipliers[unit])
        return CronSchedule(kind="every", every_ms=every_ms)

    # Plain number → treat as minutes
    if re.match(r"^every\s+\d+$", low):
        mins = int(low.split()[-1])
        return CronSchedule(kind="every", every_ms=mins * 60_000)

    # Everything else → cron expression
    return CronSchedule(kind="cron", expr=s)


def _make_provider(config):
    """Create the appropriate LLM provider from config."""
    from nanobot.providers.litellm_provider import LiteLLMProvider
    from nanobot.providers.openai_codex_provider import OpenAICodexProvider
    from nanobot.providers.custom_provider import CustomProvider

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)

    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        return OpenAICodexProvider(default_model=model)

    if provider_name == "custom":
        return CustomProvider(
            api_key=p.api_key if p else "no-key",
            api_base=config.get_api_base(model) or "http://localhost:8000/v1",
            default_model=model,
        )

    from nanobot.providers.registry import find_by_name
    spec = find_by_name(provider_name)
    if not model.startswith("bedrock/") and not (p and p.api_key) and not (spec and spec.is_oauth):
        raise RuntimeError(
            f"No API key configured for provider '{provider_name}'. "
            "Set one in config.json under providers section."
        )

    return LiteLLMProvider(
        api_key=p.api_key if p else None,
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=p.extra_headers if p else None,
        provider_name=provider_name,
    )


async def _deliver_to_supabase(conversation_id: str, content: str) -> None:
    """Write a message directly into Supabase as an assistant message.

    Used by the bus interceptor (for the `message` tool) and cron delivery.
    """
    import aiohttp

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        logger.warning("Delivery skipped: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set")
        return

    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{supabase_url}/rest/v1/messages",
                headers=headers,
                json={
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": content,
                },
            ) as resp:
                if resp.status >= 300:
                    body = await resp.text()
                    logger.error(f"Supabase delivery failed: {resp.status} {body[:200]}")
                else:
                    logger.info(f"Delivered message to conversation {conversation_id}")

            async with session.patch(
                f"{supabase_url}/rest/v1/conversations?id=eq.{conversation_id}",
                headers=headers,
                json={"last_at": "now()"},
            ) as resp:
                pass  # Non-fatal
    except Exception as e:
        logger.error(f"Supabase delivery error: {e}")


async def _deliver_cron_response(job, response: str) -> None:
    """Deliver a cron job response. The `to` field is "user_id:conversation_id"."""
    to = job.payload.to or ""
    parts = to.split(":", 1)
    if len(parts) != 2:
        logger.warning(f"Cron delivery skipped: invalid 'to' format: {to}")
        return
    await _deliver_to_supabase(parts[1], response)
