"""NanobotAdapter — connects nanobot-ai to termo-agent."""

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


class Adapter(AgentAdapter):
    """Adapter for the nanobot-ai agent framework.

    Replicates the boot sequence from ``nanobot.cli.commands:gateway()`` and
    maps every endpoint to the corresponding nanobot internal.
    """

    def __init__(self):
        self.config = None
        self.agent = None
        self.cron = None
        self.bus = None
        self.session_mgr = None
        self.heartbeat = None
        self.channels = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, config_path: str | None = None) -> None:
        from nanobot.config.loader import load_config, get_data_dir
        from nanobot.bus.queue import MessageBus
        from nanobot.agent.loop import AgentLoop
        from nanobot.channels.manager import ChannelManager
        from nanobot.session.manager import SessionManager
        from nanobot.cron.service import CronService
        from nanobot.cron.types import CronJob
        from nanobot.heartbeat.service import HeartbeatService

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

        # Wire cron callback
        async def _on_cron_job(job: CronJob) -> str | None:
            response = await self.agent.process_direct(
                job.payload.message,
                session_key=f"cron:{job.id}",
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to or "direct",
            )
            if job.payload.deliver and job.payload.to:
                from nanobot.bus.events import OutboundMessage
                await self.bus.publish_outbound(OutboundMessage(
                    channel=job.payload.channel or "cli",
                    chat_id=job.payload.to,
                    content=response or "",
                ))
            return response

        self.cron.on_job = _on_cron_job

        # Heartbeat
        async def _on_heartbeat(prompt: str) -> str:
            return await self.agent.process_direct(prompt, session_key="heartbeat")

        self.heartbeat = HeartbeatService(
            workspace=self.config.workspace_path,
            on_heartbeat=_on_heartbeat,
            interval_s=30 * 60,
            enabled=True,
        )

        # Channels
        self.channels = ChannelManager(self.config, self.bus)

        # Start background services
        await self.cron.start()
        await self.heartbeat.start()
        asyncio.create_task(self.agent.run())
        asyncio.create_task(self.channels.start_all())

        logger.info(
            "Nanobot initialized  model=%s  workspace=%s",
            self.config.agents.defaults.model,
            self.config.workspace_path,
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
            result.append({
                "id": j.id,
                "name": j.name,
                "enabled": j.enabled,
                "schedule": asdict(j.schedule),
                "payload": asdict(j.payload),
                "state": asdict(j.state),
            })
        return result

    async def add_cron(self, spec: dict) -> dict:
        from nanobot.cron.types import CronSchedule
        sched_data = spec.get("schedule", {})
        schedule = CronSchedule(
            kind=sched_data.get("kind", "every"),
            at_ms=sched_data.get("at_ms") or sched_data.get("atMs"),
            every_ms=sched_data.get("every_ms") or sched_data.get("everyMs"),
            expr=sched_data.get("expr"),
            tz=sched_data.get("tz"),
        )
        job = self.cron.add_job(
            name=spec.get("name", "unnamed"),
            schedule=schedule,
            message=spec.get("message", ""),
            deliver=spec.get("deliver", False),
            channel=spec.get("channel"),
            to=spec.get("to"),
        )
        return {"id": job.id, "name": job.name}

    async def delete_cron(self, job_id: str) -> bool:
        return self.cron.remove_job(job_id)

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------

    async def health(self) -> dict:
        return {"status": "ok"}

    async def update(self) -> dict:
        logger.info("Update requested")
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-m", "pip", "install", "--upgrade", "nanobot-ai", "termo-agent"],
                capture_output=True, text=True, timeout=120,
            )
            pip_output = result.stdout.strip() or result.stderr.strip()

            # Schedule restart after response is sent
            async def _delayed_restart():
                await asyncio.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

            asyncio.get_event_loop().create_task(_delayed_restart())

            return {
                "status": "updating",
                "pip": "ok" if result.returncode == 0 else pip_output,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def restart(self) -> None:
        logger.info("Restart requested")
        os.kill(os.getpid(), signal.SIGTERM)


# ------------------------------------------------------------------
# Provider factory (ported from nanobot.cli.commands._make_provider)
# ------------------------------------------------------------------

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
