"""HTTP REST API server for termo-agent."""

import json
import logging
from importlib.metadata import version as pkg_version

from aiohttp import web

from .adapter import AgentAdapter

logger = logging.getLogger("termo_agent")


class AgentServer:
    """Framework-agnostic HTTP API server.

    Delegates every endpoint to the provided AgentAdapter.
    """

    def __init__(
        self,
        adapter: AgentAdapter,
        host: str = "0.0.0.0",
        port: int = 8080,
        token: str | None = None,
    ):
        self.adapter = adapter
        self.host = host
        self.port = port
        self.token = token
        self._runner: web.AppRunner | None = None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _check_auth(self, request: web.Request) -> bool:
        if not self.token:
            return True
        auth = request.headers.get("Authorization", "")
        return auth == f"Bearer {self.token}"

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler):
        # Check dynamic public path prefixes from adapter
        public_prefixes = self.adapter.public_route_prefixes()
        for prefix in public_prefixes:
            if request.path.startswith(prefix):
                return await handler(request)
        if not self._check_auth(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        return await handler(request)

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def _get_version(self) -> str:
        try:
            return pkg_version("termo-agent")
        except Exception:
            return "unknown"

    async def _health(self, _request: web.Request) -> web.Response:
        data = await self.adapter.health()
        data.setdefault("version", self._get_version())
        return web.json_response(data)

    async def _send(self, request: web.Request) -> web.Response:
        body = await request.json()
        message = body.get("message", "")
        session_key = body.get("session_key", "api:direct")
        stream = body.get("stream", False)

        if not message:
            return web.json_response({"error": "message is required"}, status=400)

        if stream:
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
            await response.prepare(request)

            try:
                async for event in self.adapter.send_message_stream(
                    message=message, session_key=session_key
                ):
                    payload = {"type": event.type, "content": event.content}
                    if event.name:
                        payload["name"] = event.name
                    if event.usage:
                        payload["usage"] = event.usage
                    if event.metadata:
                        payload.update(event.metadata)  # flatten to top level
                    await response.write(
                        f"data: {json.dumps(payload)}\n\n".encode()
                    )
            except Exception as exc:
                logger.exception("Stream error")
                err = json.dumps({"type": "error", "content": str(exc)})
                await response.write(f"data: {err}\n\n".encode())

            return response
        else:
            result = await self.adapter.send_message(
                message=message, session_key=session_key
            )
            return web.json_response({"response": result, "session_key": session_key})

    async def _history(self, request: web.Request) -> web.Response:
        key = request.match_info["key"]
        messages = await self.adapter.get_history(key)
        return web.json_response({"session_key": key, "messages": messages})

    async def _list_sessions(self, _request: web.Request) -> web.Response:
        sessions = await self.adapter.list_sessions()
        return web.json_response(sessions)

    async def _get_config(self, _request: web.Request) -> web.Response:
        config = await self.adapter.get_config()
        return web.json_response(config)

    async def _patch_config(self, request: web.Request) -> web.Response:
        body = await request.json()
        await self.adapter.update_config(body)
        return web.json_response({"status": "ok"})

    async def _list_tools(self, _request: web.Request) -> web.Response:
        tools = await self.adapter.list_tools()
        return web.json_response(tools)

    async def _get_memory(self, _request: web.Request) -> web.Response:
        content = await self.adapter.get_memory()
        return web.json_response({"content": content})

    async def _patch_memory(self, request: web.Request) -> web.Response:
        body = await request.json()
        content = body.get("content")
        if content is None:
            return web.json_response({"error": "content is required"}, status=400)
        await self.adapter.update_memory(content)
        return web.json_response({"status": "ok"})

    async def _get_heartbeat(self, _request: web.Request) -> web.Response:
        data = await self.adapter.get_heartbeat()
        return web.json_response(data)

    async def _patch_heartbeat(self, request: web.Request) -> web.Response:
        body = await request.json()
        content = body.get("content")
        if content is None:
            return web.json_response({"error": "content is required"}, status=400)
        await self.adapter.update_heartbeat(content)
        return web.json_response({"status": "ok"})

    async def _update(self, _request: web.Request) -> web.Response:
        result = await self.adapter.update()
        return web.json_response(result)

    async def _restart(self, _request: web.Request) -> web.Response:
        await self.adapter.restart()
        return web.json_response({"status": "restarting"})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build_app(self) -> web.Application:
        app = web.Application(middlewares=[self._auth_middleware])
        app.router.add_get("/health", self._health)
        app.router.add_post("/api/send", self._send)
        app.router.add_post("/api/sessions/send", self._send)  # backwards compat
        app.router.add_get("/api/sessions/{key}/history", self._history)
        app.router.add_get("/api/sessions", self._list_sessions)
        app.router.add_get("/api/config", self._get_config)
        app.router.add_patch("/api/config", self._patch_config)
        app.router.add_get("/api/tools", self._list_tools)
        app.router.add_get("/api/memory", self._get_memory)
        app.router.add_patch("/api/memory", self._patch_memory)
        app.router.add_get("/api/heartbeat", self._get_heartbeat)
        app.router.add_patch("/api/heartbeat", self._patch_heartbeat)
        app.router.add_post("/api/update", self._update)
        app.router.add_post("/api/restart", self._restart)

        # Register adapter-declared extra routes
        extra = self.adapter.extra_routes()
        if extra:
            for method, path, handler in extra:
                app.router.add_route(method.upper(), path, handler)

        return app

    async def start(self) -> None:
        """Start the server and block until cancelled."""
        app = self._build_app()
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info("termo-agent listening on %s:%s", self.host, self.port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            logger.info("termo-agent stopped")
