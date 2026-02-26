"""Tests for the HTTP REST API server."""

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

from termo_agent.server import AgentServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _make_client(adapter, token=None) -> TestClient:
    """Create an aiohttp TestClient from a stub adapter."""
    server = AgentServer(adapter=adapter, host="127.0.0.1", port=0, token=token)
    app = server._build_app()
    return TestClient(TestServer(app))


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["adapter"] == "stub"
            assert "version" in data

    @pytest.mark.asyncio
    async def test_health_is_public(self, stub_adapter):
        """Health endpoint should work even with auth enabled."""
        async with await _make_client(stub_adapter, token="secret") as client:
            resp = await client.get("/health")
            assert resp.status == 200


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuth:
    @pytest.mark.asyncio
    async def test_no_auth_required_when_no_token(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.get("/api/config")
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_auth_required_with_token(self, stub_adapter):
        async with await _make_client(stub_adapter, token="secret") as client:
            resp = await client.get("/api/config")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_auth_passes_with_correct_token(self, stub_adapter):
        async with await _make_client(stub_adapter, token="secret") as client:
            resp = await client.get("/api/config", headers=_auth_headers("secret"))
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_auth_fails_with_wrong_token(self, stub_adapter):
        async with await _make_client(stub_adapter, token="secret") as client:
            resp = await client.get("/api/config", headers=_auth_headers("wrong"))
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_custom_public_prefixes(self, stub_adapter):
        stub_adapter._public_prefixes = ["/health", "/api/memory"]
        async with await _make_client(stub_adapter, token="secret") as client:
            # Memory should be public
            resp = await client.get("/api/memory")
            assert resp.status == 200
            # Config should still require auth
            resp = await client.get("/api/config")
            assert resp.status == 401


# ---------------------------------------------------------------------------
# Send message (non-streaming)
# ---------------------------------------------------------------------------

class TestSendMessage:
    @pytest.mark.asyncio
    async def test_send_non_streaming(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.post(
                "/api/send",
                json={"message": "hello", "session_key": "test:1"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["response"] == "Echo: hello"
            assert data["session_key"] == "test:1"

    @pytest.mark.asyncio
    async def test_send_empty_message_rejected(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.post("/api/send", json={"message": ""})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_send_default_session_key(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.post("/api/send", json={"message": "hi"})
            assert resp.status == 200
            data = await resp.json()
            assert data["session_key"] == "api:direct"

    @pytest.mark.asyncio
    async def test_backwards_compat_endpoint(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.post(
                "/api/sessions/send",
                json={"message": "hello"},
            )
            assert resp.status == 200


# ---------------------------------------------------------------------------
# Send message (streaming / SSE)
# ---------------------------------------------------------------------------

class TestStreaming:
    @pytest.mark.asyncio
    async def test_stream_returns_sse(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.post(
                "/api/send",
                json={"message": "hello world", "stream": True},
            )
            assert resp.status == 200
            assert resp.content_type == "text/event-stream"

            body = await resp.text()
            events = [
                json.loads(line[6:])
                for line in body.strip().split("\n")
                if line.startswith("data: ")
            ]

            # Should have token events + done
            types = [e["type"] for e in events]
            assert "token" in types
            assert "done" in types

            done = [e for e in events if e["type"] == "done"][0]
            assert done["content"] == "hello world"
            assert "usage" in done

    @pytest.mark.asyncio
    async def test_stream_metadata_flattened(self, tool_adapter):
        """Tool metadata (call_id, input, output) should be flattened to top-level SSE JSON."""
        async with await _make_client(tool_adapter) as client:
            resp = await client.post(
                "/api/send",
                json={"message": "search test", "stream": True},
            )
            body = await resp.text()
            events = [
                json.loads(line[6:])
                for line in body.strip().split("\n")
                if line.startswith("data: ")
            ]

            tool_start = [e for e in events if e["type"] == "tool_start"][0]
            assert tool_start["name"] == "web_search"
            assert tool_start["call_id"] == "call_123"
            assert tool_start["input"] == {"query": "test"}

            tool_end = [e for e in events if e["type"] == "tool_end"][0]
            assert tool_end["call_id"] == "call_123"
            assert tool_end["output"] == "result data"


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class TestSessions:
    @pytest.mark.asyncio
    async def test_history(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            # Send a message first
            await client.post(
                "/api/send",
                json={"message": "hi", "session_key": "user:conv1"},
            )
            resp = await client.get("/api/sessions/user:conv1/history")
            assert resp.status == 200
            data = await resp.json()
            assert data["session_key"] == "user:conv1"
            assert len(data["messages"]) == 2

    @pytest.mark.asyncio
    async def test_list_sessions(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            await client.post("/api/send", json={"message": "a", "session_key": "s1"})
            await client.post("/api/send", json={"message": "b", "session_key": "s2"})
            resp = await client.get("/api/sessions")
            assert resp.status == 200
            data = await resp.json()
            keys = [s["session_key"] for s in data]
            assert "s1" in keys
            assert "s2" in keys


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    @pytest.mark.asyncio
    async def test_get_config(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.get("/api/config")
            data = await resp.json()
            assert data["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_patch_config(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.patch(
                "/api/config",
                json={"model": "new-model", "temperature": 0.5},
            )
            assert resp.status == 200
            resp = await client.get("/api/config")
            data = await resp.json()
            assert data["model"] == "new-model"
            assert data["temperature"] == 0.5


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class TestTools:
    @pytest.mark.asyncio
    async def test_list_tools(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.get("/api/tools")
            data = await resp.json()
            assert len(data) == 1
            assert data[0]["name"] == "test_tool"


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestMemory:
    @pytest.mark.asyncio
    async def test_get_empty_memory(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.get("/api/memory")
            data = await resp.json()
            assert data["content"] is None

    @pytest.mark.asyncio
    async def test_patch_and_get_memory(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.patch("/api/memory", json={"content": "I like cats"})
            assert resp.status == 200
            resp = await client.get("/api/memory")
            data = await resp.json()
            assert data["content"] == "I like cats"

    @pytest.mark.asyncio
    async def test_patch_memory_requires_content(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.patch("/api/memory", json={})
            assert resp.status == 400


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_get_heartbeat(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.get("/api/heartbeat")
            data = await resp.json()
            assert data["content"] == ""
            assert data["enabled"] is False

    @pytest.mark.asyncio
    async def test_patch_and_get_heartbeat(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            await client.patch("/api/heartbeat", json={"content": "Check news"})
            resp = await client.get("/api/heartbeat")
            data = await resp.json()
            assert data["content"] == "Check news"
            assert data["enabled"] is True

    @pytest.mark.asyncio
    async def test_patch_heartbeat_requires_content(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.patch("/api/heartbeat", json={})
            assert resp.status == 400


# ---------------------------------------------------------------------------
# Update / Restart
# ---------------------------------------------------------------------------

class TestLifecycle:
    @pytest.mark.asyncio
    async def test_update(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.post("/api/update")
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_restart(self, stub_adapter):
        async with await _make_client(stub_adapter) as client:
            resp = await client.post("/api/restart")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "restarting"


# ---------------------------------------------------------------------------
# Extra routes
# ---------------------------------------------------------------------------

class TestExtraRoutes:
    @pytest.mark.asyncio
    async def test_extra_routes_registered(self, stub_adapter):
        async def custom_handler(request):
            return web.json_response({"custom": True})

        stub_adapter._extra = [("GET", "/api/custom", custom_handler)]

        async with await _make_client(stub_adapter) as client:
            resp = await client.get("/api/custom")
            assert resp.status == 200
            data = await resp.json()
            assert data["custom"] is True
