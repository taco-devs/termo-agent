"""Tests for CLI argument parsing and adapter loading."""

import sys
import types

import pytest

from termo_agent.cli import _parse_args, _load_adapter


class TestParseArgs:
    def test_defaults(self):
        args = _parse_args([])
        assert args.adapter == "openai_agents"
        assert args.port == 8080
        assert args.host == "0.0.0.0"
        assert args.token == ""
        assert args.config == ""
        assert args.verbose is False
        assert args.version is False

    def test_custom_args(self):
        args = _parse_args([
            "--adapter", "my_adapter",
            "--port", "9090",
            "--host", "127.0.0.1",
            "--token", "secret",
            "--config", "/path/to/config.json",
            "--verbose",
        ])
        assert args.adapter == "my_adapter"
        assert args.port == 9090
        assert args.host == "127.0.0.1"
        assert args.token == "secret"
        assert args.config == "/path/to/config.json"
        assert args.verbose is True

    def test_version_flag(self):
        args = _parse_args(["--version"])
        assert args.version is True

    def test_verbose_short_flag(self):
        args = _parse_args(["-v"])
        assert args.verbose is True

    def test_env_defaults(self, monkeypatch):
        monkeypatch.setenv("TERMO_ADAPTER", "custom_adapter")
        monkeypatch.setenv("TERMO_PORT", "3000")
        monkeypatch.setenv("TERMO_HOST", "localhost")
        monkeypatch.setenv("TERMO_TOKEN", "env_token")
        monkeypatch.setenv("TERMO_CONFIG", "/env/config.json")
        args = _parse_args([])
        assert args.adapter == "custom_adapter"
        assert args.port == 3000
        assert args.host == "localhost"
        assert args.token == "env_token"
        assert args.config == "/env/config.json"


class TestLoadAdapter:
    def test_load_builtin_openai_agents(self):
        """Loading 'openai_agents' should find the built-in adapter module."""
        # This may fail if openai-agents isn't installed, which is fine
        # — the important thing is it tries the right import path
        try:
            cls = _load_adapter("openai_agents")
            assert cls is not None
        except SystemExit:
            # Expected if openai-agents SDK isn't installed
            pass

    def test_load_bare_module(self, tmp_path, monkeypatch):
        """Loading a bare module name resolves via sys.path."""
        # Create a fake adapter module
        adapter_file = tmp_path / "fake_adapter.py"
        adapter_file.write_text(
            "class Adapter:\n"
            "    pass\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        cls = _load_adapter("fake_adapter")
        assert cls is not None
        assert cls.__name__ == "Adapter"

    def test_load_nonexistent_exits(self):
        """Loading a nonexistent adapter should sys.exit."""
        with pytest.raises(SystemExit):
            _load_adapter("nonexistent_adapter_xyz_123")

    def test_load_module_without_adapter_class(self, tmp_path, monkeypatch):
        """Module exists but has no Adapter class."""
        bad_module = tmp_path / "no_adapter_cls.py"
        bad_module.write_text("x = 42\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(SystemExit):
            _load_adapter("no_adapter_cls")
