"""CLI entry point: termo-agent --adapter openai_agents --port 8080."""

import argparse
import asyncio
import logging
import os
import signal
import sys

logger = logging.getLogger("termo_agent")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="termo-agent",
        description="Universal HTTP API gateway for AI agents",
    )
    parser.add_argument(
        "--adapter",
        default=os.environ.get("TERMO_ADAPTER", "openai_agents"),
        help="Adapter name (default: openai_agents). Looks for termo_agent.adapters.<name>.Adapter",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("TERMO_PORT", "8080")),
        help="HTTP port (default: 8080)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("TERMO_HOST", "0.0.0.0"),
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("TERMO_TOKEN", ""),
        help="Bearer auth token (default: none)",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("TERMO_CONFIG", ""),
        help="Path to framework config file (passed to adapter.initialize())",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )
    return parser.parse_args(argv)


def _load_adapter(name: str):
    """Import and return the Adapter class from termo_agent.adapters.<name>."""
    module_path = f"termo_agent.adapters.{name}"
    try:
        import importlib
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        print(f"Error: Could not load adapter '{name}' ({module_path}): {exc}", file=sys.stderr)
        sys.exit(1)

    adapter_cls = getattr(mod, "Adapter", None)
    if adapter_cls is None:
        print(
            f"Error: Module {module_path} does not export an 'Adapter' class",
            file=sys.stderr,
        )
        sys.exit(1)
    return adapter_cls


async def _run(args: argparse.Namespace) -> None:
    from .server import AgentServer

    adapter_cls = _load_adapter(args.adapter)
    adapter = adapter_cls()

    config_path = args.config or None
    logger.info("Initializing adapter '%s'...", args.adapter)
    await adapter.initialize(config_path=config_path)

    server = AgentServer(
        adapter=adapter,
        host=args.host,
        port=args.port,
        token=args.token or None,
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await server.start()
    logger.info(
        "termo-agent ready  adapter=%s  port=%s  auth=%s",
        args.adapter,
        args.port,
        "on" if args.token else "off",
    )

    await stop_event.wait()

    logger.info("Shutting down...")
    await server.stop()
    await adapter.shutdown()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.version:
        from importlib.metadata import version as pkg_version
        try:
            v = pkg_version("termo-agent")
        except Exception:
            v = "dev"
        print(f"termo-agent {v}")
        return

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
