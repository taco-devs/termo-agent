"""termo-agent — Open-source serverless runtime for AI agents."""

from importlib.metadata import version as _pkg_version

from .adapter import AgentAdapter, StreamEvent

__version__ = _pkg_version("termo-agent")
__all__ = ["AgentAdapter", "StreamEvent", "__version__"]
