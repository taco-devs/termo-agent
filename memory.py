"""Simple persistent memory backed by a file on the sprite's filesystem."""

from pathlib import Path

MEMORY_DIR = Path(__file__).parent / "memory"
MEMORY_FILE = MEMORY_DIR / "memory.md"


def load_memory() -> str:
    MEMORY_DIR.mkdir(exist_ok=True)
    if MEMORY_FILE.exists():
        return MEMORY_FILE.read_text()
    return ""


def save_memory(content: str) -> None:
    MEMORY_DIR.mkdir(exist_ok=True)
    MEMORY_FILE.write_text(content)
