"""Load agent config from config.json."""

import json
from pathlib import Path

AGENT_DIR = Path(__file__).parent
CONFIG_PATH = AGENT_DIR / "config.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def save_config(config: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
