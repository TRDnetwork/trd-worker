"""
Config persistence for trd-worker.

Stores: api_base, worker_id, auth_token, registered_email at
~/.trd-worker/config.json (chmod 600 — token is sensitive).
"""

from __future__ import annotations
import json
import os
import stat
from pathlib import Path
from typing import Optional, TypedDict


CONFIG_DIR = Path.home() / ".trd-worker"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_API_BASE = "https://trd-cn-backend-production.up.railway.app"


class Config(TypedDict, total=False):
    api_base: str
    worker_id: str
    auth_token: str
    email: str
    hostname: str
    base_credits_per_hour: int


def load() -> Config:
    if not CONFIG_FILE.exists():
        return {}
    try:
        with CONFIG_FILE.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save(cfg: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w") as f:
        json.dump(cfg, f, indent=2)
    # 600 — owner read/write only, token is sensitive
    try:
        os.chmod(CONFIG_FILE, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass  # Windows etc.


def clear() -> None:
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()


def api_base() -> str:
    """Override via env var TRD_CN_API for local testing."""
    return os.environ.get("TRD_CN_API") or load().get("api_base") or DEFAULT_API_BASE


def get_token() -> Optional[str]:
    return load().get("auth_token")


def is_logged_in() -> bool:
    cfg = load()
    return bool(cfg.get("auth_token") and cfg.get("worker_id"))
