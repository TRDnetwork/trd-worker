"""
Worker settings enforcement (B, May 9 2026).

Pulls user preferences from compute_worker_settings via the backend's
GET /api/compute/worker-settings endpoint and decides per-poll whether
the daemon should accept jobs.

Settings honored:
  - pause_when_on_battery: skip jobs when on battery power
  - schedule_enabled + schedule_start_hour + schedule_end_hour: only accept
    within the user's local time window
  - allowed_models: subset of supported models the user wants this worker
    to accept jobs for (None/empty = all supported models)
  - max_gpu_utilization_pct: throttle hint (informational; daemon doesn't
    enforce hard cap, but reports current util in heartbeat)

Settings refresh every SETTINGS_REFRESH_SEC (default 300s = 5 min). Failures
fall back to last known settings (or defaults if never fetched). The daemon
NEVER blocks jobs because settings couldn't be fetched — that would be a
denial-of-earnings bug.

After first successful fetch, daemon also POSTs daemon_version to
/worker-settings/ack so the dashboard can show "Settings active on worker
v0.2.0" instead of "Settings saved but worker hasn't picked them up yet".
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import requests

from . import api, config, power


SETTINGS_REFRESH_SEC = 300  # poll backend every 5 min


@dataclass
class WorkerSettings:
    max_gpu_utilization_pct: int = 90
    schedule_enabled: bool = False
    schedule_start_hour: Optional[int] = None
    schedule_end_hour: Optional[int] = None
    allowed_models: Optional[list[str]] = None  # None = all
    pause_when_on_battery: bool = True
    daemon_enforced_version: Optional[str] = None
    fetched_at: float = field(default_factory=time.monotonic)

    @classmethod
    def defaults(cls) -> "WorkerSettings":
        return cls()

    @classmethod
    def from_dict(cls, d: dict) -> "WorkerSettings":
        # Defensive: server may add fields. Only extract the ones we use.
        return cls(
            max_gpu_utilization_pct=int(d.get("max_gpu_utilization_pct") or 90),
            schedule_enabled=bool(d.get("schedule_enabled") or False),
            schedule_start_hour=_int_or_none(d.get("schedule_start_hour")),
            schedule_end_hour=_int_or_none(d.get("schedule_end_hour")),
            allowed_models=_list_or_none(d.get("allowed_models")),
            pause_when_on_battery=bool(d.get("pause_when_on_battery") if d.get("pause_when_on_battery") is not None else True),
            daemon_enforced_version=d.get("daemon_enforced_version"),
        )


def _int_or_none(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _list_or_none(v) -> Optional[list[str]]:
    if v is None:
        return None
    if isinstance(v, list):
        # Empty list = "explicitly no models allowed". Treat as None to mean
        # "no restriction" — the user almost never wants to permanently
        # disallow ALL models. Use schedule or pause-on-battery for that.
        return v if len(v) > 0 else None
    return None


# ── module-level cache ──────────────────────────────────────────────────────
_cache: WorkerSettings = WorkerSettings.defaults()
_last_fetch_attempt: float = 0.0
_first_fetch_done: bool = False


def get() -> WorkerSettings:
    """
    Returns the cached settings. Daemon should call this on every poll.
    Refresh happens in the background via maybe_refresh().
    """
    return _cache


def maybe_refresh(token: str, daemon_version: str) -> bool:
    """
    Refresh settings from the backend if SETTINGS_REFRESH_SEC has passed
    since the last attempt. Called from the daemon's main loop. Returns
    True if a refresh actually fired (success or failure), False if skipped.

    On first successful fetch, also POSTs an ack with daemon_version so
    the UI can show which version is active.

    First call always runs (regardless of timer) so the daemon picks up
    settings + acks version immediately on startup.
    """
    global _cache, _last_fetch_attempt, _first_fetch_done
    now = time.monotonic()
    # First call always runs (last_fetch_attempt == 0.0). Subsequent calls
    # only run if SETTINGS_REFRESH_SEC has elapsed.
    if _last_fetch_attempt > 0.0 and now - _last_fetch_attempt < SETTINGS_REFRESH_SEC:
        return False
    _last_fetch_attempt = now

    base = config.api_base()
    url = f"{base}/api/compute/worker-settings"
    try:
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
    except requests.RequestException:
        return True  # attempt counted; cache stays as-is
    if r.status_code != 200:
        return True

    try:
        data = r.json()
    except ValueError:
        return True

    _cache = WorkerSettings.from_dict(data)
    _maybe_announce_update(data)

    # On first successful fetch, ack the daemon version so the UI can stop
    # showing the "settings not yet active" warning.
    if not _first_fetch_done:
        _first_fetch_done = True
        try:
            requests.post(
                f"{base}/api/compute/worker-settings/ack",
                headers={"Authorization": f"Bearer {token}"},
                json={"daemon_version": daemon_version},
                timeout=5,
            )
        except requests.RequestException:
            pass  # next refresh will retry

    return True


# ── enforcement decisions ───────────────────────────────────────────────────

def _in_schedule_window(s: WorkerSettings, now: Optional[datetime] = None) -> bool:
    """
    True if the current local time falls within [schedule_start_hour,
    schedule_end_hour). Wraps midnight if start > end (e.g. start=22, end=6
    means 22:00–06:00 next morning).
    """
    if not s.schedule_enabled:
        return True
    if s.schedule_start_hour is None or s.schedule_end_hour is None:
        return True  # incomplete config = no enforcement
    h = (now or datetime.now()).hour
    start = s.schedule_start_hour % 24
    end = s.schedule_end_hour % 24
    if start == end:
        return True  # 24h window
    if start < end:
        return start <= h < end
    # Wraps midnight: e.g. 22..6 → [22,23] OR [0,5]
    return h >= start or h < end


def _pause_for_battery(s: WorkerSettings) -> bool:
    """True if we should pause due to battery state."""
    if not s.pause_when_on_battery:
        return False
    state = power.is_on_battery()
    # state == True → on battery → pause
    # state == False → on AC → don't pause
    # state == None → indeterminate (no battery, e.g. desktop) → don't pause
    return state is True


def should_accept_jobs() -> tuple[bool, Optional[str]]:
    """
    Returns (accept, reason).
      accept=True  → daemon should poll for jobs as normal
      accept=False → daemon should skip this poll cycle and send an idle
                     heartbeat with status='paused' (display-only state).

    `reason` is a short human-readable string for logging when accept=False,
    or None when accept=True.
    """
    s = _cache
    if _pause_for_battery(s):
        return False, "on battery (pause_when_on_battery=true)"
    if not _in_schedule_window(s):
        return False, f"outside schedule window {s.schedule_start_hour}-{s.schedule_end_hour}"
    return True, None


def is_model_allowed(model_name: str) -> bool:
    """
    Daemon checks this before accepting a job (defensive — backend already
    filters by supported_models, but allowed_models is a user-tightened
    subset so we re-check client-side too).
    """
    s = _cache
    if s.allowed_models is None:  # no restriction
        return True
    return model_name in s.allowed_models


# ── Auto-update notice handler (added in v0.2.1) ────────────────────────────
_update_announced = False


def _maybe_announce_update(settings_dict: dict) -> None:
    """
    Check /worker-settings response for an update_available block. Backend
    injects this when LATEST_WORKER_VERSION > daemon_enforced_version. We
    print a one-time banner to stderr so users see they should upgrade.

    Optional auto-exit: set TRD_WORKER_AUTO_EXIT_ON_OUTDATED=1 to cause
    the daemon to exit with code 2 when a warn-severity update is
    available. Useful for systemd/launchd users who want auto-restart
    after pip install -U trd-worker.
    """
    global _update_announced
    notice = settings_dict.get("update_available")
    if not notice or not notice.get("available"):
        return
    if _update_announced:
        return

    import sys
    severity = notice.get("severity", "info")
    cmd = notice.get("command", "pip install -U trd-worker")
    latest = notice.get("latest_version", "?")
    current = notice.get("current_version") or "unknown"

    prefix = "WARN" if severity == "warn" else "INFO"
    print(file=sys.stderr)
    print(
        f"[{prefix}] trd-worker update available: v{current} -> v{latest}",
        file=sys.stderr,
    )
    print(f"       Upgrade with: {cmd}", file=sys.stderr)
    if severity == "warn":
        print(
            "       (Your current version may not honor dashboard settings; "
            "please upgrade soon.)",
            file=sys.stderr,
        )
    print(file=sys.stderr)

    _update_announced = True

    import os
    if severity == "warn" and os.environ.get("TRD_WORKER_AUTO_EXIT_ON_OUTDATED") == "1":
        print(
            "TRD_WORKER_AUTO_EXIT_ON_OUTDATED=1 -> exiting so supervisor "
            "can restart on new version",
            file=sys.stderr,
        )
        sys.exit(2)

