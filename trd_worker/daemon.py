"""
Worker daemon loop.

  every HEARTBEAT_SEC: send heartbeat with status + utilization
  every POLL_SEC when idle: poll for job; if claimed, execute + submit

Network errors → log and back off. Auth errors → exit (token revoked).
Ctrl+C → final 'offline' heartbeat then exit cleanly.
"""

from __future__ import annotations
import signal
import sys
import time
from typing import Optional

import requests

from . import __version__, api, config, runner, settings


HEARTBEAT_SEC = 30
POLL_SEC = 10
BACKOFF_MAX_SEC = 60


class Daemon:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.running = True
        self.last_heartbeat = 0.0
        self.consecutive_errors = 0
        # Track the last "paused for X" reason so we don't spam logs every
        # poll cycle when nothing has changed (e.g. on battery for an hour).
        self._last_pause_reason: Optional[str] = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {msg}", flush=True)

    def _backoff(self) -> int:
        # 2, 4, 8, 16, 32, 60, 60...
        return min(BACKOFF_MAX_SEC, 2 ** min(self.consecutive_errors, 6))

    def _send_heartbeat(self, token: str, status: str, util: int) -> bool:
        try:
            api.heartbeat(token, status, util)
            self.consecutive_errors = 0
            return True
        except api.ApiError as e:
            if e.status in (401, 403):
                self._log(f"⚠ Auth rejected ({e.status}): {e.body}. Run `trd-worker login` again.")
                self.running = False
                return False
            self._log(f"⚠ Heartbeat error: {e}")
            self.consecutive_errors += 1
            return False
        except requests.RequestException as e:
            self._log(f"⚠ Heartbeat network error: {e}")
            self.consecutive_errors += 1
            return False

    def _try_run_job(self, token: str) -> bool:
        """
        Returns True if a job was processed (ran and submitted, or attempted).
        Returns False if no job was available.
        """
        try:
            job = api.poll_job(token)
        except api.ApiError as e:
            if e.status in (401, 403):
                self._log(f"⚠ Auth rejected ({e.status}). Stopping.")
                self.running = False
                return False
            self._log(f"⚠ Poll error: {e}")
            self.consecutive_errors += 1
            return False
        except requests.RequestException as e:
            self._log(f"⚠ Poll network error: {e}")
            self.consecutive_errors += 1
            return False

        if job is None:
            return False

        self._log(f"📥 [job {job.job_id[:8]}] {job.model_name} — {job.prompt[:60]}{'...' if len(job.prompt) > 60 else ''}")

        # Mark busy via heartbeat (best-effort, don't fail if it errors)
        self._send_heartbeat(token, "busy", 75)

        # Execute (real inference in Phase 3, or stub if TRD_WORKER_USE_STUB=1)
        try:
            result, duration_ms, tokens = runner.run(
                job.prompt, job.model_name, job.max_tokens, job.timeout_sec
            )
        except Exception as e:
            self._log(f"⚠ Job execution failed: {e}")
            # Report failure to backend so it can requeue or mark failed
            try:
                fr = api.fail_job(token, job.job_id, str(e))
                if fr.requeued:
                    self._log(f"   → requeued for retry (attempt {fr.retry_count})")
                else:
                    self._log(f"   → marked failed (max retries exhausted)")
            except api.ApiError as fe:
                if fe.status == 410:
                    self._log(f"   (job already cleared, no fail-report needed)")
                else:
                    self._log(f"   ⚠ fail-report rejected: {fe}")
            except requests.RequestException as fe:
                self._log(f"   ⚠ fail-report network error: {fe}")
            return True

        # Submit
        try:
            submit_result = api.submit_job(
                token, job.job_id, result, duration_ms, tokens
            )
            self._log(
                f"✅ [job {job.job_id[:8]}] +{submit_result.credits_awarded} cr "
                f"(balance: {submit_result.total_credits_balance}) — {duration_ms}ms"
            )
            self.consecutive_errors = 0
        except api.ApiError as e:
            if e.status == 410:
                self._log(f"⚠ [job {job.job_id[:8]}] timed out before submit — no credits")
            elif e.status in (401, 403):
                self._log(f"⚠ Auth rejected on submit. Stopping.")
                self.running = False
            else:
                self._log(f"⚠ Submit error: {e}")
                self.consecutive_errors += 1
        except requests.RequestException as e:
            self._log(f"⚠ Submit network error: {e}")
            self.consecutive_errors += 1

        return True

    def run(self) -> int:
        cfg = config.load()
        token = cfg.get("auth_token")
        worker_id = cfg.get("worker_id")
        if not token or not worker_id:
            print("✗ Not logged in. Run `trd-worker login` first.", file=sys.stderr)
            return 1

        self._log(f"🚀 trd-worker started — worker {worker_id[:8]}, rate {cfg.get('base_credits_per_hour', '?')} cr/hr")
        self._log(f"🔗 Backend: {config.api_base()}")

        # Catch Ctrl+C for clean shutdown
        def _stop(signum, frame):
            self._log("⏹  Stopping... sending offline heartbeat")
            self.running = False
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        # Initial heartbeat
        if not self._send_heartbeat(token, "idle", 0):
            if not self.running:
                return 1

        # Initial settings fetch (B, May 9 2026). Forces a refresh on start
        # so the daemon picks up user prefs before claiming any job.
        settings.maybe_refresh(token, __version__)

        self.last_heartbeat = time.monotonic()
        self._log("💓 Heartbeat ok — entering poll loop")

        while self.running:
            now = time.monotonic()

            # Periodic heartbeat
            if now - self.last_heartbeat >= HEARTBEAT_SEC:
                self._send_heartbeat(token, "idle", 0)
                self.last_heartbeat = now

            # Refresh settings periodically (no-op if SETTINGS_REFRESH_SEC
            # hasn't elapsed; cheap when called every loop). __version__ is
            # ack'd to the backend on first success so the UI shows
            # "Settings active on worker vX.Y.Z".
            settings.maybe_refresh(token, __version__)

            # Check user-prefs gate BEFORE polling. If paused for any reason
            # (on battery, outside schedule), skip and sleep. We log the
            # reason once per change so the user can see WHY their worker
            # is idle without log spam.
            accept, reason = settings.should_accept_jobs()
            if not accept:
                if reason != self._last_pause_reason:
                    self._log(f"⏸  Pausing — {reason}")
                    self._last_pause_reason = reason
                time.sleep(POLL_SEC)
                continue
            elif self._last_pause_reason is not None:
                self._log("▶  Resuming — settings allow jobs again")
                self._last_pause_reason = None

            # Try a job
            ran = self._try_run_job(token)

            if not self.running:
                break

            if ran:
                # Heartbeat resync after busy work
                self._send_heartbeat(token, "idle", 0)
                self.last_heartbeat = time.monotonic()
                continue  # immediately try for another job

            # No job — sleep, with backoff if errors are stacking
            sleep_for = POLL_SEC if self.consecutive_errors == 0 else self._backoff()
            if self.consecutive_errors > 0:
                self._log(f"💤 Backing off {sleep_for}s (errors: {self.consecutive_errors})")
            time.sleep(sleep_for)

        # No final heartbeat — last status (idle/busy) stays until server-side
        # timeout cron flips worker to 'offline' (Phase 2.5+)
        self._log("👋 stopped")
        return 0


def main() -> int:
    return Daemon().run()
