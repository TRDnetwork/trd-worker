"""
HTTP client for trd-cn-backend.

Thin wrappers — no retry logic here, daemon handles retries with backoff.
Network errors raise RequestException; HTTP errors raise ApiError.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any

import requests

from . import config


class ApiError(Exception):
    def __init__(self, status: int, body: Any):
        self.status = status
        self.body = body
        msg = body.get("error") if isinstance(body, dict) else str(body)
        super().__init__(f"HTTP {status}: {msg}")


@dataclass
class RegisterResult:
    worker_id: str
    auth_token: str
    base_rate_credits_per_hour: int


@dataclass
class Job:
    job_id: str
    prompt: str
    model_name: str
    max_tokens: int
    timeout_sec: int


@dataclass
class HeartbeatResult:
    server_time: str
    recommended_poll_interval_sec: int


@dataclass
class SubmitResult:
    credits_awarded: int
    total_credits_balance: int


# ── helpers ─────────────────────────────────────────────────────────────────
def _headers(token: Optional[str] = None) -> dict[str, str]:
    h = {"Content-Type": "application/json", "User-Agent": "trd-worker/0.1.0"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _check(r: requests.Response) -> Any:
    if r.status_code == 204:
        return None
    try:
        body = r.json()
    except Exception:
        body = r.text
    if r.status_code >= 400:
        raise ApiError(r.status_code, body)
    return body


# ── endpoints ───────────────────────────────────────────────────────────────
def register(
    *,
    email: str,
    hostname: str,
    gpu_vendor: str,
    gpu_model: str,
    gpu_vram_gb: int,
    supported_models: list[str],
    cli_version: str,
    cuda_version: Optional[str] = None,
    driver_version: Optional[str] = None,
    os: Optional[str] = None,
    timeout: int = 15,
) -> RegisterResult:
    url = f"{config.api_base()}/api/compute/workers/register"
    payload = {
        "email": email,
        "hostname": hostname,
        "gpu_vendor": gpu_vendor,
        "gpu_model": gpu_model,
        "gpu_vram_gb": gpu_vram_gb,
        "supported_models": supported_models,
        "cli_version": cli_version,
        "cuda_version": cuda_version,
        "driver_version": driver_version,
        "os": os,
    }
    r = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
    body = _check(r)
    return RegisterResult(
        worker_id=body["worker_id"],
        auth_token=body["auth_token"],
        base_rate_credits_per_hour=body["base_rate_credits_per_hour"],
    )


def heartbeat(
    token: str, status: str, utilization_pct: int, timeout: int = 10
) -> HeartbeatResult:
    url = f"{config.api_base()}/api/compute/workers/heartbeat"
    payload = {"status": status, "utilization_pct": utilization_pct}
    r = requests.post(url, json=payload, headers=_headers(token), timeout=timeout)
    body = _check(r)
    return HeartbeatResult(
        server_time=body["server_time"],
        recommended_poll_interval_sec=body.get("recommended_poll_interval_sec", 10),
    )


def poll_job(token: str, timeout: int = 15) -> Optional[Job]:
    """Returns Job if one was claimed, None on 204 (no jobs)."""
    url = f"{config.api_base()}/api/compute/jobs/poll"
    r = requests.get(url, headers=_headers(token), timeout=timeout)
    if r.status_code == 204:
        return None
    body = _check(r)
    return Job(
        job_id=body["job_id"],
        prompt=body["prompt"],
        model_name=body["model_name"],
        max_tokens=body["max_tokens"],
        timeout_sec=body["timeout_sec"],
    )


def submit_job(
    token: str,
    job_id: str,
    result: str,
    duration_ms: int,
    tokens_generated: int,
    timeout: int = 15,
) -> SubmitResult:
    url = f"{config.api_base()}/api/compute/jobs/submit"
    payload = {
        "job_id": job_id,
        "result": result,
        "duration_ms": duration_ms,
        "tokens_generated": tokens_generated,
    }
    r = requests.post(url, json=payload, headers=_headers(token), timeout=timeout)
    body = _check(r)
    return SubmitResult(
        credits_awarded=body["credits_awarded"],
        total_credits_balance=body["total_credits_balance"],
    )


@dataclass
class FailResult:
    requeued: bool
    retry_count: int


def fail_job(
    token: str,
    job_id: str,
    error_message: str,
    timeout: int = 10,
) -> FailResult:
    """Report job execution failure to backend. Server requeues or marks failed."""
    url = f"{config.api_base()}/api/compute/jobs/fail"
    payload = {"job_id": job_id, "error_message": error_message[:1000]}
    r = requests.post(url, json=payload, headers=_headers(token), timeout=timeout)
    body = _check(r)
    return FailResult(
        requeued=body["requeued"],
        retry_count=body["retry_count"],
    )


def revoke(token: str, timeout: int = 10) -> None:
    """Revoke this worker's token server-side. Token is dead after this call."""
    url = f"{config.api_base()}/api/compute/workers/revoke"
    r = requests.post(url, headers=_headers(token), timeout=timeout)
    _check(r)


def update_capabilities(
    token: str,
    supported_models: list[str],
    timeout: int = 10,
) -> list[str]:
    """
    Update the worker's advertised supported_models without re-registering.
    Returns the list as the server stored it (after filtering).
    """
    url = f"{config.api_base()}/api/compute/workers/update-capabilities"
    payload = {"supported_models": supported_models}
    r = requests.post(url, json=payload, headers=_headers(token), timeout=timeout)
    body = _check(r)
    return list(body.get("supported_models") or [])

