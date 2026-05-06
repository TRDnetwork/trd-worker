"""
Job runner — Phase 3 real inference via llama-cpp-python.

Loads the requested model on first call (lazy), runs the prompt, returns
generated text + duration + token count. Respects max_tokens and timeout_sec.

Errors that bubble out (so daemon can call /jobs/fail):
  - ModelNotDownloadedError: model GGUF isn't on disk
  - ModelNotSupportedError: requested model isn't in our registry
  - InferenceTimeoutError: generation exceeded timeout_sec
  - InferenceError: any other inference failure (OOM, llama.cpp crash, etc.)

The Phase 2 stub (run_job_stub) is preserved at the bottom for fallback /
testing — daemon can be flipped to use it via TRD_WORKER_USE_STUB=1 env var.
"""

from __future__ import annotations
import os
import time
import threading
from typing import Optional

from . import models


# ── exceptions ──────────────────────────────────────────────────────────────
class InferenceError(Exception):
    """Generic inference failure (OOM, llama.cpp crash, etc.)."""


class ModelNotDownloadedError(InferenceError):
    pass


class ModelNotSupportedError(InferenceError):
    pass


class InferenceTimeoutError(InferenceError):
    pass


# ── system prompt ───────────────────────────────────────────────────────────
# Phase 3 ships a single generic system prompt. Phase 7 (router integration)
# will pass per-agent system prompts through compute_jobs.metadata.
_DEFAULT_SYSTEM = (
    "You are a senior frontend engineer. Generate clean, modern HTML/CSS for "
    "the user's request. Output only the HTML. No commentary, no markdown "
    "code fences, no explanations."
)


# ── main entry point ────────────────────────────────────────────────────────
def run_job(
    prompt: str,
    model_name: str,
    max_tokens: int,
    timeout_sec: int,
    system_prompt: Optional[str] = None,
) -> tuple[str, int, int]:
    """
    Run real inference. Returns (result_text, duration_ms, tokens_generated).

    Raises one of the InferenceError subclasses on failure — daemon catches
    and reports via /jobs/fail.
    """
    spec = models._by_name(model_name)
    if not spec:
        raise ModelNotSupportedError(
            f"Model '{model_name}' is not in the worker's registry. "
            f"Available: {[m.name for m in models.list_available()]}"
        )
    if not models.is_downloaded(spec.name):
        raise ModelNotDownloadedError(
            f"Model '{spec.name}' is not downloaded. "
            f"Run: trd-worker models pull {spec.name}"
        )

    start = time.monotonic()

    # Load model (cached after first call — only first job pays the load cost)
    try:
        llm = models.load_model(spec.name)
    except Exception as e:
        raise InferenceError(f"Failed to load model '{spec.name}': {e}") from e

    sys_prompt = system_prompt or _DEFAULT_SYSTEM
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    # Run with a soft timeout via thread + join. llama-cpp-python doesn't expose
    # a native deadline; we discard results from runs that exceed timeout_sec.
    # The thread continues in the background until llama.cpp finishes — that's
    # acceptable because the daemon serializes jobs (1 GPU = 1 job at a time).
    result_holder: dict = {}
    error_holder: dict = {}

    def _run() -> None:
        try:
            out = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stream=False,
            )
            choice = out["choices"][0]
            result_holder["text"] = choice["message"]["content"] or ""
            usage = out.get("usage") or {}
            result_holder["completion_tokens"] = int(usage.get("completion_tokens") or 0)
            result_holder["finish_reason"] = choice.get("finish_reason") or "unknown"
        except Exception as e:
            error_holder["err"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        raise InferenceTimeoutError(
            f"Inference exceeded {timeout_sec}s timeout (still running in background)"
        )

    if "err" in error_holder:
        e = error_holder["err"]
        msg = str(e).lower()
        if "out of memory" in msg or ("metal" in msg and "alloc" in msg):
            raise InferenceError(f"OOM during inference: {e}") from e
        raise InferenceError(f"Inference failed: {e}") from e

    text = result_holder.get("text", "").strip()
    tokens = result_holder.get("completion_tokens", 0)

    if not text:
        raise InferenceError("Model returned empty output")

    duration_ms = int((time.monotonic() - start) * 1000)
    return text, duration_ms, tokens


# ── stub fallback (Phase 2 behavior, retained for testing) ──────────────────
import random


_HTML_STUB = """<section class="hero">
  <h1>{title}</h1>
  <p>{tagline}</p>
  <a href="#contact" class="btn">Get Started</a>
</section>"""

_TITLES = [
    "Your Vision, Beautifully Built",
    "Find Your Flow",
    "Crafted For You",
    "Where Quality Meets Care",
    "Modern Solutions, Trusted Service",
]
_TAGLINES = [
    "Professional. Local. Trusted by hundreds.",
    "Where every detail matters.",
    "Simple, fast, made for you.",
    "Excellence is our standard.",
    "Bringing your ideas to life.",
]


def run_job_stub(
    prompt: str, model_name: str, max_tokens: int, timeout_sec: int
) -> tuple[str, int, int]:
    """Phase 2 stub — kept for testing without burning through GPU cycles."""
    start = time.monotonic()
    sleep_sec = random.uniform(2.0, 8.0)
    sleep_sec = min(sleep_sec, max(1.0, timeout_sec - 2))
    time.sleep(sleep_sec)
    title = random.choice(_TITLES)
    tagline = random.choice(_TAGLINES)
    result = _HTML_STUB.format(title=title, tagline=tagline)
    tokens = min(max_tokens, max(20, len(result) // 4))
    duration_ms = int((time.monotonic() - start) * 1000)
    return result, duration_ms, tokens


# ── dispatcher used by daemon ───────────────────────────────────────────────
def run(
    prompt: str, model_name: str, max_tokens: int, timeout_sec: int
) -> tuple[str, int, int]:
    """
    Single entry point used by daemon. Picks real or stub based on env var.

    TRD_WORKER_USE_STUB=1 → use stub (no model load, fake output)
    otherwise            → real inference via llama-cpp-python
    """
    if os.environ.get("TRD_WORKER_USE_STUB") == "1":
        return run_job_stub(prompt, model_name, max_tokens, timeout_sec)
    return run_job(prompt, model_name, max_tokens, timeout_sec)
