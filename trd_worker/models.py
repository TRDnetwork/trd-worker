"""
Model manager — download GGUF files from HuggingFace, cache to disk, load into memory.

Storage: ~/.trd-worker/models/<model_name>/<filename>.gguf

Model names are short identifiers (e.g. 'qwen2.5-7b-instruct'). Each maps to a
HuggingFace repo + specific GGUF quantization. The registry is intentionally
small in Phase 3 — Qwen 2.5 7B Instruct (Q4_K_M) is the only entry. Phase 3+
adds more.
"""

from __future__ import annotations
import os
import sys
import time
import hashlib
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import config


MODELS_DIR = config.CONFIG_DIR / "models"


@dataclass(frozen=True)
class ModelSpec:
    """A model the worker can advertise. Maps short name → HF repo + file."""
    name: str               # short id used in registration / job routing
    display_name: str       # human-readable
    hf_repo: str            # huggingface repo, e.g. 'bartowski/Qwen2.5-7B-Instruct-GGUF'
    gguf_filename: str      # exact file inside that repo
    file_size_gb: float     # approximate, for disk-space pre-check
    min_vram_gb: int        # below this, refuse to load
    n_ctx: int              # context window size
    aliases: tuple[str, ...] = ()   # alt names that should resolve to this


# ── REGISTRY ────────────────────────────────────────────────────────────────
# Phase 3 ships with one model. Add more here in Phase 3+.
_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="qwen2.5-7b-instruct",
        display_name="Qwen 2.5 7B Instruct (Q4_K_M)",
        hf_repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
        gguf_filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        file_size_gb=4.7,
        min_vram_gb=8,
        n_ctx=8192,
        aliases=("qwen2.5-7b", "qwen-7b", "qwen2.5"),
    ),
)


def _by_name(name: str) -> Optional[ModelSpec]:
    n = name.lower().strip()
    for m in _MODELS:
        if m.name == n or n in m.aliases:
            return m
    return None


def list_available() -> list[ModelSpec]:
    """All models the worker knows how to download. Disk state irrelevant."""
    return list(_MODELS)


def list_local() -> list[ModelSpec]:
    """Models that are downloaded and ready to use."""
    return [m for m in _MODELS if is_downloaded(m.name)]


def model_path(name: str) -> Path:
    """Where the GGUF should live on disk (whether or not it exists yet)."""
    spec = _by_name(name)
    if not spec:
        raise ValueError(f"Unknown model: {name}")
    return MODELS_DIR / spec.name / spec.gguf_filename


def is_downloaded(name: str) -> bool:
    spec = _by_name(name)
    if not spec:
        return False
    p = model_path(spec.name)
    if not p.exists():
        return False
    # Sanity: file must be at least 80% of expected size (catches partial downloads)
    actual_gb = p.stat().st_size / (1024**3)
    expected_min = spec.file_size_gb * 0.8
    return actual_gb >= expected_min


# ── DOWNLOAD ────────────────────────────────────────────────────────────────
def _hf_url(spec: ModelSpec) -> str:
    return f"https://huggingface.co/{spec.hf_repo}/resolve/main/{spec.gguf_filename}"


def download_model(
    name: str,
    progress: bool = True,
    force: bool = False,
) -> Path:
    """
    Download a model GGUF to ~/.trd-worker/models/<name>/. Idempotent.

    Args:
        name: Model short name (e.g. 'qwen2.5-7b-instruct')
        progress: Print progress bar to stderr.
        force: Re-download even if cached.

    Returns:
        Path to the downloaded GGUF file.

    Raises:
        ValueError: unknown model name
        RuntimeError: download failed (network, disk, etc.)
    """
    spec = _by_name(name)
    if not spec:
        raise ValueError(
            f"Unknown model '{name}'. Available: "
            f"{', '.join(m.name for m in _MODELS)}"
        )

    dest = model_path(spec.name)
    if dest.exists() and not force:
        if is_downloaded(spec.name):
            return dest
        # Partial download → re-fetch
        if progress:
            print(f"⚠ Partial file detected ({dest}), re-downloading", file=sys.stderr)
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = _hf_url(spec)
    tmp = dest.with_suffix(dest.suffix + ".part")

    if progress:
        print(
            f"📥 Downloading {spec.display_name} (~{spec.file_size_gb}GB)\n"
            f"   from {url}\n"
            f"   to   {dest}",
            file=sys.stderr,
        )

    try:
        _download_with_progress(url, tmp, progress=progress)
    except urllib.error.HTTPError as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"HuggingFace returned HTTP {e.code} for {url}. "
            f"Model may have moved, or HF is rate-limiting. Try again in a few minutes."
        )
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Download failed: {e}")

    tmp.rename(dest)

    if progress:
        actual_gb = dest.stat().st_size / (1024**3)
        print(f"✓ Downloaded {actual_gb:.2f}GB → {dest}", file=sys.stderr)

    return dest


def _download_with_progress(url: str, dest: Path, progress: bool = True) -> None:
    """Stream download with simple progress bar."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "trd-worker/0.1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1024 * 1024  # 1MB
        last_print = 0.0
        with open(dest, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if progress and total > 0:
                    now = time.monotonic()
                    if now - last_print > 0.5 or downloaded == total:
                        pct = 100 * downloaded / total
                        gb_done = downloaded / (1024**3)
                        gb_total = total / (1024**3)
                        bar_w = 30
                        filled = int(bar_w * downloaded / total)
                        bar = "█" * filled + "░" * (bar_w - filled)
                        print(
                            f"\r   [{bar}] {pct:5.1f}% "
                            f"{gb_done:.2f}/{gb_total:.2f}GB",
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_print = now
        if progress:
            print("", file=sys.stderr)  # newline after final bar


# ── LOAD INTO MEMORY ────────────────────────────────────────────────────────
# Lazy import — llama_cpp is heavy, only needed when actually loading.
_loaded_cache: dict[str, "object"] = {}


def load_model(
    name: str,
    n_gpu_layers: int = -1,    # -1 = offload everything to GPU (Metal/CUDA)
    verbose: bool = False,
) -> "object":
    """
    Load a model into memory. Returns a Llama instance.

    Caches loaded models — calling load_model() twice with the same name returns
    the same instance (we only have one GPU; loading two copies of a 7B model is
    pointless and would OOM).

    Lazy-imports llama_cpp so the daemon's import path doesn't pay the cost
    until it actually needs to run inference.
    """
    spec = _by_name(name)
    if not spec:
        raise ValueError(f"Unknown model: {name}")

    if not is_downloaded(spec.name):
        raise RuntimeError(
            f"Model '{spec.name}' is not downloaded. "
            f"Run: trd-worker models pull {spec.name}"
        )

    cache_key = spec.name
    if cache_key in _loaded_cache:
        return _loaded_cache[cache_key]

    # Lazy import: avoids pulling in llama_cpp on `trd-worker --help` etc.
    from llama_cpp import Llama  # type: ignore

    path = model_path(spec.name)
    llm = Llama(
        model_path=str(path),
        n_ctx=spec.n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
        seed=-1,         # random per-call
        logits_all=False,
    )
    _loaded_cache[cache_key] = llm
    return llm


def unload_all() -> None:
    """Drop all loaded models. Frees VRAM."""
    _loaded_cache.clear()


def loaded_models() -> list[str]:
    return list(_loaded_cache.keys())
