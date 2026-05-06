"""
GPU detection across NVIDIA / Apple Silicon / AMD.

NVIDIA → nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
Apple  → system_profiler SPDisplaysDataType (Metal GPUs share unified memory)
AMD    → rocm-smi --showproductname --showmeminfo vram

Returns first detected GPU. Multi-GPU systems are Phase 3.
"""

from __future__ import annotations
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class GpuInfo:
    vendor: str          # 'nvidia' | 'apple' | 'amd'
    model: str           # e.g. 'RTX 4090', 'M3 Max', 'MI250'
    vram_gb: int
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None


def detect() -> Optional[GpuInfo]:
    """Try each detection method. Returns None if nothing found."""
    for fn in (_detect_nvidia, _detect_apple, _detect_amd):
        try:
            info = fn()
            if info:
                return info
        except Exception:
            continue
    return None


# ── NVIDIA ──────────────────────────────────────────────────────────────────
def _detect_nvidia() -> Optional[GpuInfo]:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return None
    name = parts[0]
    vram_mib = int(parts[1])
    driver = parts[2] if len(parts) >= 3 else None

    # Try CUDA version (separate command, optional)
    cuda = None
    try:
        cuda_out = subprocess.check_output(
            ["nvidia-smi", "--query"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        for line in cuda_out.splitlines():
            if "CUDA Version" in line:
                cuda = line.split(":")[-1].strip()
                break
    except Exception:
        pass

    # Strip "NVIDIA " prefix if present, keep model
    model = name.replace("NVIDIA ", "").strip()
    return GpuInfo(
        vendor="nvidia",
        model=model,
        vram_gb=round(vram_mib / 1024),
        driver_version=driver,
        cuda_version=cuda,
    )


# ── Apple Silicon ───────────────────────────────────────────────────────────
def _detect_apple() -> Optional[GpuInfo]:
    if platform.system() != "Darwin":
        return None
    if platform.machine() != "arm64":
        return None  # Intel Mac, no unified memory GPU
    if not shutil.which("system_profiler"):
        return None
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    # Parse Chipset Model + Total Number of Cores. Memory comes from unified RAM.
    model = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Chipset Model:"):
            model = line.split(":", 1)[1].strip()
            break

    if not model or "Apple" not in model:
        return None

    # Apple unified memory — use total system RAM as proxy for GPU-addressable
    ram_gb = _apple_ram_gb()
    if ram_gb is None:
        return None

    # Strip "Apple " prefix → e.g. "M3 Max"
    short = model.replace("Apple ", "").strip()
    return GpuInfo(vendor="apple", model=short, vram_gb=ram_gb)


def _apple_ram_gb() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, timeout=5
        )
        bytes_ = int(out.strip())
        return round(bytes_ / (1024**3))
    except Exception:
        return None


# ── AMD ─────────────────────────────────────────────────────────────────────
def _detect_amd() -> Optional[GpuInfo]:
    if not shutil.which("rocm-smi"):
        return None
    try:
        name_out = subprocess.check_output(
            ["rocm-smi", "--showproductname"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        mem_out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    model = None
    for line in name_out.splitlines():
        if "Card Series" in line or "Card model" in line.lower():
            parts = line.split(":")
            if len(parts) >= 2:
                model = parts[-1].strip()
                break
    if not model:
        return None

    vram_gb = 0
    for line in mem_out.splitlines():
        if "Total" in line and "VRAM" in line:
            # find the number of bytes
            digits = "".join(c for c in line if c.isdigit())
            if digits:
                try:
                    vram_gb = round(int(digits) / (1024**3))
                except ValueError:
                    pass
            break

    if vram_gb == 0:
        return None
    return GpuInfo(vendor="amd", model=model, vram_gb=vram_gb)


# ── helpers ─────────────────────────────────────────────────────────────────
def suggest_supported_models(info: GpuInfo) -> list[str]:
    """
    Given a GPU's VRAM, suggest which models it can plausibly run.
    These are placeholder names matching what TRD will route in Phase 7.
    """
    vram = info.vram_gb
    models: list[str] = []
    if vram >= 80:
        models = ["qwen3-235b", "llama3-70b", "deepseek-67b", "mixtral-8x22b"]
    elif vram >= 40:
        models = ["llama3-70b", "deepseek-67b", "mixtral-8x22b", "qwen2-32b"]
    elif vram >= 24:
        models = ["qwen2-32b", "llama3-13b", "mistral-7b", "phi-3-medium"]
    elif vram >= 16:
        models = ["llama3-13b", "mistral-7b", "phi-3-medium"]
    elif vram >= 8:
        models = ["mistral-7b", "phi-3-medium", "phi-3-mini"]
    else:
        models = ["phi-3-mini"]
    return models


def os_string() -> str:
    """e.g. 'darwin-arm64-14.5' or 'linux-x86_64-Ubuntu-22.04'."""
    return f"{platform.system().lower()}-{platform.machine()}-{platform.release()}"
