"""
Hardware detection module.

Detects CPU, GPU, VRAM, system RAM, OS, and storage.
Used by the setup view to recommend a runtime and estimate
whether a model/configuration is practical.

This module is platform-independent — it handles macOS, Linux, and Windows
with appropriate fallbacks when information cannot be determined.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    vendor: str = "unknown"       # "nvidia", "apple", "amd", "intel", "unknown"
    name: str = "Unknown GPU"
    vram_bytes: Optional[int] = None
    driver_version: Optional[str] = None
    cuda_available: bool = False
    metal_available: bool = False


@dataclass
class SystemInfo:
    """Complete hardware/system profile."""
    os_name: str = ""             # "macOS", "Linux", "Windows"
    os_version: str = ""
    architecture: str = ""        # "arm64", "x86_64"
    cpu_name: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    ram_total_bytes: int = 0
    ram_available_bytes: int = 0
    disk_free_bytes: int = 0
    gpu: GPUInfo = field(default_factory=GPUInfo)
    docker_available: bool = False
    docker_version: Optional[str] = None
    ollama_installed: bool = False
    ollama_path: Optional[str] = None


def detect_hardware() -> SystemInfo:
    """Detect all available hardware and software. Never raises."""
    info = SystemInfo()

    # ----- OS -----
    system = platform.system()
    if system == "Darwin":
        info.os_name = "macOS"
        try:
            info.os_version = platform.mac_ver()[0]
        except Exception:
            info.os_version = platform.release()
    elif system == "Linux":
        info.os_name = "Linux"
        info.os_version = platform.release()
    elif system == "Windows":
        info.os_name = "Windows"
        info.os_version = platform.version()
    else:
        info.os_name = system
        info.os_version = platform.release()

    info.architecture = platform.machine()

    # ----- CPU -----
    try:
        info.cpu_name = _get_cpu_name()
    except Exception:
        info.cpu_name = platform.processor() or "Unknown"

    info.cpu_cores = psutil.cpu_count(logical=False) or 0
    info.cpu_threads = psutil.cpu_count(logical=True) or 0

    # ----- RAM -----
    try:
        mem = psutil.virtual_memory()
        info.ram_total_bytes = mem.total
        info.ram_available_bytes = mem.available
    except Exception:
        pass

    # ----- Disk -----
    try:
        usage = shutil.disk_usage(os.path.expanduser("~"))
        info.disk_free_bytes = usage.free
    except Exception:
        pass

    # ----- GPU -----
    info.gpu = _detect_gpu(system, info.architecture)

    # ----- Docker -----
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info.docker_available = True
            info.docker_version = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # ----- Ollama -----
    ollama_path = shutil.which("ollama")
    if ollama_path:
        info.ollama_installed = True
        info.ollama_path = ollama_path

    return info


def _get_cpu_name() -> str:
    """Get the human-readable CPU name."""
    system = platform.system()

    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.strip().startswith("model name"):
                        return line.split(":")[1].strip()
        except (FileNotFoundError, PermissionError):
            pass

    return platform.processor() or "Unknown"


def _detect_gpu(system: str, architecture: str) -> GPUInfo:
    """Detect GPU information. Platform-specific."""
    gpu = GPUInfo()

    if system == "Darwin":
        if architecture == "arm64":
            gpu.vendor = "apple"
            gpu.metal_available = True
            gpu.name = _get_apple_gpu_name()
            gpu.vram_bytes = _get_apple_unified_memory()
        return gpu

    # Linux / Windows — check NVIDIA first
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            gpu.vendor = "nvidia"
            gpu.cuda_available = True
            if len(parts) >= 1:
                gpu.name = parts[0].strip()
            if len(parts) >= 2:
                try:
                    gpu.vram_bytes = int(float(parts[1].strip()) * 1024 * 1024)  # MiB → bytes
                except ValueError:
                    pass
            if len(parts) >= 3:
                gpu.driver_version = parts[2].strip()
            return gpu
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # No NVIDIA GPU detected — could be AMD or integrated
    gpu.vendor = "unknown"
    gpu.name = "No dedicated GPU detected"
    return gpu


def _get_apple_gpu_name() -> str:
    """Get Apple Silicon chip name."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            brand = result.stdout.strip()
            # Extract chip name (e.g., "Apple M2 Pro")
            if "Apple" in brand:
                return brand
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "Apple Silicon"


def _get_apple_unified_memory() -> Optional[int]:
    """
    On Apple Silicon, GPU shares unified memory with CPU.
    Report total system memory as a reference (not dedicated VRAM).
    """
    try:
        mem = psutil.virtual_memory()
        return mem.total
    except Exception:
        return None


def format_bytes(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    if size_bytes <= 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore
    return f"{size_bytes:.1f} PB"
