"""Diagnose Apple Metal/MPS visibility for local experiments."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".hf_cache" / "datasets"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Apple GPU, Metal, and PyTorch MPS.")
    parser.add_argument("--skip-metal-api", action="store_true", help="Skip the Swift Metal API check.")
    return parser.parse_args()


def run_command(args: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> None:
    args = parse_args()

    print_section("System")
    print("python:", sys.version.replace("\n", " "))
    print("executable:", sys.executable)
    print("platform:", platform.platform())
    print("machine:", platform.machine())
    code, output = run_command(["sw_vers"])
    print(output if code == 0 else f"sw_vers failed: {output}")

    print_section("Graphics")
    code, output = run_command(["system_profiler", "SPDisplaysDataType"])
    print(output if code == 0 else f"system_profiler failed: {output}")

    if not args.skip_metal_api:
        print_section("Native Metal API")
        swift_code = (
            "import Metal; "
            "if let d = MTLCreateSystemDefaultDevice() { "
            "print(\"metal_device\", d.name) "
            "} else { print(\"metal_device nil\") }"
        )
        code, output = run_command(["xcrun", "swift", "-e", swift_code], timeout=60)
        print(output if code == 0 else f"swift metal check failed: {output}")

    print_section("PyTorch")
    print("torch:", torch.__version__)
    print("mps_built:", torch.backends.mps.is_built())
    print("mps_available:", torch.backends.mps.is_available())
    print("cuda_available:", torch.cuda.is_available())
    if torch.backends.mps.is_available():
        x = torch.randn(1024, 1024, device="mps")
        y = x @ x.T
        torch.mps.synchronize()
        print("mps_tensor_device:", y.device)
        print("mps_tensor_mean:", float(y.mean().cpu()))
        print("mps_current_allocated_memory:", torch.mps.current_allocated_memory())
    else:
        print("mps_tensor_test: skipped because MPS is not available")


if __name__ == "__main__":
    main()
