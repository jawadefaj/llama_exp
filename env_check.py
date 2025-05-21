#!/usr/bin/env python3
"""
env_check.py

Quick environment diagnostics:

 • Python version & executable
 • PyTorch version, CUDA, cuDNN, GPU list
 • Hugging Face CLI login status
 • Hugging Face token file
 • **NEW:** CPU-core count (logical & physical) and total RAM
"""

import os
import sys
import subprocess
from textwrap import indent

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _human_bytes(num: int) -> str:
    """Pretty-print bytes (powers of 1024)."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024 or unit == "TiB":
            return f"{num:.1f} {unit}"
        num /= 1024


# --------------------------------------------------------------------------- #
# section 1 – Python                                                          #
# --------------------------------------------------------------------------- #
def check_python() -> None:
    print("Python executable:", sys.executable)
    print("Python version   :", sys.version.replace("\n", " "))


# --------------------------------------------------------------------------- #
# section 2 – PyTorch / CUDA                                                  #
# --------------------------------------------------------------------------- #
def check_torch() -> None:
    try:
        import torch

        print("\n--- PyTorch ---")
        print("torch version   :", torch.__version__)
        cuda_avail = torch.cuda.is_available()
        print("CUDA available  :", cuda_avail)
        if cuda_avail:
            print("CUDA version    :", torch.version.cuda)
            print("cuDNN version   :", torch.backends.cudnn.version())
            print("CUDA devices    :", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} "
                      f"({props.total_memory // 1024 ** 2} MiB)")
    except ImportError:
        print("\nPyTorch not installed.")


# --------------------------------------------------------------------------- #
# section 3 – System resources (NEW)                                          #
# --------------------------------------------------------------------------- #
def check_system() -> None:
    print("\n--- System ---")
    # CPU
    logical = os.cpu_count()
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or "n/a"
        total_ram = _human_bytes(psutil.virtual_memory().total)
    except ImportError:
        physical = "psutil-not-installed"
        total_ram = "psutil-not-installed"

    print(f"Logical CPU cores: {logical}")
    print(f"Physical CPU cores: {physical}")
    print(f"Total RAM: {total_ram}")
    if physical == "psutil-not-installed":
        print(indent("Tip: pip install psutil to see physical-core "
                     "and RAM details.", "  "))


# --------------------------------------------------------------------------- #
# section 4 – Hugging Face                                                    #
# --------------------------------------------------------------------------- #
def check_hf_cli() -> None:
    print("\n--- Hugging Face CLI ---")
    try:
        res = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            print(indent(res.stdout.strip(), "  "))
        else:
            print("huggingface-cli not logged in or not installed.")
    except FileNotFoundError:
        print("huggingface-cli is not installed.")


def check_hf_token() -> None:
    print("\n--- Hugging Face Token ---")
    token_path = os.path.expanduser("~/.huggingface/token")
    if os.path.isfile(token_path):
        with open(token_path, "r") as f:
            token = f.read().strip()
        print(f"Found token file at {token_path} "
              f"(length: {len(token)} chars)")
    else:
        print(f"No token file at {token_path}")


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    check_python()
    check_torch()
    check_system()      # ← new section
    check_hf_cli()
    check_hf_token()


if __name__ == "__main__":
    main()
