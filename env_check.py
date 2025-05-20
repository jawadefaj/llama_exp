#!/usr/bin/env python3
"""
env_check.py

Quick environment diagnostics:
 - Python version
 - PyTorch version & CUDA info
 - Hugging Face CLI login status
 - Hugging Face token in ~/.huggingface/token
"""

import sys
import subprocess
import os

def check_python():
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))

def check_torch():
    try:
        import torch
        print("\n--- PyTorch ---")
        print("torch version:", torch.__version__)
        cuda_avail = torch.cuda.is_available()
        print("CUDA available:", cuda_avail)
        if cuda_avail:
            print("CUDA version (build):", torch.version.cuda)
            print("cuDNN version:", torch.backends.cudnn.version())
            print("Number of CUDA devices:", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory//1024**2} MiB)")
    except ImportError:
        print("\nPyTorch is not installed.")

def check_hf_cli():
    print("\n--- Hugging Face CLI ---")
    try:
        res = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True,
            check=False
        )
        if res.returncode == 0:
            print("huggingface-cli whoami output:\n", res.stdout.strip())
        else:
            print("huggingface-cli not logged in or not installed.")
    except FileNotFoundError:
        print("huggingface-cli is not installed.")

def check_hf_token():
    print("\n--- Hugging Face Token ---")
    token_path = os.path.expanduser("~/.huggingface/token")
    if os.path.isfile(token_path):
        with open(token_path, "r") as f:
            token = f.read().strip()
        print("Found token in ~/.huggingface/token (length:", len(token), "characters)")
    else:
        print("No token file found at ~/.huggingface/token")

def main():
    check_python()
    check_torch()
    check_hf_cli()
    check_hf_token()

if __name__ == "__main__":
    main()
