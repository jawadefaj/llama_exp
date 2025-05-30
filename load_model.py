#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Used and distributed under the terms of the Llama 3 Community License Agreement.

import os
import logging
from pathlib import Path
import time
import json
import psutil
import torch
import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init

# ───────────────────────────── Config ──────────────────────────────
BLOCK_INDEX = int(os.environ.get("LLAMA_TRACE_BLOCK", "0"))  # which layer to trace
TRACE_DIR = Path("./log")
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Logger Setup ──────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logger.debug("===== Starting load_model_from_weight_distributed script =====")

# ───────────── CUDA patch for CPU-only environments ────────────────
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *a, **k: self
    logger.debug("CUDA not available: patched torch.Tensor.cuda to no-op.")

# ───────────── Log hardware resources on start ─────────────────────
cpu_count = os.cpu_count()
gpu_count = torch.cuda.device_count()
ram_gb = psutil.virtual_memory().total / 1_073_741_824  # GiB
logger.debug(f"Hardware: CPUs={cpu_count}, GPUs={gpu_count}, RAM={ram_gb:.1f} GiB")
if gpu_count:
    for i in range(gpu_count):
        p = torch.cuda.get_device_properties(i)
        logger.debug(f"  GPU {i}: {p.name}, {p.total_memory/1_073_741_824:.1f} GiB")

torch.set_num_threads(cpu_count)
logger.debug(f"Set PyTorch CPU thread count explicitly: {cpu_count} threads")

# ───────────── Distributed init ────────────────────────────────────
backend = "nccl" if torch.cuda.is_available() else "gloo"
dist.init_process_group(
    backend=backend,
    init_method="tcp://127.0.0.1:29500",
    rank=int(os.environ.get("RANK", "0")),
    world_size=int(os.environ.get("WORLD_SIZE", gpu_count if gpu_count else 1)),
)
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
global_rank = dist.get_rank()
world_size = dist.get_world_size()

if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device("cuda", local_rank)
else:
    DEVICE = torch.device("cpu")

logger.debug(f"Process group: rank {global_rank}/{world_size}, device={DEVICE}")

# ───────────── Model-parallel init ────────────────────────────────
fs_init.initialize_model_parallel(1)
logger.debug("Model-parallel world initialized (tensor-parallel=1)")

# ───────────── Paths & runtime hyper-params ───────────────────────
SCRIPT_ROOT = Path(__file__).resolve().parent
CKPT_DIR = SCRIPT_ROOT / "model" / "Llama-3.2-1B"
TOKENIZER_PATH = CKPT_DIR / "tokenizer.model"
MAX_SEQ_LEN = 128
MAX_BATCH = 1

logger.debug(f"Checkpoint dir: {CKPT_DIR}")
logger.debug(f"Tokenizer path: {TOKENIZER_PATH}")
logger.debug(f"Tracing block index: {BLOCK_INDEX}")

# ───────────── Import model and tokenizer ─────────────────────────
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
logger.debug("Imported llama.model + llama.tokenizer")

# ───────────── Load checkpoint shard ──────────────────────────────
ckpt_files = sorted(CKPT_DIR.glob("*.pth"))
assert ckpt_files, f"No .pth files in {CKPT_DIR}"
ckpt_file = ckpt_files[global_rank] if len(ckpt_files) == world_size else ckpt_files[0]
logger.debug(f"Selected ckpt (rank={global_rank}): {ckpt_file.name}")

raw_sd = torch.load(ckpt_file, map_location=DEVICE)
logger.debug(f"Raw state-dict loaded with {len(raw_sd)} keys (sample {list(raw_sd)[:5]})")

# print([k for k in raw_sd.keys() if 'norm' in k])



# No key remapping for original_model compatibility!
state_dict = {}
for k, v in raw_sd.items():
    nk = (
        k.replace(".attention.", ".attn.")
         .replace(".feed_forward.", ".ffn.")
         .replace(".attention_norm.", ".attn_norm.")
         .replace("norm.weight", "final_norm.weight")
    )
    state_dict[nk] = v


# ───────────── Build model & load weights ─────────────────────────
with open(CKPT_DIR / "params.json") as f:
    params = json.load(f)
params.pop("use_scaled_rope", None)  # older checkpoints may include this flag

model_args = ModelArgs(max_seq_len=MAX_SEQ_LEN, max_batch_size=MAX_BATCH, **params)
logger.debug(f"ModelArgs: {model_args}")

logger.debug("Initializing tokenizer …")
tokenizer = Tokenizer(model_path=str(TOKENIZER_PATH))

logger.debug("Constructing Transformer …")
model = Transformer(model_args).to(DEVICE)
param_total = sum(p.numel() for p in model.parameters())
logger.debug(f"Model instantiated ({param_total:,} parameters)")

# print([k for k in model.state_dict().keys() if 'norm' in k])


logger.debug("Loading checkpoint weights …")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
logger.debug(f"Loaded weights: missing={len(missing)}, unexpected={len(unexpected)}")
if missing or unexpected:
    print("Missing:", missing)
    print("Unexpected:", unexpected)

# ───────────── Helper: forward a SINGLE block ────────────────────
@torch.inference_mode()
def run_one_block(model: Transformer, tokens: torch.Tensor, *, start_pos: int = 0, block_idx: int = 0):
    """Replicates embedding + mask logic, then forwards exactly one TransformerBlock."""
    B, T = tokens.shape
    h = model.tok_embeddings(tokens)                     # (B, T, d)
    freqs = model.freqs_cis[start_pos:start_pos+T].to(h.device)

    mask = None
    if T > 1:
        causal = torch.full((T, T), float("-inf"), device=tokens.device)
        causal = torch.triu(causal, diagonal=1)
        prefix = torch.zeros((T, start_pos), device=tokens.device)
        mask = torch.hstack([prefix, causal]).type_as(h)

    block = model.layers[block_idx]
    return block(h, start_pos, freqs, mask)

# ───────────── Perfetto trace of the chosen block ────────────────
import tg4perfetto
trace_path = TRACE_DIR / f"block{BLOCK_INDEX}_trace_rank{global_rank}_{int(time.time())}.json"
logger.debug(f"Tracing ONE block → {trace_path}")

with tg4perfetto.open(trace_path):
    with torch.inference_mode():
        prompt = torch.tensor([[tokenizer.bos_id]], device=DEVICE)
        _ = run_one_block(model, prompt, start_pos=0, block_idx=BLOCK_INDEX)

logger.debug("Perfetto trace finished ✅")
print("Trace written to:", trace_path)

# ───────────── Shutdown ──────────────────────────────────────────
dist.destroy_process_group()
logger.debug("Process group destroyed – exiting.")
