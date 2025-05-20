# load_model_from_weight_distributed.py
# Adapted for multi-GPU/multi-CPU with Perfetto tracing per rank
# Includes workaround for libuv error on Windows by disabling USE_LIBUV

import os
# Disable libuv rendezvous on platforms without libuv support
os.environ.setdefault('USE_LIBUV', '0')

import logging
from pathlib import Path
import time
import json

import torch
import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init

# ───── Debug Logger Setup ─────
logging.basicConfig(level=logging.DEBUG,
                    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)
logger.debug("Starting load_model_from_weight_distributed script.")

# ───── 0. CUDA patch for CPU-only ─────
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *a, **k: self
    logger.debug("CUDA not available: patched Tensor.cuda to no-op.")

# ───── 1. Distributed init & device selection ─────
backend = "nccl" if torch.cuda.is_available() else "gloo"
# Use TCP rendezvous on platforms without libuv (e.g. Windows)
if os.environ.get('USE_LIBUV','0') == '0':
    # fallback to TCP loopback rendezvous
    dist.init_process_group(
        backend=backend,
        init_method="tcp://127.0.0.1:29500",
        rank=int(os.environ.get("RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )
else:
    dist.init_process_group(backend=backend, init_method="env://")
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = dist.get_world_size()
global_rank = dist.get_rank()

if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device("cuda", local_rank)
else:
    DEVICE = torch.device("cpu")

logger.debug(f"Initialized process group: rank {global_rank}/{world_size}, using device {DEVICE}")

# ───── 2. Model-parallel initialization ─────
fs_init.initialize_model_parallel(world_size)
logger.debug(f"Model parallel initialized with size={world_size}")





# ───── 3. Paths & hyperparameters ─────
CKPT_DIR = Path(r"C:\Users\abjaw\OneDrive\Documents\GitHub\llama_exp\models\Llama-3.2-1B\original")
TOKENIZER_PATH = CKPT_DIR / "tokenizer.model"
MAX_SEQ_LEN, MAX_BATCH_SIZE = 128, 1
logger.debug(f"Checkpoint directory: {CKPT_DIR}")
logger.debug(f"max_seq_len={MAX_SEQ_LEN}, max_batch_size={MAX_BATCH_SIZE}")

# ───── 4. Late imports ─────
from llama._model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
logger.debug("Imported Transformer and Tokenizer modules")

# ───── 5. Load checkpoint & params ─────
ckpt_files = sorted(CKPT_DIR.glob("*.pth"))
assert ckpt_files, f"No *.pth in {CKPT_DIR}"
# choose shard per rank if sharded, else use first
if len(ckpt_files) == world_size:
    ckpt_file = ckpt_files[global_rank]
else:
    ckpt_file = ckpt_files[0]
logger.debug(f"Using checkpoint file: {ckpt_file}")

raw_sd = torch.load(ckpt_file, map_location=DEVICE)
logger.debug("Original checkpoint keys sample: %s", list(raw_sd.keys())[:5])

# remap keys to match model.py naming
state_dict = {}
for k, v in raw_sd.items():
    nk = k
    nk = nk.replace('.attention.wq.', '.attn.q_proj.')
    nk = nk.replace('.attention.wk.', '.attn.k_proj.')
    nk = nk.replace('.attention.wv.', '.attn.v_proj.')
    nk = nk.replace('.attention.wo.', '.attn.o_proj.')
    nk = nk.replace('.feed_forward.w1.', '.ffn.w1.')
    nk = nk.replace('.feed_forward.w2.', '.ffn.w2.')
    nk = nk.replace('.feed_forward.w3.', '.ffn.w3.')
    nk = nk.replace('.attention_norm.', '.attn_norm.')
    state_dict[nk] = v
logger.debug("Remapped checkpoint keys sample: %s", list(state_dict.keys())[:5])

with open(CKPT_DIR / "params.json") as pf:
    params = json.load(pf)
logger.debug(f"Loaded params.json keys: {list(params.keys())}")
params.pop("use_scaled_rope", None)

model_args = ModelArgs(
    max_seq_len=MAX_SEQ_LEN,
    max_batch_size=MAX_BATCH_SIZE,
    **params,
)
logger.debug(f"Instantiated ModelArgs: {model_args}")

# ───── 6. Build model & load weights ─────
tokenizer = Tokenizer(model_path=str(TOKENIZER_PATH))
logger.debug("Tokenizer initialized")

model = Transformer(model_args).to(DEVICE)
logger.debug("Transformer model instantiated")

missing, unexpected = model.load_state_dict(state_dict, strict=False)
logger.debug(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
print(f"Loaded in {time.time() - time.time():.2f}s  ({len(missing)} missing / {len(unexpected)} unexpected keys)")

# ───── 7. Sanity check + Perfetto trace ─────
import tg4perfetto, os
os.makedirs("./log", exist_ok=True)
trace_file = f"./log/llama_trace_rank{global_rank}_{int(time.time())}.json"
logger.debug(f"Writing Perfetto trace to {trace_file}")

with tg4perfetto.open(trace_file):
    with torch.inference_mode():
        x = torch.tensor([[tokenizer.bos_id]], device=DEVICE)
        logits = model(x, start_pos=0)
logger.debug("Sanity forward done")
print("Trace written to:", trace_file)

# ───── 8. Cleanup ─────
dist.destroy_process_group()
logger.debug("Destroyed process group")
