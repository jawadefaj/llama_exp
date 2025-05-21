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
import psutil

import torch
import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init

# ───── Debug Logger Setup ─────
logging.basicConfig(level=logging.DEBUG,
                    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)
logger.debug("===== Starting load_model_from_weight_distributed script =====")

# ───── 0. CUDA patch for CPU-only ─────
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *a, **k: self
    logger.debug("CUDA not available: patched torch.Tensor.cuda to no-op.")

# ───── 0.5. Log hardware resources ─────
cpu_count = os.cpu_count()
gpu_count = torch.cuda.device_count()
total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
logger.debug(f"Hardware resources detected: CPUs={cpu_count}, GPUs={gpu_count}, RAM={total_ram_gb:.1f} GB")
if gpu_count > 0:
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        logger.debug(f" GPU {i}: {props.name}, {props.total_memory / (1024 ** 3):.1f} GB")

# ───── 1. Distributed init & device selection ─────
backend = "nccl" if torch.cuda.is_available() else "gloo"
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

local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
world_size  = dist.get_world_size()
global_rank = dist.get_rank()

if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device("cuda", local_rank)
else:
    DEVICE = torch.device("cpu")

logger.debug(f"Process group initialized: rank {global_rank}/{world_size}, device={DEVICE}")

# ───── 2. Model-parallel initialization ─────
fs_init.initialize_model_parallel(1)
logger.debug(f"Model parallel initialized with world_size={world_size}")

# ───── 3. Paths & hyperparameters ─────
SCRIPT_ROOT     = Path(__file__).resolve().parent
CKPT_DIR        = SCRIPT_ROOT / "model" / "Llama3.2-1B"
TOKENIZER_PATH  = CKPT_DIR / "tokenizer.model"
MAX_SEQ_LEN     = 128
MAX_BATCH_SIZE  = 1

logger.debug(f"Checkpoint directory: {CKPT_DIR}")
logger.debug(f"Tokenizer path:      {TOKENIZER_PATH}")
logger.debug(f"Max seq len:         {MAX_SEQ_LEN}")
logger.debug(f"Max batch size:      {MAX_BATCH_SIZE}")

# ───── 4. Late imports ─────
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
logger.debug("Imported ModelArgs, Transformer, and Tokenizer classes")

# ───── 5. Load checkpoint & params ─────
ckpt_files = sorted(CKPT_DIR.glob("*.pth"))
assert ckpt_files, f"No .pth files found in {CKPT_DIR}"
if len(ckpt_files) == world_size:
    ckpt_file = ckpt_files[global_rank]
else:
    ckpt_file = ckpt_files[0]
logger.debug(f"Selected checkpoint shard for rank {global_rank}: {ckpt_file.name}")

raw_sd = torch.load(ckpt_file, map_location=DEVICE)
logger.debug(f"Loaded raw state_dict with {len(raw_sd)} tensors; sample keys: {list(raw_sd.keys())[:5]}")

# remap keys to match model.py naming conventions
state_dict = {}
for old_k, v in raw_sd.items():
    new_k = old_k.replace('.attention.wq.',  '.attn.q_proj.') \
                 .replace('.attention.wk.',  '.attn.k_proj.') \
                 .replace('.attention.wv.',  '.attn.v_proj.') \
                 .replace('.attention.wo.',  '.attn.o_proj.') \
                 .replace('.feed_forward.w1.', '.ffn.w1.') \
                 .replace('.feed_forward.w2.', '.ffn.w2.') \
                 .replace('.feed_forward.w3.', '.ffn.w3.') \
                 .replace('.attention_norm.',   '.attn_norm.')
    state_dict[new_k] = v
logger.debug(f"Remapped state_dict keys; sample: {list(state_dict.keys())[:5]}")

with open(CKPT_DIR / "params.json") as pf:
    params = json.load(pf)
params.pop("use_scaled_rope", None)
logger.debug(f"Loaded params.json: {params}")

model_args = ModelArgs(
    max_seq_len=MAX_SEQ_LEN,
    max_batch_size=MAX_BATCH_SIZE,
    **params,
)
logger.debug(f"Constructed ModelArgs: {model_args}")

# ───── 6. Build model & load weights ─────
logger.debug("Initializing tokenizer...")
tokenizer = Tokenizer(model_path=str(TOKENIZER_PATH))
logger.debug("Tokenizer ready.")

logger.debug("Building Transformer model...")
model = Transformer(model_args).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
logger.debug(f"Transformer instantiated with {total_params:,} parameters.")

logger.debug("Loading weights into model...")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
logger.debug(f"Weight loading complete: {len(missing)} missing, {len(unexpected)} unexpected keys.")

# ───── 7. Sanity check + Perfetto trace ─────
import tg4perfetto
os.makedirs("./log", exist_ok=True)
trace_file = f"./log/llama_trace_rank{global_rank}_{int(time.time())}.json"
logger.debug(f"Starting Perfetto trace -> {trace_file}")

with tg4perfetto.open(trace_file):
    with torch.inference_mode():
        x = torch.tensor([[tokenizer.bos_id]], device=DEVICE)
        _ = model(x, start_pos=0)

logger.debug("Perfetto trace complete")
print("Trace written to:", trace_file)

# ───── 8. Cleanup ─────
dist.destroy_process_group()
logger.debug("Process group destroyed, exiting script.")
