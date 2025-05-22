#!/usr/bin/env python3
import os
os.environ.setdefault('USE_LIBUV', '0')
import logging
from pathlib import Path
import time
import json
import psutil

import torch
import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init

# ── Config ──
BLOCK_INDEX = int(os.environ.get('LLAMA_TRACE_BLOCK', '0'))
TRACE_DIR = Path('./log')
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# ── Logger ──
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)
logger.debug('Starting inference-first-block')

# ── Hardware logs ──
cpu_count = psutil.cpu_count(logical=True)
gpu_count = torch.cuda.device_count()
ram_gb    = psutil.virtual_memory().total / (1024**3)
logger.debug(f'Hardware: CPUs={cpu_count}, GPUs={gpu_count}, RAM={ram_gb:.1f}GB')
if gpu_count:
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        logger.debug(f'  GPU{i}: {props.name}, {props.total_memory/1024**3:.1f}GB')

# ── Distributed init ──
backend     = 'nccl' if torch.cuda.is_available() else 'gloo'
init_method = os.environ.get('INIT_METHOD', 'tcp://127.0.0.1:29500')
dist.init_process_group(
    backend=backend,
    init_method=init_method,
    rank=int(os.environ.get('RANK','0')),
    world_size=int(os.environ.get('WORLD_SIZE','1'))
)
local_rank = int(os.environ.get('LOCAL_RANK','0'))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
logger.debug(f'Rank init: local={local_rank}, device={device}')

# ── Model-parallel ──
fs_init.initialize_model_parallel(1)

# ── Paths & Params ──
ROOT           = Path(__file__).parent
CKPT_DIR       = ROOT / 'model' / 'Llama-3.2-1B'
TOKENIZER_PATH = CKPT_DIR / 'tokenizer.model'
logger.debug(f'Checkpoint dir: {CKPT_DIR}')
logger.debug(f'Tokenizer path: {TOKENIZER_PATH}')

# ── Load model & tokenizer ──
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

# binary checkpoint files (*.pth)
ckpts = sorted(CKPT_DIR.glob('*.pth'))
ckpt = ckpts[dist.get_rank()] if len(ckpts)==dist.get_world_size() else ckpts[0]
logger.debug(f'Loading weights from {ckpt.name}')
sd_raw = torch.load(ckpt, map_location=device)
# key remapping if needed
sd = {}
for k,v in sd_raw.items():
    nk = k.replace('.attention.wq.', '.attn.q_proj.')\
          .replace('.attention.wk.', '.attn.k_proj.')\
          .replace('.attention.wv.', '.attn.v_proj.')\
          .replace('.attention.wo.', '.attn.o_proj.')
    sd[nk] = v

with open(CKPT_DIR/'params.json') as f:
    params = json.load(f)
params.pop('use_scaled_rope', None)
model_args = ModelArgs(max_seq_len=128, max_batch_size=1, **params)

tokenizer = Tokenizer(str(TOKENIZER_PATH))
model = Transformer(model_args).to(device)
missing, unexpected = model.load_state_dict(sd, strict=False)
logger.debug(f'Loaded weights: missing={len(missing)}, unexpected={len(unexpected)}')

# ── Single-block tracer ──
def run_block(tokens: torch.Tensor, start_pos=0, block_idx=0):
    h = model.tok_embeddings(tokens)
    freqs = model.freqs_cis[start_pos:start_pos + tokens.size(1)].to(h.device)
    mask = None
    if tokens.size(1) > 1:
        causal = torch.full((tokens.size(1), tokens.size(1)), float('-inf'), device=device)
        causal = torch.triu(causal, 1)
        prefix = torch.zeros((tokens.size(1), start_pos), device=device)
        mask = torch.hstack([prefix, causal]).type_as(h)
    return model.layers[block_idx](h, start_pos, freqs, mask)

import tg4perfetto
out = TRACE_DIR / f'block{BLOCK_INDEX}_trace_rank{dist.get_rank()}_{int(time.time())}.json'
logger.debug(f'Tracing ⇒ {out}')
with tg4perfetto.open(out):
    with torch.inference_mode():
        prompt = torch.tensor([[tokenizer.bos_id]], device=device)
        _ = run_block(prompt, start_pos=0, block_idx=BLOCK_INDEX)
logger.info(f'Trace saved: {out}')

dist.destroy_process_group()
# ── End of script ──