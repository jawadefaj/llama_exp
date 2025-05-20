# load_model_from_weight.py
# Loads Llama-3.2-1B on a single-CPU machine with detailed debug logging.

import logging
from pathlib import Path
import time
import json
import torch
import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init

# ───── Debug Logger Setup ─────
logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

logger.debug("Starting load_model_from_weight script.")

# ───── 0. make .cuda() a no-op when CUDA is absent ─────
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *a, **k: self
    logger.debug("CUDA not available: patched Tensor.cuda to no-op.")

# ───── 1. paths & small hyper-params ─────
CKPT_DIR = Path(r"C:\Users\abjaw\OneDrive\Documents\GitHub\llama_exp\models\Llama-3.2-1B\original")
TOKENIZER_PATH = CKPT_DIR / "tokenizer.model"
MAX_SEQ_LEN, MAX_BATCH_SIZE = 128, 1  # Set batch size to 1 for single-sample inference
DEVICE = torch.device("cpu")
logger.debug(f"Checkpoint directory: {CKPT_DIR}")
logger.debug(f"Using device: {DEVICE}, max_seq_len={MAX_SEQ_LEN}, max_batch_size={MAX_BATCH_SIZE}")

# ───── 2. create a size-1 process group (no env vars needed) ─────
if not dist.is_initialized():
    logger.debug("Initializing distributed process group...")
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",
        rank=0,
        world_size=1,
    )
    logger.debug("Process group initialized.")
# positional arg for model_parallel_size
fs_init.initialize_model_parallel(1)
logger.debug("Model parallel initialized with size=1.")






# ───── 3. late imports (after patch & PG init) ─────
from llama._model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
logger.debug("Imported Transformer and Tokenizer modules.")

# ───── 4. load checkpoint & JSON params ─────
ckpt_files = sorted(CKPT_DIR.glob("*.pth"))
assert ckpt_files, f"No *.pth in {CKPT_DIR}"
logger.debug(f"Found checkpoint files: {ckpt_files}")
state_dict = torch.load(ckpt_files[0], map_location="cpu")
logger.debug("Original checkpoint keys sample: %s", list(state_dict.keys())[:5])
# ───── Remap checkpoint keys to match model naming ─────
new_sd = {}
for k,v in state_dict.items():
    nk = k
    nk = nk.replace('.attention.wq.', '.attn.q_proj.')
    nk = nk.replace('.attention.wk.', '.attn.k_proj.')
    nk = nk.replace('.attention.wv.', '.attn.v_proj.')
    nk = nk.replace('.attention.wo.', '.attn.o_proj.')
    nk = nk.replace('.feed_forward.w1.', '.ffn.w1.')
    nk = nk.replace('.feed_forward.w2.', '.ffn.w2.')
    nk = nk.replace('.feed_forward.w3.', '.ffn.w3.')
    nk = nk.replace('.attention_norm.', '.attn_norm.')
    new_sd[nk] = v
state_dict = new_sd
logger.debug("Remapped checkpoint keys for compatibility. New keys sample: %s", list(state_dict.keys())[:5])
logger.debug(f"Loaded state_dict from {ckpt_files[0]}")

with open(CKPT_DIR / "params.json") as f:
    params = json.load(f)
logger.debug(f"Loaded params.json: {params.keys()}")
# drop any obsolete flag
params.pop("use_scaled_rope", None)

model_args = ModelArgs(
    max_seq_len=MAX_SEQ_LEN,
    max_batch_size=MAX_BATCH_SIZE,
    **params,
)
logger.debug(f"Instantiated ModelArgs: {model_args}")






# ───── 5. build model & load weights ─────
tokenizer = Tokenizer(model_path=str(TOKENIZER_PATH))
logger.debug("Tokenizer initialized.")

start = time.time()
model = Transformer(model_args).to(DEVICE)
logger.debug("Transformer model instantiated.")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
logger.debug(f"Model.load_state_dict completed: {len(missing)} missing, {len(unexpected)} unexpected.")
print(f"Loaded in {time.time() - start:.2f}s  ({len(missing)} missing / {len(unexpected)} unexpected keys)")









# ───── 6. tiny sanity check with Perfetto trace ─────
import tg4perfetto, os
os.makedirs("./log", exist_ok=True)
trace_path = f"./log/llama_trace_{int(time.time())}.json"
logger.debug(f"Preparing to write Perfetto trace to {trace_path}")

with tg4perfetto.open(trace_path):
    with torch.inference_mode():
        x = torch.tensor([[tokenizer.bos_id]], device=DEVICE)
        logger.debug(f"Running sanity forward with bos_id={tokenizer.bos_id}")
        logits = model(x, start_pos=0)
logger.debug("Sanity forward done.")
print("Trace written to:", trace_path)

# ───── 7. tidy up dist group ─────
dist.destroy_process_group()
logger.debug("Destroyed distributed process group.")

# ───── 8. run only the first transformer block ─────
with torch.inference_mode():
    input_text = "Hello World"
    logger.debug(f"Tokenizing input text: '{input_text}'")
    tokens = torch.tensor(
        [tokenizer.encode(input_text, bos=True, eos=False)],
        device=DEVICE,
    )  # shape = (1, seq_len)
    start_pos = 0
    seq_len = tokens.shape[1]
    logger.debug(f"Encoded tokens (len={seq_len}): {tokens.tolist()}")

    # Embedding
    h = model.tok_embeddings(tokens)
    logger.debug(f"Obtained token embeddings with shape {h.shape}")

    # Rotary frequencies slice
    freqs_cis = model.freqs_cis[start_pos : start_pos + seq_len].to(h.device)
    logger.debug("Obtained rotary frequency cis tensor.")

    # Causal mask
    if seq_len > 1:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=DEVICE)
        mask = torch.triu(mask, diagonal=1)
        prefix = torch.zeros((seq_len, start_pos), device=DEVICE)
        mask = torch.hstack([prefix, mask]).type_as(h)
        logger.debug("Constructed causal attention mask.")
    else:
        mask = None
        logger.debug("Sequence length <=1, skipping mask construction.")

    # First block forward
    first_block = model.layers[0]
    logger.debug("Starting first transformer block forward pass.")
    block_out = first_block(h, start_pos, freqs_cis, mask)
    logger.debug(f"First-block output shape: {block_out.shape}")

# Ensure output matches batch size
block_out = block_out[:1]
logger.debug(f"Sliced block_out to match batch: {block_out.shape}")
print("First-block output shape:", block_out.shape)
