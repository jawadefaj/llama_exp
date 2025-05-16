# model_load_cpu.py
# Loads Llama-3.2-1B on a single-CPU machine.

from pathlib import Path
import time, json, torch, torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init

# ───── 0. make .cuda() a no-op when CUDA is absent ─────
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *a, **k: self

# ───── 1. paths & small hyper-params ─────
CKPT_DIR = Path(r"C:\Users\abjaw\OneDrive\Documents\GitHub\llama_exp\Llama-3.2-1B\original")
TOKENIZER_PATH = CKPT_DIR / "tokenizer.model"
MAX_SEQ_LEN, MAX_BATCH_SIZE = 128, 4
DEVICE = torch.device("cpu")

# ───── 2. create a size-1 process group (no env vars needed) ─────
if not dist.is_initialized():
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",   # loopback, single process
        rank=0,
        world_size=1,
    )
fs_init.initialize_model_parallel(1)           # positional arg, NOT keyword

# ───── 3. late imports (after patch & PG init) ─────
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

# ───── 4. load checkpoint & JSON params ─────
ckpt_files = sorted(CKPT_DIR.glob("*.pth"))
assert ckpt_files, f"No *.pth in {CKPT_DIR}"
state_dict = torch.load(ckpt_files[0], map_location="cpu")

with open(CKPT_DIR / "params.json") as f:
    params = json.load(f)
params.pop("use_scaled_rope", None)

model_args = ModelArgs(
    max_seq_len=MAX_SEQ_LEN,
    max_batch_size=MAX_BATCH_SIZE,
    **params,
)

# ───── 5. build model & load weights ─────
tokenizer = Tokenizer(model_path=str(TOKENIZER_PATH))
print("Tokenizer ready")

start = time.time()
model = Transformer(model_args).to(DEVICE)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Loaded in {time.time() - start:.2f}s  "
      f"({len(missing)} missing / {len(unexpected)} unexpected keys)")

# ───── 6. tiny sanity check ─────
with torch.inference_mode():
    x = torch.tensor([[tokenizer.bos_id]], device=DEVICE)
    logits = model(x, start_pos=0)
print("Logits OK – shape", logits.shape)

# ───── 7. tidy up ─────
dist.destroy_process_group()


# # ───── 6. use input "Hello World" ─────
# with torch.inference_mode():
#     input_text = "Hello World"
#     input_ids = tokenizer.encode(input_text, bos=True, eos=False)
#     print("Input IDs:", input_ids)
#     x = torch.tensor([input_ids], device=DEVICE)
#     logits = model(x, start_pos=0)
# print("Logits OK – shape", logits.shape)

# ───── 8. run only the first transformer block ─────
with torch.inference_mode():
    # ➊ Tokenise “Hello World”
    input_text = "Hello World"
    tokens = torch.tensor(
        [tokenizer.encode(input_text, bos=True, eos=False)],
        device=DEVICE,
    )                                     # shape = (1, seq_len)
    start_pos = 0
    seq_len   = tokens.shape[1]

    # ➋ Embed tokens (same as model.forward does)
    h = model.tok_embeddings(tokens)      # (1, seq_len, dim)

    # ➌ Rotary-embedding lookup for this slice of the sequence
    freqs_cis = model.freqs_cis[start_pos : start_pos + seq_len].to(h.device)

    # ➍ Causal-attention mask (identical logic to full forward pass)
    if seq_len > 1:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=DEVICE)
        mask = torch.triu(mask, diagonal=1)
        mask = torch.hstack(
            [torch.zeros((seq_len, start_pos), device=DEVICE), mask]
        ).type_as(h)
    else:
        mask = None

    # ➎ Grab layer 0 and run it
    first_block = model.layers[0]         # TransformerBlock(0)
    block_out   = first_block(h, start_pos, freqs_cis, mask)

print("First-block output shape:", block_out.shape)

