# Minimal text-completion example without python-fire
# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path
from typing import List

import torch
from llama import Llama

# ─── 1. Hard-coded paths & params ───
CKPT_DIR = Path(r"C:\Users\abjaw\OneDrive\Documents\GitHub\llama_exp\models\Llama-3.2-1B\original")
TOKENIZER_PATH = CKPT_DIR / "tokenizer.model"
MAX_SEQ_LEN   = 128
MAX_BATCH_SIZE = 4
DEVICE = torch.device("cpu")      # keeps everything on CPU

# ─── 2. Build generator ───
generator = Llama.build(
    ckpt_dir=str(CKPT_DIR),
    tokenizer_path=str(TOKENIZER_PATH),
    max_seq_len=MAX_SEQ_LEN,
    max_batch_size=MAX_BATCH_SIZE,
)

# ─── 3. Prompts & inference ───
prompts: List[str] = [
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
    """A brief message congratulating the team on the launch:

Hi everyone,

I just """,
    """Translate English to French:

sea otter => loutre de mer
peppermint => menthe poivrée
plush girafe => girafe peluche
cheese =>""",
]

results = generator.text_completion(
    prompts,
    temperature=0.6,
    top_p=0.9,
    max_gen_len=64,
)

for prompt, result in zip(prompts, results):
    print(prompt)
    print("> " + result["generation"])
    print("=" * 40)
