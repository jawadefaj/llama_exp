#!/usr/bin/env python
# pytorch_profiler.py
#
# Profile the first decoder block of Llama-3.2-1B with PyTorch-Profiler
# and write the trace in TensorBoard-compatible format.

import os
from datetime import datetime

import torch
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


# ---------------------------------------------------------------------
# 1.  Load model & tokenizer
# ---------------------------------------------------------------------
model_id = "meta-llama/Llama-3.2-1B"
# model_id = "meta-llama/Llama-3.2-3B"
print(f"🔄  Loading model …  ({model_id})")
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"✅  Model loaded ({n_params:.2f} M parameters)\n")

# ---------------------------------------------------------------------
# 2.  Prepare a tiny input and pull out the first decoder layer
# ---------------------------------------------------------------------
input_text = "Hello"  # one-token example keeps the trace minimal
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
inputs_embeds = model.model.embed_tokens(input_ids)          # (1, L, hidden)
block = model.model.layers[0]

# Rotary embeddings – ‼️ pass BOTH arguments (x **and** position_ids)
rotary = LlamaRotaryEmbedding(config=model.config)
position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
cos, sin = rotary(inputs_embeds, position_ids)             

# ---------------------------------------------------------------------
# 3.  Trace setup
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logdir = f"./log/hf_llama_block_trace/run_{timestamp}"
os.makedirs(logdir, exist_ok=True)
print(f"📂  Writing profiler trace to: {logdir}\n")

# ---------------------------------------------------------------------
# 4.  Profile one block for a few dummy steps
# ---------------------------------------------------------------------
with profile(
    activities=[ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler(logdir),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof, torch.no_grad():

    for step in range(5):
        print(f"-- Step {step + 1} --")

        with record_function("TransformerBlock"):
            # (a) Input LayerNorm
            x = block.input_layernorm(inputs_embeds)

            # (b) Self-Attention
            attn_out, _ = block.self_attn(
                hidden_states=x,
                position_embeddings=(cos, sin),   # Llama3’s RoPE tensors
                past_key_value=None,
                attention_mask=None,
                output_attentions=False,
                use_cache=False,
            )
            h = x + attn_out  # residual 1

            # (c) Post-Attention LayerNorm
            x = block.post_attention_layernorm(h)

            # (d) Gated-MLP
            gate = block.mlp.gate_proj(x)
            up = block.mlp.up_proj(x)
            act = torch.nn.functional.silu(gate) * up          # correct gating
            down = block.mlp.down_proj(act)

            # (e) Residual 2
            x = h + down

        prof.step()  # advance the profiler’s internal clock

# ---------------------------------------------------------------------
# 5.  Done!
# ---------------------------------------------------------------------
print("\n🎉  Profiling complete.")
print(f"➡️   Run: tensorboard --logdir={logdir}")
print("     then open http://localhost:6006 in your browser.\n")
