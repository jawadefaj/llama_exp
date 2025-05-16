import torch
import tg4perfetto
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datetime import datetime

# Load model and tokenizer
model_id = "meta-llama/Llama-3.2-1B"
# model_id = "meta-llama/Llama-3.2-3B"

  
print(f"Loading model: {model_id}")


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
block = model.model.layers[0]
print("Model and tokenizer loaded.\n")

# Prepare input
input_ids = tokenizer("Hello", return_tensors="pt").input_ids
inputs_embeds = model.model.embed_tokens(input_ids)
print(f"Input embeddings shape: {inputs_embeds.shape}")

# create a json_file name based on model id, and current system time
# Sanitize model_id for filename
safe_model_id = model_id.replace("/", "_").replace(".", "_")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_file_name = f"llama_block_trace_{safe_model_id}_{timestamp}.json"

# Create named tracks for tracing
track_embed = tg4perfetto.track("Embed")
track_norm1 = tg4perfetto.track("Input_LayerNorm")
track_qkv = tg4perfetto.track("QKV_Projections")
track_softmax = tg4perfetto.track("Softmax")
track_attn_out = tg4perfetto.track("Attn_Output")
track_norm2 = tg4perfetto.track("PostAttn_LayerNorm")
track_mlp = tg4perfetto.track("MLP")

with tg4perfetto.open(json_file_name):
    with torch.no_grad():
        with tg4perfetto.track("TransformerBlock").trace("Forward Pass"):
            
            # Step 1: Embedding
            with track_embed.trace("Embed_Tokens"):
                x = inputs_embeds

            # Step 2: Input LayerNorm
            with track_norm1.trace("Input_LayerNorm"):
                x = block.input_layernorm(x)

            # Step 3: Q/K/V projections
            with track_qkv.trace("Q_Projection"):
                q = block.self_attn.q_proj(x)
            with track_qkv.trace("K_Projection"):
                k = block.self_attn.k_proj(x)
            with track_qkv.trace("V_Projection"):
                v = block.self_attn.v_proj(x)

            # Simulate attention steps
            with track_softmax.trace("DotProduct_QK^T"):
                pass  # dot(Q, K^T)
            with track_softmax.trace("Softmax_Pass1_MaxSub"):
                pass
            with track_softmax.trace("Softmax_Pass2_Exp"):
                pass
            with track_softmax.trace("Softmax_Pass3_Normalize"):
                pass
            with track_softmax.trace("Attention_Weighted_Sum"):
                attn_output = block.self_attn.o_proj(q)  # simulating reuse

            # Step 4: Residual Add (simulated) + Norm
            with track_attn_out.trace("Output_Projection"):
                x = attn_output
            with track_norm2.trace("PostAttention_LayerNorm"):
                x = block.post_attention_layernorm(x)

            # Step 5: MLP sublayers
            with track_mlp.trace("Gate_Proj"):
                gate = block.mlp.gate_proj(x)
            with track_mlp.trace("Up_Proj"):
                up = block.mlp.up_proj(x)
            with track_mlp.trace("SiLU_Activation"):
                act = gate * torch.nn.functional.silu(gate)
            with track_mlp.trace("Down_Proj"):
                down = block.mlp.down_proj(act)
print("file name: ", json_file_name)
print("✅ written successfully.")
