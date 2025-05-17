# LLaMA-3.2 × Perfetto Tracing & Profiling  
*Progress snapshot – 16 May 2025*

---

## 🎯 Project Goal  
Visualise **one transformer block** of any LLaMA-3.2 checkpoint with **Perfetto**, capturing every key compute & memory step (dot-product, multi-pass soft-max, RoPE, layer norms, residual adds, KV-cache I/O, MLP, …) and embedding rich tensor-metadata in the trace.

---

## ✅ Current Progress

| # | Requirement | Status |
|---|-------------|:------:|
| **1** | Load model from Hugging Face **safetensors** and local **`.pth`** shards | **✔** |
| **2** | Auto-generate **Perfetto trace** for one block | **✔** |
| **3** | Trace **adapts to any model size / dims** | **✔** |
| **4** | Annotate each span with **op-type + I/O tensor metadata** | **▲** |

Legend — **✔ done**, **▲ partial**, **✗ todo**

---

## 🔧 Quick Start

```bash
# 1 · Create / activate a Python env
bash setup.sh

# 2 · Download weights into the models/ directory
#    (example for 1-B checkpoints)
huggingface-cli download meta-llama/Llama-3.2-1B \
    --local-dir ./models/Llama-3.2-1B

# 3 · Sanity-check – local .pth shards text-completion
python example_text_completion.py

# 4 · Generate Perfetto JSON trace on a HF checkpoint (layer 0)
python load_model_and_simulate_perfetto.py
# ↪ writes  llama_block_trace_<model>_<timestamp>.json
# ↪ open https://trace.perfetto.dev and drop the file

# 5 · (Optional) TensorBoard profile
python pytorch_profiler.py
tensorboard --logdir ./log/hf_llama_block_trace
