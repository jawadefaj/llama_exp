# LLaMA-3.2 × Perfetto Tracing & Profiling  
*Progress snapshot – 16 May 2025*

---

## 🎯 Project Goal  
Visualise **one transformer block** of any LLaMA-3.2 checkpoint with **Perfetto**, exposing every key compute & memory step (dot-product, multi-pass soft-max, RoPE, layer norms, residual adds, KV-cache I/O, MLP, …) and embedding rich tensor-metadata in the trace.

---

## 📁 Repository Layout

./
├── llama/ # Minimal fork with always-on Perfetto hooks
│ ├── generation.py # Text-completion wrapper
│ ├── model.py # Transformer + fine-grained tracing
│ └── tokenizer.py # tiktoken-based
├── example_text_completion.py # Local .pth loader demo
├── load_model_and_simulate_perfetto.py # HF loader + custom trace
├── load_model_from_weight.py # CPU-only .pth loader
├── pytorch_profiler.py # TensorBoard-compatible trace
├── test_trace.py # Tiny MLP sanity-check
└── setup.sh # One-shot env bootstrap

yaml
Copy
Edit

---

## ✅ Current Progress

| # | Requirement | Status | Notes |
|---|-------------|--------|-------|
| **1** | Load model from Hugging Face **safetensors** and local **`.pth`** shards | **✔ done** | GGUF loader **TBD** |
| **2** | Auto-generate **Perfetto trace** for one block | **✔ done** | Script still mocks Q·Kᵀ & soft-max timing |
| **3** | Trace **adapts to any model size / dims** | **✔ done** | Will inherit GGUF once loader added |
| **4** | Annotate each span with **op-type + I/O tensor metadata** | **▲ partial** | LayerNorm, RoPE, embedding to add |

Legend — **✔ done**, **▲ partial**, **✗ todo**

---

## 🔧 Quick Start

```bash
# 1  Create / activate a Python env, then:
bash setup.sh          # installs transformers, tensorboard, …

# 2  Sanity-check: local .pth shards text-completion
python example_text_completion.py

# 3  Generate Perfetto JSON trace on Hugging Face checkpoint (layer 0)
python load_model_and_simulate_perfetto.py
# → writes   llama_block_trace_<model>_<timestamp>.json
# → open trace.perfetto.dev and drop the file

# 4  Optional: TensorBoard profile (HF model)
python pytorch_profiler.py
tensorboard --logdir ./log/hf_llama_block_trace
🚧 Next Steps
GGUF loader – integrate llama_cpp / ggml bindings for quantised CPU models.

Replace placeholders with real compute – call actual matmuls & soft-max to capture true durations.

Trace completeness – add tracks & metadata for token embedding, RoPE, LayerNorm, residual adds.

Packaging – turn scripts into a small CLI:
python -m llama_trace --model meta-llama/... --layer 0.

✍️ Contributing
PRs welcome — particularly for GGUF support and fuller op-coverage.
Run black + ruff before submitting.