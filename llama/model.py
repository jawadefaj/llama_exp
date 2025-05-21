# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Llama 3 Community License Agreement.
# ---------------------------------------------------------------------------
# llama/model.py  •  CPU & GPU compatible + fine-grained Perfetto tracing
# ---------------------------------------------------------------------------
#   ▹ Four compute-engine tracks per device (GPU0-Engine-0 … GPU0-Engine-3)
#   ▹ Chunked Q/K/V projections → 4 slices in parallel
#   ▹ Per-head attention:  QK matmul → 3-pass soft-max → value matmul
#   ▹ Each slice carries rich metadata (op-type, I/O tensors, sizes, dtypes…)
# ---------------------------------------------------------------------------

from __future__ import annotations
import math, time, contextlib
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Generator

import torch
import torch.nn.functional as F
import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn
import tg4perfetto

# ─────────────────────────────  Perfetto helpers  ────────────────────────── #

N_ENGINES = 4
_world = dist.get_world_size() if dist.is_initialized() else 1

_TRACKS: List[tg4perfetto.Track] = []
for r in range(_world):
    dev = f"GPU{r}" if torch.cuda.is_available() else f"CPU{r}"
    for e in range(N_ENGINES):
        _TRACKS.append(tg4perfetto.track(f"{dev}-Engine-{e}"))

_ENGINE_FREE_AT = [0.0] * len(_TRACKS)


def _pick_engine() -> int:
    """Greedy earliest-free compute track."""
    return min(range(len(_TRACKS)), key=_ENGINE_FREE_AT.__getitem__)


def tensor_meta(name: str, x: torch.Tensor) -> Dict[str, Any]:
    """Return a dict with standard tensor metadata fields."""
    return {
        f"{name}_shape": tuple(x.shape),
        f"{name}_dtype": str(x.dtype),
        f"{name}_numel": x.numel(),
        f"{name}_bytes": x.element_size() * x.numel(),
        f"{name}_device": str(x.device),
    }


@contextlib.contextmanager
def _dispatch(name: str, op: str, **meta) -> Generator[None, None, None]:
    """
    Unified tracing context.

    • Any value in **meta that is a torch.Tensor is automatically expanded
      with tensor_meta().
    • If **meta contains an `"inputs"` mapping {alias: tensor}, each tensor
      inside is expanded the same way (keys are prefixed with the alias).
    """
    meta = dict(meta)
    meta["op"] = op

    # 1️⃣  expand plain tensor arguments
    for k, v in list(meta.items()):
        if isinstance(v, torch.Tensor):
            meta.update(tensor_meta(k, v))
            del meta[k]

    # 2️⃣  expand "inputs" payload, if provided
    inputs = meta.pop("inputs", None)
    if isinstance(inputs, dict):
        for alias, t in inputs.items():
            if isinstance(t, torch.Tensor):
                meta.update(tensor_meta(alias, t))

    # 3️⃣  add device slot
    meta["device"] = (
        f"cuda:{torch.cuda.current_device()}"
        if torch.cuda.is_available()
        else f"cpu:{dist.get_rank() if dist.is_initialized() else 0}"
    )

    eng = _pick_engine()
    trk = _TRACKS[eng]
    cm = trk.trace(name, meta)
    cm.__enter__()
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        dur = time.perf_counter_ns() - start
        _ENGINE_FREE_AT[eng] = start + dur
        cm.__exit__(None, None, None)

# ─────────────────────────────  Model primitives  ────────────────────────── #

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500_000
    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):  # FP32 math for stability
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10_000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end, dtype=torch.float32, device=freqs.device)
    return torch.polar(torch.ones_like(freqs), torch.outer(t, freqs))


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    return freqs_cis.view(
        *[d if i in (1, x.ndim - 1) else 1 for i, d in enumerate(x.shape)]
    )


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    return (
        torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq),
        torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk),
    )


def repeat_kv(x: torch.Tensor, n_rep: int):
    if n_rep == 1:
        return x
    b, s, n_kv, hd = x.shape
    return (
        x[:, :, :, None, :]
        .expand(b, s, n_kv, n_rep, hd)
        .reshape(b, s, n_kv * n_rep, hd)
    )

# ───────────────────────────  Chunked Linear helper  ─────────────────────── #

def _chunked_linear(x: torch.Tensor, w: torch.Tensor, prefix: str, n_chunks: int = N_ENGINES):
    if not torch.cuda.is_available():
        outs = []
        for i, ws in enumerate(torch.chunk(w, n_chunks, 0)):
            with _dispatch(f"{prefix}#{i}", op="Linear", inputs={"x": x, "w": ws}):
                outs.append(F.linear(x, ws))
        return torch.cat(outs, -1)

    splits = torch.chunk(w, n_chunks, 0)
    streams = [torch.cuda.Stream(device=x.device) for _ in splits]
    outs = [None] * n_chunks
    for i, (ws, st) in enumerate(zip(splits, streams)):
        with _dispatch(f"{prefix}#{i}", op="Linear", inputs={"x": x, "w": ws}):
            with torch.cuda.stream(st):
                outs[i] = F.linear(x, ws)
    torch.cuda.synchronize()
    return torch.cat(outs, -1)

# ─────────────────────────────  Attention layer  ─────────────────────────── #

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        mp = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // mp
        self.n_local_kv_heads = self.n_kv_heads // mp
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.hd = args.dim // args.n_heads

        self.q_proj = ColumnParallelLinear(args.dim, args.n_heads * self.hd, bias=False)
        self.k_proj = ColumnParallelLinear(args.dim, self.n_kv_heads * self.hd, bias=False)
        self.v_proj = ColumnParallelLinear(args.dim, self.n_kv_heads * self.hd, bias=False)
        self.o_proj = RowParallelLinear(args.n_heads * self.hd, args.dim, bias=False, input_is_parallel=True)

        shape = (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.hd)
        self.register_buffer("cache_k", torch.zeros(shape), persistent=False)
        self.register_buffer("cache_v", torch.zeros(shape), persistent=False)

    # helper
    def _proj(self, tag: str, x: torch.Tensor, layer: ColumnParallelLinear):
        out = _chunked_linear(x, layer.weight, tag)  # bias is None
        return out.view(x.size(0), x.size(1), -1, self.hd)  # [B,T,H,D]

    def forward(self, x, start_pos, freqs_cis, mask):
        B, T, _ = x.shape

        q = self._proj("Q_proj", x, self.q_proj)
        k = self._proj("K_proj", x, self.k_proj)
        v = self._proj("V_proj", x, self.v_proj)

        with _dispatch("RoPE", op="RoPE", inputs={"q": q, "k": k}):
            q, k = apply_rotary_emb(q, k, freqs_cis)

        with _dispatch("KV_write", op="MemCopy", inputs={"k": k, "v": v}):
            self.cache_k[:, start_pos:start_pos + T] = k
            self.cache_v[:, start_pos:start_pos + T] = v

        keys = repeat_kv(self.cache_k[:, :start_pos + T], self.n_rep)
        vals = repeat_kv(self.cache_v[:, :start_pos + T], self.n_rep)

        ctx = []
        inv_sqrt_d = 1.0 / math.sqrt(self.hd)

        for h in range(self.n_local_heads):
            qh, kh, vh = q[:, :, h, :], keys[:, :, h, :], vals[:, :, h, :]

            with _dispatch(f"H{h}_QK", op="MatMul", inputs={"q": qh, "k": kh}):
                scores = torch.einsum("btd,bsd->bts", qh, kh) * inv_sqrt_d

            with _dispatch(f"H{h}_SMax1", op="ReduceMax", inputs={"s": scores}):
                max_ = scores.amax(-1, keepdim=True)
            with _dispatch(f"H{h}_SMax2", op="Exp", inputs={"s": scores}):
                scores = (scores - max_).exp()
            with _dispatch(f"H{h}_SMax3", op="Normalize", inputs={"s": scores}):
                scores = scores / scores.sum(-1, keepdim=True)

            with _dispatch(f"H{h}_SV", op="MatMul", inputs={"s": scores, "v": vh}):
                ctx.append(torch.einsum("bts,bsd->btd", scores, vh))

        cat = torch.cat(ctx, -1)           # [B,T,local_heads*D]
        with _dispatch("O_proj", op="Linear", inputs={"cat": cat}):
            return self.o_proj(cat)

# ────────────────────────────  Feed-Forward MLP  ─────────────────────────── #

class FeedForward(nn.Module):
    def __init__(self, dim, hidden, multiple_of, mult):
        super().__init__()
        hidden = int(2 * hidden / 3)
        hidden = int(mult * hidden) if mult else hidden
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(dim, hidden, bias=False)
        self.w3 = ColumnParallelLinear(dim, hidden, bias=False)
        self.w2 = RowParallelLinear(hidden, dim, bias=False, input_is_parallel=True)

    def forward(self, x):
        with _dispatch("GateUp", op="Linear", inputs={"x": x}):
            g = self.w1(x)
            u = self.w3(x)
        with _dispatch("SiLU", op="Activation", inputs={"g": g}):
            act = torch.nn.functional.silu(g) * u
        with _dispatch("Down", op="Linear", inputs={"act": act}):
            return self.w2(act)

# ─────────────────────────────  Transformer block  ───────────────────────── #

class TransformerBlock(nn.Module):
    def __init__(self, idx: int, args: ModelArgs):
        super().__init__()
        self.idx = idx
        self.attn = Attention(args)
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn = FeedForward(args.dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask):
        with _dispatch(f"B{self.idx}_Res1", op="ResidualAdd", inputs={"x": x}):
            h = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        with _dispatch(f"B{self.idx}_Res2", op="ResidualAdd", inputs={"h": h}):
            return h + self.ffn(self.ffn_norm(h))

# ─────────────────────────────  Full Transformer  ────────────────────────── #

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.params = args
        self.tok_embeddings = VocabParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(args.dim, args.vocab_size, bias=False)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2, args.rope_theta),
            persistent=False,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs = self.freqs_cis[start_pos:start_pos + seqlen].to(h.device)

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, 1)
            prefix = torch.zeros((seqlen, start_pos), device=tokens.device)
            mask = torch.hstack([prefix, mask]).type_as(h)

        for block in self.layers:
            h = block(h, start_pos, freqs, mask)

        with _dispatch("FinalNorm", op="RMSNorm", inputs={"h": h}):
            h = self.norm(h)
        with _dispatch("LMHead", op="Linear", inputs={"h": h}):
            return self.output(h).float()
