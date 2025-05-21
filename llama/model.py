# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Llama 3 Community License Agreement.
# ---------------------------------------------------------------------------
# llama/model.py  •  CPU & GPU compatible + fine‑grained Perfetto tracing
# ---------------------------------------------------------------------------
#   ▹ Four compute‑engine tracks per device (GPU0‑Engine‑0 … GPU0‑Engine‑3)
#   ▹ Chunked Q/K/V projections → 4 slices in parallel
#   ▹ Per‑head attention:   QK matmul → 3‑pass soft‑max → value matmul
#   ▹ Each slice carries op‑type + tensor‑shape metadata
# ---------------------------------------------------------------------------

from __future__ import annotations
import math, os, atexit, datetime, time, contextlib
from dataclasses import dataclass
from typing import Optional, List, Generator

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

# --- Metadata helper ---
def tensor_meta(name: str, x: torch.Tensor) -> dict:
    return {
        "tensor": name,
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "numel": x.numel(),
        "bytes": x.element_size() * x.numel(),
        "device": str(x.device),
    }

# --- Perfetto tracing setup ---
N_ENGINES = 4
_world_size = dist.get_world_size() if dist.is_initialized() else 1
_TRACKS = [tg4perfetto.track(f"{'GPU' if torch.cuda.is_available() else 'CPU'}{rank}-Engine-{eng}")
           for rank in range(_world_size) for eng in range(N_ENGINES)]
_ENGINE_FREE_AT = [0.0] * len(_TRACKS)

def _pick_engine() -> int:
    return min(range(len(_TRACKS)), key=_ENGINE_FREE_AT.__getitem__)

@contextlib.contextmanager
def _dispatch(name: str, op: str, **meta):
    meta = dict(meta)
    meta["op"] = op
    meta["device"] = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() \
                     else f"cpu:{dist.get_rank() if dist.is_initialized() else 0}"
    eng = _pick_engine()
    track = _TRACKS[eng]
    cm = track.trace(name, meta)
    cm.__enter__()
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        dur = time.perf_counter_ns() - start
        _ENGINE_FREE_AT[eng] = start + dur
        cm.__exit__(None, None, None)

# --- Chunked Linear Projections ---
def _chunked_linear(x, weight, name_prefix, n_chunks=N_ENGINES):
    if not torch.cuda.is_available():
        return torch.cat([F.linear(x, w) for w in torch.chunk(weight, n_chunks, dim=0)], dim=-1)
    w_splits = torch.chunk(weight, n_chunks, dim=0)
    streams = [torch.cuda.Stream(device=x.device) for _ in w_splits]
    outs = [None] * n_chunks
    for idx, (w, st) in enumerate(zip(w_splits, streams)):
        with _dispatch(f"{name_prefix}#{idx}", op="Linear", **tensor_meta("x", x)):
            with torch.cuda.stream(st):
                outs[idx] = F.linear(x, w)
    torch.cuda.synchronize()
    return torch.cat(outs, dim=-1)

def proj_with_chunking(name: str, x: torch.Tensor, layer: ColumnParallelLinear) -> torch.Tensor:
    out = _chunked_linear(x, layer.weight, name_prefix=name)
    hd = out.shape[-1] // (x.shape[-1] // layer.input_size_per_partition)
    return out.view(x.size(0), x.size(1), -1, hd)

# --- Model Args ---
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

# --- Norm ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

# --- RoPE ---
def precompute_freqs_cis(dim, end, theta):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
    t = torch.arange(end, dtype=torch.float32, device=freqs.device)
    return torch.polar(torch.ones_like(freqs), torch.outer(t, freqs))

def reshape_for_broadcast(freqs_cis, x):
    return freqs_cis.view(*[d if i in (1, x.ndim - 1) else 1 for i, d in enumerate(x.shape)])

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    return (
        torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq),
        torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk),
    )

# --- Attention ---
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        mp = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // mp
        self.n_local_kv_heads = self.n_kv_heads // mp
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.q_proj = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.k_proj = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = RowParallelLinear(args.n_heads * self.head_dim, args.dim, bias=False, input_is_parallel=True)
        shape = (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)
        self.register_buffer("cache_k", torch.zeros(shape), persistent=False)
        self.register_buffer("cache_v", torch.zeros(shape), persistent=False)

    def forward(self, x, start_pos, freqs_cis, mask):
        B, T, _ = x.shape
        hd = self.head_dim
        q = proj_with_chunking("Q_proj", x, self.q_proj)
        k = proj_with_chunking("K_proj", x, self.k_proj)
        v = proj_with_chunking("V_proj", x, self.v_proj)

        with _dispatch("RoPE", op="RoPE", **tensor_meta("q", q)):
            q, k = apply_rotary_emb(q, k, freqs_cis)

        with _dispatch("KV_write", op="MemCopy", **tensor_meta("k", k)):
            self.cache_k[:, start_pos:start_pos+T] = k
            self.cache_v[:, start_pos:start_pos+T] = v

        keys = self.cache_k[:, :start_pos+T].repeat(1, 1, self.n_rep, 1).view(B, -1, self.n_local_heads, hd)
        values = self.cache_v[:, :start_pos+T].repeat(1, 1, self.n_rep, 1).view(B, -1, self.n_local_heads, hd)

        ctx = []
        for h in range(self.n_local_heads):
            qh, kh, vh = q[:, :, h, :], keys[:, :, h, :], values[:, :, h, :]
            with _dispatch(f"H{h}_QK", op="MatMul", **tensor_meta("qh", qh)):
                score_h = torch.einsum("btd,bsd->bts", qh, kh) / math.sqrt(hd)
            with _dispatch(f"H{h}_SoftmaxP1", op="ReduceMax", **tensor_meta("score", score_h)):
                max_h = score_h.max(-1, keepdim=True).values
            with _dispatch(f"H{h}_SoftmaxP2", op="ElementwiseExp"):
                score_h = (score_h - max_h).exp()
            with _dispatch(f"H{h}_SoftmaxP3", op="Normalize"):
                score_h = score_h / score_h.sum(-1, keepdim=True)
            with _dispatch(f"H{h}_ValueMM", op="MatMul"):
                ctx_h = torch.einsum("bts,bsd->btd", score_h, vh)
            ctx.append(ctx_h)
        cat = torch.cat(ctx, dim=-1)
        with _dispatch("O_proj", op="Linear", **tensor_meta("cat", cat)):
            return self.o_proj(cat)

# --- FeedForward ---
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True)

    def forward(self, x):
        with _dispatch("GateUp", op="Linear", **tensor_meta("x", x)):
            g = self.w1(x)
            u = self.w3(x)
        with _dispatch("SiLU", op="Activation", **tensor_meta("g", g)):
            act = F.silu(g) * u
        with _dispatch("Down", op="Linear", **tensor_meta("act", act)):
            return self.w2(act)

# --- TransformerBlock ---
class TransformerBlock(nn.Module):
    def __init__(self, idx: int, args: ModelArgs):
        super().__init__()
        self.idx = idx
        self.attn = Attention(args)
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn = FeedForward(args.dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask):
        with _dispatch(f"B{self.idx}_Res1", op="ResidualAdd", **tensor_meta("x", x)):
            h = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        with _dispatch(f"B{self.idx}_Res2", op="ResidualAdd", **tensor_meta("h", h)):
            return h + self.ffn(self.ffn_norm(h))

# --- Transformer ---
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.params = args
        self.tok_embeddings = VocabParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(args.dim, args.vocab_size, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2, args.rope_theta), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen].to(h.device)
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            prefix = torch.zeros((seqlen, start_pos), device=tokens.device)
            mask = torch.hstack([prefix, mask]).type_as(h)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        with _dispatch("FinalNorm", op="RMSNorm", **tensor_meta("h", h)):
            h = self.norm(h)
        with _dispatch("LMHead", op="Linear", **tensor_meta("h", h)):
            return self.output(h).float()
