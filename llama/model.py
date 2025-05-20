# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Llama 3 Community License Agreement.
# ---------------------------------------------------------------------------
# llama/model.py  •  CPU & GPU compatible + always-on Perfetto tracing
#   • Each slice carries detailed metadata: compute op, input/output shapes,
#     dtypes, and byte counts.
# ---------------------------------------------------------------------------

from __future__ import annotations
import math, os, atexit, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Generator
import torch
import torch.nn.functional as F
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn
import tg4perfetto
from contextlib import contextmanager

# --------------------------------------------------------------------------- #
# 🗂️  GLOBAL PERFETTO TRACKS (one per compute engine)                         #
# --------------------------------------------------------------------------- #
N_ENGINES = 4
_TRACKS = [tg4perfetto.track(f"Engine-{i}") for i in range(N_ENGINES)]
_ENGINE_FREE_AT = [0.0] * N_ENGINES  # wall-clock ends of last slice

def _pick_engine() -> int:
    """Greedy earliest-free scheduler."""
    return min(range(N_ENGINES), key=_ENGINE_FREE_AT.__getitem__)

@contextmanager
def trace_op(name: str, **meta) -> Generator[None, None, None]:
    """
    Context-manager that drops the slice on whichever engine is free.
    Usage:
        with trace_op("Q_proj", op="Linear", inputs=..., output=...):
            ...
    """
    eng = _pick_engine()
    trk = _TRACKS[eng]
    cm = trk.trace(name, meta)
    cm.__enter__()
    start_ns = tg4perfetto.now_ns()
    try:
        yield
    finally:
        dur = tg4perfetto.now_ns() - start_ns
        _ENGINE_FREE_AT[eng] = start_ns + dur
        cm.__exit__(None, None, None)

# --------------------------------------------------------------------------- #
# MODEL ARGUMENTS                                                             #
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# LAYER HELPERS                                                               #
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10_000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end, dtype=torch.float32, device=freqs.device)
    return torch.polar(torch.ones_like(freqs), torch.outer(t, freqs))  # complex64

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
    bs, slen, n_kv_heads, hd = x.shape
    return (
        x[:, :, :, None, :]
         .expand(bs, slen, n_kv_heads, n_rep, hd)
         .reshape(bs, slen, n_kv_heads * n_rep, hd)
    )

# --------------------------------------------------------------------------- #
# ATTENTION WITH PER-HEAD SLICES + 3-PASS SOFTMAX                             #
# --------------------------------------------------------------------------- #
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
        self.o_proj = RowParallelLinear(
            args.n_heads * self.head_dim, args.dim, bias=False, input_is_parallel=True
        )

        shape = (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)
        self.register_buffer("cache_k", torch.zeros(shape), persistent=False)
        self.register_buffer("cache_v", torch.zeros(shape), persistent=False)

    def forward(self, x, start_pos, freqs_cis, mask):
        B, T, _ = x.shape
        head_dim = self.head_dim

        # 1. Projections
        with trace_op("Q_proj", op="Linear", inputs={"x": x}, output_shape=(B, T, self.n_local_heads, head_dim)):
            q = self.q_proj(x).view(B, T, self.n_local_heads, head_dim)
        with trace_op("K_proj", op="Linear", inputs={"x": x}, output_shape=(B, T, self.n_local_kv_heads, head_dim)):
            k = self.k_proj(x).view(B, T, self.n_local_kv_heads, head_dim)
        with trace_op("V_proj", op="Linear", inputs={"x": x}, output_shape=(B, T, self.n_local_kv_heads, head_dim)):
            v = self.v_proj(x).view(B, T, self.n_local_kv_heads, head_dim)

        # 2. Rotary positional encoding
        with trace_op("RoPE", op="RoPE", inputs={"q": q, "k": k}, output_shape=q.shape):
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # 3. Write to KV cache
        with trace_op("KV_write", op="MemCopy", inputs={"K": k, "V": v}):
            self.cache_k[:, start_pos:start_pos+T] = k
            self.cache_v[:, start_pos:start_pos+T] = v

        keys = repeat_kv(self.cache_k[:, :start_pos+T], self.n_rep)
        values = repeat_kv(self.cache_v[:, :start_pos+T], self.n_rep)

        attn_out = []
        for h in range(self.n_local_heads):
            qh = q[..., h, :]
            kh = keys[..., h, :, :]
            vh = values[..., h, :, :]

            # Dot-product
            with trace_op(f"Dot_H{h}", op="MatMul", inputs={"q": qh, "k": kh}):
                scores = torch.einsum("btd,bSd->bts", qh, kh) / math.sqrt(head_dim)

            # 3-pass softmax
            with trace_op(f"Softmax_MaxSub_H{h}", op="MaxSub", inputs={"scores": scores}):
                scores = scores - scores.amax(-1, keepdim=True)
            with trace_op(f"Softmax_Exp_H{h}", op="Exp", inputs={"scores": scores}):
                scores = scores.exp()
            with trace_op(f"Softmax_Norm_H{h}", op="Normalize", inputs={"scores": scores}):
                scores = scores / scores.sum(-1, keepdim=True)

            # Attention-weighted value
            with trace_op(f"AttnV_H{h}", op="MatMul", inputs={"scores": scores, "v": vh}):
                out_h = torch.einsum("bts,bSd->btd", scores, vh)
            attn_out.append(out_h)

        # 4. Concat & output projection
        cat = torch.cat(attn_out, dim=-1)
        with trace_op("O_proj", op="Linear", inputs={"cat": cat}, output_shape=cat.shape):
            return self.o_proj(cat)

# --------------------------------------------------------------------------- #
# FEED FORWARD WITH METADATA TRACING                                          #
# --------------------------------------------------------------------------- #
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
        with trace_op("GateUp", op="Linear", inputs={"X": x}):
            g = self.w1(x)
            u = self.w3(x)
        with trace_op("SiLU", op="Activation", inputs={"G": g}):
            act = torch.nn.functional.silu(g) * g
        with trace_op("Down", op="Linear", inputs={"Act": act}):
            return self.w2(act)

# --------------------------------------------------------------------------- #
# TRANSFORMER BLOCK + MODEL                                                   #
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    def __init__(self, idx: int, args: ModelArgs):
        super().__init__()
        self.idx = idx
        self.attn = Attention(args)
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn = FeedForward(
            args.dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier
        )
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask):
        with trace_op(f"Block{self.idx}_Attn", op="ResidualAdd", inputs={"X": x}):
            h = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        with trace_op(f"Block{self.idx}_MLP", op="ResidualAdd", inputs={"H": h}):
            return h + self.ffn(self.ffn_norm(h))

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.params = args
        self.tok_embeddings = VocabParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(i, args) for i in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(args.dim, args.vocab_size, bias=False)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                args.dim // args.n_heads, args.max_seq_len * 2, args.rope_theta
            ),
            persistent=False,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen].to(h.device)

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        return self.output(self.norm(h)).float()
