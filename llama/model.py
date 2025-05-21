# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Llama 3 Community License Agreement.
# ---------------------------------------------------------------------------
# llama/model.py  •  CPU & GPU compatible + fine‑grained Perfetto tracing
# ---------------------------------------------------------------------------
#   ▸ Four compute‑engine tracks per device (GPU0‑Engine‑0 … GPU0‑Engine‑3)
#   ▸ Chunked Q/K/V projections → 4 slices in parallel
#   ▸ Per‑head attention:   QK matmul → 3‑pass soft‑max → value matmul
#   ▸ Each slice carries op‑type + tensor‑shape metadata
# ---------------------------------------------------------------------------

from __future__ import annotations
import math, os, atexit, datetime, time, contextlib
from dataclasses import dataclass
from pathlib import Path
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

# --------------------------------------------------------------------------- #
# 🗂️  GLOBAL PERFETTO TRACKS (one per compute engine, per device / rank)      #
# --------------------------------------------------------------------------- #
N_ENGINES = 4                                         # configurable ‑‑n_engines flag later
_world_size = dist.get_world_size() if dist.is_initialized() else 1

_TRACKS: List[tg4perfetto.Track] = []
for rank in range(_world_size):
    dev_name = f"GPU{rank}" if torch.cuda.is_available() else f"CPU{rank}"
    for eng in range(N_ENGINES):
        _TRACKS.append(tg4perfetto.track(f"{dev_name}-Engine-{eng}"))

_ENGINE_FREE_AT: List[float] = [0.0] * len(_TRACKS)

def _pick_engine() -> int:
    """Greedy earliest‑free engine across *all* devices/tracks."""
    return min(range(len(_TRACKS)), key=_ENGINE_FREE_AT.__getitem__)

@contextlib.contextmanager
def trace_op(name: str, **meta) -> Generator[None, None, None]:
    """Emit a Perfetto slice on the earliest‑free compute engine."""
    meta = dict(meta)  # make a local copy
    if torch.cuda.is_available():
        meta["device"] = f"cuda:{torch.cuda.current_device()}"
    else:
        meta["device"] = f"cpu:{dist.get_rank() if dist.is_initialized() else 0}"

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

# --------------------------------------------------------------------------- #
# Convenience: dispatch helper that wraps the engine scheduling + metadata    #
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _dispatch(name: str, op: str, **meta):
    with trace_op(name, op=op, **meta):
        yield

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
    return x[:, :, :, None, :].expand(b, s, n_kv, n_rep, hd).reshape(b, s, n_kv * n_rep, hd)

# --------------------------------------------------------------------------- #
# INTERNAL helper – chunked linear (no bias) with dispatch per chunk          #
# --------------------------------------------------------------------------- #

def _chunked_linear(x: torch.Tensor, weight: torch.Tensor, name_prefix: str, n_chunks: int = N_ENGINES):
    """Split *weight* rows into n_chunks and run F.linear for each, traced."""
    w_splits = torch.chunk(weight, n_chunks, dim=0)
    outs = []
    for idx, w in enumerate(w_splits):
        with _dispatch(f"{name_prefix}#{idx}", op="Linear", w_shape=w.shape, x_shape=x.shape):
            outs.append(F.linear(x, w))
    return torch.cat(outs, dim=-1)

# --------------------------------------------------------------------------- #
# ATTENTION – per‑head slices + 3‑pass softmax + chunked projections          #
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

        # Column/Row‑parallel layers hold parameters; we may bypass their forward
        self.q_proj = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.k_proj = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = RowParallelLinear(args.n_heads * self.head_dim, args.dim, bias=False, input_is_parallel=True)

        # KV‑cache (local to this rank)
        shape = (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)
        self.register_buffer("cache_k", torch.zeros(shape), persistent=False)
        self.register_buffer("cache_v", torch.zeros(shape), persistent=False)

    # --------------------------------------------------------------------- #
    def _proj(self, name: str, x: torch.Tensor, layer: ColumnParallelLinear):
        """Chunked projection → cat → reshape."""
        # We reuse layer.weight to keep parameter sharing; bias is None.
        out = _chunked_linear(x, layer.weight, name_prefix=name)  # [B,T, out_dim]
        hd = self.head_dim
        n_heads = (out.shape[-1] // hd)
        return out.view(x.size(0), x.size(1), n_heads, hd)

    # --------------------------------------------------------------------- #
    def forward(self, x, start_pos, freqs_cis, mask):
        B, T, _ = x.shape
        hd = self.head_dim

        q = self._proj("Q_proj", x, self.q_proj)
        k = self._proj("K_proj", x, self.k_proj)
        v = self._proj("V_proj", x, self.v_proj)

        with _dispatch("RoPE", op="RoPE", q_shape=q.shape, k_shape=k.shape):
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # ----------------------------------------------------------------- #
        # KV cache write (memcpy slice)
        with _dispatch("KV_write", op="MemCopy", T=T):
            self.cache_k[:, start_pos:start_pos + T] = k
            self.cache_v[:, start_pos:start_pos + T] = v

        keys   = repeat_kv(self.cache_k[:, :start_pos + T], self.n_rep)
        values = repeat_kv(self.cache_v[:, :start_pos + T], self.n_rep)

        # ----------------------------------------------------------------- #
        # Per‑head attention
        inv_sqrt_d = 1.0 / math.sqrt(hd)
        ctx_heads: List[torch.Tensor] = []

        for h in range(self.n_local_heads):
            qh, kh, vh = q[:, :, h, :], keys[:, :, h, :], values[:, :, h, :]

            # ① Q·Kᵀ
            with _dispatch(f"H{h}_QK", op="MatMul", q_shape=qh.shape, k_shape=kh.shape):
                score_h = torch.einsum("btd,bsd->bts", qh, kh) * inv_sqrt_d

            # ② 3‑pass soft‑max
            with _dispatch(f"H{h}_SoftmaxP1", op="ReduceMax", in_shape=score_h.shape):
                max_h = score_h.max(-1, keepdim=True)
            with _dispatch(f"H{h}_SoftmaxP2", op="ElementwiseExp"):
                score_h = (score_h - max_h).exp()
            with _dispatch(f"H{h}_SoftmaxP3", op="Normalize"):
                score_h = score_h / score_h.sum(-1, keepdim=True)

            # ③ Attention · V
            with _dispatch(f"H{h}_ValueMM", op="MatMul", score_shape=score_h.shape, v_shape=vh.shape):
                ctx_h = torch.einsum("bts,bsd->btd", score_h, vh)
            ctx_heads.append(ctx_h)

        cat = torch.cat(ctx_heads, dim=-1)          # [B,T, n_local_heads*hd]
        with _dispatch("O_proj", op="Linear", cat_shape=cat.shape):
            return self.o_proj(cat)

# --------------------------------------------------------------------------- #
# FEED‑FORWARD                                                                #
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
        with _dispatch("GateUp", op="Linear", X_shape=x.shape):
            g = self.w1(x)
            u = self.w3(x)
        with _dispatch("SiLU", op="Activation", G_shape=g.shape):
            act = torch.nn.functional.silu(g) * u
        with _dispatch("Down", op="Linear", Act_shape=act.shape):
            return self.w2(act)

# --------------------------------------------------------------------------- #
# TRANSFORMER BLOCK                                                           #
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    def __init__(self, idx: int, args: ModelArgs):
        super().__init__()
        self.idx = idx
        self.attn = Attention(args)
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn = FeedForward(args.dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask):
        with _dispatch(f"Block{self.idx}_Residual1", op="ResidualAdd", X_shape=x.shape):
            h = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        with _dispatch(f"Block{self.idx}_Residual2", op="ResidualAdd", H_shape=h.shape):
            return h + self.ffn(self.ffn_norm(h))

# --------------------------------------------------------------------------- #
# TOP‑LEVEL TRANSFORMER                                                       #
# --------------------------------------------------------------------------- #
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
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen].to(h.device)

        # Generate an autoregressive mask if we are decoding more than one token
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            prefix = torch.zeros((seqlen, start_pos), device=tokens.device)
            mask = torch.hstack([prefix, mask]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        with _dispatch("FinalNorm", op="RMSNorm", H_shape=h.shape):
            h = self.norm(h)
        with _dispatch("LMHead", op="Linear", H_shape=h.shape):
            return self.output(h).float()
