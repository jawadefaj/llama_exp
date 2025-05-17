# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Llama 3 Community License Agreement.
# ---------------------------------------------------------------------------
# llama/model.py  •  CPU- and GPU-compatible + opt-in Perfetto spans
# ---------------------------------------------------------------------------

from __future__ import annotations
import math, os, tempfile, atexit
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn
# --------------------------------------------------------------------------- #
# ⏱️  PERFETTO TRACING (hard-coded ON)                                        #
# --------------------------------------------------------------------------- #
import datetime, atexit, tg4perfetto as t4p

TRACING = True                                   # ← always trace
# Build a readable file name:  llama_trace_20250516_184512.json
TRACE_PATH = (
    Path(__file__).parent /                      # same folder as model.py
    f"llama_trace_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
).as_posix()

_TRACE_CTX = t4p.open(TRACE_PATH)
_TRACE_CTX.__enter__()                           # open once per process

TR_ATT = t4p.track("Attention")                  # named tracks
TR_MLP = t4p.track("MLP")
TR_MEM = t4p.track("KVCache")

atexit.register(_TRACE_CTX.__exit__, None, None, None)

from contextlib import contextmanager
@contextmanager
def span(track, name, **args):
    with track.trace(name, **({"ts": args} if args else {})):
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
# LAYER HELPER CLASSES                                                        #
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):               # fp32 for stability
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10_000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end, dtype=torch.float32, device=freqs.device)
    return torch.polar(torch.ones_like(freqs), torch.outer(t, freqs))  # complex64

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    return freqs_cis.view(*[d if i in (1, ndim - 1) else 1 for i, d in enumerate(x.shape)])

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    return (
        torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq),
        torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk),
    )

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, hd = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, hd)
        .reshape(bs, slen, n_kv_heads * n_rep, hd)
    )

# --------------------------------------------------------------------------- #
# ATTENTION BLOCK WITH TRACING                                                #
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        mp = fs_init.get_model_parallel_world_size()
        self.n_local_heads   = args.n_heads   // mp
        self.n_local_kv_heads = self.n_kv_heads // mp
        self.n_rep  = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim,  bias=False)
        self.wk = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = RowParallelLinear(args.n_heads * self.head_dim, args.dim, bias=False, input_is_parallel=True)

        cache_shape = (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape), persistent=False)
        self.register_buffer("cache_v", torch.zeros(cache_shape), persistent=False)

    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.shape

        with span(TR_ATT, "QKV_proj"):
            xq = self.wq(x).view(bsz, seqlen, self.n_local_heads,    self.head_dim)
            xk = self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        with span(TR_MEM, "KV_write",
                  k_shape=list(xk.shape), v_shape=list(xv.shape)):
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys   = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        with span(TR_MEM, "KV_repeat", rep=self.n_rep): 
            keys   = repeat_kv(keys,   self.n_rep)
            values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)    # (bs, heads, seq, hd)
        keys   = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        with span(TR_ATT, "QK_matmul"):
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
        with span(TR_ATT, "Softmax"):
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        with span(TR_ATT, "AttnV"):
            out = torch.matmul(scores, values)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        with span(TR_ATT, "OutProj"):
            return self.wo(out)

# --------------------------------------------------------------------------- #
# FEED-FORWARD WITH TRACING                                                   #
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
        with span(TR_MLP, "GateUp"):
            g = self.w1(x)
            u = self.w3(x)
        with span(TR_MLP, "SiLU"):
            act = torch.nn.functional.silu(g) * g
        with span(TR_MLP, "Down"):
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
        self.ffn  = FeedForward(args.dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask):
        with span(TR_ATT, f"Block{self.idx}_Attn"):
            h = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        with span(TR_MLP, f"Block{self.idx}_MLP"):
            return h + self.ffn(self.ffn_norm(h))

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.params = args
        self.tok_embeddings = VocabParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm   = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(args.dim, args.vocab_size, bias=False)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2, args.rope_theta),
            persistent=False,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen].to(h.device)

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mask]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        return self.output(self.norm(h)).float()
