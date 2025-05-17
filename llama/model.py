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
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn
import tg4perfetto as t4p

# --------------------------------------------------------------------------- #
# ⏱️  PERFETTO TRACING (hard-coded ON)                                        #
# --------------------------------------------------------------------------- #
import datetime, atexit, tg4perfetto as t4p
from pathlib import Path

LOG_DIR = Path(__file__).parent / "log"      # ← new ❶
LOG_DIR.mkdir(exist_ok=True)                # ← new ❷

timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")   # ← new ❸
TRACE_PATH = LOG_DIR / f"llama_trace_{timestamp}.json"           # ← new ❹

_TRACE_CTX = t4p.open(str(TRACE_PATH))
_TRACE_CTX.__enter__()         # start the global trace file

TR_ATT = t4p.track("Attention")
TR_MLP = t4p.track("MLP")
TR_MEM = t4p.track("KVCache")

atexit.register(_TRACE_CTX.__exit__, None, None, None)

# ---------- helpers to embed tensor metadata in each slice ------------------
from contextlib import contextmanager


def _tinfo(t: torch.Tensor, name: str):
    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "bytes": t.numel() * t.element_size(),
    }


@contextmanager
def trace_op(track, name: str, *, op: str,
             inputs: dict[str, torch.Tensor] | None = None,
             output: torch.Tensor | None = None):
    meta = {"op": op}
    if inputs:
        meta["inputs"] = [_tinfo(t, n) for n, t in inputs.items()]
    if output is not None:
        meta["output"] = _tinfo(output, "out")
    with track.trace(name, ts=meta):
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
# ATTENTION WITH METADATA TRACING                                             #
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

        self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim, args.dim, bias=False, input_is_parallel=True
        )

        shape = (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)
        self.register_buffer("cache_k", torch.zeros(shape), persistent=False)
        self.register_buffer("cache_v", torch.zeros(shape), persistent=False)

    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.shape

        with trace_op(TR_ATT, "QKV_proj", op="Linear", inputs={"X": x}):
            xq = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        with trace_op(TR_MEM, "KV_write", op="MemCopy", inputs={"K": xk, "V": xv}):
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        with trace_op(TR_MEM, "KV_repeat", op="Repeat", inputs={"K": keys, "V": values}):
            keys = repeat_kv(keys, self.n_rep)
            values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys, values = keys.transpose(1, 2), values.transpose(1, 2)

        with trace_op(TR_ATT, "QK_matmul", op="MatMul",
                      inputs={"Q": xq, "Kᵀ": keys.transpose(2, 3)}):
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
        with trace_op(TR_ATT, "Softmax", op="Softmax", inputs={"Scores": scores}):
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        with trace_op(TR_ATT, "AttnV", op="MatMul", inputs={"P": scores, "V": values}):
            out = torch.matmul(scores, values)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        with trace_op(TR_ATT, "OutProj", op="Linear", inputs={"Context": out}):
            return self.wo(out)


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
        with trace_op(TR_MLP, "GateUp", op="Linear", inputs={"X": x}):
            g = self.w1(x)
            u = self.w3(x)
        with trace_op(TR_MLP, "SiLU", op="Activation", inputs={"G": g}):
            act = torch.nn.functional.silu(g) * g
        with trace_op(TR_MLP, "Down", op="Linear", inputs={"Act": act}):
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
        with trace_op(TR_ATT, f"Block{self.idx}_Attn", op="ResidualAdd",
                      inputs={"X": x}):
            h = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        with trace_op(TR_MLP, f"Block{self.idx}_MLP", op="ResidualAdd",
                      inputs={"H": h}):
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
        _bsz, seqlen = tokens.shape
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
