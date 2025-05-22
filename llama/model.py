# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
# llama/model_cpu_timed.py  •  CPU-only fine-grained Perfetto tracing with real timings

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn

import tg4perfetto

# ─── PERFETTO SETUP ─────────────────────────────────────────────────────────────
_COMPUTE_TRACKS = [tg4perfetto.track(f"Engine_{i}") for i in range(4)]

def timed_op(fn, *args, **kwargs):
    """Runs fn(*args, **kwargs) on CPU and returns (output, elapsed_ms)."""
    t0 = time.time_ns()
    out = fn(*args, **kwargs)
    t1 = time.time_ns()
    return out, (t1 - t0) / 1e6

def _dispatch_timed(name: str, op: str, engine: int = 0, fn=None, **meta):
    """
    Runs fn() and emits a Perfetto slice on track Engine_{engine}
    with real start/end timestamps plus duration_ms in metadata.
    """
    track    = _COMPUTE_TRACKS[engine % len(_COMPUTE_TRACKS)]
    start_ns = time.time_ns()
    out, ms  = timed_op(fn)
    end_ns   = start_ns + int(ms * 1e6)
    track.trace_slice_ts(
        name,
        start_ns,
        end_ns,
        {**meta, "op": op, "duration_ms": ms, "device": "cpu"}
    )
    return out

def _tensor_meta(name: str, x: torch.Tensor) -> dict:
    return {
        f"{name}_shape": tuple(x.shape),
        f"{name}_dtype": str(x.dtype),
        f"{name}_numel": x.numel(),
        f"{name}_bytes": x.element_size() * x.numel(),
        "device": "cpu",
    }

# ────────────────────────────────────────────────────────────────────────────────

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
    rope_theta: float = 500000
    max_batch_size: int = 32
    max_seq_len: int = 2048

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t     = torch.arange(end, dtype=torch.float32, device=freqs.device)
    return torch.polar(torch.ones((end, dim//2), device=freqs.device), torch.outer(t, freqs))

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    shape = [d if i in (1, ndim - 1) else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    meta = {**_tensor_meta("xq_in", xq), **_tensor_meta("xk_in", xk)}
    def _fn():
        xq_c   = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_c   = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_b= reshape_for_broadcast(freqs_cis, xq_c)
        return (torch.view_as_real(xq_c * freqs_b).flatten(3).type_as(xq),
                torch.view_as_real(xk_c * freqs_b).flatten(3).type_as(xk))
    return _dispatch_timed("RoPE", "RoPE", engine=3, fn=_fn, **meta)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    def _fn():
        bs, sl, nh, hd = x.shape
        x2 = x[:, :, :, None, :].expand(bs, sl, nh, n_rep, hd)
        return x2.reshape(bs, sl, nh * n_rep, hd)
    return _dispatch_timed("RepeatKV", "Reshape", engine=0, fn=_fn, **_tensor_meta("kv_in", x))

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads         = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        mp                      = fs_init.get_model_parallel_world_size()
        self.n_local_heads      = args.n_heads // mp
        self.n_local_kv_heads   = self.n_kv_heads // mp
        self.n_rep              = self.n_local_heads // self.n_local_kv_heads
        self.head_dim           = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(args.dim, args.n_heads*self.head_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.wk = ColumnParallelLinear(args.dim, self.n_kv_heads*self.head_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.wv = ColumnParallelLinear(args.dim, self.n_kv_heads*self.head_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.wo = RowParallelLinear(args.n_heads*self.head_dim, args.dim, bias=False, input_is_parallel=True, init_method=lambda x: x)

        self.register_buffer('cache_k', torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim))
        self.register_buffer('cache_v', torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim))

    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.shape

        xq = _dispatch_timed("Q_proj", "Linear", engine=0, fn=lambda: self.wq(x), **_tensor_meta("x", x))
        xk = _dispatch_timed("K_proj", "Linear", engine=1, fn=lambda: self.wk(x), **_tensor_meta("x", x))
        xv = _dispatch_timed("V_proj", "Linear", engine=2, fn=lambda: self.wv(x), **_tensor_meta("x", x))

        xq = xq.view(bsz, seqlen, self.n_local_heads,    self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        self.cache_k[:bsz, start_pos:start_pos+seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos+seqlen] = xv

        keys   = repeat_kv(self.cache_k[:bsz, :start_pos+seqlen], self.n_rep)
        values = repeat_kv(self.cache_v[:bsz, :start_pos+seqlen], self.n_rep)

        xq      = xq.transpose(1,2)
        keys    = keys.transpose(1,2)
        values  = values.transpose(1,2)

        meta_qk = {**_tensor_meta("xq", xq), **_tensor_meta("keysT", keys.transpose(2,3))}
        scores  = _dispatch_timed(
            "QK_matmul", "MatMul", engine=0,
            fn=lambda: torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim),
            **meta_qk
        )

        if mask is not None:
            scores = _dispatch_timed(
                "AddMask", "Add", engine=1,
                fn=lambda: scores + mask,
                **_tensor_meta("scores_in", scores)
            )

        scores = _dispatch_timed(
            "Softmax", "Softmax", engine=1,
            fn=lambda: F.softmax(scores.float(), dim=-1).type_as(xq),
            **_tensor_meta("scores", scores)
        )

        context = _dispatch_timed(
            "V_matmul", "MatMul", engine=2,
            fn=lambda: torch.matmul(scores, values),
            **{**_tensor_meta("scores", scores), **_tensor_meta("values", values)}
        )

        out = _dispatch_timed(
            "ConcatHeads", "Reshape", engine=2,
            fn=lambda: context.transpose(1,2).contiguous().view(bsz, seqlen, -1),
            **_tensor_meta("ctx", context)
        )

        return _dispatch_timed(
            "O_proj", "Linear", engine=3,
            fn=lambda: self.wo(out),
            **_tensor_meta("out", out)
        )

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden = int(ffn_dim_multiplier * hidden)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(dim, hidden, bias=False, gather_output=False, init_method=lambda x: x)
        self.w2 = RowParallelLinear(hidden, dim, bias=False, input_is_parallel=True, init_method=lambda x: x)
        self.w3 = ColumnParallelLinear(dim, hidden, bias=False, gather_output=False, init_method=lambda x: x)

    def forward(self, x):
        h1 = _dispatch_timed("FFN_w1", "Linear", engine=0, fn=lambda: self.w1(x), **_tensor_meta("x", x))
        h1 = _dispatch_timed("FFN_silu", "Activation", engine=0, fn=lambda: F.silu(h1), **_tensor_meta("h1", h1))
        h3 = _dispatch_timed("FFN_w3", "Linear", engine=1, fn=lambda: self.w3(x), **_tensor_meta("x", x))
        h  = _dispatch_timed("FFN_mul", "Mul", engine=1, fn=lambda: h1 * h3, **{**_tensor_meta("h1", h1), **_tensor_meta("h3", h3)})
        return _dispatch_timed("FFN_w2", "Linear", engine=2, fn=lambda: self.w2(h), **_tensor_meta("h", h))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm  = RMSNorm(args.dim, eps=args.norm_eps)
        self.attn      = Attention(args)
        self.ffn       = FeedForward(args.dim, 4*args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.layer_id  = layer_id

    def forward(self, x, start_pos, freqs_cis, mask):
        x_norm = _dispatch_timed(
            f"Layer{self.layer_id}_AttnNorm", "Norm", engine=0,
            fn=lambda: self.attn_norm(x), **_tensor_meta("x", x)
        )
        h      = _dispatch_timed(
            f"Layer{self.layer_id}_Attention", "Attention", engine=0,
            fn=lambda: x + self.attn(x_norm, start_pos, freqs_cis, mask)
        )
        h2     = _dispatch_timed(
            f"Layer{self.layer_id}_FFNNorm", "Norm", engine=1,
            fn=lambda: self.ffn_norm(h), **_tensor_meta("h", h)
        )
        out    = _dispatch_timed(
            f"Layer{self.layer_id}_FeedForward", "FFN", engine=1,
            fn=lambda: h + self.ffn(h2)
        )
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params         = params
        self.tok_embeddings = VocabParallelEmbedding(params.vocab_size, params.dim, init_method=lambda x: x)
        self.layers         = nn.ModuleList([TransformerBlock(i, params) for i in range(params.n_layers)])
        self.final_norm     = RMSNorm(params.dim, eps=params.norm_eps)
        self.output         = ColumnParallelLinear(params.dim, params.vocab_size, bias=False, init_method=lambda x: x)
        self.freqs_cis      = precompute_freqs_cis(params.dim//params.n_heads, params.max_seq_len*2, params.rope_theta)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        bsz, seqlen = tokens.shape
        h = _dispatch_timed(
            "Embed", "Embedding", engine=0,
            fn=lambda: self.tok_embeddings(tokens),
            **_tensor_meta("tokens", tokens)
        )
        freqs = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            def _mk_mask():
                m  = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
                mm = torch.triu(m, 1)
                return torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mm]).type_as(h)
            mask = _dispatch_timed("MakeMask", "Mask", engine=1, fn=_mk_mask, **_tensor_meta("h_in", h))
        out = h
        for layer in self.layers:
            out = layer(out, start_pos, freqs, mask)
        out = _dispatch_timed("FinalNorm", "Norm", engine=2, fn=lambda: self.final_norm(out), **_tensor_meta("out_in", out))
        logits = _dispatch_timed("OutputProj", "Linear", engine=3, fn=lambda: self.output(out).float(), **_tensor_meta("out2", out))
        return logits
