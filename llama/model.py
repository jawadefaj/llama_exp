from __future__ import annotations
import math, time, contextlib
from dataclasses import dataclass
from typing import Optional, Tuple, List

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
# Use tg4perfetto.open in your entrypoint to start/stop recording:
#   with tg4perfetto.open("llama.perfetto-trace"):
#       <run your model forward/training here>
# Tracks will be saved automatically by tg4perfetto.open

_CPU_TRACK = tg4perfetto.track("CPU")
_GPU_TRACK = tg4perfetto.track("GPU")

def trace_op(name: str, **meta):(name: str, **meta):
    # choose track by device type
    device = meta.get("device")
    track = _GPU_TRACK if device and device.startswith("cuda") else _CPU_TRACK
    cm = track.trace(name, meta)
    cm.__enter__()
    try:
        yield
    finally:
        cm.__exit__(None, None, None)

# context manager wrapper
@contextlib.contextmanager
def _dispatch(name: str, op: str, **meta):
    meta = dict(meta)
    if torch.cuda.is_available():
        meta["device"] = f"cuda:{torch.cuda.current_device()}"
    else:
        meta["device"] = f"cpu:{dist.get_rank() if dist.is_initialized() else 0}"
    meta["op"] = op
    with trace_op(name, **meta):
        yield

# --- Model Arguments ---
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

# --- RMSNorm & RoPE ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10_000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32, device=freqs.device)
    return torch.polar(torch.ones_like(freqs), torch.outer(t, freqs))

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    return freqs_cis.view(*[d if i in (1, x.ndim - 1) else 1 for i, d in enumerate(x.shape)])

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs = reshape_for_broadcast(freqs_cis, xq_c)
    return (
        torch.view_as_real(xq_c * freqs).flatten(3).type_as(xq),
        torch.view_as_real(xk_c * freqs).flatten(3).type_as(xk),
    )

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bs, sl, nk, hd = x.shape
    return x[:, :, :, None, :].expand(bs, sl, nk, n_rep, hd).reshape(bs, sl, nk * n_rep, hd)

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
        self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = RowParallelLinear(args.n_heads * self.head_dim, args.dim, bias=False, input_is_parallel=True)
        buf_shape = (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)
        self.register_buffer("cache_k", torch.zeros(buf_shape), persistent=False)
        self.register_buffer("cache_v", torch.zeros(buf_shape), persistent=False)

    def forward(self, x, start_pos, freqs_cis, mask=None):
        B, T, _ = x.shape
        # projections
        with _dispatch("Q_proj", op="Linear", **tensor_meta("x", x)):
            xq = self.wq(x).view(B, T, self.n_local_heads, self.head_dim)
        with _dispatch("K_proj", op="Linear", **tensor_meta("x", x)):
            xk = self.wk(x).view(B, T, self.n_local_kv_heads, self.head_dim)
        with _dispatch("V_proj", op="Linear", **tensor_meta("x", x)):
            xv = self.wv(x).view(B, T, self.n_local_kv_heads, self.head_dim)
        # RoPE
        with _dispatch("RoPE", op="RoPE", **tensor_meta("xq", xq)):
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        # KV cache
        with _dispatch("KV_write", op="MemCopy", **tensor_meta("xk", xk)):
            self.cache_k[:, start_pos:start_pos+T] = xk
            self.cache_v[:, start_pos:start_pos+T] = xv
        keys = repeat_kv(self.cache_k[:, :start_pos+T], self.n_rep)
        vals = repeat_kv(self.cache_v[:, :start_pos+T], self.n_rep)
        # attention per head
        heads_out: List[torch.Tensor] = []
        inv = 1.0 / math.sqrt(self.head_dim)
        for h in range(self.n_local_heads):
            qh = xq[:, :, h, :]
            kh = keys[:, :, h, :]
            vh = vals[:, :, h, :]
            with _dispatch(f"H{h}_QK", op="MatMul", **tensor_meta("qh", qh)):
                score = torch.einsum("btd,bsd->bts", qh, kh) * inv
            # softmax 3-pass
            with _dispatch(f"H{h}_Max", op="ReduceMax", **tensor_meta("score", score)):
                m = score.max(-1, keepdim=True).values
            with _dispatch(f"H{h}_Exp", op="ElementwiseExp"): score = (score - m).exp()
            with _dispatch(f"H{h}_Norm", op="Normalize"): score = score / score.sum(-1, keepdim=True)
            with _dispatch(f"H{h}_VMM", op="MatMul", **tensor_meta("score", score)):
                heads_out.append(torch.einsum("bts,bsd->btd", score, vh))
        cat = torch.cat(heads_out, dim=-1)
        with _dispatch("O_proj", op="Linear", **tensor_meta("cat", cat)):
            return self.wo(cat)

# --- FeedForward ---
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        super().__init__()
        hd = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier: hd = int(ffn_dim_multiplier * hd)
        hd = multiple_of * ((hd + multiple_of - 1) // multiple_of)
        self.w1 = ColumnParallelLinear(dim, hd, bias=False)
        self.w3 = ColumnParallelLinear(dim, hd, bias=False)
        self.w2 = RowParallelLinear(hd, dim, bias=False, input_is_parallel=True)
    def forward(self, x):
        with _dispatch("Gate", op="Linear", **tensor_meta("x", x)):
            g = self.w1(x); u = self.w3(x)
        with _dispatch("SiLU", op="Activation", **tensor_meta("g", g)):
            act = F.silu(g) * u
        with _dispatch("Down", op="Linear", **tensor_meta("act", act)):
            return self.w2(act)

# --- TransformerBlock & Transformer ---
class TransformerBlock(nn.Module):
    def __init__(self, idx: int, args: ModelArgs):
        super().__init__()
        self.attn = Attention(args)
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn = FeedForward(args.dim, 4*args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.idx = idx
    def forward(self, x, start_pos, freqs_cis, mask=None):
        with _dispatch(f"Block{self.idx}_Res1", op="ResidualAdd", **tensor_meta("x", x)):
            h = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        with _dispatch(f"Block{self.idx}_Res2", op="ResidualAdd", **tensor_meta("h", h)):
            return h + self.ffn(self.ffn_norm(h))

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.tok_embeddings = VocabParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(args.dim, args.vocab_size, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args.dim//args.n_heads, args.max_seq_len*2, args.rope_theta), persistent=False)
    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs = self.freqs_cis[start_pos:start_pos+seqlen].to(h.device)
        mask = None
        if seqlen>1:
            m = torch.full((seqlen,seqlen), float("-inf"), device=h.device)
            m = torch.triu(m, diagonal=1)
            prefix = torch.zeros((seqlen,start_pos), device=h.device)
            mask = torch.hstack([prefix, m]).type_as(h)
        for l in self.layers:
            h = l(h, start_pos, freqs, mask)
        with _dispatch("FinalNorm", op="RMSNorm", **tensor_meta("h", h)):
            h = self.norm(h)
        with _dispatch("LMHead", op="Linear", **tensor_meta("h", h)):
            return self.output(h).float()
