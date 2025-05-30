# -----------------------------------------------------------------------------
# model.py (in llama/model.py)
# -----------------------------------------------------------------------------
import math
from dataclasses import dataclass
from typing import Optional

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn

from llama.perfetto import perfetto_tracer

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
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32, device=freqs.device)
    return torch.polar(torch.ones((end, dim//2), device=freqs.device), torch.outer(t, freqs))


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    shape = [d if i in (1, ndim - 1) else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    meta = {**perfetto_tracer.tensor_meta("xq_in", xq), **perfetto_tracer.tensor_meta("xk_in", xk)}
    with perfetto_tracer.dispatch("RoPE", "RoPE", **meta):
        xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_b = reshape_for_broadcast(freqs_cis, xq_c)
        xq_out = torch.view_as_real(xq_c * freqs_b).flatten(3).type_as(xq)
        xk_out = torch.view_as_real(xk_c * freqs_b).flatten(3).type_as(xk)
    return xq_out, xk_out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    meta = perfetto_tracer.tensor_meta("kv_in", x)
    with perfetto_tracer.dispatch("RepeatKV", "Reshape", **meta):
        bs, sl, nh, hd = x.shape
        x2 = x[:, :, :, None, :].expand(bs, sl, nh, n_rep, hd)
        return x2.reshape(bs, sl, nh * n_rep, hd)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        mp = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // mp
        self.n_local_kv_heads = self.n_kv_heads // mp
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(args.dim, args.n_heads*self.head_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.wk = ColumnParallelLinear(args.dim, self.n_kv_heads*self.head_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.wv = ColumnParallelLinear(args.dim, self.n_kv_heads*self.head_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.wo = RowParallelLinear(args.n_heads*self.head_dim, args.dim, bias=False, input_is_parallel=True, init_method=lambda x: x)

        self.register_buffer('cache_k', torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim))
        self.register_buffer('cache_v', torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim))

    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.shape

        with perfetto_tracer.dispatch("Q_proj", "MatMul", **perfetto_tracer.tensor_meta("x", x)):
            xq = self.wq(x)
        with perfetto_tracer.dispatch("K_proj", "MatMul", **perfetto_tracer.tensor_meta("x", x)):
            xk = self.wk(x)
        with perfetto_tracer.dispatch("V_proj", "MatMul", **perfetto_tracer.tensor_meta("x", x)):
            xv = self.wv(x)

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

        meta_qk = {**perfetto_tracer.tensor_meta("xq", xq), **perfetto_tracer.tensor_meta("keysT", keys.transpose(2,3))}
        with perfetto_tracer.dispatch("QK_matmul", "MatMul", **meta_qk):
            scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)

        if mask is not None:
            with perfetto_tracer.dispatch("AddMask", "Add", **perfetto_tracer.tensor_meta("scores_in", scores)):
                scores = scores + mask

        with perfetto_tracer.dispatch("Softmax", "Softmax", **perfetto_tracer.tensor_meta("scores", scores)):
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        with perfetto_tracer.dispatch("V_matmul", "MatMul", **{**perfetto_tracer.tensor_meta("scores", scores), **perfetto_tracer.tensor_meta("values", values)}):
            context = torch.matmul(scores, values)

        with perfetto_tracer.dispatch("ConcatHeads", "Reshape", **perfetto_tracer.tensor_meta("ctx", context)):
            out = context.transpose(1,2).contiguous().view(bsz, seqlen, -1)

        with perfetto_tracer.dispatch("O_proj", "MatMul", **perfetto_tracer.tensor_meta("out", out)):
            return self.wo(out)


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
        with perfetto_tracer.dispatch("FFN_w1", "Linear", **perfetto_tracer.tensor_meta("x", x)):
            h1 = self.w1(x)
        with perfetto_tracer.dispatch("FFN_silu", "Activation", **perfetto_tracer.tensor_meta("h1", h1)):
            h1 = F.silu(h1)
        with perfetto_tracer.dispatch("FFN_w3", "Linear", **perfetto_tracer.tensor_meta("x", x)):
            h3 = self.w3(x)
        with perfetto_tracer.dispatch("FFN_mul", "Mul", **{**perfetto_tracer.tensor_meta("h1", h1), **perfetto_tracer.tensor_meta("h3", h3)}):
            h = h1 * h3
        with perfetto_tracer.dispatch("FFN_w2", "Linear", **perfetto_tracer.tensor_meta("h", h)):
            return self.w2(h)

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm  = RMSNorm(args.dim, eps=args.norm_eps)
        self.attn      = Attention(args)
        self.ffn       = FeedForward(args.dim, 4*args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.layer_id  = layer_id

    def forward(self, x, start_pos, freqs_cis, mask):
        with perfetto_tracer.dispatch(f"Layer{self.layer_id}_AttnNorm", "Norm", **perfetto_tracer.tensor_meta("x", x)):
            x_norm = self.attn_norm(x)
        with perfetto_tracer.dispatch(f"Layer{self.layer_id}_Attention", "Attention"):
            h = x + self.attn(x_norm, start_pos, freqs_cis, mask)
        with perfetto_tracer.dispatch(f"Layer{self.layer_id}_FFNNorm", "Norm", **perfetto_tracer.tensor_meta("h", h)):
            h2 = self.ffn_norm(h)
        with perfetto_tracer.dispatch(f"Layer{self.layer_id}_FeedForward", "FFN"):
            out = h + self.ffn(h2)
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
        with perfetto_tracer.dispatch("Embed", "Embedding", **perfetto_tracer.tensor_meta("tokens", tokens)):
            h = self.tok_embeddings(tokens)
        freqs = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            def _mk_mask():
                m  = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
                mm = torch.triu(m, 1)
                return torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mm]).type_as(h)
            with perfetto_tracer.dispatch("MakeMask", "Mask", **perfetto_tracer.tensor_meta("h_in", h)):
                mask = _mk_mask()
        out = h
        for layer in self.layers:
            out = layer(out, start_pos, freqs, mask)
        with perfetto_tracer.dispatch("FinalNorm", "Norm", **perfetto_tracer.tensor_meta("out_in", out)):
            out = self.final_norm(out)
        with perfetto_tracer.dispatch("OutputProj", "Linear", **perfetto_tracer.tensor_meta("out2", out)):
            logits = self.output(out).float()
        return logits
