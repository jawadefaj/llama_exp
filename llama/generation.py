# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of
# the Llama 3 Community License Agreement.
# ---------------------------------------------------------------------------
# generation.py – single version that works on               ­
#  • CUDA builds  (NCCL backend, half/BF16, GPU tensors) and
#  • CPU-only builds (GLOO backend, FP32, CPU tensors)
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama._model import ModelArgs, Transformer
from llama.tokenizer import ChatFormat, Dialog, Message, Tokenizer


# --------------------------------------------------------------------------- #
# typed-dict helpers for public API                                           #
# --------------------------------------------------------------------------- #
class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]
    logprobs: List[float]


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]
    logprobs: List[float]


# --------------------------------------------------------------------------- #
# main wrapper                                                                
# --------------------------------------------------------------------------- #
class Llama:
    # --------------------------------------------------------------------- #
    # BUILD                                                                  #
    # --------------------------------------------------------------------- #
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """Load weights, tokenizer and return ready-to-use generator."""
        assert 1 <= max_seq_len <= 8192, "max_seq_len must be 1-8192"
        assert os.path.isdir(ckpt_dir), f"Checkpoint dir '{ckpt_dir}' not found"
        assert os.path.isfile(tokenizer_path), f"Tokenizer '{tokenizer_path}' missing"

        # ---- 0. distributed init on working backend ----------------------
        if not torch.distributed.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            if backend == "nccl":
                torch.distributed.init_process_group("nccl")
            else:  # single-process rendez-vous, needs no env vars
                torch.distributed.init_process_group(
                    backend="gloo",
                    init_method=f"file://{tempfile.mkstemp()[1]}",
                    rank=0,
                    world_size=1,
                )

        # ---- 1. model-parallel init --------------------------------------
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        torch.manual_seed(seed)
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        # ---- 2. load checkpoint ------------------------------------------
        t0 = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert checkpoints, f"No .pth files in {ckpt_dir}"
        assert model_parallel_size == len(checkpoints), (
            f"Checkpoint shards={len(checkpoints)} but MP={model_parallel_size}"
        )

        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        with open(Path(ckpt_dir) / "params.json") as f:
            params = json.load(f)
        params.pop("use_scaled_rope", None)  # not used here

        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )

        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words

        # half / bf16 only on GPU
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            else:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Transformer(model_args).to(device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - t0:.2f}s")

        return Llama(model, tokenizer)

    # --------------------------------------------------------------------- #
    # INIT / helpers                                                        #
    # --------------------------------------------------------------------- #
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)

    # --------------------------------------------------------------------- #
    # TOKEN-LEVEL GENERATION                                                #
    # --------------------------------------------------------------------- #
    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size

        min_prompt_len = min(map(len, prompt_tokens))
        max_prompt_len = max(map(len, prompt_tokens))
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        device = next(self.model.parameters()).device
        pad_id = self.tokenizer.pad_id

        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.zeros(bsz, dtype=torch.bool, device=device)
        input_text_mask = tokens != pad_id

        if min_prompt_len == total_len:
            logits = self.model(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                logits.transpose(1, 2), tokens, reduction="none", ignore_index=pad_id
            )

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens), device=device)

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model(tokens[:, prev_pos:cur_pos], prev_pos)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.view(-1)
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    logits.transpose(1, 2),
                    tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )

            eos_reached |= (~input_text_mask[:, cur_pos]) & torch.isin(next_token, stop_tokens)
            prev_pos = cur_pos
            if eos_reached.all():
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()

        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]

            for stop_token in self.tokenizer.stop_tokens:
                if stop_token in toks:
                    idx = toks.index(stop_token)
                    toks = toks[:idx]
                    if probs is not None:
                        probs = probs[:idx]
                    break

            out_tokens.append(toks)
            out_logprobs.append(probs)

        return out_tokens, (out_logprobs if logprobs else None)

    # --------------------------------------------------------------------- #
    # HIGH-LEVEL HELPERS                                                    #
    # --------------------------------------------------------------------- #
    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1

        prompt_tokens = [self.tokenizer.encode(p, bos=True, eos=False) for p in prompts]
        gen_tokens, gen_logprobs = self.generate(
            prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )

        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(toks),
                    "tokens": [self.tokenizer.decode([x]) for x in toks],
                    "logprobs": lp,
                }
                for toks, lp in zip(gen_tokens, gen_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(toks)} for toks in gen_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1

        prompt_tokens = [self.formatter.encode_dialog_prompt(d) for d in dialogs]
        gen_tokens, gen_logprobs = self.generate(
            prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )

        if logprobs:
            return [
                {
                    "generation": {"role": "assistant", "content": self.tokenizer.decode(t)},
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": lp,
                }
                for t, lp in zip(gen_tokens, gen_logprobs)
            ]
        return [
            {"generation": {"role": "assistant", "content": self.tokenizer.decode(t)}}
            for t in gen_tokens
        ]


# --------------------------------------------------------------------------- #
# utilities                                                                   #
# --------------------------------------------------------------------------- #
def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling on a probability vector."""
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_cum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_cum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)
