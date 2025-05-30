# perfetto.py
import contextlib
import time
import torch
import torch.distributed as dist
import tg4perfetto
import torch.nn.functional as F

class PerfettoTracer:
    def __init__(self, engines_config):
        """
        engines_config: list of dicts, e.g.
          [
            {'name': 'DotProductEngine', 'optimized_for': 'MatMul'},
            {'name': 'SoftmaxEngine',    'optimized_for': 'Softmax'},
            {'name': 'LayerNormEngine',  'optimized_for': 'Norm'},
            {'name': 'PosEncEngine',     'optimized_for': 'RoPE'},
          ]
        """
        self.engines_config = engines_config
        self.world_size     = dist.get_world_size() if dist.is_initialized() else 1
        self._tracks        = [
            tg4perfetto.track(
                f"{'GPU' if torch.cuda.is_available() else 'CPU'}{rank}-{eng['name']}"
            )
            for rank in range(self.world_size)
            for eng  in self.engines_config
        ]
        self._engine_free_at = [0.0] * len(self._tracks)

    def _pick_engine(self, op_type):
        preferred = [
            i for i, e in enumerate(self.engines_config)
            if e['optimized_for'] == op_type
        ]
        candidates = preferred if preferred else list(range(len(self._tracks)))
        return min(candidates, key=lambda i: self._engine_free_at[i])

    @contextlib.contextmanager
    def dispatch(self, name: str, op: str, **meta):
        meta = dict(meta)
        meta["op"]      = op
        meta["device"]  = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else f"cpu:{dist.get_rank() if dist.is_initialized() else 0}"
        )
        eng   = self._pick_engine(op)
        track = self._tracks[eng]
        cm    = track.trace(name, meta)
        cm.__enter__()
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            dur = time.perf_counter_ns() - start
            self._engine_free_at[eng] = start + dur
            cm.__exit__(None, None, None)

    def tensor_meta(self, name: str, x: torch.Tensor) -> dict:
        return {
            f"{name}_shape": tuple(x.shape),
            f"{name}_dtype": str(x.dtype),
            f"{name}_numel": x.numel(),
            f"{name}_bytes": x.element_size() * x.numel(),
            "device": "cpu",
        }

    def chunked_linear(self, x, weight, name_prefix, n_chunks=None):
        if n_chunks is None:
            n_chunks = len(self.engines_config)
        if not torch.cuda.is_available():
            return torch.cat([F.linear(x, w) for w in torch.chunk(weight, n_chunks, dim=0)], dim=-1)
        w_splits = torch.chunk(weight, n_chunks, dim=0)
        streams  = [torch.cuda.Stream(device=x.device) for _ in w_splits]
        outs     = [None] * n_chunks
        for idx, (w, st) in enumerate(zip(w_splits, streams)):
            with self.dispatch(f"{name_prefix}#{idx}", op="MatMul", **self.tensor_meta("x", x)):
                with torch.cuda.stream(st):
                    outs[idx] = F.linear(x, w)
        torch.cuda.synchronize()
        return torch.cat(outs, dim=-1)

    def proj_with_chunking(self, name: str, x: torch.Tensor, layer, head_dim: int) -> torch.Tensor:
        out = self.chunked_linear(x, layer.weight, name_prefix=name)
        return out.view(x.size(0), x.size(1), -1, head_dim)

# instantiate global tracer
perfetto_tracer = PerfettoTracer([
    {'name': 'DotProductEngine', 'optimized_for': 'MatMul'},
    {'name': 'SoftmaxEngine',    'optimized_for': 'Softmax'},
    {'name': 'LayerNormEngine',  'optimized_for': 'Norm'},
    {'name': 'PosEncEngine',     'optimized_for': 'RoPE'},
])
