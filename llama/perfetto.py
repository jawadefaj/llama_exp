import contextlib
import time
import torch
import torch.distributed as dist
import tg4perfetto
import torch.nn.functional as F

class PerfettoTracer:
    def __init__(self, n_engines=4):
        self.n_engines = n_engines
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._tracks = [
            tg4perfetto.track(
                f"{'GPU' if torch.cuda.is_available() else 'CPU'}{rank}-Engine-{eng}"
            )
            for rank in range(self.world_size)
            for eng in range(self.n_engines)
        ]
        self._engine_free_at = [0.0] * len(self._tracks)

    def _pick_engine(self):
        # Pick the engine that's available soonest
        return min(range(len(self._tracks)), key=self._engine_free_at.__getitem__)

    @contextlib.contextmanager
    def dispatch(self, name: str, op: str, **meta):
        meta = dict(meta)
        meta["op"] = op
        meta["device"] = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else f"cpu:{dist.get_rank() if dist.is_initialized() else 0}"
        eng = self._pick_engine()
        track = self._tracks[eng]
        cm = track.trace(name, meta)
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

    # Optional: chunked linear (from your previous pattern)
    def chunked_linear(self, x, weight, name_prefix, n_chunks=None):
        if n_chunks is None:
            n_chunks = self.n_engines
        if not torch.cuda.is_available():
            return torch.cat([F.linear(x, w) for w in torch.chunk(weight, n_chunks, dim=0)], dim=-1)
        w_splits = torch.chunk(weight, n_chunks, dim=0)
        streams = [torch.cuda.Stream(device=x.device) for _ in w_splits]
        outs = [None] * n_chunks
        for idx, (w, st) in enumerate(zip(w_splits, streams)):
            with self.dispatch(f"{name_prefix}#{idx}", op="Linear", **self.tensor_meta("x", x)):
                with torch.cuda.stream(st):
                    outs[idx] = F.linear(x, w)
        torch.cuda.synchronize()
        return torch.cat(outs, dim=-1)

    def proj_with_chunking(self, name: str, x: torch.Tensor, layer, head_dim: int) -> torch.Tensor:
        out = self.chunked_linear(x, layer.weight, name_prefix=name)
        return out.view(x.size(0), x.size(1), -1, head_dim)

# Create a global tracer
perfetto_tracer = PerfettoTracer(n_engines=4)
