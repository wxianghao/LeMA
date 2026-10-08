import time
import torch
import torch.distributed as dist

from collections import defaultdict
from contextlib import contextmanager

PHASES = ("jacobian", "gram", "vjp", "solve", "comm", "other")


class PhaseTimer:
    """Attribute the wall-clock time of ``LeMA.step`` to phases.

    Rather than copying the step into this repository, ``patch`` temporarily wraps
    the operations that ``lema/lema.py`` calls by attribute lookup:

        jacobian  LeMA.jacrev            Jacobian shard evaluation
        gram      LeMA._gram_block       Gram block products of the dual form (split-K or torch.mm)
                  LeMA._gram_update      J^T J and J^T r accumulation of the standard form
        vjp       LeMA.vjp               J^T v of the dual form
        solve     torch.linalg.solve     damped linear solve
        comm      dist.all_reduce, dist.all_gather_single, dist.all_gather_object

    Everything else (loss evaluation, element-wise ops) is reported as "other".
    Every wrapped call synchronizes the device before and after, so a collective's
    time includes waiting for the slowest rank. Nested calls (e.g. a collective inside
    another wrapped call) are attributed to the outermost wrapped call.
    """

    def __init__(self, optim):
        self.optim = optim
        self.times = defaultdict(float)
        self._depth = 0

    def _wrap(self, fn, phase):
        def wrapped(*args, **kwargs):
            if self._depth > 0:
                return fn(*args, **kwargs)
            self._depth += 1
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                self.times[phase] += time.perf_counter() - start
                self._depth -= 1

        return wrapped

    @contextmanager
    def patch(self):
        targets = [
            (self.optim, "jacrev", "jacobian"),
            (self.optim, "vjp", "vjp"),
            (self.optim, "_gram_block", "gram"),
            (self.optim, "_gram_update", "gram"),
            (torch.linalg, "solve", "solve"),
            (dist, "all_reduce", "comm"),
            (dist, "all_gather_single", "comm"),
            (dist, "all_gather_object", "comm"),
        ]
        originals = []
        try:
            for obj, name, phase in targets:
                orig = getattr(obj, name)
                originals.append((obj, name, orig))
                setattr(obj, name, self._wrap(orig, phase))
            yield self
        finally:
            for obj, name, orig in originals:
                if obj is self.optim:
                    # Drop the instance attribute to fall back to the class method
                    delattr(obj, name)
                else:
                    setattr(obj, name, orig)

    def breakdown(self, total: float) -> dict:
        phases = {phase: self.times.get(phase, 0.0) for phase in PHASES if phase != "other"}
        phases["other"] = max(0.0, total - sum(phases.values()))
        return phases
