"""LM whose standard form is solved by preconditioned conjugate gradients (PCG)."""

import os
import sys

import torch

# Ensure correct import of lema
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

import lema

from lema.util import iter_batches


class PCGLM(lema.LeMA):
    """LM that solves (J^T J + lambda D) u = J^T r matrix-free on a single GPU.

    Each CG iteration multiplies by J^T J through a JVP and a VJP over the batch, in slices
    of L samples, so that neither J nor J^T J is formed. D = diag(J^T J) comes from Jacobian
    shards and is clamped as in LeMA, and it also serves as the Jacobi preconditioner. The
    damping strategy and the acceptance test are those of LeMA.
    """

    def __init__(self, *args, cg_iters: int = 100, cg_tol: float = 1e-6, **kwargs) -> None:
        super().__init__(*args, form="standard", **kwargs)
        if self._world_size != 1:
            raise RuntimeError("PCGLM runs on a single process.")
        self._cg_iters = cg_iters
        self._cg_tol = cg_tol
        self.cg_iterations: list = []

    def _gauss_newton(self, x, y, v, slice_size):
        """J^T J v, slice by slice."""
        out = torch.zeros_like(self._flat)
        for start, end in iter_batches(x.size(0), slice_size):
            out.add_(self.vjp(x[start:end], y[start:end], self.jvp(x[start:end], y[start:end], v)))
        return out

    def _pcg(self, x, y, b, d, damp, slice_size):
        """Solve (J^T J + damp D) u = b with the preconditioner (1 + damp) D."""
        precond = 1.0 / ((1.0 + damp) * d)
        u = torch.zeros_like(b)
        res = b.clone()
        z = precond * res
        p = z.clone()
        rz = torch.dot(res, z)
        b_norm = b.norm()
        for k in range(1, self._cg_iters + 1):
            ap = self._gauss_newton(x, y, p, slice_size).add_(d * p, alpha=damp)
            alpha = rz / torch.dot(p, ap)
            u.add_(p, alpha=alpha)
            res.sub_(ap, alpha=alpha)
            if res.norm() <= self._cg_tol * b_norm:
                break
            z = precond * res
            rz_new = torch.dot(res, z)
            p.mul_(rz_new / rz).add_(z)
            rz = rz_new
        return u, k

    def step(self, x, y, shard_size=None, slice_size=None):
        n = x.size(0)
        shard_size = shard_size or n
        slice_size = slice_size or shard_size
        # D = diag(J^T J) and the residuals from Jacobian shards
        d = torch.zeros_like(self._flat)
        r = self._template.new_empty(n)
        for start, end in iter_batches(n, shard_size):
            j_shard, r[start:end] = self.jacrev(x[start:end], y[start:end], has_residual=True, slice_size=slice_size)
            d.add_(j_shard.square_().sum(dim=0))
            del j_shard
        d.clamp_(min=self._d_min_ratio * d.mean())
        # b = J^T r
        b = torch.zeros_like(self._flat)
        for start, end in iter_batches(n, slice_size):
            b.add_(self.vjp(x[start:end], y[start:end], r[start:end]))
        loss = r.square().sum().item()

        iterations, terminate = 0, False
        while iterations < self._max_iters:
            iterations += 1
            update, cg_iters = self._pcg(x, y, b, d, self._damping.val, slice_size)
            self.cg_iterations.append(cg_iters)
            self._flat.sub_(update)
            new_loss = self._compute_loss(x, y, slice_size)
            if new_loss < loss:
                loss = new_loss
                self._damping.on_success()
                self._backup.copy_(self._flat)
                break
            self._flat.copy_(self._backup)
            self._damping.on_fail()
            if self._damping.terminate():
                terminate = True
                break
        return lema.LeMAResult(loss=loss / n, iterations=iterations, terminate=terminate, damp=self._damping.val,
                               overdetermined=True, batch_size=n)
