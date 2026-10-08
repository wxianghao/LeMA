"""
E7: parameter update of the standard form vs the dual form on one GPU.

From the same parameters, batch, and damping factor, LeMA performs one trial step
with each form, and the script records the update of the trial, i.e., theta' - theta,
before LeMA accepts or rejects it. An FP64 run of the cheaper form serves as the
reference. Each record is one JSON line in --out with the relative differences

    diff_forms   ||u_standard - u_dual|| / ||u_ref||
    err_standard ||u_standard - u_ref|| / ||u_ref||
    err_dual     ||u_dual - u_ref|| / ||u_ref||

and the numbers of diagonal elements of J^T J that are zero, i.e., parameters without
curvature such as those of dead ReLU units, and that both forms clamp to
LeMA._d_min_ratio of their mean.

Launch (from the repository root):
    torchrun --standalone --nproc_per_node 1 paper-experiments/update_equivalence/run.py --out e7.jsonl
"""

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist
import torchvision

from torch import nn

# Ensure correct import of lema and paper-experiments/common
exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_dir = os.path.dirname(exp_dir)
for path in (repo_dir, exp_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

import lema

from lema.util import iter_batches
from common.models import LeNet5, residual_fn


class MLP(nn.Module):
    """784-16-10 MLP with 12,730 parameters, small enough for N > M on MNIST."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 16), nn.ReLU(), nn.Linear(16, 10), nn.LogSoftmax(dim=1))

    def forward(self, x):
        return self.net(x)


MODELS = {"lenet5": LeNet5, "mlp": MLP}
# model, global batch size N; N < M for LeNet-5, and both N < M and N > M for the MLP
CONFIGS = [("lenet5", 1024), ("lenet5", 8192), ("mlp", 4096), ("mlp", 16384)]


def load_mnist(data_dir: str):
    trainset = torchvision.datasets.MNIST(data_dir, train=True, download=False)
    x = trainset.data.unsqueeze(1).float().div_(255).sub_(0.1307).div_(0.3081)
    return x, trainset.targets


def trial_update(name, form, damp, dtype, x, y, shard, slice_, seed):
    """Update of one trial step of LeMA from freshly initialized parameters."""
    torch.manual_seed(seed)
    with torch.device(x.device):
        model = MODELS[name]()
    optim = lema.LeMA(model=model, residual_fn=residual_fn, model_dtype=dtype, optim_dtype=dtype, max_iters=1,
                      damp_start=damp, form=form)
    updates = []
    compute_loss = optim._compute_loss

    def record_update(*args, **kwargs):
        # Called right after theta' = theta + delta, before the step is accepted or rejected
        updates.append((optim._flat - optim._backup).double())
        return compute_loss(*args, **kwargs)

    optim._compute_loss = record_update
    res = optim.step(x.to(dtype), y, shard_size=shard, slice_size=slice_)
    assert res.overdetermined == (form == "standard") and len(updates) == 1
    return updates[0], optim


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=[f"{name}:{n}" for name, n in CONFIGS],
                        help="model:N pairs")
    parser.add_argument("--damps", type=float, nargs="+", default=[1e-3, 1.0, 1e3], help="damping factors")
    parser.add_argument("--shard-size", type=int, default=1024, help="shard size S, at most N")
    parser.add_argument("--slice-size", type=int, default=256, help="slice size L")
    parser.add_argument("--data-dir", type=str, default=".data")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    x_all, y_all = load_mnist(args.data_dir)
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for config in args.configs:
        name, n = config.split(":")[0], int(config.split(":")[1])
        x, y = x_all[:n].to(device), y_all[:n].to(device)
        s = min(args.shard_size, n)
        l = min(args.slice_size, s)

        # Fractions of diag(J^T J) without curvature and under the clamp of both forms, in FP64
        torch.manual_seed(args.seed)
        with torch.device(device):
            probe = lema.JacobianModel(MODELS[name](), residual_fn, model_dtype=torch.float64)
        d = torch.zeros_like(probe._flat)
        for start, end in iter_batches(n, s):
            d.add_(probe.jacrev(x[start:end].double(), y[start:end], slice_size=l).square().sum(dim=0))
        m = d.numel()
        zero = int((d == 0).sum())
        clamped = int((d < lema.LeMA._d_min_ratio * d.mean()).sum())
        del probe, d
        # FP64 reference with the form whose Gram matrix is smaller
        ref_form = "dual" if m > n else "standard"

        for damp in args.damps:
            u_ref, _ = trial_update(name, ref_form, damp, torch.float64, x, y, s, l, args.seed)
            u_std, _ = trial_update(name, "standard", damp, torch.float32, x, y, s, l, args.seed)
            torch.cuda.empty_cache()
            u_dual, _ = trial_update(name, "dual", damp, torch.float32, x, y, s, l, args.seed)
            torch.cuda.empty_cache()
            norm = u_ref.norm().item()
            record = {
                "model": name, "params": m, "batch_size": n, "shard_size": s, "slice_size": l, "damp": damp,
                "ref_form": ref_form, "update_norm": norm,
                "diff_forms": (u_std - u_dual).norm().item() / norm,
                "err_standard": (u_std - u_ref).norm().item() / norm,
                "err_dual": (u_dual - u_ref).norm().item() / norm,
                "num_zero": zero, "num_clamped": clamped, **env,
            }
            print(f"{name:6s} M={m:6,} N={n:6,} damp={damp:7.0e} | ||u||={norm:.3e} | std vs dual "
                  f"{record['diff_forms']:.1e} | std vs ref {record['err_standard']:.1e} | dual vs ref "
                  f"{record['err_dual']:.1e} | D = 0: {zero:,} ({zero / m:.2%}), clamped: {clamped:,} "
                  f"({clamped / m:.2%})",
                  flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(record) + "\n")
            del u_ref, u_std, u_dual

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
