"""
E8: peak memory of one LM trial step against the model size on one GPU.

Each invocation measures one (variant, model) pair, where the variants form a 2 x 2
design of the form of the LM equation and the treatment of the Jacobian:

    full-standard   standard form with the full N x M Jacobian, i.e., the classic LM
    full-dual       dual form with the full N x M Jacobian
    lema-standard   LeMA's standard form, with Jacobian shards of S rows
    lema-dual       LeMA's dual form, with Jacobian shards of S rows

The full-Jacobian variants are written as lean as their algorithm allows: one Jacobian,
D from its column norms without a temporary copy, in-place scaling and damping. All
variants evaluate per-sample VJPs in slices of L samples. The script appends one JSON
line to --out with the peak allocated memory, or oom = true if the step ran out of memory.

Launch (from the repository root):
    torchrun --standalone --nproc_per_node 1 paper-experiments/memory_scaling/run.py \
        --variant lema-dual --hidden 108378 --out memory.jsonl
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torchvision

# Ensure correct import of lema and paper-experiments/common
exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_dir = os.path.dirname(exp_dir)
for path in (repo_dir, exp_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

import lema

from common.models import WideCNN, residual_fn

VARIANTS = ("full-standard", "full-dual", "lema-standard", "lema-dual")


def load_mnist(data_dir: str, n: int):
    trainset = torchvision.datasets.MNIST(data_dir, train=True, download=False)
    x = trainset.data[:n].unsqueeze(1).float().div_(255).sub_(0.1307).div_(0.3081)
    return x, trainset.targets[:n]


def is_oom(e: Exception) -> bool:
    # cuDNN reports a failed workspace allocation as an internal error
    return isinstance(e, torch.OutOfMemoryError) or "ALLOCATION_FAILED" in str(e) or "out of memory" in str(e)


def full_standard_step(jm, x, y, damp, slice_size):
    j, r = jm.jacrev(x, y, has_residual=True, slice_size=slice_size)
    jtj = j.T @ j
    jtr = j.T @ r
    del j
    d = jtj.diagonal().clamp(min=lema.LeMA._d_min_ratio * jtj.diagonal().mean())
    jtj.diagonal().add_(d, alpha=damp)
    jm._flat.sub_(torch.linalg.solve(jtj, jtr))


def full_dual_step(jm, x, y, damp, slice_size):
    j, r = jm.jacrev(x, y, has_residual=True, slice_size=slice_size)
    d = torch.linalg.vector_norm(j, dim=0).square_()
    d_inv_sqrt = d.clamp_(min=lema.LeMA._d_min_ratio * d.mean()).rsqrt_()
    # J D^{-1/2} in place, so that (J D^{-1/2})(J D^{-1/2})^T = J D^{-1} J^T
    j.mul_(d_inv_sqrt)
    gram = j @ j.T
    gram.diagonal().add_(damp)
    v = torch.linalg.solve(gram, r)
    jm._flat.sub_(d_inv_sqrt * (j.T @ v))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--hidden", type=int, required=True, help="fc1 width of WideCNN")
    parser.add_argument("--batch-size", type=int, default=256, help="global batch size N")
    parser.add_argument("--shard-size", type=int, default=4, help="shard size S of the LeMA variants")
    parser.add_argument("--slice-size", type=int, default=4, help="slice size L")
    parser.add_argument("--damp", type=float, default=1e-3, help="damping factor of the trial step")
    parser.add_argument("--data-dir", type=str, default=".data")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    n, s, l = args.batch_size, args.shard_size, args.slice_size
    x, y = load_mnist(args.data_dir, n)
    x, y = x.to(device), y.to(device)

    torch.manual_seed(args.seed)
    with torch.device(device):
        model = WideCNN(args.hidden)
    m = sum(p.numel() for p in model.parameters())
    record = {"variant": args.variant, "hidden": args.hidden, "params": m, "batch_size": n, "slice_size": l,
              "shard_size": s if args.variant.startswith("lema") else n,
              "gpu": torch.cuda.get_device_name(device), "gpu_mem_gb": torch.cuda.get_device_properties(device).total_memory / 1e9,
              "torch": torch.__version__}
    print(f"{args.variant} | M: {m:,} | N: {n} | S: {record['shard_size']} | L: {l}", flush=True)

    torch.cuda.reset_peak_memory_stats(device)
    tic = time.perf_counter()
    try:
        if args.variant.startswith("lema"):
            form = args.variant.split("-")[1]
            optim = lema.LeMA(model=model, residual_fn=residual_fn, max_iters=1, damp_start=args.damp, form=form)
            optim.step(x, y, shard_size=s, slice_size=l)
        else:
            jm = lema.JacobianModel(model, residual_fn)
            step = full_standard_step if args.variant == "full-standard" else full_dual_step
            with torch.no_grad():
                step(jm, x, y, args.damp, l)
                # Evaluate the loss of the trial step, as LeMA does
                residual_fn(jm(x), y).square().sum().item()
        torch.cuda.synchronize(device)
        record.update(oom=False, time_s=time.perf_counter() - tic)
    except Exception as e:
        if not is_oom(e):
            raise
        record.update(oom=True, error=str(e).splitlines()[0])
    record["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    print(("OOM" if record["oom"] else f"time {record['time_s']:.1f} s") + f" | peak memory {record['peak_mem_gb']:.2f} GB",
          flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(record) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
