"""
E4: strong scaling of LeMA on a fixed global batch.

Every invocation measures one (model, P) configuration. The global batch of N
samples is identical for every P: step k uses MNIST training samples
[k*N, (k+1)*N), and process p takes the p-th contiguous N/P of them. Each LM
iteration performs exactly one trial solve (max_iters=1), so the work per step is
independent of whether the trial is accepted.

Per configuration, the script runs
    --warmup steps            not recorded
    --steps steps             end-to-end step time, no instrumentation
    --breakdown-steps steps   per-phase time, synchronized instrumentation
and appends one JSON line per recorded step to --out (rank 0 only).

Launch (from the repository root):
    torchrun --standalone --nproc_per_node 8 paper-experiments/strong_scaling/run.py \
        --model cnn-1b --batch-size 256 --shard-size 4 --slice-size 4
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torchvision

from contextlib import nullcontext

# Ensure correct import of lema and paper-experiments/common
exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_dir = os.path.dirname(exp_dir)
for path in (repo_dir, exp_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

import lema

from common.models import WideCNN, PRESETS, residual_fn
from common.timing import PhaseTimer


def load_mnist(data_dir: str):
    trainset = torchvision.datasets.MNIST(data_dir, train=True, download=False)
    x = trainset.data.unsqueeze(1).float().div_(255).sub_(0.1307).div_(0.3081)
    return x, trainset.targets


def all_reduce_max(value: float, group) -> float:
    t = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return t.item()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", choices=list(PRESETS), default="cnn-1b")
    parser.add_argument("--hidden", type=int, default=None, help="fc1 width, overrides --model")
    parser.add_argument("--batch-size", type=int, default=256, help="global batch size N")
    parser.add_argument("--shard-size", type=int, default=4, help="shard size S")
    parser.add_argument("--slice-size", type=int, default=4, help="slice size L")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--breakdown-steps", type=int, default=1)
    parser.add_argument("--data-dir", type=str, default=".data")
    parser.add_argument("--out", type=str, default="paper-experiments/strong_scaling/results/strong_scaling.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    gloo = dist.new_group(backend="gloo")

    n, s, l = args.batch_size, args.shard_size, args.slice_size
    if n % world_size != 0 or (n // world_size) % s != 0 or s % l != 0:
        raise ValueError(f"Require P | N, S | N/P and L | S, got N={n}, P={world_size}, S={s}, L={l}")
    block_size = n // world_size
    total_steps = args.warmup + args.steps + args.breakdown_steps

    x_all, y_all = load_mnist(args.data_dir)
    if total_steps * n > x_all.size(0):
        raise ValueError(f"{total_steps} steps of {n} samples exceed the training set")

    # Initialize the model directly on the device to avoid a host copy of 4 GB per process
    hidden = args.hidden if args.hidden is not None else PRESETS[args.model]
    model_name = args.model if args.hidden is None else f"cnn-h{hidden}"
    torch.manual_seed(args.seed)
    with torch.device(device):
        model = WideCNN(hidden)
    optim = lema.LeMA(model=model, residual_fn=residual_fn, max_iters=1)
    num_params = optim._flat.numel()

    if rank == 0:
        print(
            f"model: {model_name} | M: {num_params:,} | P: {world_size} | N: {n} | S: {s} | L: {l} | "
            f"Gram: {n * n * 4 / 1e9:.3f} GB | 2 shards: {2 * s * num_params * 4 / 1e9:.2f} GB",
            flush=True,
        )

    env = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }
    records = []
    for k in range(total_steps):
        kind = "warmup" if k < args.warmup else "timed" if k < args.warmup + args.steps else "breakdown"
        start = k * n + rank * block_size
        x = x_all[start : start + block_size].to(device, non_blocking=True)
        y = y_all[start : start + block_size].to(device, non_blocking=True)

        timer = PhaseTimer(optim) if kind == "breakdown" else None
        dist.barrier()
        torch.cuda.synchronize(device)
        tic = time.perf_counter()
        with timer.patch() if timer is not None else nullcontext():
            res = optim.step(x, y, shard_size=s, slice_size=l)
        torch.cuda.synchronize(device)
        elapsed = all_reduce_max(time.perf_counter() - tic, gloo)

        if rank == 0:
            print(f"step {k} ({kind}): {elapsed:.3f} s | loss: {res.loss:.4e}", flush=True)
        if kind == "warmup":
            continue
        record = {
            "model": model_name,
            "params": num_params,
            "world_size": world_size,
            "batch_size": n,
            "shard_size": s,
            "slice_size": l,
            "kind": kind,
            "step": k,
            "time_s": elapsed,
            "loss": res.loss,
        }
        if timer is not None:
            # Phases of rank 0; collectives include waiting for the other ranks
            record["phases"] = timer.breakdown(elapsed)
        records.append(record)

    peak_gb = all_reduce_max(torch.cuda.max_memory_allocated(device) / 1e9, gloo)
    if rank == 0:
        print(f"peak memory (max over ranks): {peak_gb:.2f} GB", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            for record in records:
                f.write(json.dumps({**record, "peak_mem_gb": peak_gb, **env}) + "\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
