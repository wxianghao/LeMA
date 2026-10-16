"""
E11: training CNN-1B on MNIST with LeMA on P GPUs, to show that LeMA trains a
billion-parameter model with exact LM steps.

The run takes --steps LM steps on global batches of N samples. The k-th global batch is
the k-th group of N samples of a seeded permutation of the MNIST training set, with a new
permutation per epoch, and process p takes the p-th contiguous N/P of it. Each LM step
evaluates the Gram matrix once and then solves with increasing damping until a trial
reduces the loss on the batch, for at most max_iters trials (LeMA defaults, as in E9).
Before training and after every step, the processes evaluate together the CE loss on the
full training and test sets and the test accuracy. The script appends one JSON line per
evaluation to --out as soon as it is made, so an interrupted run keeps its records.

Launch (from the repository root):
    torchrun --standalone --nproc_per_node 8 paper-experiments/large_training/train.py \
        --model cnn-1b --batch-size 256 --shard-size 4 --slice-size 4 --out e11.jsonl
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision

# Ensure correct import of lema and paper-experiments/common
exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_dir = os.path.dirname(exp_dir)
for path in (repo_dir, exp_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

import lema

from common.models import MODELS, build_model, residual_fn


def load_mnist(data_dir: str, train: bool, device):
    dataset = torchvision.datasets.MNIST(data_dir, train=train, download=False)
    x = dataset.data.unsqueeze(1).float().div_(255).sub_(0.1307).div_(0.3081)
    return x.to(device), dataset.targets.to(device)


def all_reduce_max(value: float, group) -> float:
    t = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return t.item()


@torch.no_grad()
def evaluate(model, x, y, rank, world_size, chunk):
    """Mean CE loss and accuracy on (x, y), each process evaluating a contiguous 1/P of it."""
    n = x.size(0)
    lo, hi = n * rank // world_size, n * (rank + 1) // world_size
    stats = torch.zeros(2, dtype=torch.float64, device=x.device)
    for i in range(lo, hi, chunk):
        out = model(x[i : min(i + chunk, hi)])
        target = y[i : min(i + chunk, hi)]
        stats[0] += F.nll_loss(out, target, reduction="sum").double()
        stats[1] += (out.argmax(dim=1) == target).sum()
    dist.all_reduce(stats)
    return stats[0].item() / n, stats[1].item() / n


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", choices=MODELS, default="cnn-1b")
    parser.add_argument("--steps", type=int, default=30, help="number of LM steps")
    parser.add_argument("--batch-size", type=int, default=256, help="global batch size N")
    parser.add_argument("--shard-size", type=int, default=4, help="shard size S")
    parser.add_argument("--slice-size", type=int, default=4, help="slice size L")
    parser.add_argument("--damp-start", type=float, default=1e-3, help="initial damping factor (LeMA default)")
    parser.add_argument("--damp-min", type=float, default=1e-1, help="lower bound of the damping factor (LeMA default)")
    parser.add_argument("--eval-chunk", type=int, default=500, help="samples per forward pass of the evaluation")
    parser.add_argument("--data-dir", type=str, default=".data")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    gloo = dist.new_group(backend="gloo")

    n, s, l = args.batch_size, args.shard_size, args.slice_size
    if n % world_size != 0 or (n // world_size) % s != 0 or s % l != 0:
        raise ValueError(f"Require P | N, S | N/P and L | S, got N={n}, P={world_size}, S={s}, L={l}")
    block = n // world_size

    x_train, y_train = load_mnist(args.data_dir, True, device)
    x_test, y_test = load_mnist(args.data_dir, False, device)
    steps_per_epoch = x_train.size(0) // n  # drop the last partial batch

    # Initialize the model directly on the device, with the same parameters as in E4
    torch.manual_seed(args.seed)
    with torch.device(device):
        model = build_model(args.model)
    optim = lema.LeMA(model=model, residual_fn=residual_fn, damp_start=args.damp_start, damp_min=args.damp_min)
    num_params = optim._flat.numel()

    # Record the damping factor and the loss of every trial; LeMA calls _compute_loss once
    # per trial, after theta' = theta + delta and before accepting or rejecting it
    trials = []
    compute_loss = optim._compute_loss

    def record_trial(*args, **kwargs):
        damp = optim._damping.val
        loss = compute_loss(*args, **kwargs)
        trials.append((damp, loss))
        return loss

    optim._compute_loss = record_trial

    base = {"model": args.model, "params": num_params, "world_size": world_size, "batch_size": n, "shard_size": s,
            "slice_size": l, "max_iters": optim._max_iters, "damp_start": args.damp_start,
            "damp_min": args.damp_min, "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__}
    if rank == 0:
        print(f"model: {args.model} | M: {num_params:,} | P: {world_size} | N: {n} | S: {s} | L: {l} | "
              f"steps: {args.steps}", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def log(step, train_time, extra):
        tic = time.perf_counter()
        train_loss, _ = evaluate(model, x_train, y_train, rank, world_size, args.eval_chunk)
        test_loss, test_acc = evaluate(model, x_test, y_test, rank, world_size, args.eval_chunk)
        if rank == 0:
            record = {**base, "step": step, "epoch": step / steps_per_epoch, "time_s": train_time,
                      "train_loss": train_loss, "test_loss": test_loss, "test_acc": test_acc,
                      "eval_time_s": time.perf_counter() - tic, **extra}
            with open(args.out, "a") as f:
                f.write(json.dumps(record) + "\n")
            summary = " | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in extra.items()
                                 if k not in ("trial_damps", "trial_losses"))
            print(f"step {step:3d} | time {train_time:7.1f} s | train loss {train_loss:.4f} | test loss "
                  f"{test_loss:.4f} | test acc {test_acc:.2%}" + (f" | {summary}" if summary else ""), flush=True)

    train_time = 0.0
    log(0, train_time, {})
    perm = None
    for k in range(args.steps):
        epoch, j = divmod(k, steps_per_epoch)
        if j == 0:
            perm = torch.randperm(x_train.size(0), generator=torch.Generator().manual_seed(args.seed + epoch))
            perm = perm.to(device)
        idx = perm[j * n + rank * block : j * n + (rank + 1) * block]
        x, y = x_train[idx], y_train[idx]

        trials.clear()
        dist.barrier()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        tic = time.perf_counter()
        res = optim.step(x, y, shard_size=s, slice_size=l)
        torch.cuda.synchronize(device)
        elapsed = all_reduce_max(time.perf_counter() - tic, gloo)
        peak_gb = all_reduce_max(torch.cuda.max_memory_allocated(device) / 1e9, gloo)
        train_time += elapsed

        # An accepted trial becomes the loss of the step, otherwise the step keeps the initial loss
        accepted = bool(trials) and trials[-1][1] / res.batch_size == res.loss
        log(k + 1, train_time, {
            "form": "standard" if res.overdetermined else "dual",
            "step_time_s": elapsed,
            "batch_loss": res.loss,
            "trials": res.iterations,
            "rejected": res.iterations - int(accepted),
            "accepted": accepted,
            "terminate": res.terminate,
            "damp": res.damp,
            "trial_damps": [d for d, _ in trials],
            "trial_losses": [loss / res.batch_size for _, loss in trials],
            "peak_mem_gb": peak_gb,
        })

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
