"""
E9: training a 784-64-10 MLP on MNIST with LeMA (dual or standard form), LM with a PCG solver, and Adam.

All methods start from the same parameters and minimize the same CE loss. The LM methods
take steps on global batches of N samples, the k-th of which is the same for every
method and every number of processes P, and process p takes the p-th contiguous N/P of
it. Adam takes steps on mini-batches of B samples on one GPU. Every eval_every steps,
rank 0 evaluates the CE loss on the full training set and the accuracy on the test set;
the reported time accumulates only the training steps. The peak memory covers the
training steps only, not the datasets or the evaluations. Each record is one JSON line in
--out.

Launch (from the repository root):
    torchrun --standalone --nproc_per_node 8 paper-experiments/convergence/train.py --method lema-dual --out e9.jsonl
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
for path in (repo_dir, exp_dir, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import lema

from common.models import MLP, residual_fn
from pcg import PCGLM

METHODS = ("adam", "lema-dual", "lema-standard", "pcg")


def load_mnist(data_dir: str, train: bool, device):
    dataset = torchvision.datasets.MNIST(data_dir, train=train, download=False)
    x = dataset.data.unsqueeze(1).float().div_(255).sub_(0.1307).div_(0.3081)
    return x.to(device), dataset.targets.to(device)


@torch.no_grad()
def evaluate(model, x_train, y_train, x_test, y_test, chunk=2048):
    model.eval()
    train_loss = sum(F.nll_loss(model(x_train[i : i + chunk]), y_train[i : i + chunk], reduction="sum").item()
                     for i in range(0, x_train.size(0), chunk)) / x_train.size(0)
    correct = sum((model(x_test[i : i + chunk]).argmax(dim=1) == y_test[i : i + chunk]).sum().item()
                  for i in range(0, x_test.size(0), chunk))
    model.train()
    return train_loss, correct / x_test.size(0)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--hidden", type=int, default=64, help="hidden neurons of the MLP")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096, help="global batch size N of the LM methods")
    parser.add_argument("--shard-size", type=int, default=512, help="shard size S of the LM methods")
    parser.add_argument("--slice-size", type=int, default=256, help="slice size L of the LM methods")
    parser.add_argument("--cg-iters", type=int, default=100, help="maximum PCG iterations per solve")
    parser.add_argument("--adam-batch-size", type=int, default=128, help="mini-batch size B of Adam")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of Adam")
    parser.add_argument("--eval-every", type=int, default=None, help="steps between evaluations (default: 1 for "
                        "the LM methods, 50 for Adam)")
    parser.add_argument("--data-dir", type=str, default=".data")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.method in ("adam", "pcg") and world_size != 1:
        raise ValueError(f"{args.method} runs on a single GPU")

    x_train, y_train = load_mnist(args.data_dir, True, device)
    x_test, y_test = load_mnist(args.data_dir, False, device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(args.seed)
    with torch.device(device):
        model = MLP(args.hidden)
    num_params = sum(p.numel() for p in model.parameters())

    is_lm = args.method != "adam"
    batch = args.batch_size if is_lm else args.adam_batch_size
    if is_lm:
        if batch % world_size != 0:
            raise ValueError(f"Require P | N, got N={batch}, P={world_size}")
        if args.method == "pcg":
            optim = PCGLM(model=model, residual_fn=residual_fn, cg_iters=args.cg_iters)
        else:
            optim = lema.LeMA(model=model, residual_fn=residual_fn, form=args.method.split("-")[1])
    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    eval_every = args.eval_every or (1 if is_lm else 50)
    block = batch // world_size
    steps_per_epoch = x_train.size(0) // batch  # drop the last partial batch
    label = args.method + (f"-{args.cg_iters}" if args.method == "pcg" else "")
    base = {"method": label, "world_size": world_size, "model": f"mlp-{args.hidden}", "params": num_params, "batch_size": batch,
            "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__}
    if is_lm:
        base.update(shard_size=args.shard_size, slice_size=args.slice_size)
    if rank == 0:
        print(f"{label} | P: {world_size} | M: {num_params:,} | batch: {batch} | steps per epoch: {steps_per_epoch}",
              flush=True)

    records, train_time, peak, step = [], 0.0, 0, 0

    def log(epoch_frac, extra):
        nonlocal peak
        torch.cuda.synchronize(device)
        peak = max(peak, torch.cuda.max_memory_allocated(device))
        dist.barrier()
        if rank == 0:
            train_loss, test_acc = evaluate(model, x_train, y_train, x_test, y_test)
            record = {**base, "step": step, "epoch": epoch_frac, "time_s": train_time, "train_loss": train_loss,
                      "test_acc": test_acc, **extra}
            records.append(record)
            print(f"step {step:5d} | epoch {epoch_frac:5.2f} | time {train_time:8.2f} s | train loss "
                  f"{train_loss:.4f} | test acc {test_acc:.2%} | " + " | ".join(f"{k} {v}" for k, v in extra.items()),
                  flush=True)
        dist.barrier()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    log(0.0, {})
    for epoch in range(args.epochs):
        perm = torch.randperm(x_train.size(0), generator=torch.Generator().manual_seed(args.seed + epoch)).to(device)
        for k in range(steps_per_epoch):
            idx = perm[k * batch + rank * block : k * batch + (rank + 1) * block]
            x, y = x_train[idx], y_train[idx]
            dist.barrier()
            torch.cuda.synchronize(device)
            tic = time.perf_counter()
            extra = {}
            if is_lm:
                res = optim.step(x, y, shard_size=args.shard_size, slice_size=args.slice_size)
                extra = {"trials": res.iterations, "damp": res.damp}
            else:
                optim.zero_grad(set_to_none=True)
                F.nll_loss(model(x), y).backward()
                optim.step()
            torch.cuda.synchronize(device)
            train_time += time.perf_counter() - tic
            step += 1
            if step % eval_every == 0 or k == steps_per_epoch - 1:
                if args.method == "pcg":
                    extra["cg_iters"] = optim.cg_iterations[-res.iterations:]
                log(epoch + (k + 1) / steps_per_epoch, extra)

    # Peak memory per GPU over the training steps, maximized over the processes
    torch.cuda.synchronize(device)
    peak = torch.tensor([max(peak, torch.cuda.max_memory_allocated(device)) / 1e9], dtype=torch.float64, device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"peak memory (max over ranks): {peak.item():.2f} GB | training time {train_time:.1f} s", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            for record in records:
                f.write(json.dumps({**record, "peak_mem_gb": peak.item()}) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
