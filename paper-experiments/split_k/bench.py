"""
E5 microbenchmark: one Gram block product a @ b.T of two S x M shards on one GPU.

Every product goes through LeMA._gram_block, either as a plain torch.mm (split_k=1)
or with split-K into 16 chunks per SM, the default of LeMA. Per (M, S), the script
records both, and for a few shapes it also sweeps the number of chunks. Each record
is one JSON line in --out:

    kind=block    method (mm | split), m, s, chunks, time_s, bytes, flops, rel_diff
    kind=chunks   m, s, chunks, time_s, bytes, flops

where rel_diff is the relative difference of the split-K result from torch.mm.

Launch (from the repository root):
    torchrun --standalone --nproc_per_node 1 paper-experiments/split_k/bench.py --out bench.jsonl
"""

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

from torch import nn

# Ensure correct import of lema
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

import lema

# Parameter counts of CNN-100M and CNN-1B in E4
MODEL_SIZES = (100_002_598, 1_000_022_632)


def make_optim(device: torch.device, split_k: int) -> lema.LeMA:
    # Only _gram_block is used, so a tiny model suffices
    model = nn.Linear(1, 1, device=device)
    return lema.LeMA(model=model, residual_fn=lambda y_hat, y: y_hat - y, split_k=split_k)


def timeit(fn, warmup: int, min_time: float, max_reps: int) -> float:
    """Mean time of fn in seconds, with enough repetitions to run for min_time."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    reps, elapsed = 1, 0.0
    while True:
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1e3
        if elapsed >= min_time or reps >= max_reps:
            return elapsed / reps
        reps = min(max_reps, max(2 * reps, int(reps * min_time / max(elapsed, 1e-6)) + 1))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-sizes", type=int, nargs="+", default=list(MODEL_SIZES), help="inner dimensions M")
    parser.add_argument("--shard-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64], help="rows S")
    parser.add_argument("--chunks-per-sm", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64],
                        help="chunk counts of the sweep, per SM")
    parser.add_argument("--sweep-shard-size", type=int, default=4, help="S of the chunk sweep")
    # Two 8 x 1e9 shards of CNN-1B take 64.0007 GB, which still fit into an 80 GB GPU
    parser.add_argument("--max-gb", type=float, default=66.0, help="skip shapes whose two shards exceed this")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--min-time", type=float, default=0.5, help="seconds of timed repetitions per point")
    parser.add_argument("--max-reps", type=int, default=50)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    sms = torch.cuda.get_device_properties(device).multi_processor_count

    mm = make_optim(device, split_k=1)
    split = make_optim(device, split_k=16 * sms)
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
           "sms": sms}
    print(f"{env['gpu']} | SMs: {sms} | default chunks: {split._split_k}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_file = open(args.out, "a")

    def record(**fields):
        out_file.write(json.dumps({**fields, **env}) + "\n")
        out_file.flush()

    def bench(optim, a, b, out):
        return timeit(lambda: optim._gram_block(a, b, out=out), args.warmup, args.min_time, args.max_reps)

    torch.manual_seed(args.seed)
    for m in args.model_sizes:
        for s in args.shard_sizes:
            if 2 * s * m * 4 > args.max_gb * 1e9:
                print(f"M={m:,} S={s}: skipped, two shards exceed {args.max_gb} GB", flush=True)
                continue
            a = torch.randn(s, m, device=device)
            b = torch.randn(s, m, device=device)
            # A non-contiguous output, as the blocks of jjt_block in LeMA
            buf = torch.empty(s, 2 * s, device=device)
            out = buf[:, :s]
            shape = {"m": m, "s": s, "bytes": 2 * s * m * 4, "flops": 2 * s * s * m}

            t_mm = bench(mm, a, b, out)
            ref = out.clone()
            record(kind="block", method="mm", chunks=1, time_s=t_mm, rel_diff=0.0, **shape)
            t_split = bench(split, a, b, out)
            diff = ((out - ref).norm() / ref.norm()).item()
            record(kind="block", method="split", chunks=split._split_k, time_s=t_split, rel_diff=diff, **shape)
            print(f"M={m:>13,} S={s:2d} | mm {t_mm * 1e3:8.2f} ms {shape['bytes'] / t_mm / 1e9:6.0f} GB/s "
                  f"| split {t_split * 1e3:8.2f} ms {shape['bytes'] / t_split / 1e9:6.0f} GB/s "
                  f"| {t_mm / t_split:5.2f}x | rel diff {diff:.1e}", flush=True)

            if s == args.sweep_shard_size:
                for per_sm in args.chunks_per_sm:
                    optim = make_optim(device, split_k=per_sm * sms)
                    t = bench(optim, a, b, out)
                    record(kind="chunks", chunks=optim._split_k, time_s=t, **shape)
                    print(f"M={m:>13,} S={s:2d} | {per_sm:2d} chunks/SM ({optim._split_k:5d}) {t * 1e3:8.2f} ms "
                          f"{shape['bytes'] / t / 1e9:6.0f} GB/s", flush=True)
            del a, b, ref
            torch.cuda.empty_cache()

    out_file.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
