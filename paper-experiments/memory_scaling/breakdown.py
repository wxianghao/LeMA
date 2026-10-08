"""
E8 breakdown: the components of the peak memory of one LM trial step on one GPU.

The script records every allocation of the step with its Python call site, replays the
allocations and frees to find the peak, and attributes each allocation that is live at
the peak to a component of Table I by the source line in lema/ or run.py that made it:

    Gram matrix   J^T J or J D^-1 J^T, its damped copy, and the LU factorization of the solve
    Jacobian      the full Jacobian or its shards, including the per-sample gradients of a slice
    Vectors       M-dimensional vectors, i.e., the parameters, their backup, D, J^T r, and updates
    Activations   activations of the forward passes
    Other         everything else, e.g., the batch and small buffers

It appends one JSON line per run to --out, with the components in GB and the peak of the
replay next to torch.cuda.max_memory_allocated as a check.

Launch (from the repository root):
    torchrun --standalone --nproc_per_node 1 paper-experiments/memory_scaling/breakdown.py \
        --variant lema-dual --hidden 108378 --out breakdown.jsonl
"""

import argparse
import json
import linecache
import os
import sys
import time

import torch
import torch.distributed as dist

here = os.path.dirname(os.path.abspath(__file__))
if here not in sys.path:
    sys.path.insert(0, here)

import lema

from run import VARIANTS, WideCNN, full_dual_step, full_standard_step, load_mnist, residual_fn

COMPONENTS = ("Gram matrix", "Jacobian", "Vectors", "Activations", "Other")


def classify(frames, phase) -> str:
    """Component of an allocation, by the innermost call site in lema/ or run.py.

    Backward passes of CUDA tensors run on autograd's device threads without Python frames,
    so such allocations fall back to the phase in which they happen.
    """
    for frame in frames:
        path = frame["filename"]
        if not (path.endswith(os.path.join("lema", "lema.py")) or path.endswith(os.path.join("lema", "jacobian.py"))
                or path.endswith(os.path.join("memory_scaling", "run.py"))):
            continue
        line = linecache.getline(path, frame["line"])
        if path.endswith("jacobian.py"):
            if "vjp_fn(" in line or "torch.vmap(" in line or ")(self._flat, x, y)" in line:
                return "Jacobian"  # per-sample gradients and the assembled slices
            if "torch.func.vjp(" in line or "functional_call" in line or "_residual_fn" in line:
                return "Activations"
            if "parameters_to_vector" in line or "self._flat" in line:
                return "Vectors"
            continue
        if "jacrev(" in line:
            return "Jacobian"
        if any(k in line for k in ("jtj", "jjt", "gram", "linalg.solve", "_gram_block", "damped")):
            # Including the LU factorization of the solve; its output is preallocated
            return "Gram matrix"
        if "_compute_loss" in line or "self._model(" in line or "jm(x)" in line or "residual_fn(" in line:
            return "Activations"
        if any(k in line for k in ("_flat", "_backup", "d_inv", "update", "jtr", "vector_norm", "rsqrt", "d.clamp",
                                   ".T @ v", ".T @ r", "new_zeros(model_size)", "WideCNN(", "JacobianModel(")):
            return "Vectors"
        return "Other"
    return {"jacobian": "Jacobian", "vjp": "Vectors"}.get(phase, "Other")


def begin_jacobian(device):
    torch.empty(1, dtype=torch.uint8, device=device)


def begin_vjp(device):
    torch.empty(1, dtype=torch.uint8, device=device)


def end_phase(device):
    torch.empty(1, dtype=torch.uint8, device=device)


MARKERS = {"begin_jacobian": "jacobian", "begin_vjp": "vjp", "end_phase": None}


def mark_phases(obj, device):
    """Wrap jacrev and vjp of obj with allocations that mark their phase in the trace."""
    for name, begin in (("jacrev", begin_jacobian), ("vjp", begin_vjp)):
        fn = getattr(obj, name)

        def wrapped(*args, _fn=fn, _begin=begin, **kwargs):
            _begin(device)
            try:
                return _fn(*args, **kwargs)
            finally:
                end_phase(device)

        object.__setattr__(obj, name, wrapped)


def marker_of(ev):
    return next((MARKERS[f["name"]] if f["name"] in MARKERS else None for f in ev["frames"][:3]
                 if f["name"] in MARKERS), False)


def peak_breakdown(trace):
    """Replay the allocation trace and attribute the allocations live at its peak."""
    live, total, peak, peak_index = {}, 0, 0, -1
    for i, ev in enumerate(trace):
        if ev["action"] == "alloc" and marker_of(ev) is False:
            live[ev["addr"]] = ev["size"]
            total += ev["size"]
            if total > peak:
                peak, peak_index = total, i
        elif ev["action"] in ("free_requested", "free_completed") and ev["addr"] in live:
            total -= live.pop(ev["addr"])
    # Second pass up to the peak, keeping the call sites and the phase of each allocation
    live, phase = {}, None
    for ev in trace[: peak_index + 1]:
        if ev["action"] == "alloc":
            marker = marker_of(ev)
            if marker is not False:
                phase = marker
                continue
            live[ev["addr"]] = (ev, phase)
        elif ev["action"] in ("free_requested", "free_completed"):
            live.pop(ev["addr"], None)
    components = dict.fromkeys(COMPONENTS, 0)
    sites = {}
    for ev, phase in live.values():
        comp = classify(ev["frames"], phase)
        components[comp] += ev["size"]
        site = next((f"{os.path.basename(f['filename'])}:{f['line']}" for f in ev["frames"]
                     if f["filename"].endswith((".py",)) and "site-packages" not in f["filename"]), "?")
        sites[(comp, site)] = sites.get((comp, site), 0) + ev["size"]
    top = sorted(sites.items(), key=lambda kv: -kv[1])[:8]
    return peak, components, [{"component": c, "site": s, "gb": b / 1e9} for (c, s), b in top]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--hidden", type=int, required=True, help="fc1 width of WideCNN")
    parser.add_argument("--batch-size", type=int, default=256, help="global batch size N")
    parser.add_argument("--shard-size", type=int, default=4, help="shard size S of the LeMA variants")
    parser.add_argument("--slice-size", type=int, default=4, help="slice size L")
    parser.add_argument("--damp", type=float, default=1e-3, help="damping factor of the trial step")
    parser.add_argument("--max-entries", type=int, default=4_000_000, help="allocation events to record")
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

    # Record from before the model exists, so that the parameters are attributed too
    torch.cuda.memory._record_memory_history(max_entries=args.max_entries, stacks="python")
    torch.manual_seed(args.seed)
    with torch.device(device):
        model = WideCNN(args.hidden)
    m = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats(device)
    tic = time.perf_counter()
    if args.variant.startswith("lema"):
        optim = lema.LeMA(model=model, residual_fn=residual_fn, max_iters=1, damp_start=args.damp,
                          form=args.variant.split("-")[1])
        mark_phases(optim, device)
        optim.step(x, y, shard_size=s, slice_size=l)
    else:
        jm = lema.JacobianModel(model, residual_fn)
        mark_phases(jm, device)
        step = full_standard_step if args.variant == "full-standard" else full_dual_step
        with torch.no_grad():
            step(jm, x, y, args.damp, l)
            residual_fn(jm(x), y).square().sum().item()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - tic
    max_allocated = torch.cuda.max_memory_allocated(device)
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)

    trace = snapshot["device_traces"][local_rank]
    if len(trace) >= args.max_entries:
        raise RuntimeError(f"The trace filled all {args.max_entries} entries; increase --max-entries.")
    peak, components, top = peak_breakdown(trace)
    record = {"variant": args.variant, "hidden": args.hidden, "params": m, "batch_size": n, "slice_size": l,
              "shard_size": s if args.variant.startswith("lema") else n, "time_s": elapsed, "events": len(trace),
              "peak_gb": peak / 1e9, "max_allocated_gb": max_allocated / 1e9,
              "components_gb": {k: v / 1e9 for k, v in components.items()}, "top_sites": top,
              "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__}
    print(f"{args.variant} | M: {m:,} | peak {peak / 1e9:.2f} GB (max allocated {max_allocated / 1e9:.2f} GB) | "
          + " | ".join(f"{k} {v / 1e9:.2f}" for k, v in components.items()), flush=True)
    for t in top:
        print(f"    {t['component']:12s} {t['site']:28s} {t['gb']:8.3f} GB", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(record) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
