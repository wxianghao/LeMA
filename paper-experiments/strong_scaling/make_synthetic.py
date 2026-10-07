"""Write fake results in the schema of run.py, for placeholder figures only."""

import json
import os
import random

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic", "strong_scaling.jsonl")

# model: (params, N, S, L, time at P=1 in seconds, parallel efficiency per P)
MODELS = {
    "cnn-1b": (1_000_022_632, 256, 4, 4, 170.0, {1: 1.0, 2: 0.99, 4: 0.98, 8: 0.96}),
    "cnn-100m": (100_002_598, 2048, 32, 32, 130.0, {1: 1.0, 2: 0.97, 4: 0.93, 8: 0.88}),
}
# Share of step time per phase at P=1, and the communication share at P=8
SHARES = {"jacobian": 0.70, "gram": 0.27, "vjp": 0.012, "solve": 0.001}
COMM = {1: 0.002, 2: 0.010, 4: 0.020, 8: 0.035}


def main():
    random.seed(0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for model, (params, n, s, l, t1, eff) in MODELS.items():
            for p, e in eff.items():
                t = t1 / (p * e)
                base = {"model": model, "params": params, "world_size": p, "batch_size": n,
                        "shard_size": s, "slice_size": l, "peak_mem_gb": 0.0, "synthetic": True}
                for step in range(1, 4):
                    f.write(json.dumps({**base, "kind": "timed", "step": step,
                                        "time_s": t * random.uniform(0.99, 1.01)}) + "\n")
                phases = {k: v * t * (1 - COMM[p]) for k, v in SHARES.items()}
                phases["comm"] = COMM[p] * t
                phases["other"] = t - sum(phases.values())
                f.write(json.dumps({**base, "kind": "breakdown", "step": 4, "time_s": t, "phases": phases}) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
