"""Write fake results in the schema of run.sh, for placeholder figures only.

The numbers roughly follow a smoke test of LeNet-5 at P = 1 on an A100-SXM4-80GB:
the standard form spends most of its step in the redundant M x M solve.
"""

import json
import os
import random

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic", "forms.jsonl")
BASE = {"model": "lenet5", "params": 61_706, "batch_size": 8192, "shard_size": 1024, "slice_size": 256,
        "split_k": None, "synthetic": True}


def step_time(form: str, p: int) -> float:
    if form == "standard":
        # Redundant solve, partitioned Jacobian and Gram accumulation, all-reduce of M^2 elements
        return 9.2 + 3.4 / p + (0.15 if p > 1 else 0.0)
    return 1.47 / (p * {1: 1.0, 2: 0.98, 4: 0.96, 8: 0.93}[p])


def main():
    random.seed(0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for form, mem in (("standard", 45.8), ("dual", 1.4)):
            for p in (1, 2, 4, 8):
                t = step_time(form, p) * random.uniform(0.99, 1.01)
                record = {**BASE, "form": form, "world_size": p, "kind": "timed", "step": 1, "time_s": t,
                          "peak_mem_gb": mem}
                f.write(json.dumps(record) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
