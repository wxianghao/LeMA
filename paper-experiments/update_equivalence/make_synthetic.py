"""Write fake results in the schema of run.sh, for placeholder tables only.

The numbers at a damping factor of 1 roughly follow a smoke test on an A100-SXM4-80GB.
"""

import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic", "updates.jsonl")

# (model, M, N): {damp: (err_standard, err_dual)}
ERRORS = {
    ("lenet5", 61_706, 1024): {1e-3: (3e-2, 5e-5), 1.0: (3.4e-3, 6e-6), 1e3: (2e-5, 1e-6)},
    ("lenet5", 61_706, 8192): {1e-3: (5e-2, 1e-4), 1.0: (5e-3, 1e-5), 1e3: (3e-5, 2e-6)},
    ("mlp", 12_730, 4096): {1e-3: (2e-2, 3e-4), 1.0: (2.8e-3, 2.7e-5), 1e3: (1e-5, 1e-6)},
    ("mlp", 12_730, 16384): {1e-3: (2e-4, 1e-2), 1.0: (2e-5, 1e-3), 1e3: (1e-6, 1e-5)},
}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for (name, m, n), errors in ERRORS.items():
            for damp, (err_std, err_dual) in errors.items():
                record = {"model": name, "params": m, "batch_size": n, "damp": damp,
                          "ref_form": "dual" if m > n else "standard", "diff_forms": max(err_std, err_dual),
                          "err_standard": err_std, "err_dual": err_dual, "synthetic": True}
                f.write(json.dumps(record) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
