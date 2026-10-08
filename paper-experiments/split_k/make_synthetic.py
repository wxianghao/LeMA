"""Write fake results in the schema of run.sh, for placeholder figures only.

The bandwidths roughly follow a preliminary benchmark on an A100-SXM4-80GB.
"""

import json
import os
import random

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic")
SMS = 108
ENV = {"synthetic": True, "sms": SMS}

# Effective bandwidth in GB/s of one block product per shard size S
MM_GBPS = {1: 1230, 2: 125, 4: 245, 8: 465, 16: 895, 32: 1335, 64: 890}
SPLIT_GBPS = {1: 1600, 2: 690, 4: 1400, 8: 990, 16: 780, 32: 150, 64: 300}
# Effective bandwidth in GB/s of split-K with S = 4 per chunks per SM
CHUNKS_GBPS = {1: 280, 2: 530, 4: 970, 8: 1500, 16: 1400, 32: 1400, 64: 1390}
MODEL_SIZES = (100_002_598, 1_000_022_632)

# CNN-1B at P = 8: step time and phase shares, without and with split-K
E2E = {
    1: (95.0, {"jacobian": 0.285, "gram": 0.700, "vjp": 0.001, "solve": 0.00001, "comm": 0.0007}),
    None: (39.8, {"jacobian": 0.682, "gram": 0.294, "vjp": 0.0026, "solve": 0.00002, "comm": 0.0016}),
}


def main():
    random.seed(0)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "bench.jsonl"), "w") as f:
        for m in MODEL_SIZES:
            for s in MM_GBPS:
                if 2 * s * m * 4 > 64e9:
                    continue
                shape = {"m": m, "s": s, "bytes": 2 * s * m * 4, "flops": 2 * s * s * m}
                for method, gbps, chunks in (("mm", MM_GBPS, 1), ("split", SPLIT_GBPS, 16 * SMS)):
                    t = shape["bytes"] / (gbps[s] * 1e9) * random.uniform(0.97, 1.03)
                    f.write(json.dumps({"kind": "block", "method": method, "chunks": chunks, "time_s": t,
                                        "rel_diff": 0.0 if method == "mm" else 1e-6, **shape, **ENV}) + "\n")
                if s == 4:
                    for per_sm, gbps in CHUNKS_GBPS.items():
                        t = shape["bytes"] / (gbps * 1e9) * random.uniform(0.97, 1.03)
                        f.write(json.dumps({"kind": "chunks", "chunks": per_sm * SMS, "time_s": t, **shape,
                                            **ENV}) + "\n")
    with open(os.path.join(OUT, "e2e.jsonl"), "w") as f:
        for split_k, (total, shares) in E2E.items():
            base = {"model": "cnn-1b", "params": 1_000_022_632, "world_size": 8, "batch_size": 256,
                    "shard_size": 4, "slice_size": 4, "split_k": split_k, "peak_mem_gb": 60.0, **ENV}
            f.write(json.dumps({**base, "kind": "timed", "step": 1, "time_s": total}) + "\n")
            phases = {k: v * total for k, v in shares.items()}
            phases["other"] = total - sum(phases.values())
            f.write(json.dumps({**base, "kind": "breakdown", "step": 2, "time_s": total, "phases": phases}) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
