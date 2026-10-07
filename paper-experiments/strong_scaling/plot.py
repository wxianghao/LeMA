"""
Plot E4 strong scaling: (a) speedup and (b) per-phase share of the step time.

    python paper-experiments/strong_scaling/plot.py              # real results
    python paper-experiments/strong_scaling/plot.py --synthetic  # placeholder
"""

import argparse
import json
import os
import statistics

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# Categorical slots of the reference palette, in fixed order
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED = "#0b0b0b", "#52514e"

MODEL_LABELS = {"cnn-1b": "CNN-1B", "cnn-100m": "CNN-100M"}
PHASE_LABELS = {
    "jacobian": "Jacobian",
    "gram": "Gram product",
    "vjp": r"$\mathbf{J}^T\mathbf{v}$",
    "solve": "Solve",
    "comm": "Communication",
    "other": "Other",
}
HATCHES = ["", "////", "", "\\\\\\\\", "xxxx", ""]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.linewidth": 0.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "legend.frameon": False,
    "hatch.linewidth": 0.4,
    "pdf.fonttype": 42,
})


def load(path):
    timed = defaultdict(list)
    breakdown = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = (r["model"], r["world_size"])
            if r["kind"] == "timed":
                timed[key].append(r["time_s"])
            elif r["kind"] == "breakdown":
                breakdown[key] = r["phases"]
    times = {key: statistics.median(v) for key, v in timed.items()}
    return times, breakdown


def plot(times, breakdown, out, watermark):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.55), gridspec_kw={"width_ratios": [1, 1.15]})

    # (a) Speedup over one GPU
    models = [m for m in MODEL_LABELS if any(k[0] == m for k in times)]
    procs = sorted({p for _, p in times})
    ax1.plot(procs, procs, color=MUTED, lw=0.8, ls=(0, (3, 2)), label="Ideal")
    for i, m in enumerate(models):
        ps = sorted(p for mm, p in times if mm == m)
        speedup = [times[(m, 1)] / times[(m, p)] for p in ps]
        ax1.plot(ps, speedup, color=SLOTS[i], lw=1.2, marker="o", ms=3, label=MODEL_LABELS[m])
        # Direct label: speedup and efficiency at the largest P, first model above the line
        p, sp = ps[-1], speedup[-1]
        ax1.annotate(f"{sp:.1f}$\\times$ ({sp / p:.0%})", (p, sp), xytext=(-5, 3 if i == 0 else -3),
                     textcoords="offset points", ha="right", va="bottom" if i == 0 else "top",
                     fontsize=6, color=INK)
    ax1.set_xticks(procs, [str(p) for p in procs])
    ax1.set_yticks(procs, [str(p) for p in procs])
    ax1.set_xlim(0.5, procs[-1] + 0.5)
    ax1.set_ylim(0.5, procs[-1] + 1.2)
    ax1.set_xlabel("Number of GPUs $P$")
    ax1.set_ylabel("Speedup")
    ax1.legend(loc="lower right", fontsize=6, handlelength=1.6, borderaxespad=0.2)
    ax1.set_title("(a) Speedup", fontsize=7, pad=3)

    # (b) Share of each phase in the step time of the largest model
    m = models[0]
    ps = sorted(p for mm, p in breakdown if mm == m)
    bottom = [0.0] * len(ps)
    for i, phase in enumerate(PHASE_LABELS):
        shares = [100 * breakdown[(m, p)][phase] / sum(breakdown[(m, p)].values()) for p in ps]
        ax2.bar(range(len(ps)), shares, bottom=bottom, width=0.62, color=SLOTS[i], hatch=HATCHES[i],
                edgecolor="white", linewidth=0.4, label=PHASE_LABELS[phase])
        bottom = [b + s for b, s in zip(bottom, shares)]
    ax2.set_xticks(range(len(ps)), [str(p) for p in ps])
    ax2.set_xlabel("Number of GPUs $P$")
    ax2.set_ylabel("Share of step time (%)")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=5.5, handlelength=1.0,
               handleheight=0.9, labelspacing=0.3, borderaxespad=0.2)
    ax2.set_title(f"(b) Breakdown of {MODEL_LABELS[m]}", fontsize=7, pad=3)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
        ax.grid(axis="y", color="#e4e3df", lw=0.4)
        ax.set_axisbelow(True)

    if watermark:
        fig.text(0.5, 0.5, "SYNTHETIC DATA", fontsize=22, color="#d0021b", alpha=0.18,
                 ha="center", va="center", rotation=15, weight="bold")

    fig.tight_layout(pad=0.2, w_pad=0.8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="plot fake data with a watermark")
    parser.add_argument("--input", default=None)
    parser.add_argument("--out", default=os.path.join(REPO, "ipdps_2027_lema", "figures", "strong_scaling.pdf"))
    args = parser.parse_args()
    default = os.path.join(HERE, "synthetic" if args.synthetic else "results", "strong_scaling.jsonl")
    times, breakdown = load(args.input or default)
    plot(times, breakdown, args.out, watermark=args.synthetic)


if __name__ == "__main__":
    main()
