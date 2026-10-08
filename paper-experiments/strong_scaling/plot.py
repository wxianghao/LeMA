"""
Plot E4 strong scaling: (a) step time against ideal scaling and (b) per-phase share of the step time.

    python paper-experiments/strong_scaling/plot.py --input paper-experiments/strong_scaling/results/<run>/strong_scaling.jsonl
    python paper-experiments/strong_scaling/plot.py --synthetic  # placeholder
"""

import argparse
import json
import math
import os
import statistics

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PAPER = os.path.join(os.path.dirname(REPO), "2027-ipdps-lema")

# Categorical slots of the reference palette, in fixed order
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, NEUTRAL = "#0b0b0b", "#52514e", "#b9b8b2"

MODEL_LABELS = {"cnn-1b": "CNN-1B", "cnn-100m": "CNN-100M"}
# Stacked bottom to top: label -> (timer phases, color, hatch). The minor phases
# (J^T v, solve, loss evaluation) are folded into "Other" since each is a few percent at most.
PHASE_GROUPS = {
    "Jacobian": (("jacobian",), SLOTS[0], ""),
    "Gram product": (("gram",), SLOTS[1], "////"),
    "Other": (("vjp", "solve", "other"), NEUTRAL, ""),
    "Communication": (("comm",), SLOTS[2], "\\\\\\\\"),
}

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


def spread(ys, gap):
    """Move label positions apart to at least ``gap``, keeping their mean."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + gap)
    shift = (sum(ys) - sum(pos)) / len(pos)
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k] + shift
    return out


def plot(times, breakdown, out, watermark):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.6), gridspec_kw={"width_ratios": [1.2, 1]})

    # (a) Step time on log-log axes, against ideal scaling from one GPU
    models = [m for m in MODEL_LABELS if any(k[0] == m for k in times)]
    procs = sorted({p for _, p in times})
    ends = []
    for i, m in enumerate(models):
        ps = sorted(p for mm, p in times if mm == m)
        t1 = times[(m, 1)]
        ax1.plot(ps, [times[(m, p)] for p in ps], color=SLOTS[i], lw=1.2, marker="o", ms=3, label=MODEL_LABELS[m])
        # Ideal on top, so that it stays visible where the measurement coincides with it
        ax1.plot(ps, [t1 / p for p in ps], color=INK, lw=0.6, ls=(0, (2, 2)), zorder=3,
                 label="Ideal" if i == 0 else None)
        ends.append((ps[-1], times[(m, ps[-1])], t1 / times[(m, ps[-1])]))
    # Direct labels right of the last points: speedup and parallel efficiency
    ys = spread([math.log10(t) for _, t, _ in ends], gap=0.1)
    for (p, _, sp), y in zip(ends, ys):
        ax1.text(p * 1.12, 10**y, f"{sp:.2f}$\\times$ ({sp / p:.1%})", ha="left", va="center", fontsize=5.5,
                 color=INK)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(procs, [str(p) for p in procs])
    ax1.minorticks_off()
    lo, hi = min(times.values()), max(times.values())
    yticks = [v for v in (25, 50, 100, 200, 400) if lo / 1.3 <= v <= hi * 1.3]
    ax1.set_yticks(yticks, [str(v) for v in yticks])
    ax1.set_xlim(procs[0] / 1.25, procs[-1] * 4.6)
    ax1.set_ylim(lo / 1.3, hi * 1.3)
    ax1.spines["bottom"].set_bounds(procs[0], procs[-1])
    ax1.set_xlabel("Number of GPUs $P$")
    ax1.set_ylabel("Step time (s)")
    # Legend in the empty upper right, models before the ideal
    handles, labels = ax1.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda k: labels[k] == "Ideal")
    ax1.legend([handles[k] for k in order], [labels[k] for k in order], loc="upper right", fontsize=5.5,
               handlelength=1.6, borderaxespad=0.2)
    ax1.set_title("(a) Step time", fontsize=7, pad=3)

    # (b) Share of each phase in the step time of the largest model
    m = models[0]
    ps = sorted(p for mm, p in breakdown if mm == m)
    bottom = [0.0] * len(ps)
    for label, (phases, color, hatch) in PHASE_GROUPS.items():
        shares = [100 * sum(breakdown[(m, p)][ph] for ph in phases) / sum(breakdown[(m, p)].values()) for p in ps]
        ax2.bar(range(len(ps)), shares, bottom=bottom, width=0.66, color=color, hatch=hatch,
                edgecolor="white", linewidth=0.4, label=label)
        bottom = [b + s for b, s in zip(bottom, shares)]
    ax2.set_xticks(range(len(ps)), [str(p) for p in ps])
    ax2.set_xlabel("Number of GPUs $P$")
    ax2.set_ylabel("Share of step time (%)")
    ax2.set_ylim(0, 100)
    # Legend in stacking order, top segment first
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=5.5,
               handlelength=1.0, handleheight=0.9, labelspacing=0.3, borderaxespad=0.2)
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
    parser.add_argument("--input", default=None, help="strong_scaling.jsonl of a run.sh invocation")
    parser.add_argument("--out", default=os.path.join(PAPER, "figures", "strong_scaling.pdf"))
    args = parser.parse_args()
    if args.input is None and not args.synthetic:
        parser.error("--input is required unless --synthetic is given")
    times, breakdown = load(args.input or os.path.join(HERE, "synthetic", "strong_scaling.jsonl"))
    plot(times, breakdown, args.out, watermark=args.synthetic)


if __name__ == "__main__":
    main()
