"""
Plot E8: peak memory of one LM trial step against the model size, for the standard and
the dual form, each with the full Jacobian and with Jacobian shards, and optionally the
components of the peak memory of selected runs of breakdown.py beside it.

    python paper-experiments/memory_scaling/plot.py --input paper-experiments/memory_scaling/results/<run>/memory.jsonl \
        [--breakdown paper-experiments/memory_scaling/results/<run>/breakdown.jsonl]
    python paper-experiments/memory_scaling/plot.py --synthetic  # placeholder
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PAPER = os.path.join(os.path.dirname(REPO), "2027-ipdps-lema")

# Categorical slots of the reference palette, in fixed order, as in strong_scaling/plot.py
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, NEUTRAL = "#0b0b0b", "#52514e", "#b9b8b2"

# Color by form as in standard_vs_dual/plot.py, line style by the treatment of the Jacobian
VARIANTS = {
    "full-standard": ("Standard, full $\\mathbf{J}$", SLOTS[1], (0, (3, 1.5)), "o"),
    "lema-standard": ("Standard, sharded $\\mathbf{J}$", SLOTS[1], "-", "s"),
    "full-dual": ("Dual, full $\\mathbf{J}$", SLOTS[0], (0, (3, 1.5)), "o"),
    "lema-dual": ("Dual, sharded $\\mathbf{J}$ (LeMA)", SLOTS[0], "-", "s"),
}

# Components of breakdown.py: (color, hatch), with the Jacobian, the Gram matrix, and Other as
# in the phase breakdowns of strong_scaling/plot.py and split_k/plot.py
COMPONENTS = {
    "Jacobian": (SLOTS[0], ""),
    "Gram matrix": (SLOTS[1], "////"),
    "Vectors": (SLOTS[3], ""),
    "Activations": (SLOTS[4], "\\\\\\\\"),
    "Other": (NEUTRAL, ""),
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
    with open(path) as f:
        return [json.loads(line) for line in f]


def sci(m):
    """$M = 3.0 \\times 10^{7}$"""
    exp = len(str(int(m))) - 1
    return f"$M = {m / 10**exp:.1f} \\times 10^{{{exp}}}$"


def plot_breakdown(ax, breakdown, capacity):
    """Horizontal stacked bars of the components of the peak memory, one bar per run."""
    labels, left = [], [0.0] * len(breakdown)
    for comp, (color, hatch) in COMPONENTS.items():
        widths = [r["components_gb"][comp] for r in breakdown]
        ax.barh(range(len(breakdown)), widths, left=left, height=0.62, color=color, hatch=hatch, edgecolor="white",
                linewidth=0.4, label=comp)
        left = [a + w for a, w in zip(left, widths)]
    for i, r in enumerate(breakdown):
        # Inside the end of bars that reach the capacity line, right of the others
        if left[i] > 0.8 * capacity:
            ax.text(left[i] - 1, i, f"{r['peak_gb']:.1f}", va="center", ha="right", fontsize=5.5, color="white")
        else:
            ax.text(left[i] + 1, i, f"{r['peak_gb']:.1f}", va="center", ha="left", fontsize=5.5, color=INK)
        # The variant without the "(LeMA)" of the legend of (a), and the model size
        labels.append(f"{VARIANTS[r['variant']][0].replace(' (LeMA)', '')}, {sci(r['params'])}")
    ax.axvline(capacity, color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
    ax.set_yticks(range(len(breakdown)), labels, fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlim(0, capacity * 1.08)
    ax.set_xlabel("Peak memory (GB)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=5.5, handlelength=1.0,
              handleheight=0.9, columnspacing=0.8, borderaxespad=0.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
    ax.grid(axis="x", color="#e4e3df", lw=0.4)
    ax.set_axisbelow(True)


def plot(records, out, watermark, breakdown=None):
    if breakdown:
        # Side by side across both columns, as the figure of convergence/plot.py
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.16, 1.85), gridspec_kw={"width_ratios": [1, 1]})
    else:
        fig, ax = plt.subplots(figsize=(3.5, 1.9))
    # Usable capacity reported by PyTorch, e.g., 85.1 GB = 79.25 GiB for an 80 GB A100
    capacity = max(r["gpu_mem_gb"] for r in records)
    ax.axhline(capacity, color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
    ax.text(min(r["params"] for r in records), capacity * 1.12, "GPU memory capacity", fontsize=5.5, color=MUTED,
            va="bottom")
    for variant, (label, color, ls, marker) in VARIANTS.items():
        rs = sorted((r for r in records if r["variant"] == variant), key=lambda r: r["params"])
        ok = [r for r in rs if not r["oom"]]
        if ok:
            ax.plot([r["params"] for r in ok], [r["peak_mem_gb"] for r in ok], color=color, ls=ls, lw=1.1,
                    marker=marker, ms=3, mfc="white" if ls != "-" else color, mew=0.8, label=label)
        # Cross on the capacity line at the first model size that runs out of memory
        for r in rs:
            if r["oom"]:
                ax.plot([r["params"]], [capacity], color=color, marker="x", ms=4.5, mew=1.1, ls="none", zorder=4)
                if ok:
                    ax.plot([ok[-1]["params"], r["params"]], [ok[-1]["peak_mem_gb"], capacity], color=color, ls=":",
                            lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of parameters $M$")
    ax.set_ylabel("Peak memory (GB)")
    ax.set_ylim(top=capacity * 2.2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(loc="lower right", fontsize=5.5, handlelength=2.2, borderaxespad=0.2, labelspacing=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
    ax.grid(axis="y", color="#e4e3df", lw=0.4, which="major")
    ax.set_axisbelow(True)
    if breakdown:
        # Same title padding in both panels, to clear the legend above (b)
        ax.set_title("(a) Peak memory", fontsize=7, pad=11)
        plot_breakdown(ax2, breakdown, capacity)
        ax2.set_title("(b) Breakdown", fontsize=7, pad=11)

    if watermark:
        fig.text(0.5, 0.5, "SYNTHETIC DATA", fontsize=22, color="#d0021b", alpha=0.18,
                 ha="center", va="center", rotation=15, weight="bold")

    fig.tight_layout(pad=0.2, w_pad=1.0)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="plot fake data with a watermark")
    parser.add_argument("--input", default=None, help="memory.jsonl of a run.sh invocation")
    parser.add_argument("--breakdown", default=None, help="breakdown.jsonl of breakdown.sh, plotted beside")
    parser.add_argument("--out", default=os.path.join(PAPER, "figures", "memory_scaling.pdf"))
    args = parser.parse_args()
    if args.input is None and not args.synthetic:
        parser.error("--input is required unless --synthetic is given")
    breakdown = load(args.breakdown) if args.breakdown else None
    plot(load(args.input or os.path.join(HERE, "synthetic", "memory.jsonl")), args.out, watermark=args.synthetic,
         breakdown=breakdown)


if __name__ == "__main__":
    main()
