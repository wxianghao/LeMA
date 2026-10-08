"""
Plot E5 split-K: (a) time of one Gram block product against the shard size S, with
torch.mm and with split-K, and (b) the step time of CNN-1B by phase, without and with
split-K.

    python paper-experiments/split_k/plot.py --input paper-experiments/split_k/results/<run>
    python paper-experiments/split_k/plot.py --synthetic  # placeholder
"""

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PAPER = os.path.join(os.path.dirname(REPO), "2027-ipdps-lema")

# Categorical slots of the reference palette, in fixed order, as in strong_scaling/plot.py
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, NEUTRAL, SHADE = "#0b0b0b", "#52514e", "#b9b8b2", "#efeeea"

METHODS = {"mm": ("torch.mm", MUTED, (0, (3, 1.5))), "split": ("Split-K", INK, "-")}
# One marker per inner dimension M, in increasing order
MARKERS = ["o", "s", "^", "D"]
# Same phases and colors as strong_scaling/plot.py, stacked left to right
PHASE_GROUPS = {
    "Jacobian": (("jacobian",), SLOTS[0], ""),
    "Gram product": (("gram",), SLOTS[1], "////"),
    "Other": (("vjp", "solve", "other"), NEUTRAL, ""),
    "Communication": (("comm",), SLOTS[2], "\\\\\\\\"),
}
# Shards to which LeMA applies split-K by default
SPLIT_K_MAX_ROWS = 8

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


def load(run_dir):
    with open(os.path.join(run_dir, "bench.jsonl")) as f:
        bench = [json.loads(line) for line in f]
    e2e = {}
    with open(os.path.join(run_dir, "e2e.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            method = "mm" if r.get("split_k") == 1 else "split"
            e2e.setdefault(method, {})[r["kind"]] = r
    return bench, e2e


def size_label(m):
    """M rounded to one significant digit, e.g. $M \\approx 10^9$ or $M \\approx 2 \\times 10^6$."""
    exp = int(math.floor(math.log10(m)))
    lead = round(m / 10**exp)
    if lead == 10:
        lead, exp = 1, exp + 1
    return f"$M \\approx 10^{{{exp}}}$" if lead == 1 else f"$M \\approx {lead} \\times 10^{{{exp}}}$"


def plot(bench, e2e, out, watermark):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.6), gridspec_kw={"width_ratios": [1, 1.1]})

    # (a) Time of one block product on log-log axes
    blocks = [r for r in bench if r["kind"] == "block"]
    sizes = sorted({r["s"] for r in blocks})
    ax1.axvspan(sizes[0] / 1.4, math.sqrt(SPLIT_K_MAX_ROWS * 2 * SPLIT_K_MAX_ROWS), color=SHADE, lw=0, zorder=0)
    model_sizes = sorted({r["m"] for r in blocks})
    for m, marker in zip(model_sizes, MARKERS):
        for method, (_, color, ls) in METHODS.items():
            rs = sorted((r for r in blocks if r["m"] == m and r["method"] == method), key=lambda r: r["s"])
            if rs:
                ax1.plot([r["s"] for r in rs], [1e3 * r["time_s"] for r in rs], color=color, ls=ls,
                         lw=1.0, marker=marker, ms=2.8, mfc="white" if method == "mm" else color, mew=0.8)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(sizes, [str(s) for s in sizes])
    ax1.minorticks_off()
    ax1.set_xlim(sizes[0] / 1.4, sizes[-1] * 1.4)
    lo, hi = (1e3 * f(r["time_s"] for r in blocks) for f in (min, max))
    ax1.set_ylim(lo / 1.6, hi * 1.6)
    yticks = [v for v in (0.1, 1, 10, 100, 1000) if lo / 1.6 <= v <= hi * 1.6]
    ax1.set_yticks(yticks, [f"{v:g}" for v in yticks])
    ax1.set_xlabel("Shard size $S$")
    ax1.set_ylabel("Time (ms)")
    handles = [Line2D([], [], color=color, ls=ls, lw=1.0, label=label) for label, color, ls in METHODS.values()]
    handles += [Line2D([], [], color=INK, ls="none", marker=marker, ms=2.8, label=size_label(m))
                for m, marker in zip(model_sizes, MARKERS)]
    # Legend above the axes, as in (b), since the curves span the whole plot area
    ax1.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.5,
               handlelength=1.6, labelspacing=0.2, columnspacing=0.8, borderaxespad=0.1)
    ax1.set_title("(a) One block product", fontsize=7, pad=17)

    # (b) Step time of CNN-1B by phase: shares of the breakdown step, scaled to the timed step
    methods = [m for m in ("mm", "split") if m in e2e]
    left = [0.0] * len(methods)
    totals = [e2e[m]["timed"]["time_s"] for m in methods]
    for label, (phases, color, hatch) in PHASE_GROUPS.items():
        widths = []
        for m, total in zip(methods, totals):
            ph = e2e[m]["breakdown"]["phases"]
            widths.append(total * sum(ph[p] for p in phases) / sum(ph.values()))
        ax2.barh(range(len(methods)), widths, left=left, height=0.55, color=color, hatch=hatch,
                 edgecolor="white", linewidth=0.4, label=label)
        left = [a + w for a, w in zip(left, widths)]
    # Direct labels: step time, and the speedup of split-K
    for i, (m, total) in enumerate(zip(methods, totals)):
        text = f"{total:.1f} s"
        if m == "split" and "mm" in e2e:
            text += f" ({e2e['mm']['timed']['time_s'] / total:.2f}$\\times$)"
        ax2.text(total + 0.02 * max(totals), i, text, ha="left", va="center", fontsize=5.5, color=INK)
    ax2.set_yticks(range(len(methods)), [METHODS[m][0] for m in methods])
    ax2.set_ylim(-0.6, len(methods) - 0.4)
    ax2.set_xlim(0, max(totals) * 1.45)
    ax2.set_xlabel("Step time (s)")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.5, handlelength=1.0,
               handleheight=0.9, labelspacing=0.2, columnspacing=0.8, borderaxespad=0.1)
    p = e2e[methods[0]]["timed"]["world_size"]
    ax2.set_title(f"(b) Step of CNN-1B, $P = {p}$", fontsize=7, pad=17)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
        ax.set_axisbelow(True)
    ax1.grid(axis="y", color="#e4e3df", lw=0.4)
    ax2.grid(axis="x", color="#e4e3df", lw=0.4)

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
    parser.add_argument("--input", default=None, help="run directory of run.sh")
    parser.add_argument("--out", default=os.path.join(PAPER, "figures", "split_k.pdf"))
    args = parser.parse_args()
    if args.input is None and not args.synthetic:
        parser.error("--input is required unless --synthetic is given")
    bench, e2e = load(args.input or os.path.join(HERE, "synthetic"))
    plot(bench, e2e, args.out, watermark=args.synthetic)


if __name__ == "__main__":
    main()
