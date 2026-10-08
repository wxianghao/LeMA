"""
Plot E10: (a) step time of the dual form against the shard size S for several slice sizes L,
and (b) the peak memory against S, measured and predicted by Table I as 3SM + 3M elements.

    python paper-experiments/sl_sweep/plot.py --input paper-experiments/sl_sweep/results/<run>
"""

import argparse
import glob
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
INK, MUTED = "#0b0b0b", "#52514e"

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
    "pdf.fonttype": 42,
})


def load(run_dir):
    """Timed records keyed by (S, L)."""
    timed = {}
    for path in glob.glob(os.path.join(run_dir, "runs", "*.jsonl")):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] == "timed":
                    timed[(r["shard_size"], r["slice_size"])] = r
    return timed


def plot(timed, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.6))
    sizes = sorted({s for s, _ in timed})
    params = next(iter(timed.values()))["params"]

    # (a) Step time against S for L = 1, L = min(S, 8), which reaches S = 64, and L = S, with 1/S from the smallest S
    lines = {"$L = 1$": lambda s: 1, "$L = \\min(S, 8)$": lambda s: min(s, 8), "$L = S$": lambda s: s}
    for (label, slice_of), color in zip(lines.items(), SLOTS):
        pts = [(s, timed[(s, slice_of(s))]["time_s"]) for s in sizes if (s, slice_of(s)) in timed]
        ax1.plot([s for s, _ in pts], [t for _, t in pts], color=color, lw=1.1, marker="o", ms=2.8, label=label)
    s0 = sizes[0]
    t0 = timed[(s0, 1)]["time_s"]
    ax1.plot(sizes, [t0 * s0 / s for s in sizes], color=INK, lw=0.6, ls=(0, (2, 2)), label="$\\propto 1/S$")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(sizes, [str(s) for s in sizes])
    ax1.minorticks_off()
    yticks = [v for v in (1, 3, 10, 30, 100) if min(r["time_s"] for r in timed.values()) / 1.5 <= v]
    ax1.set_yticks(yticks, [str(v) for v in yticks])
    ax1.set_xlabel("Shard size $S$")
    ax1.set_ylabel("Step time (s)")
    ax1.legend(loc="upper right", fontsize=5.5, handlelength=1.6, borderaxespad=0.2, labelspacing=0.25)
    ax1.set_title("(a) Step time", fontsize=7, pad=3)

    # (b) Peak memory against S: all slice sizes, and the prediction of Table I
    for (s, l), r in sorted(timed.items()):
        ax2.plot([s], [r["peak_mem_gb"]], color=SLOTS[0], marker="o", ms=2.8, ls="none")
    pred = [(3 * s + 3) * params * 4 / 1e9 for s in sizes]
    ax2.plot(sizes, pred, color=INK, lw=0.6, ls=(0, (2, 2)), label="$3SM + 3M$")
    ax2.plot([], [], color=SLOTS[0], marker="o", ms=2.8, ls="none", label="Measured, all $L$")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(sizes, [str(s) for s in sizes])
    ax2.minorticks_off()
    ax2.set_yticks([3, 10, 30, 80], ["3", "10", "30", "80"])
    ax2.set_xlabel("Shard size $S$")
    ax2.set_ylabel("Peak memory (GB)")
    ax2.legend(loc="upper left", fontsize=5.5, handlelength=1.6, borderaxespad=0.2, labelspacing=0.25)
    ax2.set_title("(b) Memory", fontsize=7, pad=3)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
        ax.grid(axis="y", color="#e4e3df", lw=0.4)
        ax.set_axisbelow(True)

    fig.tight_layout(pad=0.2, w_pad=0.8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="run directory of run.sh")
    parser.add_argument("--out", default=os.path.join(PAPER, "figures", "sl_sweep.pdf"))
    args = parser.parse_args()
    plot(load(args.input), args.out)


if __name__ == "__main__":
    main()
