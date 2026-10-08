"""
Plot E9: (a) training loss and (b) test accuracy against the training time, and (c) the peak
memory per GPU, for LeMA's dual and standard form on 1 and 8 GPUs, LM with PCG, and Adam.

    python paper-experiments/convergence/plot.py --input paper-experiments/convergence/results/<run>
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
DASHED = (0, (3, 1.5))

# (method, P): label, color, line style; colors by method as in standard_vs_dual/plot.py
RUNS = {
    ("lema-dual", 1): ("LeMA dual, 1 GPU", SLOTS[0], "-"),
    ("lema-dual", 8): ("LeMA dual, 8 GPUs", SLOTS[0], DASHED),
    ("lema-standard", 1): ("LeMA standard, 1 GPU", SLOTS[1], "-"),
    ("lema-standard", 8): ("LeMA standard, 8 GPUs", SLOTS[1], DASHED),
    ("pcg-100", 1): ("PCG-100, 1 GPU", SLOTS[2], "-"),
    ("pcg-10", 1): ("PCG-10, 1 GPU", SLOTS[2], DASHED),
    ("adam", 1): ("Adam, 1 GPU", MUTED, "-"),
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
    "pdf.fonttype": 42,
})


def load(run_dir):
    """Evaluation records of each run, keyed by (method, P)."""
    runs = {}
    for path in glob.glob(os.path.join(run_dir, "runs", "*.jsonl")):
        with open(path) as f:
            records = [json.loads(line) for line in f]
        if records:
            runs[(records[0]["method"], records[0]["world_size"])] = sorted(records, key=lambda r: r["step"])
    return runs


def plot(runs, out):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 1.85), gridspec_kw={"width_ratios": [1, 1, 0.85]})
    keys = [k for k in RUNS if k in runs]

    # (a, b) Training loss and test accuracy against the training time, from the first step on
    for key in keys:
        label, color, ls = RUNS[key]
        rs = [r for r in runs[key] if r["step"] > 0]
        t = [r["time_s"] for r in rs]
        ax1.plot(t, [r["train_loss"] for r in rs], color=color, ls=ls, lw=1.0, label=label)
        ax2.plot(t, [100 * r["test_acc"] for r in rs], color=color, ls=ls, lw=1.0)
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plain)
        ax.set_xlabel("Training time (s)")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(plain)
    ax1.set_ylabel("Training loss")
    ax1.set_title("(a) Training loss", fontsize=7, pad=3)
    ax2.set_ylim(90, 98.5)
    ax2.set_ylabel("Test accuracy (%)")
    ax2.set_title("(b) Test accuracy", fontsize=7, pad=3)

    # (c) Peak memory per GPU, labeled with its value
    mem = [runs[k][-1]["peak_mem_gb"] for k in keys]
    ax3.barh(range(len(keys)), mem, height=0.62, color=[RUNS[k][1] for k in keys], edgecolor="white", linewidth=0.4,
             hatch=["////" if RUNS[k][2] == DASHED else "" for k in keys])
    for i, m in enumerate(mem):
        ax3.text(m * 1.15, i, f"{m:.2f}" if m < 10 else f"{m:.1f}", va="center", ha="left", fontsize=5.5, color=INK)
    ax3.set_xscale("log")
    ax3.set_xlim(min(mem) / 1.5, max(mem) * 6)
    ax3.set_xticks([v for v in (0.1, 1, 10, 100) if min(mem) / 1.5 <= v <= max(mem) * 6])
    ax3.xaxis.set_major_formatter(plain)
    ax3.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax3.set_yticks(range(len(keys)), [RUNS[k][0] for k in keys], fontsize=5.5)
    ax3.invert_yaxis()
    ax3.set_xlabel("Peak memory per GPU (GB)")
    ax3.set_title("(c) Memory", fontsize=7, pad=3)

    for ax in (ax1, ax2, ax3):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
        ax.set_axisbelow(True)
    ax1.grid(axis="y", color="#e4e3df", lw=0.4)
    ax2.grid(axis="y", color="#e4e3df", lw=0.4)
    fig.legend(*ax1.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.36, 1.0), ncol=4,
               fontsize=6, handlelength=2.0, columnspacing=1.0)
    fig.tight_layout(pad=0.2, w_pad=1.0)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="run directory of run.sh")
    parser.add_argument("--out", default=os.path.join(PAPER, "figures", "convergence.pdf"))
    args = parser.parse_args()
    plot(load(args.input), args.out)


if __name__ == "__main__":
    main()
