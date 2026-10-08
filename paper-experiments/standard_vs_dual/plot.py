"""
Plot E6 standard vs dual form: (a) step time against the number of GPUs, with ideal
scaling from one GPU, and (b) peak memory per GPU.

    python paper-experiments/standard_vs_dual/plot.py --input paper-experiments/standard_vs_dual/results/<run>/forms.jsonl
    python paper-experiments/standard_vs_dual/plot.py --synthetic  # placeholder
"""

import argparse
import json
import math
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

FORMS = {"standard": ("Standard form", SLOTS[1]), "dual": ("Dual form", SLOTS[0])}

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


def load(path):
    """Step time of the timed step and peak memory, keyed by (form, P)."""
    times, memory = {}, {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "timed":
                times[(r["form"], r["world_size"])] = r["time_s"]
                memory[(r["form"], r["world_size"])] = r["peak_mem_gb"]
    return times, memory


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


def plot(times, memory, out, watermark):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.6), gridspec_kw={"width_ratios": [1.25, 1]})
    forms = [f for f in FORMS if any(k[0] == f for k in times)]
    procs = sorted({p for _, p in times})

    # (a) Step time on log-log axes, against ideal scaling from one GPU
    ends = []
    for i, form in enumerate(forms):
        label, color = FORMS[form]
        ps = sorted(p for f, p in times if f == form)
        t1 = times[(form, 1)]
        ax1.plot(ps, [times[(form, p)] for p in ps], color=color, lw=1.2, marker="o", ms=3, label=label)
        ax1.plot(ps, [t1 / p for p in ps], color=INK, lw=0.6, ls=(0, (2, 2)), zorder=3,
                 label="Ideal" if i == 0 else None)
        ends.append((ps[-1], times[(form, ps[-1])], t1 / times[(form, ps[-1])]))
    # Direct labels right of the last points: speedup and parallel efficiency
    ys = spread([math.log10(t) for _, t, _ in ends], gap=0.2)
    for (p, _, sp), y in zip(ends, ys):
        ax1.text(p * 1.12, 10**y, f"{sp:.2f}$\\times$ ({sp / p:.1%})", ha="left", va="center", fontsize=5.5,
                 color=INK)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(procs, [str(p) for p in procs])
    ax1.minorticks_off()
    lo, hi = min(times.values()), max(times.values())
    yticks = [v for v in (0.1, 0.3, 1, 3, 10, 30, 100) if lo / 1.5 <= v <= hi * 1.5]
    ax1.set_yticks(yticks, [f"{v:g}" for v in yticks])
    ax1.set_xlim(procs[0] / 1.25, procs[-1] * 4.6)
    ax1.set_ylim(lo / 1.5, hi * 1.5)
    ax1.spines["bottom"].set_bounds(procs[0], procs[-1])
    ax1.set_xlabel("Number of GPUs $P$")
    ax1.set_ylabel("Step time (s)")
    # Legend above the axes, models before the ideal
    handles, labels = ax1.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda k: labels[k] == "Ideal")
    ax1.legend([handles[k] for k in order], [labels[k] for k in order], loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=5.5, handlelength=1.6, columnspacing=0.8,
               borderaxespad=0.1)
    ax1.set_title("(a) Step time", fontsize=7, pad=11)

    # (b) Peak memory per GPU, grouped by P, labeled at P = 1
    width = 0.8 / len(forms)
    for i, form in enumerate(forms):
        label, color = FORMS[form]
        xs = [j + (i - (len(forms) - 1) / 2) * width for j in range(len(procs))]
        mem = [memory[(form, p)] for p in procs]
        ax2.bar(xs, mem, width=width * 0.9, color=color, edgecolor="white", linewidth=0.4, label=label)
        ax2.text(xs[0], mem[0] + 0.02 * max(memory.values()), f"{mem[0]:.1f}", ha="center", va="bottom",
                 fontsize=5.5, color=INK)
    ax2.set_xticks(range(len(procs)), [str(p) for p in procs])
    ax2.set_ylim(0, max(memory.values()) * 1.15)
    ax2.set_xlabel("Number of GPUs $P$")
    ax2.set_ylabel("Peak memory per GPU (GB)")
    ax2.set_title("(b) Memory", fontsize=7, pad=11)

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
    parser.add_argument("--input", default=None, help="forms.jsonl of a run.sh invocation")
    parser.add_argument("--out", default=os.path.join(PAPER, "figures", "standard_vs_dual.pdf"))
    args = parser.parse_args()
    if args.input is None and not args.synthetic:
        parser.error("--input is required unless --synthetic is given")
    times, memory = load(args.input or os.path.join(HERE, "synthetic", "forms.jsonl"))
    plot(times, memory, args.out, watermark=args.synthetic)


if __name__ == "__main__":
    main()
