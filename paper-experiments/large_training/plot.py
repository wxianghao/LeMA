"""
Plot E11: (a) the CE loss on the full training and test sets and (b) the test accuracy
against the LM step, for LeMA training CNN-1B on 8 GPUs.
Also prints the numbers quoted in the paper.

    python paper-experiments/large_training/plot.py --input paper-experiments/large_training/results/<run>/training.jsonl
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
INK, MUTED = "#0b0b0b", "#52514e"
DASHED = (0, (3, 1.5))

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
        return sorted((json.loads(line) for line in f), key=lambda r: r["step"])


def summarize(records):
    first, last = records[0], records[-1]
    steps = records[1:]
    rejected = sum(r["rejected"] for r in steps)
    trials = sum(r["trials"] for r in steps)
    print(f"{last['model']} | M: {last['params']:,} | P: {last['world_size']} | N: {last['batch_size']} | "
          f"S: {last['shard_size']} | L: {last['slice_size']} | steps: {len(steps)}")
    print(f"train loss {first['train_loss']:.4f} -> {last['train_loss']:.4f} | test loss {first['test_loss']:.4f} -> "
          f"{last['test_loss']:.4f} | test acc {first['test_acc']:.2%} -> {last['test_acc']:.2%}")
    print(f"accepted steps {sum(r['accepted'] for r in steps)} of {len(steps)} | trials {trials} | rejected {rejected} "
          f"({rejected / trials:.1%}) | steps with rejections {sum(r['rejected'] > 0 for r in steps)} | "
          f"terminated {sum(r['terminate'] for r in steps)}")
    times = [r["step_time_s"] for r in steps]
    print(f"step time mean {sum(times) / len(times):.1f} s (min {min(times):.1f}, max {max(times):.1f}) | "
          f"training time {last['time_s'] / 60:.1f} min | peak memory {max(r['peak_mem_gb'] for r in steps):.2f} GB | "
          f"evaluation {sum(r['eval_time_s'] for r in records):.0f} s in total")
    print("damping per step:", " ".join(f"{r['damp']:.0e}" for r in steps))
    increases = [b["train_loss"] > a["train_loss"] for a, b in zip(records, records[1:])]
    print(f"steps that increase the full training loss: {sum(increases)}")


def plot(records, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.6))
    steps = records[1:]

    # (a) Loss on the full sets, from the initial model on
    k = [r["step"] for r in records]
    ax1.plot(k, [r["train_loss"] for r in records], color=SLOTS[0], lw=1.0, marker="o", ms=1.8, label="Training")
    ax1.plot(k, [r["test_loss"] for r in records], color=SLOTS[1], lw=1.0, ls=DASHED, label="Test")
    ax1.set_yscale("log")
    # Label 1-2-3-5 ticks within the range of the data, since it spans about one decade
    losses = [r[key] for r in records for key in ("train_loss", "test_loss")]
    ax1.set_yticks([v for v in (0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5) if min(losses) / 1.1 <= v <= max(losses) * 1.1])
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_xlabel("LM step")
    ax1.set_ylabel("CE loss")
    ax1.set_title("(a) Loss", fontsize=7)
    ax1.legend(loc="upper right", fontsize=5.5, handlelength=2.0, borderaxespad=0.2)

    # (b) Test accuracy, from the initial model on; the trials go to the printed summary,
    # since with M >> N every step is accepted at its first trial
    ax2.plot(k, [100 * r["test_acc"] for r in records], color=SLOTS[0], lw=1.0, marker="o", ms=1.8)
    ax2.set_xlabel("LM step")
    ax2.set_ylabel("Test accuracy (%)")
    ax2.set_title("(b) Test accuracy", fontsize=7)

    for ax in (ax1, ax2):
        ax.set_xlim(-0.8, len(steps) + 0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
        ax.grid(axis="y", color="#e4e3df", lw=0.4)
        ax.set_axisbelow(True)

    fig.tight_layout(pad=0.2, w_pad=1.0)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="training.jsonl of a run.sh invocation")
    parser.add_argument("--out", default=os.path.join(PAPER, "figures", "large_training.pdf"))
    args = parser.parse_args()
    records = load(args.input)
    summarize(records)
    plot(records, args.out)


if __name__ == "__main__":
    main()
