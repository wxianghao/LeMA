"""
Write the E7 table: relative differences of the trial updates of both forms and their
errors against the FP64 reference, as a LaTeX tabular for \\input in the paper.

    python paper-experiments/update_equivalence/table.py --input paper-experiments/update_equivalence/results/<run>/updates.jsonl
    python paper-experiments/update_equivalence/table.py --synthetic  # placeholder in red
"""

import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PAPER = os.path.join(os.path.dirname(REPO), "2027-ipdps-lema")

MODEL_LABELS = {"lenet5": "LeNet-5", "mlp": "MLP"}


def sci(x: float) -> str:
    """Two significant digits in LaTeX, e.g. $3.4 \\times 10^{-3}$."""
    if x == 0:
        return "$0$"
    exp = math.floor(math.log10(x))
    mant = x / 10**exp
    if round(mant, 1) >= 10:
        mant, exp = mant / 10, exp + 1
    return f"${mant:.1f} {{\\times}} 10^{{{exp}}}$"


def power(x: float) -> str:
    exp = round(math.log10(x))
    return "$1$" if exp == 0 else f"$10^{{{exp}}}$"


def table(records, placeholder: bool) -> str:
    mark = (lambda s: f"\\textcolor{{red}}{{{s}}}") if placeholder else (lambda s: s)
    lines = [
        "\\begin{tabular}{llcccc}",
        "    \\hline",
        "    Model & $N$ & $\\lambda$ & Std.\\ vs.\\ dual & Std.\\ vs.\\ FP64 & Dual vs.\\ FP64 \\\\",
    ]
    # One group of rows per (model, N), in the order of the records
    groups = list(dict.fromkeys((r["model"], r["batch_size"]) for r in records))
    last_model = None
    for model, n in groups:
        rows = sorted((r for r in records if (r["model"], r["batch_size"]) == (model, n)), key=lambda r: r["damp"])
        lines.append("    \\hline")
        for i, r in enumerate(rows):
            model_cell = MODEL_LABELS.get(model, model) if model != last_model and i == 0 else ""
            n_cell = f"{n:,}" if i == 0 else ""
            cells = [model_cell, n_cell, power(r["damp"])]
            cells += [mark(sci(r[k])) for k in ("diff_forms", "err_standard", "err_dual")]
            lines.append("    " + " & ".join(cells) + " \\\\")
        last_model = model
    lines += ["    \\hline", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="tabulate fake data in red")
    parser.add_argument("--input", default=None, help="updates.jsonl of a run.sh invocation")
    parser.add_argument("--out", default=os.path.join(PAPER, "tables", "update_equivalence.tex"))
    args = parser.parse_args()
    if args.input is None and not args.synthetic:
        parser.error("--input is required unless --synthetic is given")
    with open(args.input or os.path.join(HERE, "synthetic", "updates.jsonl")) as f:
        records = [json.loads(line) for line in f]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(table(records, placeholder=args.synthetic))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
