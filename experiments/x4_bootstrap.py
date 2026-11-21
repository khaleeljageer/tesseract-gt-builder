#!/usr/bin/env python3
"""Paired bootstrap for the X4 scale and font-count runs (IJDAR revision).

Each X4 run is trained with the corrected workflow (seed 0, 100,000
iterations) on a subset of the released images (experiments/x4_lists.py)
and its checkpoint is selected on the selection split by
sweep_checkpoints.py. The selected checkpoints' predictions are stored in
results/revision/x4_preds/ and scored here line by line with the paper's
scorer; Tesseract reference systems are read from results/<model>.json.
Resampling as in paired_bootstrap.py: reporting split, by line, 10,000
resamples, seed string "<seed>:<label>:line".

    python3 experiments/x4_bootstrap.py [--n 10000] [--seed 0]
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import tamil_ocr_eval as ev  # noqa: E402
from paired_bootstrap import METRICS, per_line, rate  # noqa: E402

TEST = HERE / "testset"
PREDS = HERE / "results" / "revision" / "x4_preds"
OUT = HERE / "results" / "revision" / "x4_bootstrap.json"

FULL = "tamv3_i1400"          # full corpus, seed 0, selected, no dictionary
S50, S34, F10 = "x4:x4_s50_i11800", "x4:x4_s34_i5900", "x4:x4_f10_i7200"

# (label, A, B): A is the system tested for being better (lower error).
COMPARISONS = [
    ("full_vs_s50", FULL, S50),
    ("full_vs_s34", FULL, S34),
    ("faces29_vs_faces10", S34, F10),
    ("s50_vs_stock", S50, "stock"),
    ("s34_vs_stock", S34, "stock"),
    ("f10_vs_stock", F10, "stock"),
]


def load(system, names):
    if system.startswith("x4:"):
        d = PREDS / system.split(":", 1)[1]
        return {n: ev.score_pair(ev.read(TEST / "gt" / f"{n}.gt.txt"),
                                 ev.read(d / f"{n}.txt")) for n in names}
    rows = per_line(system)
    return {n: rows[n] for n in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sweep = json.loads((HERE / "results" / "checkpoint_sweep.json").read_text())
    names = sweep["reporting_split"]
    cache = {}
    out = {"n_resamples": args.n, "seed": args.seed, "view": "rep202",
           "unit": "line",
           "note": "A is better in a resample when its rate is strictly lower; "
                   "diff = A - B in percentage points",
           "comparisons": {}}
    for label, a, b in COMPARISONS:
        for m in (a, b):
            if m not in cache:
                cache[m] = load(m, names)
        rng = random.Random(f"{args.seed}:{label}:line")
        draws = [[rng.choice(names) for _ in names] for _ in range(args.n)]
        row = {"A": a, "B": b, "lines": len(names)}
        for met, (e, r) in METRICS.items():
            ra = rate([cache[a][n] for n in names], e, r)
            rb = rate([cache[b][n] for n in names], e, r)
            diffs, wins = [], 0
            for s in draws:
                da = rate([cache[a][n] for n in s], e, r)
                db = rate([cache[b][n] for n in s], e, r)
                diffs.append(100 * (da - db))
                wins += da < db
            diffs.sort()
            row[met] = {"A": round(100 * ra, 2), "B": round(100 * rb, 2),
                        "diff": round(100 * (ra - rb), 2), "a_better": wins,
                        "ci95": [round(diffs[int(0.025 * args.n)], 2),
                                 round(diffs[int(0.975 * args.n) - 1], 2)]}
        out["comparisons"][label] = row
        print(f"{label:20s} " + "  ".join(
            f"{m} {row[m]['A']:6.2f} vs {row[m]['B']:6.2f} "
            f"{row[m]['diff']:+6.2f} {row[m]['a_better']:>5d} "
            f"[{row[m]['ci95'][0]:+.2f},{row[m]['ci95'][1]:+.2f}]"
            for m in METRICS))
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
