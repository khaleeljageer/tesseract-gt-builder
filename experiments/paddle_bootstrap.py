#!/usr/bin/env python3
"""Paired bootstrap for the PaddleOCR comparisons (IJDAR revision, X5).

Scores each stored PaddleOCR prediction directory line by line with the same
scorer as score_preds.py, and compares pairs of systems on the reporting
split with the resampling of paired_bootstrap.py (same seed convention,
10,000 resamples, by line). Tesseract systems are read from their stored
per-line results (results/<model>.json).

    python3 experiments/paddle_bootstrap.py [--n 10000] [--seed 0]
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
PREDS = HERE / "results" / "revision" / "paddle_preds"
OUT = HERE / "results" / "revision" / "paddle_bootstrap.json"

# (label, A, B); A is the system tested for being better (lower error).
# "paddle:<dir>" names a PaddleOCR prediction directory, anything else a
# Tesseract result file.
COMPARISONS = [
    ("ft_model_vs_zeroshot", "paddle:ft_model_s4000", "paddle:zeroshot"),
    ("ft_rebuilt_vs_zeroshot", "paddle:ft_rebuilt_s9000", "paddle:zeroshot"),
    ("model_vs_rebuilt_dict", "paddle:ft_model_s4000", "paddle:ft_rebuilt_s9000"),
    ("tess_release_vs_zeroshot", "tamv3_dawg_corpus", "paddle:zeroshot"),
    ("tess_release_vs_ft_model", "tamv3_dawg_corpus", "paddle:ft_model_s4000"),
]


def load(system, names):
    if system.startswith("paddle:"):
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
        print(f"{label:26s} " + "  ".join(
            f"{m} {row[m]['A']:6.2f} vs {row[m]['B']:6.2f} "
            f"{row[m]['diff']:+6.2f} {row[m]['a_better']:>5d} "
            f"[{row[m]['ci95'][0]:+.2f},{row[m]['ci95'][1]:+.2f}]"
            for m in METRICS))
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
