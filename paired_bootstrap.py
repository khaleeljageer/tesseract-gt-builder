#!/usr/bin/env python3
"""Paired bootstrap for every model-versus-model comparison in the paper.

Written for the IJDAR revision (reviewer 2, item 2): the paper quoted four
resample counts for one CER comparison (9,609, 9,588, 9,603, 0.959) without
saying which comparison each belongs to. This script is now the single
source for all of them. It reads the stored per-line edit counts
(results/<model>.json, written by tamil_ocr_eval.py), restricts them to a
test-set view, and for N resamples (fixed seed) draws the same lines --- or
the same whole source pages --- for both systems, computing micro-averaged
grapheme CER and WER for each.

Reported per comparison and metric: the point difference A - B, the number
of resamples in which A is strictly better (lower), and the percentile 95%
interval on the difference.

    python3 paired_bootstrap.py [--n 10000] [--seed 0] \
        [--json results/paired_bootstrap.json]
"""
import argparse
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

# (label, A, B, view, stratum or None); A is the system claimed better.
COMPARISONS = [
    ("release_vs_stock", "tamv3_dawg_corpus", "stock", "rep202", None),
    ("nodict_vs_stock", "tamv3_i1400", "stock", "rep202", None),
    ("stockdawg_vs_stock", "tamv3_dawg_stock", "stock", "rep202", None),
    ("uniondawg_vs_stock", "tamv3_dawg_union", "stock", "rep202", None),
    ("corpus_effect", "tam_unset", "tamsyn", "rep202", None),
    ("workflow_effect", "tamv3_dawg_corpus", "tam_unset", "rep202", None),
    ("defaults_vs_stock", "stock", "tam_unset", "rep202", None),
] + [(f"release_vs_stock_{s}", "tamv3_dawg_corpus", "stock", "rep202", s)
     for s in ("newsprint", "books", "forms", "degraded", "signage")]

METRICS = {"cer": ("grapheme_edits", "grapheme_ref"),
           "wer": ("word_edits", "word_ref")}


def per_line(model):
    return {r["name"]: r for r in
            json.loads((RES / f"{model}.json").read_text())["per_line"]}


def rate(rows, e, r):
    ref = sum(x[r] for x in rows)
    return sum(x[e] for x in rows) / ref if ref else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=str(RES / "paired_bootstrap.json"))
    args = ap.parse_args()

    sweep = json.loads((RES / "checkpoint_sweep.json").read_text())
    views = {"rep202": sweep["reporting_split"], "sel98": sweep["selection_split"]}
    with open(HERE / "testset" / "manifest.csv", newline="") as fh:
        meta = {r["line_id"]: r for r in csv.DictReader(fh)}
    cache = {}
    out = {"n_resamples": args.n, "seed": args.seed,
           "note": "A is better in a resample when its rate is strictly lower; "
                   "diff = A - B in percentage points",
           "comparisons": {}}

    for label, a, b, view, stratum in COMPARISONS:
        for m in (a, b):
            cache.setdefault(m, per_line(m))
        names = [n for n in views[view]
                 if stratum is None or meta[n]["stratum"] == stratum]
        pages = {}
        for n in names:
            pages.setdefault(meta[n]["source_page"], []).append(n)
        page_list = list(pages.values())
        row = {"A": a, "B": b, "view": view, "stratum": stratum,
               "lines": len(names), "pages": len(page_list)}
        for unit in ("line", "page"):
            rng = random.Random(f"{args.seed}:{label}:{unit}")
            draws = []
            for _ in range(args.n):
                if unit == "line":
                    sample = [rng.choice(names) for _ in names]
                else:
                    sample = [n for _ in page_list for n in rng.choice(page_list)]
                draws.append(sample)
            for met, (e, r) in METRICS.items():
                pa = [cache[a][n] for n in names]
                pb = [cache[b][n] for n in names]
                point = 100 * (rate(pa, e, r) - rate(pb, e, r))
                diffs, wins = [], 0
                for s in draws:
                    da = rate([cache[a][n] for n in s], e, r)
                    db = rate([cache[b][n] for n in s], e, r)
                    diffs.append(100 * (da - db))
                    wins += da < db
                diffs.sort()
                lo = diffs[int(0.025 * args.n)]
                hi = diffs[int(0.975 * args.n) - 1]
                row[f"{met}_{unit}"] = {"diff": round(point, 2), "a_better": wins,
                                        "ci95": [round(lo, 2), round(hi, 2)]}
        out["comparisons"][label] = row
        print(f"{label:28s} {len(names):3d} lines {len(page_list):3d} pages  "
              + "  ".join(f"{k} {v['diff']:+.2f} {v['a_better']:>5d} "
                          f"[{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]"
                          for k, v in row.items() if isinstance(v, dict)),
              flush=True)
    Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
