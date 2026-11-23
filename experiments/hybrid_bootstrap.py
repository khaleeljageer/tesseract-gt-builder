#!/usr/bin/env python3
"""Hybrid real-plus-synthetic runs and the real-only control (IJDAR revision, R2.7).

Each hybrid run trains the corrected workflow on its seed's own synthetic
list.train with the 445 real lines of hybrid/ inserted 25 times each at
random positions (5.4% of the stream); the real-only control trains stock
tam on the 445 lines alone (10,000 iterations, list shuffled by
tesstrain's shuffle.py with the seed, three seeds). Checkpoints are selected on
the selection split by sweep_checkpoints.py; the selected checkpoints'
predictions are stored in results/revision/hybrid_preds/ and scored here
line by line with the paper's scorer. Resampling as in paired_bootstrap.py:
reporting split, by line, 10,000 resamples, seed string "<seed>:<label>:line".

"pooled" systems sum the edit and reference counts of the three seeds line
by line, so a resample compares three seeds of one arm with three seeds of
the other on the same lines.

    python3 experiments/hybrid_bootstrap.py [--n 10000] [--seed 0]
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import tamil_ocr_eval as ev  # noqa: E402
from paired_bootstrap import METRICS, per_line, rate  # noqa: E402

TEST = HERE / "testset"
PREDS = HERE / "results" / "revision" / "hybrid_preds"
OUT = HERE / "results" / "revision" / "hybrid_bootstrap.json"

SYN = ["p:tam_x9_i1400", "p:tam_v3_s1_i4700", "p:tam_v3_s2_i6500"]
HYB = ["p:tam_hyb_s0_i6000", "p:tam_hyb_s1_i5700", "p:tam_hyb_s2_i3900"]
REAL = ["p:tam_realsh_s0_i1000", "p:tam_realsh_s1_i3200", "p:tam_realsh_s2_i1400"]
RELEASE = "tamv3_dawg_corpus"

COMPARISONS = (
    [(f"hyb_s{k}_vs_syn_s{k}", HYB[k], SYN[k]) for k in range(3)]
    + [("hyb_pooled_vs_syn_pooled", "pool:hyb", "pool:syn")]
    + [(f"hyb_s{k}_vs_stock", HYB[k], "stock") for k in range(3)]
    + [("hyb_pooled_vs_stock", "pool:hyb", "stock"),
       ("hyb_pooled_vs_release", "pool:hyb", RELEASE),
       ("real_pooled_vs_stock", "pool:real", "stock"),
       ("real_pooled_vs_syn_pooled", "pool:real", "pool:syn"),
       ("real_pooled_vs_release", "pool:real", RELEASE),
       ("real_pooled_vs_hyb_pooled", "pool:real", "pool:hyb")]
    + [(f"real_s{k}_vs_stock", REAL[k], "stock") for k in range(3)]
)
STRATA = ("newsprint", "books", "forms", "degraded", "signage")


def load(system, names, cache):
    if system in cache:
        return cache[system]
    if system.startswith("p:"):
        d = PREDS / system[2:]
        rows = {n: ev.score_pair(ev.read(TEST / "gt" / f"{n}.gt.txt"),
                                 ev.read(d / f"{n}.txt")) for n in names}
    elif system.startswith("pool:"):
        parts = [load(s, names, cache) for s in
                 {"pool:hyb": HYB, "pool:syn": SYN, "pool:real": REAL}[system]]
        rows = {n: {k: sum(p[n][k] for p in parts) for k in
                    ("grapheme_edits", "grapheme_ref", "word_edits", "word_ref")}
                for n in names}
    else:
        allrows = per_line(system)
        rows = {n: allrows[n] for n in names}
    cache[system] = rows
    return rows


def compare(label, a, b, names, cache, n, seed):
    A, B = load(a, names, cache), load(b, names, cache)
    rng = random.Random(f"{seed}:{label}:line")
    draws = [[rng.choice(names) for _ in names] for _ in range(n)]
    row = {"A": a, "B": b, "lines": len(names)}
    for met, (e, r) in METRICS.items():
        ra, rb = rate([A[x] for x in names], e, r), rate([B[x] for x in names], e, r)
        diffs, wins = [], 0
        for s in draws:
            da, db = rate([A[x] for x in s], e, r), rate([B[x] for x in s], e, r)
            diffs.append(100 * (da - db))
            wins += da < db
        diffs.sort()
        row[met] = {"A": round(100 * ra, 2), "B": round(100 * rb, 2),
                    "diff": round(100 * (ra - rb), 2), "a_better": wins,
                    "ci95": [round(diffs[int(0.025 * n)], 2),
                             round(diffs[int(0.975 * n) - 1], 2)]}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sweep = json.loads((HERE / "results" / "checkpoint_sweep.json").read_text())
    names = sweep["reporting_split"]
    with open(TEST / "manifest.csv", newline="") as fh:
        meta = {r["line_id"]: r for r in csv.DictReader(fh)}
    cache = {}
    out = {"n_resamples": args.n, "seed": args.seed, "view": "rep202", "unit": "line",
           "note": "A is better in a resample when its rate is strictly lower; "
                   "diff = A - B in percentage points; pool = three seeds summed "
                   "line by line",
           "systems": {"syn": SYN, "hyb": HYB, "real": REAL, "release": RELEASE},
           "comparisons": {}, "strata": {}}
    for label, a, b in COMPARISONS:
        row = compare(label, a, b, names, cache, args.n, args.seed)
        out["comparisons"][label] = row
        print(f"{label:26s} " + "  ".join(
            f"{m} {row[m]['A']:6.2f} vs {row[m]['B']:6.2f} {row[m]['diff']:+6.2f} "
            f"{row[m]['a_better']:>5d} [{row[m]['ci95'][0]:+.2f},{row[m]['ci95'][1]:+.2f}]"
            for m in METRICS))
    systems = ["stock", RELEASE, "pool:syn", "pool:hyb", "pool:real"] + SYN + HYB + REAL
    for st in STRATA:
        sub = [n for n in names if meta[n]["stratum"] == st]
        out["strata"][st] = {"lines": len(sub), "rates": {}, "comparisons": {}}
        for s in systems:
            rows = load(s, names, cache)
            out["strata"][st]["rates"][s] = {
                m: round(100 * rate([rows[x] for x in sub], e, r), 2)
                for m, (e, r) in METRICS.items()}
        for label, a, b in (("hyb_pooled_vs_syn_pooled", "pool:hyb", "pool:syn"),
                            ("real_pooled_vs_syn_pooled", "pool:real", "pool:syn"),
                            ("real_pooled_vs_hyb_pooled", "pool:real", "pool:hyb")):
            out["strata"][st]["comparisons"][label] = compare(
                f"{label}_{st}", a, b, sub, cache, args.n, args.seed)
        r = out["strata"][st]
        print(f"{st:10s} n={len(sub):3d} " + "  ".join(
            f"{s.split(':')[-1]} {r['rates'][s]['cer']:.2f}/{r['rates'][s]['wer']:.2f}"
            for s in ["stock", RELEASE, "pool:syn", "pool:hyb", "pool:real"]))
        for label, c in r["comparisons"].items():
            print(f"    {label:26s} CER {c['cer']['diff']:+6.2f} {c['cer']['a_better']:>5d}"
                  f"  WER {c['wer']['diff']:+6.2f} {c['wer']['a_better']:>5d}")
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
