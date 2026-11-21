#!/usr/bin/env python3
"""Tesseract-version check for the IJDAR revision (X10).

The 300 test lines were recognised on the desktop (AMD Ryzen 9 9900X) by
results/revision/x10_preds/x10.sh with three models -- stock tam, the
released model (corpus DAWG) and the released checkpoint without a
dictionary -- under four Tesseract builds: 5.5.0 (the paper's version), 5.5.1,
the 5.5.2 release tag, and the 5.5.2-146-gd4f61b62 development build.
Invocation as in sweep_checkpoints.py: --psm 13, OMP_THREAD_LIMIT=1.

This script scores every directory with the paper's scorer, counts the
lines on which each run differs from the paper's stored predictions
(results/<model>.json), and repeats the release-vs-stock paired bootstrap
(reporting split, by line, 10,000 resamples, seed string
"<seed>:<label>:line") under each build.

    python3 experiments/x10_versions.py
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import tamil_ocr_eval as ev  # noqa: E402
from paired_bootstrap import METRICS, per_line, rate  # noqa: E402

TEST = HERE / "testset"
PREDS = HERE / "results" / "revision" / "x10_preds"
OUT = HERE / "results" / "revision" / "x10_versions.json"
BUILDS = ["v550", "v551", "v552", "v552dev"]
MODELS = {"tam": "stock", "tamv3dcorpus": "tamv3_dawg_corpus",
          "tamv3i1400": "tamv3_i1400"}
PAIRS = [("release_vs_stock", "tamv3dcorpus", "tam"),
         ("nodict_vs_stock", "tamv3i1400", "tam")]


def agg(rows, names):
    a = ev.aggregate([rows[n] for n in names])
    return {"cer": round(100 * a["cer_grapheme_micro"], 2),
            "wer": round(100 * a["wer_micro"], 2)}


def main(n_res=10000, seed=0):
    sweep = json.loads((HERE / "results" / "checkpoint_sweep.json").read_text())
    rep = sweep["reporting_split"]
    names = sorted(p.name[:-len(".gt.txt")] for p in (TEST / "gt").glob("*.gt.txt"))
    out = {"builds": {b: (PREDS / f"version_{b}.txt").read_text().splitlines()[0]
                      for b in BUILDS},
           "models_md5": (PREDS / "models.md5").read_text().split("\n"),
           "scores": {}, "bootstrap": {}}
    rows = {}
    for b in BUILDS:
        for m, ref in MODELS.items():
            r = {n: ev.score_pair(ev.read(TEST / "gt" / f"{n}.gt.txt"),
                                  ev.read(PREDS / b / m / f"{n}.txt"))
                 for n in names}
            rows[b, m] = r
            stored = per_line(ref)
            changed = sum((r[n]["grapheme_edits"], r[n]["word_edits"]) !=
                          (stored[n]["grapheme_edits"], stored[n]["word_edits"])
                          for n in names)
            text_vs_550 = sum((PREDS / b / m / f"{n}.txt").read_text() !=
                              (PREDS / "v550" / m / f"{n}.txt").read_text()
                              for n in names)
            out["scores"][f"{b}/{m}"] = {
                "all300": agg(r, names), "rep202": agg(r, rep),
                "lines_score_differs_from_paper": changed,
                "lines_text_differs_from_v550": text_vs_550}
            print(f"{b:8s} {m:13s} all {agg(r, names)} rep {agg(r, rep)} "
                  f"score-diff {changed:3d} text-diff-vs-550 {text_vs_550:3d}")
    for b in BUILDS:
        for label, a, c in PAIRS:
            rng = random.Random(f"{seed}:{label}:line")
            draws = [[rng.choice(rep) for _ in rep] for _ in range(n_res)]
            ra, rc = rows[b, a], rows[b, c]
            res = {}
            for met, (e, t) in METRICS.items():
                diffs, wins = [], 0
                for s in draws:
                    da, dc = rate([ra[n] for n in s], e, t), rate([rc[n] for n in s], e, t)
                    diffs.append(100 * (da - dc))
                    wins += da < dc
                diffs.sort()
                full = 100 * (rate([ra[n] for n in rep], e, t) - rate([rc[n] for n in rep], e, t))
                res[met] = {"diff": round(full, 2), "a_better": wins,
                            "ci95": [round(diffs[int(0.025 * n_res)], 2),
                                     round(diffs[int(0.975 * n_res) - 1], 2)]}
            out["bootstrap"][f"{b}/{label}"] = res
            print(f"{b:8s} {label:17s} " + "  ".join(
                f"{m} {res[m]['diff']:+.2f} {res[m]['a_better']:5d} {res[m]['ci95']}"
                for m in METRICS))
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
