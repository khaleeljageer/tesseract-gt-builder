#!/usr/bin/env python3
"""The Anuvaad project's Tamil Tesseract models on our test set (X6).

Saini et al. (arXiv:2205.02543) do not release their synthetic images; the
paper gives no download location, and none is listed by the authors'
organisation (project-anuvaad on GitHub). The same group does publish the
Tesseract models it fine-tuned (github.com/project-anuvaad/anuvaad-ocr-model):
  anuvaad_tam.traineddata                 md5 c62a83e0d3bd3f706ba880321aedc22a
  anuvad_tam_scene_text_real.traineddata  md5 fcbeceb3d87ba63ebd3af8f2d5a81bb4
both downloaded 2026-11-21 from anuvaad-pubnet-weights.s3.amazonaws.com
(LSTM only, no DAWGs, version 4.1.1). Their training data is not documented
beyond the paper. They were run on the 300 test lines with Tesseract 5.5.0,
--psm 13 (make_testset.py predict), predictions in
results/revision/x6_preds/, and are compared here with stock and the
released model by paired bootstrap as in paired_bootstrap.py (reporting
split, by line, 10,000 resamples, seed string "<seed>:<label>:line").

    python3 experiments/x6_anuvaad.py
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
PREDS = HERE / "results" / "revision" / "x6_preds"
OUT = HERE / "results" / "revision" / "x6_anuvaad.json"
DOC, SCENE = "av:anuvaad_tam", "av:anuvad_tam_scene_text_real"
COMPARISONS = [
    ("anuvaad_vs_stock", DOC, "stock"),
    ("scene_vs_stock", SCENE, "stock"),
    ("release_vs_anuvaad", "tamv3_dawg_corpus", DOC),
    ("nodict_vs_anuvaad", "tamv3_i1400", DOC),
    ("release_vs_scene", "tamv3_dawg_corpus", SCENE),
]


def load(system, names):
    if system.startswith("av:"):
        d = PREDS / system.split(":", 1)[1]
        return {n: ev.score_pair(ev.read(TEST / "gt" / f"{n}.gt.txt"),
                                 ev.read(d / f"{n}.txt")) for n in names}
    rows = per_line(system)
    return {n: rows[n] for n in names}


def main(n_res=10000, seed=0):
    sweep = json.loads((HERE / "results" / "checkpoint_sweep.json").read_text())
    rep, sel = sweep["reporting_split"], sweep["selection_split"]
    allnames = sorted(p.name[:-len(".gt.txt")] for p in (TEST / "gt").glob("*.gt.txt"))
    out = {"n_resamples": n_res, "seed": seed, "scores": {}, "by_stratum": {},
           "comparisons": {}}
    cache = {}
    for m in (DOC, SCENE, "stock", "tamv3_dawg_corpus", "tamv3_i1400"):
        cache[m] = load(m, allnames)
        row = {}
        for v, ns in (("all300", allnames), ("sel98", sel), ("rep202", rep)):
            a = ev.aggregate([cache[m][n] for n in ns])
            row[v] = [round(100 * a["cer_grapheme_micro"], 2), round(100 * a["wer_micro"], 2)]
        out["scores"][m] = row
        strata = {}
        for s in sorted({n.split("_")[0] for n in rep}):
            a = ev.aggregate([cache[m][n] for n in rep if n.startswith(s + "_")])
            strata[s] = [round(100 * a["cer_grapheme_micro"], 2), round(100 * a["wer_micro"], 2)]
        out["by_stratum"][m] = strata
        print(f"{m:32s} {row}  {strata}")
    for label, a, b in COMPARISONS:
        rng = random.Random(f"{seed}:{label}:line")
        draws = [[rng.choice(rep) for _ in rep] for _ in range(n_res)]
        res = {"A": a, "B": b}
        for met, (e, r) in METRICS.items():
            diffs, wins = [], 0
            for s in draws:
                da = rate([cache[a][n] for n in s], e, r)
                db = rate([cache[b][n] for n in s], e, r)
                diffs.append(100 * (da - db))
                wins += da < db
            diffs.sort()
            full = 100 * (rate([cache[a][n] for n in rep], e, r) - rate([cache[b][n] for n in rep], e, r))
            res[met] = {"diff": round(full, 2), "a_better": wins,
                        "ci95": [round(diffs[int(0.025 * n_res)], 2),
                                 round(diffs[int(0.975 * n_res) - 1], 2)]}
        out["comparisons"][label] = res
        print(f"{label:20s} " + "  ".join(
            f"{m} {res[m]['diff']:+.2f} {res[m]['a_better']:5d} {res[m]['ci95']}" for m in METRICS))
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
