#!/usr/bin/env python3
"""Score the local vision-language models on the test set (R2.8).

Predictions come from vlm_predict.py (results/revision/vlm_preds/<dir>/,
each with a _meta.json of raw outputs and timings). The candidate is chosen
on the 98-line selection split by grapheme CER (WER breaks ties), as the
paper chooses checkpoints and dictionaries, and only the chosen model is
compared on the 202-line reporting split, by the paired bootstrap of
paired_bootstrap.py (10,000 resamples by line, seed string
"<seed>:<label>:line").

Failure modes counted on all 300 lines: empty output, output with less than
half its letters in the Tamil block, output more than 1.5x the reference
length in graphemes (run-on or hallucinated text), and outputs cut at the
token limit.

    python3 experiments/vlm_score.py
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
PREDS = HERE / "results" / "revision" / "vlm_preds"
PADDLE = HERE / "results" / "revision" / "paddle_preds"
OUT = HERE / "results" / "revision" / "vlm_score.json"
COST = HERE / "results" / "revision" / "vlm_cost.json"


def from_dir(d, names):
    return {n: ev.score_pair(ev.read(TEST / "gt" / f"{n}.gt.txt"),
                             ev.read(d / f"{n}.txt")) for n in names}


def pct(rows, names):
    a = ev.aggregate([rows[n] for n in names])
    return [round(100 * a["cer_grapheme_micro"], 2), round(100 * a["wer_micro"], 2)]


def tamil_share(s):
    letters = [c for c in s if c.isalpha() or 0x0B80 <= ord(c) <= 0x0BFF]
    if not letters:
        return 0.0
    return sum(0x0B80 <= ord(c) <= 0x0BFF for c in letters) / len(letters)


def failures(d, names):
    meta = json.loads((d / "_meta.json").read_text())
    f = {"empty": [], "non_tamil": [], "overlong": [], "truncated": []}
    for n in names:
        hyp = ev.read(d / f"{n}.txt").strip()
        gt = ev.read(TEST / "gt" / f"{n}.gt.txt").strip()
        if not hyp:
            f["empty"].append(n)
            continue
        if tamil_share(hyp) < 0.5:
            f["non_tamil"].append(n)
        if len(ev.graphemes(hyp)) > 1.5 * len(ev.graphemes(gt)):
            f["overlong"].append(n)
        if meta["lines"][n]["done_reason"] == "length":
            f["truncated"].append(n)
    return {k: len(v) for k, v in f.items()} | {"lines": f,
            "n_cleaned": meta["summary"]["n_cleaned"]}


def compare(cache, rep, label, a, b, n_res, seed):
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
        full = 100 * (rate([cache[a][n] for n in rep], e, r)
                      - rate([cache[b][n] for n in rep], e, r))
        res[met] = {"diff": round(full, 2), "a_better": wins,
                    "ci95": [round(diffs[int(0.025 * n_res)], 2),
                             round(diffs[int(0.975 * n_res) - 1], 2)]}
    print(f"{label:28s} " + "  ".join(
        f"{m} {res[m]['diff']:+.2f} {res[m]['a_better']:5d} {res[m]['ci95']}"
        for m in METRICS))
    return res


def main(n_res=10000, seed=0):
    sweep = json.loads((HERE / "results" / "checkpoint_sweep.json").read_text())
    rep, sel = sweep["reporting_split"], sweep["selection_split"]
    allnames = sorted(p.name[:-len(".gt.txt")] for p in (TEST / "gt").glob("*.gt.txt"))
    out = {"n_resamples": n_res, "seed": seed, "candidates": {}, "selection": None,
           "scores": {}, "by_stratum": {}, "comparisons": {}, "failures": {}}
    cache = {}
    # candidates: every prediction directory scored on at least the 98 lines
    for d in sorted(p for p in PREDS.iterdir() if p.is_dir() and not p.name.endswith("_cpu")):
        have = {p.stem for p in d.glob("*.txt")}
        if not set(sel) <= have:
            continue
        rows = from_dir(d, sorted(have))
        out["candidates"][d.name] = {"sel98": pct(rows, sel),
                                     "complete": set(allnames) <= have}
        print(f"candidate {d.name:24s} sel98 {out['candidates'][d.name]['sel98']}")
    best = min(out["candidates"], key=lambda k: tuple(out["candidates"][k]["sel98"]))
    out["selection"] = best
    print("selected on sel98:", best)
    full = [k for k, v in out["candidates"].items() if v["complete"]]
    for k in full:
        cache["vlm:" + k] = from_dir(PREDS / k, allnames)
        out["failures"][k] = failures(PREDS / k, allnames)
    cache["paddle:zeroshot"] = from_dir(PADDLE / "zeroshot", allnames)
    for m in ("stock", "tamv3_dawg_corpus", "tamv3_i1400"):
        rows = per_line(m)
        cache[m] = {n: rows[n] for n in allnames}
    for m, rows in cache.items():
        out["scores"][m] = {v: pct(rows, ns) for v, ns in
                            (("all300", allnames), ("sel98", sel), ("rep202", rep))}
        out["by_stratum"][m] = {s: pct(rows, [n for n in rep if n.startswith(s + "_")])
                                for s in sorted({n.split("_")[0] for n in rep})}
        print(f"{m:28s} {out['scores'][m]}")
    v = "vlm:" + best
    for label, a, b in (("vlm_vs_stock", v, "stock"),
                        ("release_vs_vlm", "tamv3_dawg_corpus", v),
                        ("nodict_vs_vlm", "tamv3_i1400", v),
                        ("paddle_zs_vs_vlm", "paddle:zeroshot", v)):
        out["comparisons"][label] = compare(cache, rep, label, a, b, n_res, seed)
    # cost: GPU runs of every candidate, CPU-only runs where present
    cost = {}
    for d in sorted(p for p in PREDS.iterdir() if p.is_dir()):
        m = json.loads((d / "_meta.json").read_text())
        cost[d.name] = m["summary"] | {"model": m["model"], "cpu": m["cpu"],
                                       "ollama_ps": m.get("ollama_ps", "")}
    COST.write_text(json.dumps(cost, indent=2))
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
