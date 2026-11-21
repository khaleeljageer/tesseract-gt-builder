#!/usr/bin/env python3
"""X3 (IJDAR revision): corpus and workflow effects at matched weight updates.

R3: "The corpus-versus-workflow comparison is confounded by unequal training
updates." tesstrain's 100,000-iteration cap gave the fixed-width defaults run
(tam_new) 19,171 weight updates and the paragraph-aware one (tam_unset)
38,673. No new training is needed: every swept checkpoint's update count is
known (tam_unset* and tam_v3_s*: the second number in the checkpoint name;
tam_new and tam_v3: the training logs, "At iteration L/T/S" with L = updates).

All CERs are grapheme CER in percent, no DAWG, scored as in the sweeps. Two
rules choose a checkpoint within a budget of B updates:
  sel98   - lowest CER on the 98-line selection split (the paper's rule);
  last    - the last checkpoint at or below B (what a run stopped at B gives).
Reported numbers are always on the 202-line reporting split.

tam_new was swept at only 14 checkpoints against 64-68 for the tam_unset
runs, which would hand tam_unset more candidates under sel98. The "grid"
variant therefore restricts every tam_unset run to the checkpoint nearest
each tam_new sweep point (by updates, at or below it).

    python3 experiments/matched_budget.py   ->  results/revision/matched_budget.json
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
REV = RES / "revision"


def log_updates(path):
    m = {}
    for line in open(path):
        r = re.match(r"At iteration (\d+)/(\d+)/(\d+)", line)
        if r:
            m[int(r[2])] = int(r[1])
    return m


def load_named(name):
    """Sweep whose checkpoint names carry the update count."""
    c = json.load(open(REV / f"sweep_{name}.json"))["checkpoints"]
    return sorted(
        (int(v["checkpoint"].split("_")[-2]), v["cer_sel98"], v["cer_rep202"],
         v["wer_rep202"], int(k))
        for k, v in c.items())


def load_old(key, log):
    """checkpoint_sweep.json entries (fractions), updates from the log."""
    cs = json.load(open(RES / "checkpoint_sweep.json"))[key]
    m = log_updates(RES / "training" / log)
    return sorted(
        (m[int(k)], 100 * v["cer_selection_split"], 100 * v["cer_reporting_split"],
         100 * v["wer_reporting_split"], int(k))
        for k, v in cs.items())


RUNS = {
    "fixed_defaults": load_old("v2_checkpoints", "tam_new_training.log"),
    "para_defaults_s0": load_named("tam_unset_orig"),
    "para_defaults_s1": load_named("tam_unset_s1"),
    "para_defaults_s2": load_named("tam_unset_s2"),
    "para_corrected_s0": load_old("checkpoints", "tam_v3_training.log"),
    "para_corrected_s1": load_named("tam_v3_s1"),
    "para_corrected_s2": load_named("tam_v3_s2"),
}
GRID = [p[0] for p in RUNS["fixed_defaults"]]


def pick(points, budget, rule):
    cand = [p for p in points if p[0] <= budget]
    if not cand:
        return None
    p = min(cand, key=lambda p: p[1]) if rule == "sel98" else cand[-1]
    return {"updates": p[0], "iteration": p[4], "cer_sel98": round(p[1], 2),
            "cer_rep202": round(p[2], 2), "wer_rep202": round(p[3], 2)}


def on_grid(points):
    out = []
    for g in GRID:
        below = [p for p in points if p[0] <= g]
        if below and below[-1] not in out:
            out.append(below[-1])
    return out


def spread(rows):
    v = [r["cer_rep202"] for r in rows if r]
    return {"values": v, "mean": round(sum(v) / len(v), 2), "min": min(v), "max": max(v)}


out = {"note": __doc__.strip().splitlines()[0],
       "budget_fixed_defaults_total": 19171, "grid_updates": GRID,
       "runs_n_checkpoints": {k: len(v) for k, v in RUNS.items()}}

# A. Corpus effect, defaults workflow, at the fixed-width run's whole budget.
B = max(GRID)  # 17,107: last tam_new checkpoint swept, inside its 19,171
A = {"budget": B}
for rule in ("sel98", "last"):
    A[rule] = {"fixed_defaults": pick(RUNS["fixed_defaults"], B, rule)}
    for s in ("s0", "s1", "s2"):
        A[rule][f"para_defaults_{s}"] = pick(RUNS[f"para_defaults_{s}"], B, rule)
        A[rule][f"para_defaults_{s}_grid"] = pick(on_grid(RUNS[f"para_defaults_{s}"]), B, rule)
    paras = [A[rule][f"para_defaults_{s}"] for s in ("s0", "s1", "s2")]
    A[rule]["para_defaults_seeds"] = spread(paras)
    A[rule]["corpus_effect_mean"] = round(
        A[rule]["fixed_defaults"]["cer_rep202"] - A[rule]["para_defaults_seeds"]["mean"], 2)
out["corpus_effect_at_matched_budget"] = A

# B. Both curves at each of tam_new's sweep points (last checkpoint <= g).
curve = []
for g in GRID:
    row = {"updates": g, "fixed_defaults": pick(RUNS["fixed_defaults"], g, "last")["cer_rep202"]}
    p = [pick(RUNS[f"para_defaults_{s}"], g, "last") for s in ("s0", "s1", "s2")]
    row["para_defaults"] = [x["cer_rep202"] for x in p]
    row["para_defaults_updates"] = [x["updates"] for x in p]
    q = [pick(RUNS[f"para_corrected_{s}"], g, "last") for s in ("s0", "s1", "s2")]
    row["para_corrected"] = [x["cer_rep202"] if x and x["updates"] >= 0.8 * g else None for x in q]
    curve.append(row)
out["curve_at_fixed_defaults_points"] = curve

# C. Workflow effect at matched budget: corrected vs defaults, same corpus.
C = []
for b in (1059, 3000, 5000, 10000, 17107):
    row = {"budget": b}
    for rule in ("sel98", "last"):
        cor = [pick(RUNS[f"para_corrected_{s}"], b, rule) for s in ("s0", "s1", "s2")]
        dfl = [pick(RUNS[f"para_defaults_{s}"], b, rule) for s in ("s0", "s1", "s2")]
        row[rule] = {"corrected": spread(cor), "defaults": spread(dfl),
                     "corrected_updates": [x["updates"] for x in cor],
                     "workflow_effect_mean": round(spread(dfl)["mean"] - spread(cor)["mean"], 2)}
    C.append(row)
out["workflow_effect_at_matched_budget"] = C

dst = REV / "matched_budget.json"
dst.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(json.dumps(out, indent=1)[:200000])
