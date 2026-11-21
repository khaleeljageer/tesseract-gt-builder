#!/usr/bin/env python3
"""T7 (IJDAR revision, R2.7): the corpus against real Tamil, distributionally.

R2: "coverage is measured against the 247-unit syllabary, not against unit
frequencies in the test lines". This compares the released corpus
(data/corpus.txt, 218,125 lines) with the 300 hand-transcribed test lines,
both under the evaluation's own normalisation and grapheme segmentation:

  * Jensen-Shannon divergence (base 2, in [0, 1]) between grapheme
    distributions, overall and per capture stratum, with two yardsticks: the
    sampling floor (corpus against random corpus samples of the test set's
    size) and the spread between the corpus's own sources is not available
    here, so we use the spread between test strata instead;
  * where test graphemes fall in the corpus's exemplar tiers (never seen,
    1-28, 29-289, 290+ occurrences);
  * character classes (Tamil, Latin, digits, punctuation, space);
  * word types and tokens absent from the corpus (OOV).

    python3 experiments/distribution.py  ->  results/revision/distribution.json
"""
import csv
import json
import math
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from tamil_ocr_eval import graphemes, normalize  # noqa: E402
from corpus_stats import syllabary  # noqa: E402

SEED = 20260824


def jsd(p, q):
    keys = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    d = 0.0
    for k in keys:
        a, b = p.get(k, 0) / sp, q.get(k, 0) / sq
        m = (a + b) / 2
        if a:
            d += 0.5 * a * math.log2(a / m)
        if b:
            d += 0.5 * b * math.log2(b / m)
    return d


def cls(g):
    c = g[0]
    if c == " ":
        return "space"
    if "஀" <= c <= "௿":
        return "tamil_digit" if unicodedata.category(c) == "Nd" else "tamil"
    if c.isdigit():
        return "digit"
    if c.isascii() and c.isalpha():
        return "latin"
    if unicodedata.category(c).startswith("P") or unicodedata.category(c).startswith("S"):
        return "punct_symbol"
    return "other"


def gcount(lines, keep_space=False):
    c = Counter()
    for l in lines:
        for g in graphemes(normalize(l)):
            if keep_space or g != " ":
                c[g] += 1
    return c


corpus_lines = (ROOT / "data" / "corpus.txt").read_text(encoding="utf-8").splitlines()
man = {r["line_id"]: r["stratum"] for r in csv.DictReader(open(ROOT / "testset" / "manifest.csv"))}
test = {lid: (ROOT / "testset" / "gt" / f"{lid}.gt.txt").read_text(encoding="utf-8").strip()
        for lid in man}
strata = sorted(set(man.values()))

C = gcount(corpus_lines)
T = gcount(test.values())
n_test = sum(T.values())
out = {"corpus_lines": len(corpus_lines), "test_lines": len(test),
       "corpus_graphemes": sum(C.values()), "test_graphemes": n_test,
       "distinct": {"corpus": len(C), "test": len(T),
                    "test_not_in_corpus": sorted(g for g in T if g not in C)}}

# JSD overall, per stratum, and the sampling floor at the test set's size.
rng = random.Random(SEED)
pool = [g for g, n in C.items() for _ in range(n)]
floor = []
for _ in range(200):
    floor.append(jsd(C, Counter(rng.choices(pool, k=n_test))))
floor.sort()
out["jsd"] = {"corpus_vs_test": round(jsd(C, T), 4),
              "sampling_floor_mean": round(sum(floor) / len(floor), 4),
              "sampling_floor_p95": round(floor[int(0.95 * len(floor))], 4),
              "per_stratum": {}}
S = {s: gcount([test[l] for l in test if man[l] == s]) for s in strata}
for s in strata:
    ns = sum(S[s].values())
    fl = sorted(jsd(C, Counter(rng.choices(pool, k=ns))) for _ in range(200))
    rest = Counter()
    for t in strata:
        if t != s:
            rest.update(S[t])
    out["jsd"]["per_stratum"][s] = {
        "graphemes": ns, "vs_corpus": round(jsd(C, S[s]), 4),
        "floor_mean": round(sum(fl) / len(fl), 4),
        "vs_other_strata": round(jsd(rest, S[s]), 4)}

# Tamil-only JSD: removes the effect of punctuation, digits and Latin.
Ct = Counter({g: n for g, n in C.items() if cls(g) == "tamil"})
Tt = Counter({g: n for g, n in T.items() if cls(g) == "tamil"})
out["jsd"]["tamil_only"] = round(jsd(Ct, Tt), 4)
pool_t = [g for g, n in Ct.items() for _ in range(n)]
flt = sorted(jsd(Ct, Counter(rng.choices(pool_t, k=sum(Tt.values())))) for _ in range(200))
out["jsd"]["tamil_only_floor_mean"] = round(sum(flt) / len(flt), 4)

# Largest per-grapheme contributions (where the two differ).
sc, st = sum(C.values()), n_test
diff = sorted(((T.get(g, 0) / st - C.get(g, 0) / sc, g) for g in set(C) | set(T)))
out["over_in_test"] = [[g, round(100 * T.get(g, 0) / st, 3), round(100 * C.get(g, 0) / sc, 3)]
                       for d, g in diff[::-1][:12]]
out["under_in_test"] = [[g, round(100 * T.get(g, 0) / st, 3), round(100 * C.get(g, 0) / sc, 3)]
                        for d, g in diff[:12]]

# Character classes.
def classes(cnt):
    k = Counter()
    for g, n in cnt.items():
        k[cls(g)] += n
    s = sum(k.values())
    return {c: round(100 * n / s, 2) for c, n in sorted(k.items())}

out["classes_pct"] = {"corpus": classes(gcount(corpus_lines, True)),
                      "test": classes(gcount(test.values(), True))}

# Test graphemes by the corpus's exemplar tiers.
syl = [u for v in syllabary().values() for u in v]
tiers = [("never", 0, 1), ("1-28", 1, 29), ("29-289", 29, 290), ("290+", 290, None)]
tt = {}
for name, lo, hi in tiers:
    units = [u for u in syl if C.get(u, 0) >= lo and (hi is None or C.get(u, 0) < hi)]
    occ = sum(T.get(u, 0) for u in units)
    tt[name] = {"units": len(units), "units_in_test": sum(1 for u in units if T.get(u)),
                "test_occurrences": occ,
                "test_units": sorted(u for u in units if T.get(u)) if name != "290+" else None}
non_syl = sum(n for g, n in T.items() if g not in set(syl))
out["test_by_corpus_tier"] = tt
out["test_graphemes_outside_247"] = non_syl

# Words.
def words(lines):
    w = Counter()
    for l in lines:
        w.update(normalize(l).split())
    return w

CW, TW = words(corpus_lines), words(test.values())
oov_types = [w for w in TW if w not in CW]
out["words"] = {"test_types": len(TW), "test_tokens": sum(TW.values()),
                "oov_types": len(oov_types),
                "oov_tokens": sum(TW[w] for w in oov_types),
                "oov_token_pct": round(100 * sum(TW[w] for w in oov_types) / sum(TW.values()), 2),
                "per_stratum_oov_token_pct": {}}
for s in strata:
    sw = words([test[l] for l in test if man[l] == s])
    o = sum(n for w, n in sw.items() if w not in CW)
    out["words"]["per_stratum_oov_token_pct"][s] = round(100 * o / sum(sw.values()), 2)

dst = ROOT / "results" / "revision" / "distribution.json"
dst.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
