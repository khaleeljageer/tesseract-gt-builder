#!/usr/bin/env python3
"""How much of the headline gain is punctuation (IJDAR revision 3).

Rescores stock, the no-dictionary checkpoint and the released model on the
reporting split under three views, all with the paper's scorer
(tamil_ocr_eval.score_pair) applied after the view's transformation to both
reference and hypothesis:

  as_reported      the paper's scoring;
  no_punct_tokens  whitespace tokens with no letter, mark or digit dropped
                   (stray marks such as a lone "-" or "|");
  text_only        additionally every character that is not a letter, mark,
                   digit or ZWNJ/ZWJ deleted, so only the Tamil (and Latin)
                   text and digits are scored.

Paired bootstrap as in paired_bootstrap.py: reporting split, by line,
10,000 resamples, seed string "<seed>:<label>:line". Also counts lines whose
output ends with punctuation the reference line does not end with, and lines
that start with stray marks. Predictions: results/revision/x10_preds/v550
(Tesseract 5.5.0; they reproduce every stored prediction).

    python3 experiments/punct_decomposition.py [--n 10000] [--seed 0]
"""
import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import tamil_ocr_eval as ev  # noqa: E402

TEST = HERE / "testset"
PREDS = HERE / "results" / "revision" / "x10_preds" / "v550"
OUT = HERE / "results" / "revision" / "punct_decomposition.json"
SYSTEMS = {"stock": "tam", "nodict": "tamv3i1400", "released": "tamv3dcorpus"}
METRICS = {"cer": ("grapheme_edits", "grapheme_ref"), "wer": ("word_edits", "word_ref")}


def is_text(c):
    return unicodedata.category(c)[0] in "LMN" or c in "‌‍"


VIEWS = {
    "as_reported": lambda t: t,
    "no_punct_tokens": lambda t: " ".join(w for w in t.split() if any(map(is_text, w))),
    "text_only": lambda t: " ".join("".join(c for c in w if is_text(c))
                                    for w in t.split() if any(map(is_text, w))),
}


def rate(rows, e, r):
    return sum(x[e] for x in rows) / sum(x[r] for x in rows)


def edge_marks(text):
    t = ev.normalize(text).strip()
    end = t[-1:] and not is_text(t[-1]) and t[-1] not in ")\"'”"
    start = t[:1] and not is_text(t[0]) and t[0] not in "(\"'“"
    return bool(end), bool(start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names = json.loads((HERE / "results" / "checkpoint_sweep.json").read_text())["reporting_split"]
    ref = {n: ev.read(TEST / "gt" / f"{n}.gt.txt") for n in names}
    hyp = {s: {n: ev.read(PREDS / d / f"{n}.txt") for n in names} for s, d in SYSTEMS.items()}
    out = {"n_resamples": a.n, "seed": a.seed, "view_split": "rep202", "unit": "line",
           "note": "diff = A - B in percentage points; a_better = resamples with A strictly lower",
           "rates": {}, "comparisons": {}, "edges": {}}
    for view, f in VIEWS.items():
        sc = {s: {n: ev.score_pair(f(ref[n]), f(hyp[s][n])) for n in names} for s in SYSTEMS}
        out["rates"][view] = {s: {m: round(100 * rate(list(sc[s].values()), e, r), 2)
                                  for m, (e, r) in METRICS.items()} for s in SYSTEMS}
        for A in ("released", "nodict"):
            label = f"{A}_vs_stock_{view}"
            # the as-reported view reuses paired_bootstrap.py's labels, so its
            # counts are the paper's (Table 3, Table 2)
            seed_label = {"released": "release_vs_stock", "nodict": "nodict_vs_stock"}[A] \
                if view == "as_reported" else label
            rng = random.Random(f"{a.seed}:{seed_label}:line")
            draws = [[rng.choice(names) for _ in names] for _ in range(a.n)]
            row = {}
            for m, (e, r) in METRICS.items():
                wins, diffs = 0, []
                for s in draws:
                    da = rate([sc[A][x] for x in s], e, r)
                    db = rate([sc["stock"][x] for x in s], e, r)
                    wins += da < db
                    diffs.append(100 * (da - db))
                diffs.sort()
                full = 100 * (rate(list(sc[A].values()), e, r) - rate(list(sc["stock"].values()), e, r))
                row[m] = {"diff": round(full, 2),
                          "a_better": wins,
                          "ci95": [round(diffs[int(0.025 * a.n)], 2), round(diffs[int(0.975 * a.n) - 1], 2)]}
            out["comparisons"][label] = row
            print(f"{label:30s} CER {row['cer']['diff']:+.2f} ({row['cer']['a_better']})"
                  f"  WER {row['wer']['diff']:+.2f} ({row['wer']['a_better']})")
    for s in ("ref",) + tuple(SYSTEMS):
        texts = ref if s == "ref" else hyp[s]
        marks = [edge_marks(texts[n]) for n in names]
        out["edges"][s] = {"ends_with_punct": sum(e for e, _ in marks),
                           "starts_with_mark": sum(b for _, b in marks)}
    # lines ending in punctuation the reference does not end with
    for s in SYSTEMS:
        out["edges"][s]["ends_with_punct_ref_does_not"] = sum(
            edge_marks(hyp[s][n])[0] and not edge_marks(ref[n])[0] for n in names)
        out["edges"][s]["starts_with_mark_ref_does_not"] = sum(
            edge_marks(hyp[s][n])[1] and not edge_marks(ref[n])[1] for n in names)
    print(json.dumps(out["rates"], indent=1))
    print(json.dumps(out["edges"], indent=1))
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
