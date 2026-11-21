#!/usr/bin/env python3
"""T8 (IJDAR revision, R3): the headline comparisons on page-disjoint views.

R3: "The real-document test set is relatively small and not fully
page-disjoint." Two things can share a page: two test lines (so lines are
not independent), and a selection line with a reporting line (so the
checkpoint chosen on sel98 has seen the page of a line it is scored on).
Views, all subsets of the 202-line reporting split:

  rep_disjoint  - reporting lines whose page contributes no selection line;
  one_per_page  - of those, one line per page (seeded draw), so every line
                  is from a different page and none from a selection page.

Same paired bootstrap as paired_bootstrap.py (10,000 line resamples; in
one_per_page a line resample is a page resample). The rep202 rows reuse its
seed strings and reproduce its counts exactly; SEED only draws the line kept
per page.

    python3 experiments/page_disjoint.py -> results/revision/page_disjoint.json
"""
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paired_bootstrap import METRICS, per_line, rate  # noqa: E402

N, SEED = 10000, 20260824
PAIRS = [("release_vs_stock", "tamv3_dawg_corpus", "stock"),
         ("nodict_vs_stock", "tamv3_i1400", "stock"),
         ("workflow_effect", "tamv3_dawg_corpus", "tam_unset"),
         ("corpus_effect", "tam_unset", "tamsyn"),
         ("defaults_vs_stock", "stock", "tam_unset")]

sweep = json.loads((ROOT / "results" / "checkpoint_sweep.json").read_text())
sel, rep = sweep["selection_split"], sweep["reporting_split"]
meta = {r["line_id"]: r for r in csv.DictReader(open(ROOT / "testset" / "manifest.csv"))}
page = {n: meta[n]["source_page"] for n in meta}

sel_pages = {page[n] for n in sel}
rep_pages = {page[n] for n in rep}
all_pages = {page[n] for n in meta}
rep_disjoint = [n for n in rep if page[n] not in sel_pages]
rng = random.Random(SEED)
by_page = {}
for n in sorted(rep_disjoint):
    by_page.setdefault(page[n], []).append(n)
one_per_page = sorted(rng.choice(v) for _, v in sorted(by_page.items()))

out = {"n_resamples": N, "seed": SEED,
       "pages": {"all300": len(all_pages), "sel98": len(sel_pages), "rep202": len(rep_pages),
                 "shared_sel_rep": len(sel_pages & rep_pages),
                 "rep_lines_on_shared_pages": sum(page[n] in sel_pages for n in rep),
                 "multi_line_pages_all300": sum(1 for p in all_pages
                                                if sum(page[n] == p for n in meta) > 1)},
       "views": {"rep202": len(rep), "rep_disjoint": len(rep_disjoint),
                 "one_per_page": len(one_per_page)},
       "one_per_page_lines": one_per_page, "comparisons": {}}

cache = {}
for view, names in (("rep202", rep), ("rep_disjoint", rep_disjoint), ("one_per_page", one_per_page)):
    for label, a, b in PAIRS:
        for m in (a, b):
            cache.setdefault(m, per_line(m))
        # rep202 uses paired_bootstrap.py's own seed string, so its counts are
        # the paper's (Table 3); the subsets get their own streams.
        key = f"0:{label}:line" if view == "rep202" else f"0:{label}:{view}:line"
        r = random.Random(key)
        draws = [[r.choice(names) for _ in names] for _ in range(N)]
        row = {"A": a, "B": b, "lines": len(names)}
        for met, (e, k) in METRICS.items():
            ra = rate([cache[a][n] for n in names], e, k)
            rb = rate([cache[b][n] for n in names], e, k)
            diffs, wins = [], 0
            for s in draws:
                da, db = rate([cache[a][n] for n in s], e, k), rate([cache[b][n] for n in s], e, k)
                diffs.append(100 * (da - db))
                wins += da < db
            diffs.sort()
            row[met] = {"A": round(100 * ra, 2), "B": round(100 * rb, 2),
                        "diff": round(100 * (ra - rb), 2), "a_better": wins,
                        "ci95": [round(diffs[int(0.025 * N)], 2), round(diffs[int(0.975 * N) - 1], 2)]}
        out["comparisons"].setdefault(view, {})[label] = row
        print(f"{view:13s} {label:18s} n={len(names):3d} " + "  ".join(
            f"{m} {row[m]['A']:.2f}/{row[m]['B']:.2f} {row[m]['diff']:+.2f} {row[m]['a_better']:5d} "
            f"[{row[m]['ci95'][0]:+.2f},{row[m]['ci95'][1]:+.2f}]" for m in METRICS), flush=True)

print(json.dumps(out["pages"]), out["views"])
(ROOT / "results" / "revision" / "page_disjoint.json").write_text(json.dumps(out, indent=1) + "\n")
