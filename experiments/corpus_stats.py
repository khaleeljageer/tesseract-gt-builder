"""Corpus statistics for the Corpus Construction section.

The headline question a resource paper has to answer is not "how big is it"
but "does it exercise the script". For Tamil that has a precise form: the
traditional syllabary is 247 units -- 12 vowels (uyir), 18 consonants in
their pure/virama form (mei), 216 consonant-vowel combinations (uyirmei),
and the aytham. A training corpus that never shows the recogniser a given
unit cannot teach it, so per-unit coverage and the frequency of the rarest
units are what matter.

Grantha letters, used for Sanskrit and English loanwords, are counted
separately: they are not part of the 247 but do appear in modern Tamil text
and must be recognised.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import corpus as corpus_mod  # noqa: E402
from tamil_ocr_eval import graphemes  # noqa: E402

UYIR = list("அஆஇஈஉஊஎஏஐஒஓஔ")
MEY_CONS = list("கஙசஞடணதநபமயரலவழளறன")
GRANTHA_CONS = list("ஜஷஸஹ")
AYTHAM = "ஃ"
VIRAMA = "\u0BCD"
# Dependent vowel signs in syllabary order after the inherent 'a'.
VOWEL_SIGNS = ["", "\u0BBE", "\u0BBF", "\u0BC0", "\u0BC1", "\u0BC2",
               "\u0BC6", "\u0BC7", "\u0BC8", "\u0BCA", "\u0BCB", "\u0BCC"]


def syllabary(consonants=None):
    """The 247 units, as grapheme strings."""
    cons = consonants if consonants is not None else MEY_CONS
    units = {"uyir": list(UYIR), "aytham": [AYTHAM],
             "mey": [c + VIRAMA for c in cons],
             "uyirmey": [c + v for c in cons for v in VOWEL_SIGNS]}
    return units


def coverage(counts, units):
    """Which units appear, and how often the rarest do."""
    present = {u: counts.get(u, 0) for u in units}
    seen = {u: n for u, n in present.items() if n > 0}
    return {
        "total": len(units),
        "covered": len(seen),
        "missing": sorted(u for u, n in present.items() if n == 0),
        "rarest": sorted(seen.items(), key=lambda kv: kv[1])[:12],
        "occurrences": sum(seen.values()),
    }


def report(lines, label):
    counts = Counter(g for ln in lines for g in graphemes(ln))
    units = syllabary()
    out = {"label": label, "lines": len(lines),
           "graphemes": sum(counts.values()),
           "distinct_graphemes": len(counts)}

    for name, us in units.items():
        out[name] = coverage(counts, us)

    grantha_units = [c + v for c in GRANTHA_CONS for v in VOWEL_SIGNS] \
        + [c + VIRAMA for c in GRANTHA_CONS]
    out["grantha"] = coverage(counts, grantha_units)

    all_247 = [u for us in units.values() for u in us]
    out["syllabary_247"] = coverage(counts, all_247)
    out["top_graphemes"] = counts.most_common(15)
    out["_counts"] = counts
    return out


def figure(counts, out_path, top=40):
    """Rank-frequency plot of the grapheme distribution."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        print("[skip] matplotlib unavailable")
        return

    ranked = counts.most_common()
    y = [n for _, n in ranked]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax1.plot(range(1, len(y) + 1), y, color="#0072B2", linewidth=1.4)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Grapheme rank")
    ax1.set_ylabel("Occurrences")
    ax1.grid(alpha=0.25, linewidth=0.5)
    ax1.spines[["top", "right"]].set_visible(False)

    # Tamil glyphs on the tick labels need a Tamil-capable face.
    tfont = None
    for cand in ("fonts/NotoSerifTamil.ttf", "fonts/AnekTamil.ttf",
                 "fonts/Catamaran.ttf"):
        if Path(cand).exists():
            tfont = font_manager.FontProperties(fname=cand, size=7)
            break

    head = ranked[:top]
    ax2.bar(range(len(head)), [n for _, n in head], color="#0072B2", width=0.8)
    ax2.set_xticks(range(len(head)))
    ax2.set_xticklabels([g for g, _ in head], fontproperties=tfont, rotation=0)
    ax2.set_xlabel(f"{top} most frequent graphemes")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.25, linewidth=0.5, axis="y")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.margins(x=0.01)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def exemplar_tiers(counts, n_fonts):
    """How many syllabary units are too rare to be learned properly.

    Under round-robin font assignment a unit occurring k times appears in at
    most min(k, n_fonts) typefaces. A unit seen fewer than n_fonts times
    therefore cannot be shown to the recogniser in every font it will meet at
    inference, however large the corpus is overall.
    """
    units = [u for us in syllabary().values() for u in us]
    present = [(u, counts.get(u, 0)) for u in units]
    # Disjoint bands, so the counts add up to 247 and a reader can subtract.
    # Cumulative bands read as though the tail were three times larger than
    # it is: the 20 units that never occur also satisfy "fewer than 27".
    bands = ((0, 1, "never occur"),
             (1, n_fonts, f"1--{n_fonts - 1} (cannot appear in every font)"),
             (n_fonts, n_fonts * 10,
              f"{n_fonts}--{n_fonts * 10 - 1} ($<10$ exemplars per font)"),
             (n_fonts * 10, None, f"{n_fonts * 10} or more"))
    tiers = []
    for lo, hi, label in bands:
        n = sum(1 for _, c in present if lo <= c and (hi is None or c < hi))
        tiers.append({"low": lo, "high": hi, "label": label, "units": n})
    rare = sorted([(u, c) for u, c in present if 0 < c < n_fonts],
                  key=lambda x: x[1])
    total = sum(c for _, c in present)
    return {"tiers": tiers, "rare_units": rare,
            "rare_share": sum(c for _, c in rare) / total if total else 0.0,
            "n_fonts": n_fonts}


# Source stems carry underscores, which LaTeX reads as subscripts outside
# math mode. Give each a display name rather than escaping, since the reader
# wants the publication's name and not our filename.
DISPLAY = {
    "wikisource-ta": "Tamil Wikisource",
    "stories": "\\texttt{tamil\\_stories}",
    "theekkathir_content_tamil_only": "Theekkathir",
    "maattru": "Maattru",
    "wikinews-ta": "Tamil Wikinews",
}


def latex_coverage(tier_info, out_path):
    body = "\n".join(
        f"{t['label']} & {t['units']} \\\\" for t in tier_info["tiers"])
    rare = "  ".join(f"\\ta{{{u}}}~({c})" for u, c in tier_info["rare_units"])
    tex = f"""% generated by experiments/corpus_stats.py -- do not edit
\\begin{{table}}[t]
\\centering\\small
\\begin{{tabular}}{{lr}}
\\toprule
Occurrences in the corpus & Units \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Training exposure of the 247 syllabary units. Under round-robin
assignment a unit occurring $k$ times is rendered in at most
$\\min(k, {tier_info['n_fonts']})$ typefaces, so units below that threshold
cannot be shown in every font regardless of corpus size. The
{len(tier_info['rare_units'])} units in the middle tier account for only
{tier_info['rare_share'] * 100:.4f}\\% of syllabary occurrences:
{rare}.}}
\\label{{tab:coverage}}
\\end{{table}}
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(tex, encoding="utf-8")
    print(f"wrote {out_path}")


def latex(full, per_source, out_path):
    rows = []
    for r in per_source:
        s = r["syllabary_247"]
        rows.append(f"{DISPLAY.get(r['label'], r['label'])} & "
                    f"{r['lines']:,} & {r['graphemes']:,} & "
                    f"{s['covered']}/{s['total']} \\\\")
    body = "\n".join(rows)
    s = full["syllabary_247"]
    g = full["grantha"]
    tex = f"""% generated by experiments/corpus_stats.py -- do not edit
\\begin{{table}}[t]
\\centering\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{lrrc}}
\\toprule
Source & Lines & Graphemes & Syllabary \\\\
\\midrule
{body}
\\midrule
\\textbf{{Released corpus}} & \\textbf{{{full['lines']:,}}} &
\\textbf{{{full['graphemes']:,}}} & \\textbf{{{s['covered']}/{s['total']}}} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Corpus composition and syllabary coverage. Per-source counts are
lines in the text pool before deduplication, so they sum to more than the
released total. The final column counts
how many of the 247 traditional Tamil syllabary units (12 \\emph{{uyir}},
18 \\emph{{mey}}, 216 \\emph{{uyirmey}}, and the aytham) occur at least once.
The full corpus additionally covers {g['covered']} of {g['total']} Grantha
forms used for loanwords.}}
\\label{{tab:corpus-stats}}
\\end{{table}}
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(tex, encoding="utf-8")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="raw_data")
    ap.add_argument("--font-dir", default="fonts")
    ap.add_argument("--json", default="results/corpus_stats.json")
    ap.add_argument("--latex", default="../paper/tables/corpus_stats.tex")
    ap.add_argument("--figure", default="../paper/figures/grapheme_freq.pdf")
    ap.add_argument("--repair", action="store_true",
                    help="apply repair_tamil and drop lines that stay "
                         "ill-formed. OFF by default: the released v3 corpus "
                         "was built before repair_tamil existed, so leaving "
                         "this on described a corpus that was never released.")
    args = ap.parse_args()

    pool = corpus_mod.build_pool(args.raw_dir, repair=args.repair)
    per_source = [report(v, k) for k, v in
                  sorted(pool.items(), key=lambda kv: -len(kv[1]))]

    # The headline statistics must describe the corpus that is actually
    # released, not the raw pool: deduplication, the Tamil-content filter and
    # the typeface-coverage filter all remove lines, and coverage claims made
    # against the pool would overstate what a model can be trained on.
    released = corpus_mod.select(pool, n_lines=None, seed=0, dedup=True,
                                 require_well_formed=args.repair)
    try:
        import render as render_mod
        coverage = render_mod.common_coverage(args.font_dir)
        released, _dropped = render_mod.renderable(released, coverage)
    except Exception as exc:
        print(f"[warn] typeface-coverage filter skipped ({exc}); "
              f"statistics describe the pre-coverage set")
    full = report(released, "released")

    print(f"{'source':<34} {'lines':>9} {'graphemes':>12} {'247':>8} {'grantha':>8}")
    print("-" * 76)
    for r in per_source:
        print(f"{r['label']:<34} {r['lines']:>9,} {r['graphemes']:>12,} "
              f"{r['syllabary_247']['covered']:>4}/247 "
              f"{r['grantha']['covered']:>4}/{r['grantha']['total']}")
    print("-" * 76)
    print(f"{'RELEASED':<34} {full['lines']:>9,} {full['graphemes']:>12,} "
          f"{full['syllabary_247']['covered']:>4}/247 "
          f"{full['grantha']['covered']:>4}/{full['grantha']['total']}")

    print(f"\nDistinct graphemes: {full['distinct_graphemes']:,}")
    for part in ("uyir", "mey", "uyirmey", "aytham"):
        c = full[part]
        print(f"  {part:<9} {c['covered']:>3}/{c['total']:<3} covered", end="")
        if c["missing"]:
            print(f"   missing: {' '.join(c['missing'][:10])}")
        else:
            print()

    print("\nRarest syllabary units in the full corpus:")
    for u, n in full["syllabary_247"]["rarest"]:
        print(f"    {u}   {n:>8,}")

    raw_lines = [l for v in pool.values() for l in v]
    dedup_removed = len(raw_lines) - len(corpus_mod.deduplicate(raw_lines))
    print(f"\nRaw pool: {len(raw_lines):,} lines; exact duplicates removed: "
          f"{dedup_removed:,} ({dedup_removed / len(raw_lines) * 100:.2f}%)")
    print(f"Released after all filters: {full['lines']:,} lines")

    counts = full.pop("_counts")
    for r in per_source:
        r.pop("_counts", None)
    payload = {"full": full, "per_source": per_source,
               "duplicate_lines": dedup_removed}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    print(f"\nwrote {args.json}")

    # Derive the typeface count rather than hardcoding it: the tier
    # boundaries are "can this unit appear in every font", so a stale count
    # silently mislabels every tier.
    n_fonts = len([f for f in Path(args.font_dir).iterdir()
                   if f.suffix.lower() in (".ttf", ".otf")])
    tiers = exemplar_tiers(counts, n_fonts)
    payload["exemplar_tiers"] = {k: v for k, v in tiers.items() if k != "rare_units"}
    payload["exemplar_tiers"]["rare_units"] = tiers["rare_units"]
    Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                               encoding="utf-8")

    latex(full, per_source, args.latex)
    latex_coverage(tiers, str(Path(args.latex).with_name("coverage.tex")))
    figure(counts, args.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
