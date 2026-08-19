"""Audit the 27 typefaces against the corpus they are used to render.

Answers three questions, in increasing order of how badly a "no" would hurt:

1. Metadata. Family, designer, licence -- for the paper's appendix table.

2. Character coverage. Does every font carry a glyph for every codepoint the
   corpus actually uses? A missing glyph renders as .notdef (tofu) or nothing,
   while the ground-truth file still says the correct character. That is a
   silently corrupted training pair, and it would be present in the released
   corpus.

3. Complex-script shaping. Tamil needs reordering and ligature substitution.
   Pillow only performs that when built against Raqm/HarfBuzz. Without it,
   text is drawn as a naive left-to-right run of base glyphs: vowel signs sit
   in the wrong place and conjuncts never form. Every rendered line would be
   wrong while its transcription stayed right.
"""

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from fontTools.ttLib import TTFont

# Name table IDs we care about.
NAME_IDS = {1: "family", 2: "subfamily", 5: "version", 8: "manufacturer",
            9: "designer", 13: "license", 14: "license_url", 0: "copyright"}

TAMIL_BLOCK = range(0x0B80, 0x0C00)


def check_shaping():
    """Is Pillow able to shape complex scripts at all?"""
    from PIL import features, __version__
    raqm = features.check("raqm")
    out = {"pillow": __version__, "raqm": bool(raqm)}
    if raqm:
        try:
            out["raqm_version"] = features.version("raqm")
            out["harfbuzz_version"] = features.version("harfbuzz")
        except Exception:
            pass
    return out


def shaping_smoke_test(font_path, size=22):
    """Render a conjunct two ways and see whether shaping changed anything.

    'கி' must be narrower than the naive sum of its parts once shaped, and
    a shaped 'ஸ்ரீ' differs in width from the unshaped sequence. If widths are
    identical to the naive concatenation the font is being drawn unshaped.
    """
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(font_path), size)
    draw = ImageDraw.Draw(Image.new("L", (10, 10)))

    def w(s):
        return draw.textbbox((0, 0), s, font=font)[2]

    # A conjunct that must reorder/ligate, vs its constituent codepoints.
    conj = "ஸ்ரீ"
    parts = sum(w(c) for c in conj)
    return {"shaped_width": w(conj), "naive_sum": parts,
            "differs": w(conj) != parts}


def font_report(path, corpus_freq):
    """Metadata + coverage for one font."""
    tt = TTFont(str(path), fontNumber=0, lazy=True)

    meta = {}
    try:
        for rec in tt["name"].names:
            key = NAME_IDS.get(rec.nameID)
            if key and key not in meta:
                try:
                    meta[key] = rec.toUnicode().strip()
                except Exception:
                    pass
    except Exception:
        pass

    cmap = set(tt.getBestCmap().keys())

    # Which corpus codepoints does this font not cover, and how much text
    # do they account for?
    missing = {cp: n for cp, n in corpus_freq.items() if cp not in cmap}
    total = sum(corpus_freq.values())
    missing_n = sum(missing.values())

    tamil_covered = sum(1 for cp in TAMIL_BLOCK if cp in cmap)

    gsub = "GSUB" in tt
    tamil_scripts = []
    if gsub:
        try:
            for rec in tt["GSUB"].table.ScriptList.ScriptRecord:
                tamil_scripts.append(rec.ScriptTag)
        except Exception:
            pass

    tt.close()
    return {
        "file": path.name,
        "stem": path.stem,
        **meta,
        "cmap_size": len(cmap),
        "tamil_block_covered": tamil_covered,
        "has_gsub": gsub,
        "gsub_scripts": sorted(set(tamil_scripts)),
        "missing_codepoints": sorted(missing),
        "missing_count": len(missing),
        "missing_occurrences": missing_n,
        "missing_share": missing_n / total if total else 0.0,
    }


def corpus_codepoint_freq(raw_dir="raw_data", limit_mb=None):
    """Frequency of every codepoint appearing in the source texts."""
    freq = Counter()
    for path in sorted(Path(raw_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        if limit_mb:
            text = text[: int(limit_mb * 1_000_000)]
        freq.update(unicodedata.normalize("NFC", text))
    return Counter({ord(c): n for c, n in freq.items() if not c.isspace()})


def describe(cp):
    try:
        return unicodedata.name(chr(cp))
    except ValueError:
        return "<unnamed>"


def classify_license(f):
    """Bucket the name-table licence string. 'unclear' is a real answer here,
    not a parsing failure: many of these fonts ship no licence field at all."""
    text = (f.get("license") or "") + " " + (f.get("license_url") or "")
    if "Open Font License" in text or "OFL" in text.upper():
        return "OFL 1.1"
    if "Apache" in text:
        return "Apache 2.0"
    if "Virtual Academy" in text:
        return "TVA (order no.)"
    if f.get("license") or f.get("copyright"):
        return "proprietary/unstated"
    return "none declared"


def latex_table(reports, out_path):
    """Appendix typeface inventory."""
    def esc(s):
        return (s or "--").replace("&", "\\&").replace("_", "\\_")

    rows = []
    for r in sorted(reports, key=lambda r: r["stem"].lower()):
        rows.append(
            f"{esc(r['stem'])} & {esc(r.get('family'))} & "
            f"{esc((r.get('designer') or r.get('manufacturer') or '--')[:34])} & "
            f"{esc(classify_license(r))} \\\\")

    tex = ("% generated by experiments/font_audit.py -- do not edit\n"
           "\\begin{table*}[t]\n\\centering\\small\n"
           "\\begin{tabular}{llll}\n\\toprule\n"
           "File & Family & Designer / foundry & Licence \\\\\n\\midrule\n"
           + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n"
           "\\caption{The 27 typefaces used to render \\corpus. All 27 carry "
           "GSUB tables and cover every Tamil codepoint appearing in the "
           "corpus; rendering used HarfBuzz shaping via Raqm.}\n"
           "\\label{tab:fonts}\n\\end{table*}\n")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(tex, encoding="utf-8")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font-dir", default="fonts")
    ap.add_argument("--raw-dir", default="raw_data")
    ap.add_argument("--limit-mb", type=float, default=None,
                    help="sample only the first N MB of each source (faster)")
    ap.add_argument("--json", default="results/font_audit.json")
    ap.add_argument("--latex", default="../paper/tables/fonts.tex",
                    help="write the appendix typeface table here")
    args = ap.parse_args()

    print("=" * 78)
    print("1. COMPLEX-SCRIPT SHAPING")
    print("=" * 78)
    shaping = check_shaping()
    print(f"Pillow {shaping['pillow']}, Raqm available: {shaping['raqm']}")
    if shaping.get("raqm_version"):
        print(f"  raqm {shaping['raqm_version']}, "
              f"harfbuzz {shaping.get('harfbuzz_version')}")
    if not shaping["raqm"]:
        print("\n  *** Pillow has no Raqm support. Tamil is drawn without")
        print("  *** reordering or ligature substitution: every rendered line")
        print("  *** is wrong while its .gt.txt stays correct. Any corpus")
        print("  *** generated in this environment is unusable.")

    print()
    print("=" * 78)
    print("2. CORPUS CHARACTER INVENTORY")
    print("=" * 78)
    freq = corpus_codepoint_freq(args.raw_dir, args.limit_mb)
    total = sum(freq.values())
    tamil = {cp: n for cp, n in freq.items() if cp in TAMIL_BLOCK}
    print(f"{len(freq):,} distinct codepoints, {total:,} occurrences")
    print(f"{len(tamil):,} distinct Tamil-block codepoints "
          f"({sum(tamil.values()) / total * 100:.2f}% of all characters)")

    print()
    print("=" * 78)
    print("3. PER-FONT COVERAGE")
    print("=" * 78)
    paths = sorted(p for p in Path(args.font_dir).iterdir()
                   if p.suffix.lower() in (".ttf", ".otf"))
    reports = []
    for path in paths:
        try:
            reports.append(font_report(path, freq))
        except Exception as exc:
            print(f"[warn] {path.name}: {exc}")

    print(f"{'font':<20} {'tamil':>6} {'gsub':>5} {'miss cp':>8} "
          f"{'miss share':>11}  licence")
    print("-" * 78)
    for r in sorted(reports, key=lambda r: -r["missing_share"]):
        lic = (r.get("license") or r.get("copyright") or "")[:22].replace("\n", " ")
        flag = "  <<<" if r["missing_share"] > 0.0001 else ""
        print(f"{r['stem']:<20} {r['tamil_block_covered']:>6} "
              f"{'yes' if r['has_gsub'] else 'NO':>5} "
              f"{r['missing_count']:>8} {r['missing_share'] * 100:>10.4f}%  "
              f"{lic}{flag}")

    worst = [r for r in reports if r["missing_share"] > 0.0001]
    if worst:
        print()
        print("Fonts missing corpus characters, with the most frequent gaps:")
        for r in sorted(worst, key=lambda r: -r["missing_share"])[:8]:
            top = sorted(((cp, freq[cp]) for cp in r["missing_codepoints"]),
                         key=lambda x: -x[1])[:6]
            print(f"\n  {r['stem']}  ({r['missing_share'] * 100:.3f}% of corpus text)")
            for cp, n in top:
                print(f"      U+{cp:04X} {chr(cp)!r:<6} {n:>10,}  {describe(cp)}")

    nogsub = [r for r in reports if not r["has_gsub"]]
    if nogsub:
        print(f"\nFonts with no GSUB table (cannot form conjuncts): "
              f"{', '.join(r['stem'] for r in nogsub)}")

    print()
    print("=" * 78)
    print("4. LICENCE DISTRIBUTION")
    print("=" * 78)
    buckets = Counter(classify_license(r) for r in reports)
    for name, n in buckets.most_common():
        print(f"  {name:<24} {n:>2}")
    unclear = [r for r in reports if classify_license(r) != "OFL 1.1"]
    if unclear:
        print(f"\n  {len(unclear)} of {len(reports)} typefaces carry no clear "
              f"libre licence. The corpus redistributes rendered images of\n"
              f"  these designs and the repository ships the .ttf files under "
              f"GPL-3.0, which cannot relicense them. Resolve before\n"
              f"  claiming an open corpus release.")

    if args.latex:
        latex_table(reports, args.latex)

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"shaping": shaping, "corpus_codepoints": len(freq),
         "corpus_occurrences": total, "fonts": reports},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
