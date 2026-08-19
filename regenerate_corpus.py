#!/usr/bin/env python3
"""Regenerate the released corpus deterministically.

Supersedes the generate-gt.py path for producing the published dataset. Two
things differ, and both are the point of this script:

1. FONT ASSIGNMENT. generate-gt.py computes a round-robin pairing and then
   discards it — create_a4_tiff_image calls random.choice on the per-page
   window instead of using the pairing it was handed, and the module-level
   random is never seeded. The v1 corpus was therefore produced by unseeded
   random sampling with replacement: font usage spans roughly 8% between the
   least- and most-used typeface, and the corpus cannot be regenerated.
   Here every line i is rendered in font i mod 27, exactly balanced and
   reproducible.

2. LINE SELECTION. The reduction from the full pool to 105,738 lines was
   previously undocumented. Here it is a seeded shuffle followed by
   truncation, so the exact line set is a pure function of (SEED, N_LINES)
   and the source texts.

Shuffling before truncating matters: the sources are concatenated in size
order, so taking a prefix without shuffling would draw almost entirely from
Wikisource.

    python3 regenerate_corpus.py            # full run
    python3 regenerate_corpus.py --dry-run  # report the selection, render nothing
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

import corpus as corpus_mod  # noqa: E402
import render as render_mod  # noqa: E402

SEED = 0
N_LINES = 105_738          # matches the v1 release
OUT_DIR = Path("gt")
CORPUS_TXT = Path("data/corpus-105738.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-lines", type=int, default=N_LINES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--font-dir", default="fonts")
    args = ap.parse_args()

    started = time.time()
    print("Building per-source line pools...")
    pool = corpus_mod.build_pool()
    raw = sum(len(v) for v in pool.values())

    lines = corpus_mod.select(pool, sources=None, n_lines=args.n_lines,
                              seed=args.seed, dedup=True)
    uniq = len(corpus_mod.select(pool, n_lines=None, seed=args.seed, dedup=True))

    print(f"  raw pool          {raw:,} lines")
    print(f"  after dedup       {uniq:,} lines  ({raw - uniq:,} duplicates removed)")
    print(f"  selected          {len(lines):,} lines  (seed={args.seed})")

    # Provenance of the selection, so the composition is reportable.
    index = {}
    for src, src_lines in pool.items():
        for ln in src_lines:
            index.setdefault(ln, src)
    provenance = Counter(index.get(ln, "?") for ln in lines)
    print("\n  selected lines by source:")
    for src, n in provenance.most_common():
        print(f"    {src:<34} {n:>7,}  {n / len(lines) * 100:5.1f}%")

    if args.dry_run:
        print(f"\nDry run — nothing rendered. ({time.time() - started:.1f}s)")
        return 0

    corpus_mod.write_corpus(lines, CORPUS_TXT)
    print(f"\nWrote {CORPUS_TXT}")

    print(f"\nRendering to {args.out}/ ...")
    manifest = render_mod.generate(
        lines, args.out, font_dir=args.font_dir, font_names=None,
        assignment="round-robin", seed=args.seed)

    counts = manifest["font_line_counts"]
    lo, hi = min(counts.values()), max(counts.values())
    print(f"\n  crops written     {manifest['crops_written']:,}"
          f" / {manifest['requested_lines']:,} requested")
    print(f"  typefaces         {manifest['n_fonts']}")
    print(f"  per-font lines    {lo:,} to {hi:,}"
          f"  (spread {(hi - lo) / (sum(counts.values()) / len(counts)) * 100:.2f}%)")

    check = render_mod.validate(args.out)
    print(f"  images/gt pairs   {check['images']:,} / {check['transcriptions']:,}")
    if check["orphan_images"] or check["orphan_transcriptions"]:
        print(f"  ORPHANS           {check['orphan_images'][:3]} "
              f"{check['orphan_transcriptions'][:3]}")

    meta = {
        "seed": args.seed,
        "n_lines_requested": args.n_lines,
        "raw_pool_lines": raw,
        "deduplicated_lines": uniq,
        "selection": "seeded shuffle then truncate (experiments/corpus.py:select)",
        "font_assignment": "round-robin, line i -> font i mod n",
        "provenance": dict(provenance),
        "manifest": manifest,
        "validation": check,
        "wall_clock_s": round(time.time() - started, 1),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/regeneration.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote results/regeneration.json  ({meta['wall_clock_s'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
