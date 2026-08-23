#!/usr/bin/env python3
"""Find ground-truth lines that Tesseract cannot normalise.

unicharset_extractor reports these as

    Invalid start of grapheme sequence:M=0xbbe
    Normalization failed for string '...'

and then skips the line, so it contributes nothing to the unicharset. The
line still reaches lstmtraining through list.train, where it is trained on
as written -- an image of a broken glyph cluster labelled with the broken
codepoint sequence. In v3 that was 123 lines of 218,125 (0.056%), too few to
move the model but not something to publish.

The defects are artifacts of the 8-bit Tamil encodings (Bamini, TAB, TSCII)
the source sites used before Unicode; see corpus.py's TAMIL_REPAIRS for the
mechanism. Most are mechanically repairable, but repairing the text here
would not do -- the .tif was rendered from the broken text and depicts the
broken glyphs, so a repaired .gt.txt would no longer describe its image. The
repair belongs in corpus.py, before rendering. What this tool does for an
already-rendered build is find the affected lines and, with --delete, remove
each one's .tif/.gt.txt/.box/.lstmf together.

Usage:
    python3 audit_gt.py gt                  # report only
    python3 audit_gt.py gt --list bad.txt   # also write the line ids
    python3 audit_gt.py gt --delete         # remove the whole quadruple
"""

import argparse
import collections
import sys
import unicodedata
from pathlib import Path

CONSONANTS = frozenset(chr(c) for c in range(0x0B95, 0x0BBA))
MARKS = frozenset(chr(c) for c in range(0x0BBE, 0x0BCE)) | {"ௗ"}
SUFFIXES = (".gt.txt", ".tif", ".box", ".lstmf")


def defects(text):
    """[(char, index, kind)] for every mark with no consonant to attach to."""
    found = []
    prev = ""
    for i, ch in enumerate(unicodedata.normalize("NFC", text)):
        if ch in MARKS and prev not in CONSONANTS:
            kind = "mark-after-mark" if prev in MARKS else "dangling-mark"
            found.append((ch, i, kind))
        prev = ch
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt_dir", help="directory of *.gt.txt")
    ap.add_argument("--list", metavar="FILE", help="write affected line ids here")
    ap.add_argument("--delete", action="store_true",
                    help="remove .tif/.gt.txt/.box/.lstmf for each affected line")
    ap.add_argument("--show", type=int, default=10,
                    help="print this many offending lines (default 10)")
    args = ap.parse_args(argv)

    root = Path(args.gt_dir)
    files = sorted(root.glob("*.gt.txt"))
    if not files:
        sys.exit(f"no *.gt.txt under {root}")

    kinds = collections.Counter()
    contexts = collections.Counter()
    bad = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        d = defects(text)
        if not d:
            continue
        bad.append(path)
        norm = unicodedata.normalize("NFC", text)
        for ch, i, kind in d:
            kinds[kind] += 1
            contexts[(norm[i - 1] if i else "") + ch] += 1

    print(f"{len(files):,} lines scanned")
    print(f"{len(bad):,} affected ({len(bad) / len(files):.3%}), "
          f"{sum(kinds.values()):,} defective marks")
    for kind, n in kinds.most_common():
        print(f"  {kind:16s} {n}")
    if contexts:
        print("\nmost common contexts:")
        for ctx, n in contexts.most_common(10):
            codes = " ".join(f"U+{ord(c):04X}" for c in ctx)
            print(f"  {n:5d}  {ctx!r:12s} {codes}")
    for path in bad[:args.show]:
        print(f"\n  {path.name}\n    {path.read_text(encoding='utf-8').strip()}")
    if len(bad) > args.show:
        print(f"\n  ... {len(bad) - args.show} more")

    if args.list:
        Path(args.list).write_text(
            "\n".join(p.name[:-len(".gt.txt")] for p in bad) + "\n")
        print(f"\nwrote {args.list}")

    if args.delete:
        removed = 0
        for path in bad:
            stem = path.name[:-len(".gt.txt")]
            for suffix in SUFFIXES:
                target = root / (stem + suffix)
                if target.exists():
                    target.unlink()
                    removed += 1
        print(f"\ndeleted {removed} files for {len(bad)} lines")
        print("regenerate list.train/list.eval before training again")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
