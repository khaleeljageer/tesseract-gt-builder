#!/usr/bin/env python3
"""Check that the pipeline regenerates the released TamilSynth v3 line set.

Rebuilds the line list exactly as `regenerate_corpus.py --no-repair` does
(no rendering) and compares it, file by file, against the transcriptions in
the released archive (Zenodo 10.5281/zenodo.22093954, tam_new-ground-truth.7z,
md5 9846e262bdc22332f3c21df43286b8d4). Line i is written to
page_{i//50+1:06d}_line_{i%50+1:03d}.gt.txt, so a match means both the text
and its position in the release are reproduced.

    python3 verify_release.py --gt-dir /path/to/extracted/gt

Exit status 0 only if all 218,125 transcriptions are byte-identical.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "experiments"))
import corpus as C  # noqa: E402
import render as R  # noqa: E402

LINES_PER_PAGE = 50
EXPECTED = 218_125


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = C.build_pool(raw_dir=str(HERE / "raw_data"), repair=False)
    lines = C.select(pool, n_lines=None, seed=args.seed, dedup=True,
                     require_well_formed=False)
    lines, _ = R.renderable(lines, R.common_coverage(str(HERE / "fonts")))

    gt = Path(args.gt_dir)
    same = differ = missing = 0
    examples = []
    for i, line in enumerate(lines):
        p = gt / (f"page_{i // LINES_PER_PAGE + 1:06d}"
                  f"_line_{i % LINES_PER_PAGE + 1:03d}.gt.txt")
        if not p.exists():
            missing += 1
            continue
        if p.read_text(encoding="utf-8").rstrip("\n") == line:
            same += 1
        else:
            differ += 1
            if len(examples) < 5:
                examples.append(p.name)

    print(f"regenerated {len(lines):,}  identical {same:,}  "
          f"differ {differ:,}  missing {missing:,}")
    for name in examples:
        print(f"  differs: {name}")
    ok = len(lines) == EXPECTED and same == EXPECTED
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
