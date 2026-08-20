"""Ablation grids for Experiments 3-5.

Each grid varies exactly one thing and holds everything else fixed. The
fixed quantities are not incidental — they are what makes the comparison
mean anything:

  Exp 3 (fonts)   line count fixed, font count varies.
                  Without the fix, "more fonts" also means "more data" and
                  the result is uninterpretable.

  Exp 4 (size)    font count fixed at 29, line count varies.

  Exp 5 (domain)  line count fixed at the size of the SMALLER register arm,
                  source set varies. The corpus holds 167,245 literary lines
                  against 34,269 newsprint lines; comparing them at natural
                  size would measure volume, not register.

Every variant is evaluated on the same held-out real-document test set, which
is never regenerated.

Usage:
    export TESSTRAIN_DIR=/path/to/tesstrain
    export TESSDATA_DIR=/path/to/tessdata_best

    python experiments/run_ablation.py --list
    python experiments/run_ablation.py fonts  --test-dir testset
    python experiments/run_ablation.py size   --test-dir testset
    python experiments/run_ablation.py domain --test-dir testset
    python experiments/run_ablation.py all    --test-dir testset
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus  # noqa: E402
from render import load_fonts  # noqa: E402
from runner import Variant, run_grid  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed quantities shared across grids.

FONT_ABLATION_LINES = 50_000   # held constant across the font grid
SIZE_ABLATION_FONTS = None     # None = all 29
DOMAIN_ABLATION_LINES = 34_000 # capped to the newsprint arm (34,269 available)
SEED = 0


def font_subset(all_fonts, k, seed=SEED):
    """Pick k fonts deterministically, spread across the sorted set.

    Taking a contiguous prefix would bias the subset toward one foundry
    (the TAU-* family alone is 10 of the 29). Even spacing over the sorted
    list gives a subset that is reproducible and typographically spread.
    """
    if k >= len(all_fonts):
        return None                      # None = use everything
    step = len(all_fonts) / k
    return [all_fonts[int(i * step)] for i in range(k)]


def grid_fonts(all_fonts):
    """Experiment 3: does typographic diversity drive the gain?"""
    out = []
    for k in (5, 10, 29):
        sub = font_subset(all_fonts, k)
        out.append(Variant(
            experiment="fonts",
            name=f"f{k:02d}",
            hypothesis=(
                f"At a fixed {FONT_ABLATION_LINES:,} lines, training on {k} "
                f"typefaces yields lower CER than fewer typefaces; if CER is "
                f"flat across k, the gain attributed to font diversity is "
                f"really just data volume."),
            n_lines=FONT_ABLATION_LINES,
            font_names=sub,
            seed=SEED,
        ))
    return out


def grid_size(pool):
    """Experiment 4: how much corpus is actually needed?"""
    available = sum(len(v) for v in pool.values())
    out = []
    for n in (10_000, 50_000, 198_880):
        if n > available:
            print(f"[warn] skipping size={n:,}: only {available:,} lines available")
            continue
        out.append(Variant(
            experiment="size",
            name=f"n{n // 1000:03d}k",
            hypothesis=(
                f"With all 29 fonts, {n:,} training lines gives lower CER than "
                f"fewer lines, with diminishing returns; the curve tells "
                f"reusers how much of the corpus they need."),
            n_lines=n,
            font_names=SIZE_ABLATION_FONTS,
            seed=SEED,
        ))
    return out


def grid_domain(pool):
    """Experiment 5: does training register transfer across register?"""
    out = []
    for register, sources in corpus.REGISTERS.items():
        present = [s for s in sources if s in pool]
        have = sum(len(pool[s]) for s in present)
        if have < DOMAIN_ABLATION_LINES:
            print(f"[warn] register {register!r} has only {have:,} lines; "
                  f"reduce DOMAIN_ABLATION_LINES below {have:,}")
            continue
        out.append(Variant(
            experiment="domain",
            name=register,
            hypothesis=(
                f"A model trained only on {register} text ({DOMAIN_ABLATION_LINES:,} "
                f"lines, 29 fonts) underperforms on the other register, showing "
                f"that source register — not just volume — shapes the model."),
            n_lines=DOMAIN_ABLATION_LINES,
            sources=present,
            seed=SEED,
        ))

    # Mixed control at the same line count: isolates register composition
    # from the simple fact of having two sources.
    out.append(Variant(
        experiment="domain",
        name="mixed",
        hypothesis=(
            f"A register-balanced mix at the same {DOMAIN_ABLATION_LINES:,} lines "
            f"matches or beats either single-register arm on the combined test "
            f"set."),
        n_lines=DOMAIN_ABLATION_LINES,
        sources=sorted({s for ss in corpus.REGISTERS.values() for s in ss
                        if s in pool}),
        seed=SEED,
    ))
    return out


def build(which, pool, all_fonts):
    grids = {
        "fonts": lambda: grid_fonts(all_fonts),
        "size": lambda: grid_size(pool),
        "domain": lambda: grid_domain(pool),
    }
    if which == "all":
        return [v for g in grids.values() for v in g()]
    return grids[which]()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?", default="all",
                    choices=["fonts", "size", "domain", "all"])
    ap.add_argument("--test-dir", default="testset",
                    help="held-out test set with images/ and gt/")
    ap.add_argument("--font-dir", default="fonts")
    ap.add_argument("--list", action="store_true",
                    help="print the grid and exit without running anything")
    ap.add_argument("--force", action="store_true",
                    help="recompute variants that already have result.json")
    ap.add_argument("--keep-images", action="store_true",
                    help="retain rendered crops (large: ~2.5 GB per 100k lines)")
    args = ap.parse_args()

    pool = corpus.build_pool()
    all_fonts = [name for name, _ in load_fonts(args.font_dir, 22)]
    variants = build(args.experiment, pool, all_fonts)

    print(f"{len(all_fonts)} fonts available, "
          f"{sum(len(v) for v in pool.values()):,} corpus lines\n")
    print(f"{'experiment':<10} {'variant':<10} {'lines':>9} {'fonts':>6}  sources")
    print("-" * 74)
    for v in variants:
        nf = len(v.font_names) if v.font_names else len(all_fonts)
        src = ",".join(s[:12] for s in v.sources) if v.sources else "all"
        print(f"{v.experiment:<10} {v.name:<10} {v.n_lines:>9,} {nf:>6}  {src}")
    print()

    if args.list:
        return 0

    test_dir = Path(args.test_dir)
    if not (test_dir / "gt").is_dir() or not (test_dir / "images").is_dir():
        print(f"ERROR: {test_dir}/ must contain images/ and gt/.\n"
              f"Build it first — see paper/TESTSET.md. Every number in these "
              f"grids is measured against it, so it has to exist before any "
              f"of this runs.", file=sys.stderr)
        return 1

    tessdata = os.environ.get("TESSDATA_DIR")
    if not tessdata:
        print("ERROR: TESSDATA_DIR is not set.", file=sys.stderr)
        return 1

    run_grid(variants, test_dir, tessdata,
             force=args.force, keep_images=args.keep_images)

    print("\nAggregate with:  python experiments/aggregate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
