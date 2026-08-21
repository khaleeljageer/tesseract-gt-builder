"""Variant orchestration: generate, train, recognise, score, record.

One variant = one point in an ablation grid. Each gets its own directory and
its own result.json. A variant whose result.json already exists is skipped, so
a sweep that dies partway through resumes where it stopped rather than
recomputing everything.

Every variant also appends one row to results/journal.jsonl recording the
hypothesis it tests and what came back, so the exploration tree survives into
the writeup.
"""

import json
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import corpus  # noqa: E402
import render  # noqa: E402
import train as trainer  # noqa: E402
from tamil_ocr_eval import score_pair, aggregate, bootstrap_ci, confusion_counts  # noqa: E402

RESULTS = Path("results")
JOURNAL = RESULTS / "journal.jsonl"

# Computed once: the codepoints every typeface can render. Shared by all
# variants so the candidate line pool is identical across the grid.
_COVERAGE = render.common_coverage("fonts")


@dataclass
class Variant:
    """One point in an ablation grid."""
    experiment: str                 # "fonts" | "size" | "domain"
    name: str                       # unique, becomes the directory name
    hypothesis: str                 # what this variant is meant to show
    n_lines: int
    font_names: list = None         # None = all 29
    sources: list = None            # None = all
    seed: int = 0
    # Must match the headline model's budget (\S6) or Table 1 and the
    # ablation tables are not on the same footing.
    max_iterations: int = 100_000
    start_model: str = "tam"

    def dir(self):
        return RESULTS / self.experiment / self.name


def log_journal(entry):
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def score(gt_dir, pred_dir, bootstrap=2000, top_confusions=30):
    """Score one prediction directory against its ground truth."""
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    per_line = []
    for gt_file in sorted(gt_dir.glob("*.gt.txt")):
        stem = gt_file.name[: -len(".gt.txt")]
        pred = pred_dir / f"{stem}.txt"
        if not pred.exists():
            continue
        r = score_pair(gt_file.read_text(encoding="utf-8"),
                       pred.read_text(encoding="utf-8"))
        r["name"] = stem
        per_line.append(r)

    if not per_line:
        raise RuntimeError(f"no scoreable pairs between {gt_dir} and {pred_dir}")

    agg = aggregate(per_line)
    ci = bootstrap_ci(per_line, "grapheme_edits", "grapheme_ref", bootstrap)
    wci = bootstrap_ci(per_line, "word_edits", "word_ref", bootstrap)
    if ci:
        agg["cer_grapheme_ci95"] = list(ci)
        agg["wer_ci95"] = list(wci)

    subs, dels, ins = confusion_counts(per_line)
    agg["top_substitutions"] = [
        {"ref": a, "hyp": b, "count": n} for (a, b), n in subs.most_common(top_confusions)
    ]
    agg["top_deletions"] = [{"ref": c, "count": n} for c, n in dels.most_common(15)]
    agg["top_insertions"] = [{"hyp": c, "count": n} for c, n in ins.most_common(15)]
    return agg


def run_variant(v, pool, test_dir, tessdata_dir, force=False, keep_images=False):
    """Generate -> train -> recognise -> score for one variant.

    test_dir is the fixed held-out real-document test set. It is never
    regenerated and never varies between variants; that is the whole point.
    """
    vdir = v.dir()
    result_path = vdir / "result.json"

    if result_path.exists() and not force:
        print(f"[skip] {v.experiment}/{v.name} already has result.json")
        return json.loads(result_path.read_text(encoding="utf-8"))

    vdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"\n=== {v.experiment}/{v.name} ===")
    print(f"    {v.hypothesis}")

    # 1. corpus slice
    #
    # The typeface-coverage filter uses the FULL font set, not the variant's
    # subset, deliberately. Filtering per-variant would give each variant a
    # slightly different candidate pool, so the font ablation would vary two
    # things at once. Holding the pool fixed means only the tested variable
    # moves, and it matches how the released corpus was built.
    lines = corpus.select(pool, sources=v.sources, n_lines=None, seed=v.seed)
    lines, _unrenderable = render.renderable(lines, _COVERAGE)
    if v.n_lines is not None:
        if v.n_lines > len(lines):
            raise ValueError(
                f"{v.experiment}/{v.name} wants {v.n_lines:,} lines but only "
                f"{len(lines):,} survive filtering for sources={v.sources}")
        lines = lines[: v.n_lines]
    corpus.write_corpus(lines, vdir / "corpus.txt")

    # 2. render
    gt_dir = vdir / "gt"
    manifest = render.generate(
        lines, gt_dir, font_names=v.font_names,
        assignment="round-robin", seed=v.seed)

    # 3. train
    model_name = f"{v.experiment}_{v.name}"
    model_path = trainer.train(
        gt_dir, model_name, vdir / "model",
        start_model=v.start_model, max_iterations=v.max_iterations,
        log_path=vdir / "train.log")

    # tesseract wants the model discoverable by -l <name> in a tessdata dir
    staging = vdir / "tessdata"
    staging.mkdir(exist_ok=True)
    shutil.copy(model_path, staging / f"{model_name}.traineddata")

    # 4. recognise the FIXED test set
    pred_dir = vdir / "pred"
    infer = trainer.recognise(
        Path(test_dir) / "images", pred_dir, model_name, staging)

    # 5. score
    agg = score(Path(test_dir) / "gt", pred_dir)

    result = {
        "variant": asdict(v),
        "manifest": manifest,
        "inference": infer,
        "metrics": agg,
        "model": str(model_path),
        "wall_clock_s": round(time.time() - started, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    # Snapshot the code that produced this, so the run reproduces even after
    # the scripts change.
    snap = vdir / "code_snapshot"
    snap.mkdir(exist_ok=True)
    for mod in ("corpus.py", "render.py", "train.py", "runner.py"):
        src = Path(__file__).parent / mod
        if src.exists():
            shutil.copy(src, snap / mod)

    if not keep_images:
        shutil.rmtree(gt_dir, ignore_errors=True)

    log_journal({
        "experiment": v.experiment,
        "variant": v.name,
        "hypothesis": v.hypothesis,
        "n_lines": v.n_lines,
        "n_fonts": manifest["n_fonts"],
        "sources": v.sources,
        "cer_grapheme_micro": agg["cer_grapheme_micro"],
        "wer_micro": agg["wer_micro"],
        "ci95": agg.get("cer_grapheme_ci95"),
        "wall_clock_s": result["wall_clock_s"],
        "timestamp": result["timestamp"],
    })

    print(f"    grapheme CER {agg['cer_grapheme_micro'] * 100:.2f}%   "
          f"WER {agg['wer_micro'] * 100:.2f}%   "
          f"({result['wall_clock_s'] / 60:.1f} min)")
    return result


def run_grid(variants, test_dir, tessdata_dir, force=False, keep_images=False):
    pool = corpus.build_pool()
    results = []
    for v in variants:
        try:
            results.append(run_variant(v, pool, test_dir, tessdata_dir,
                                       force=force, keep_images=keep_images))
        except Exception as exc:                       # keep the sweep alive
            print(f"[FAIL] {v.experiment}/{v.name}: {exc}")
            log_journal({
                "experiment": v.experiment, "variant": v.name,
                "status": "failed", "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    return results
