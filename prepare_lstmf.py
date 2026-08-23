#!/usr/bin/env python3
"""Generate tesstrain .box and .lstmf files in parallel.

tesstrain's Makefile creates one target per training line. At corpus scale
that is ~200k targets, and GNU make spends nearly all its time evaluating the
dependency graph rather than running recipes: with -j 10 on a 12-core machine
the observed load average was 1.6 and throughput ~117 lines/min, an ETA of
about 28 hours for a stage that is embarrassingly parallel.

This script does exactly what the Makefile's two pattern rules do --

    %.box:   $(GENERATE_BOX_SCRIPT) -i <img> -t <img>.gt.txt > <img>.box
    %.lstmf: tesseract <img> <stem> --psm 13 lstm.train

-- across a process pool, then exits. Afterwards `make training` finds every
.lstmf already present, skips straight past the expensive stage, and proceeds
to build the file lists and train.

    python3 prepare_lstmf.py --gt-dir gt --jobs 10
    cd $TESSTRAIN_DIR && make training MODEL_NAME=... GROUND_TRUTH_DIR=...

Safe to interrupt and re-run: completed pairs are skipped.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PSM = "13"          # tesstrain default: raw line, no layout analysis

# The Makefile picks the box generator from LANG_TYPE, and getting this wrong
# is silent: character-level boxes are produced, training runs, and the model
# is merely worse. For Indic and RTL scripts tesstrain uses WordStr boxes --
# one record carrying the whole line -- instead of one record per character,
# because Tamil's combining vowel signs do not correspond to the units the
# character-level generator emits. generate_line_box.py groups a mark with
# the previous letter only when its canonical combining class is non-zero,
# and every Tamil vowel sign has class 0 (they are spacing marks, category
# Mc), so it splits லி into ல + ி while keeping க் whole.
#
# LANG_TYPE also sets --pass_through_recoder for combine_lang_model, which
# this script does not control: pass LANG_TYPE=Indic to `make training` too,
# or the recoder will be rebuilt in decomposing form.
BOX_SCRIPTS = {
    "Indic": "generate_wordstr_box.py",
    "RTL": "generate_wordstr_box.py",
    "": "generate_line_box.py",
}


def one(args):
    """Produce .box then .lstmf for a single image. Returns (stem, status)."""
    img, tesstrain_dir, box_script = args
    img = Path(img)
    stem = img.with_suffix("")
    box = stem.with_suffix(".box")
    lstmf = stem.with_suffix(".lstmf")
    gt = Path(str(stem) + ".gt.txt")

    if lstmf.exists() and lstmf.stat().st_size > 0:
        return stem.name, "skip"
    if not gt.exists():
        return stem.name, "no-gt"

    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_THREAD_LIMIT="1")

    if not (box.exists() and box.stat().st_size > 0):
        r = subprocess.run(
            [sys.executable, str(Path(tesstrain_dir) / box_script),
             "-i", str(img), "-t", str(gt)],
            capture_output=True, env=env)
        if r.returncode != 0:
            return stem.name, f"box-fail: {r.stderr.decode()[:70]}"
        box.write_bytes(r.stdout)

    r = subprocess.run(
        ["tesseract", str(img), str(stem), "--psm", PSM, "lstm.train"],
        capture_output=True, env=env)
    if r.returncode != 0:
        return stem.name, f"lstmf-fail: {r.stderr.decode()[:70]}"
    if not lstmf.exists():
        return stem.name, "lstmf-missing"
    return stem.name, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True,
                    help="directory of paired .tif / .gt.txt files")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--tesstrain-dir",
                    default=os.environ.get("TESSTRAIN_DIR", ""))
    ap.add_argument("--lang-type", default="Indic", choices=sorted(BOX_SCRIPTS),
                    help="as in tesstrain's Makefile; selects the box "
                         "generator. Tamil is Indic (default)")
    ap.add_argument("--box-script", default=None,
                    help="override the generator LANG_TYPE would select")
    args = ap.parse_args()
    if args.box_script is None:
        args.box_script = BOX_SCRIPTS[args.lang_type]

    if not args.tesstrain_dir:
        sys.exit("set TESSTRAIN_DIR or pass --tesstrain-dir")
    script = Path(args.tesstrain_dir) / args.box_script
    if not script.exists():
        sys.exit(f"missing {script}")

    gt_dir = Path(args.gt_dir)
    images = sorted(gt_dir.glob("*.tif"))
    if not images:
        sys.exit(f"no .tif files in {gt_dir}")

    todo = [p for p in images
            if not (p.with_suffix(".lstmf").exists()
                    and p.with_suffix(".lstmf").stat().st_size > 0)]
    print(f"LANG_TYPE={args.lang_type or '(blank)'} -> {args.box_script}")
    print(f"{len(images):,} images, {len(images) - len(todo):,} already done, "
          f"{len(todo):,} to process on {args.jobs} workers")
    if not todo:
        print("Nothing to do.")
        return 0

    started = time.time()
    done = failed = 0
    problems = []
    payload = [(str(p), args.tesstrain_dir, args.box_script) for p in todo]

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(one, p) for p in payload]
        for fut in as_completed(futures):
            name, status = fut.result()
            done += 1
            if status not in ("ok", "skip"):
                failed += 1
                if len(problems) < 10:
                    problems.append(f"{name}: {status}")
            if done % 2000 == 0 or done == len(todo):
                el = time.time() - started
                rate = done / el
                eta = (len(todo) - done) / rate / 60 if rate else 0
                print(f"  {done:>7,}/{len(todo):,}  "
                      f"{rate * 60:>6.0f} lines/min  "
                      f"ETA {eta:>5.1f} min  failed {failed}", flush=True)

    el = time.time() - started
    print(f"\nprocessed {done:,} in {el / 60:.1f} min "
          f"({done / el * 60:.0f} lines/min), {failed} failed")
    for p in problems:
        print(f"  {p}")
    n = len(list(gt_dir.glob("*.lstmf")))
    print(f"\n{n:,} .lstmf files now in {gt_dir}")
    print("Next: cd $TESSTRAIN_DIR && make training MODEL_NAME=... "
          f"GROUND_TRUTH_DIR=... MAX_ITERATIONS=... LANG_TYPE={args.lang_type}")
    if args.lang_type:
        print(f"      LANG_TYPE={args.lang_type} is not optional there -- it also "
              "sets --pass_through_recoder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
