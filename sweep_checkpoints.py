#!/usr/bin/env python3
"""Score every checkpoint of one tesstrain run on the real-document test set.

Reproduces the procedure behind results/checkpoint_sweep.json (IJDAR
revision, new runs X1/X2/X9):

  1. each <model>_<err>_<learning>_<training>.checkpoint is converted with
     `lstmtraining --stop_training` against the run's own starter
     traineddata (<data_dir>/<model>/<model>.traineddata), which carries no
     word list, so every point is scored without a dictionary;
  2. every test line is recognised with `tesseract --psm 13`,
     OMP_THREAD_LIMIT=1, exactly as make_testset.py predict does;
  3. grapheme CER and WER are computed with tamil_ocr_eval on all 300 lines,
     the 98-line selection split and the 202-line reporting split.

The iteration reported is the training iteration (the third number in the
checkpoint name), as in the paper.

    python3 sweep_checkpoints.py --data-dir ~/revision/runs --model tam_x1b \
        --out results/sweep_tam_x1b.json --jobs 20

Idempotent: converted models and predictions are cached under
<data_dir>/<model>/sweep/, so an interrupted sweep resumes.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tamil_ocr_eval as ev  # noqa: E402

TEST = HERE / "testset"
CKPT_RE = re.compile(r"_(?P<err>[\d.]+)_(?P<learn>\d+)_(?P<train>\d+)\.checkpoint$")


def convert(ckpt, proto, out):
    if out.exists() and out.stat().st_size > 0:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["lstmtraining", "--stop_training",
                        "--continue_from", str(ckpt),
                        "--traineddata", str(proto),
                        "--model_output", str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"convert {ckpt.name}: {r.stderr[-300:]}")


def recognise(img, tessdata, lang, dest):
    target = dest / f"{img.stem}.txt"
    if target.exists():
        return
    env = dict(os.environ, OMP_THREAD_LIMIT="1", TESSDATA_PREFIX=str(tessdata))
    r = subprocess.run(["tesseract", str(img), "stdout", "-l", lang,
                        "--psm", "13"], capture_output=True, text=True, env=env)
    text = " ".join(r.stdout.split()) if r.returncode == 0 else ""
    target.write_text(text, encoding="utf-8")


def score(pred_dir, names):
    per_line = []
    for stem in names:
        r = ev.score_pair(ev.read(TEST / "gt" / f"{stem}.gt.txt"),
                          ev.read(pred_dir / f"{stem}.txt"))
        per_line.append(r)
    agg = ev.aggregate(per_line)
    return round(100 * agg["cer_grapheme_micro"], 2), round(100 * agg["wer_micro"], 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--every", type=int, default=0,
                    help="if >0, keep only checkpoints at least this many "
                         "training iterations apart (plus the last)")
    args = ap.parse_args()

    run = Path(args.data_dir).expanduser() / args.model
    proto = run / f"{args.model}.traineddata"
    ckpts = []
    for c in (run / "checkpoints").glob(f"{args.model}_*.checkpoint"):
        m = CKPT_RE.search(c.name)
        if m:
            ckpts.append((int(m["train"]), c))
    ckpts.sort()
    if not ckpts:
        sys.exit(f"no checkpoints in {run / 'checkpoints'}")
    if args.every:
        kept, last = [], -10**9
        for it, c in ckpts:
            if it - last >= args.every:
                kept.append((it, c))
                last = it
        if kept[-1] != ckpts[-1]:
            kept.append(ckpts[-1])
        ckpts = kept

    sweep = json.loads((HERE / "results" / "checkpoint_sweep.json").read_text())
    views = {"all300": sorted(p.name[:-len(".gt.txt")]
                              for p in (TEST / "gt").glob("*.gt.txt")),
             "sel98": sweep["selection_split"],
             "rep202": sweep["reporting_split"]}
    images = sorted((TEST / "images").glob("*.tif"))
    assert len(images) == 300 and len(views["all300"]) == 300

    tver = subprocess.run(["tesseract", "--version"], capture_output=True,
                          text=True).stdout.split("\n")[0]
    results = {"model": args.model, "tesseract": tver,
               "procedure": "lstmtraining --stop_training against the run's "
                            "starter traineddata (no DAWG); tesseract --psm 13",
               "checkpoints": {}}
    out = Path(args.out)
    if out.exists():
        results["checkpoints"] = json.loads(out.read_text()).get("checkpoints", {})

    for it, c in ckpts:
        key = str(it)
        if key in results["checkpoints"]:
            continue
        tess = run / "sweep" / f"i{it}"
        lang = f"{args.model}_i{it}"
        convert(c, proto, tess / f"{lang}.traineddata")
        preds = run / "sweep" / f"preds_i{it}"
        preds.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(args.jobs) as ex:
            list(ex.map(lambda im: recognise(im, tess, lang, preds), images))
        row = {"checkpoint": c.name}
        for v, names in views.items():
            cer, wer = score(preds, names)
            row[f"cer_{v}"], row[f"wer_{v}"] = cer, wer
        results["checkpoints"][key] = row
        out.write_text(json.dumps(results, indent=2))
        print(f"i{it:>6}  all300 {row['cer_all300']:6.2f}  sel98 {row['cer_sel98']:6.2f}"
              f"  rep202 {row['cer_rep202']:6.2f} / WER {row['wer_rep202']:6.2f}",
              flush=True)

    best = min(results["checkpoints"].items(), key=lambda kv: kv[1]["cer_sel98"])
    results["selected_on_sel98"] = {"iteration": int(best[0]), **best[1]}
    out.write_text(json.dumps(results, indent=2))
    print(f"selected on sel98: i{best[0]}  rep202 CER {best[1]['cer_rep202']}  "
          f"WER {best[1]['wer_rep202']}")


if __name__ == "__main__":
    main()
