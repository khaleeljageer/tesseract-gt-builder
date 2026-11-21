#!/usr/bin/env python3
"""Score one prediction directory on all300 / sel98 / rep202 (IJDAR revision).

Same scorer and splits as sweep_checkpoints.py and reconcile_splits.py, so
PaddleOCR predictions are scored exactly like Tesseract's.

    python3 score_preds.py PRED_DIR [--json OUT.json] [--label NAME]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import tamil_ocr_eval as ev  # noqa: E402

TEST = HERE / "testset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("--json")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    pred = Path(args.pred_dir)
    sweep = json.loads((HERE / "results" / "checkpoint_sweep.json").read_text())
    views = {"all300": sorted(p.name[:-len(".gt.txt")]
                              for p in (TEST / "gt").glob("*.gt.txt")),
             "sel98": sweep["selection_split"],
             "rep202": sweep["reporting_split"]}
    row = {"label": args.label or pred.name, "pred_dir": str(pred)}
    for v, names in views.items():
        per = [ev.score_pair(ev.read(TEST / "gt" / f"{s}.gt.txt"),
                             ev.read(pred / f"{s}.txt")) for s in names]
        agg = ev.aggregate(per)
        row[f"cer_{v}"] = round(100 * agg["cer_grapheme_micro"], 2)
        row[f"wer_{v}"] = round(100 * agg["wer_micro"], 2)
    print(json.dumps(row))
    if args.json:
        out = Path(args.json)
        rows = json.loads(out.read_text()) if out.exists() else []
        rows = [r for r in rows if r["label"] != row["label"]] + [row]
        out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
