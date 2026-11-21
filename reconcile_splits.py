#!/usr/bin/env python3
"""Score every stored prediction set on all three test-set views.

Written for the IJDAR revision (reviewer 2, item 2): the paper quotes some
models on all 300 lines and others on the 202-line reporting split without
saying which. This prints, for each preds_* directory, grapheme CER and WER
on

    all300   the full test set (the "diagnostic" view used in early sections)
    sel98    the selection split (model choices only)
    rep202   the reporting split (headline numbers)

using tamil_ocr_eval's own scorer and the split lists stored in
results/checkpoint_sweep.json, so the figures are the paper's figures.

    python3 reconcile_splits.py [--json results/reconciliation.json]
"""
import argparse
import json
from pathlib import Path

import tamil_ocr_eval as ev

HERE = Path(__file__).resolve().parent
GT_DIR = HERE / "testset" / "gt"


def score(pred_dir, names):
    per_line = []
    for stem in names:
        gt = GT_DIR / f"{stem}.gt.txt"
        pred = pred_dir / f"{stem}.txt"
        if not pred.exists():
            raise FileNotFoundError(f"{pred} missing")
        r = ev.score_pair(ev.read(gt), ev.read(pred))
        r["name"] = stem
        per_line.append(r)
    agg = ev.aggregate(per_line)
    return {"lines": len(per_line),
            "cer": round(100 * agg["cer_grapheme_micro"], 2),
            "wer": round(100 * agg["wer_micro"], 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="also write the table here")
    args = ap.parse_args()

    sweep = json.loads((HERE / "results" / "checkpoint_sweep.json").read_text())
    views = {
        "all300": sorted(p.name[: -len(".gt.txt")] for p in GT_DIR.glob("*.gt.txt")),
        "sel98": sweep["selection_split"],
        "rep202": sweep["reporting_split"],
    }
    assert len(views["all300"]) == 300, len(views["all300"])
    assert not set(views["sel98"]) & set(views["rep202"])

    table = {}
    for pred_dir in sorted(HERE.glob("preds_*")):
        if not pred_dir.is_dir():
            continue
        table[pred_dir.name] = {v: score(pred_dir, names)
                                for v, names in views.items()}

    print(f"{'predictions':16s} {'all300 CER/WER':>16s} {'sel98 CER':>10s} "
          f"{'rep202 CER/WER':>16s}")
    for name, r in table.items():
        print(f"{name:16s} {r['all300']['cer']:7.2f} {r['all300']['wer']:7.2f} "
              f"{r['sel98']['cer']:10.2f} {r['rep202']['cer']:7.2f} "
              f"{r['rep202']['wer']:7.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
