#!/usr/bin/env python3
"""Rescore the test set under Vasantharajan et al.'s evaluation profile (X8).

What their paper (arXiv:2109.05952v3, IALP 2022, Sec. III-C) states:
  * CER and WER are computed with the `fastwer` package (PyPI, v0.2.0),
    S+D+I over reference length, against manually labelled ground truth;
  * every test sample is one rendered image of 762 characters / 77 words
    in one font, read with `--oem 3 --psm 1`, and Table II reports the
    mean over 11 fonts of the per-image rates.
Not stated: any Unicode, zero-width or whitespace normalisation (fastwer
itself applies none: it compares Python code points, spaces included), and
the pre-processing ("advanced steps by using OpenCV"). Their test images
and transcriptions are not released, so only the metric can be reproduced.

Profiles scored here, for stock, the released model and the released
checkpoint without a dictionary (Tesseract 5.5.0 predictions,
results/revision/x10_preds/v550):
  paper      the paper's scorer: NFC, zero-width stripped, whitespace
             collapsed, grapheme CER, micro over lines;
  fw_lines   fastwer on the raw line strings, micro over lines;
  fw_lines_nozw  the same after deleting zero-width characters only (stock
             tam emits them; the ground truth and our model almost never);
  fw_pages   fastwer per source page (its lines joined by a space), then the
             unweighted mean over pages -- the analogue of their mean over
             per-font sample images;
  fw_strata  fastwer micro within each capture condition, then the
             unweighted mean over the five conditions.

    <venv with fastwer>/bin/python experiments/x8_fastwer.py
"""
import collections
import json
import sys
from pathlib import Path

import fastwer

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import tamil_ocr_eval as ev  # noqa: E402

TEST = HERE / "testset"
PREDS = HERE / "results" / "revision" / "x10_preds" / "v550"
OUT = HERE / "results" / "revision" / "x8_fastwer.json"
MODELS = {"stock": "tam", "release": "tamv3dcorpus", "nodict": "tamv3i1400"}


def raw(path):
    return path.read_text(encoding="utf-8").strip()


ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))


def fw(hyps, refs):
    return (round(fastwer.score(hyps, refs, char_level=True), 2),
            round(fastwer.score(hyps, refs, char_level=False), 2))


def mean(pairs):
    return tuple(round(sum(p[i] for p in pairs) / len(pairs), 2) for i in (0, 1))


def profiles(model, names):
    ref = {n: raw(TEST / "gt" / f"{n}.gt.txt") for n in names}
    hyp = {n: raw(PREDS / model / f"{n}.txt") for n in names}
    rows = [ev.score_pair(ev.read(TEST / "gt" / f"{n}.gt.txt"),
                          ev.read(PREDS / model / f"{n}.txt")) for n in names]
    a = ev.aggregate(rows)
    out = {"paper": (round(100 * a["cer_grapheme_micro"], 2),
                     round(100 * a["wer_micro"], 2)),
           "fw_lines": fw([hyp[n] for n in names], [ref[n] for n in names]),
           "fw_lines_nozw": fw([hyp[n].translate(ZW) for n in names],
                               [ref[n].translate(ZW) for n in names]),
           "zw_in_hyp": sum(len(hyp[n]) - len(hyp[n].translate(ZW)) for n in names)}
    pages, strata = collections.defaultdict(list), collections.defaultdict(list)
    for n in names:
        pages[n.rsplit("_", 1)[0]].append(n)
        strata[n.split("_", 1)[0]].append(n)
    out["fw_pages"] = mean([fw([" ".join(hyp[n] for n in ns)],
                               [" ".join(ref[n] for n in ns)])
                            for ns in pages.values()])
    out["fw_strata"] = mean([fw([hyp[n] for n in ns], [ref[n] for n in ns])
                             for ns in strata.values()])
    return out, len(pages)


def main():
    sweep = json.loads((HERE / "results" / "checkpoint_sweep.json").read_text())
    views = {"rep202": sweep["reporting_split"],
             "all300": sorted(p.name[:-len(".gt.txt")]
                              for p in (TEST / "gt").glob("*.gt.txt"))}
    result = {"fastwer_version": "0.2.0", "tesseract": "5.5.0",
              "units": "[CER, WER] in percent", "views": {}}
    for v, names in views.items():
        result["views"][v] = {}
        for label, model in MODELS.items():
            res, n_pages = profiles(model, names)
            result["views"][v][label] = res
            print(v, label, res)
        result["views"][v]["pages"] = n_pages
    OUT.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
