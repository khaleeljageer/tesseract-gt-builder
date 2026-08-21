#!/usr/bin/env python3
"""Turn photographed or scanned pages into a line-level test set.

Takes page images, deskews and binarises them, segments them into text lines,
and writes the tesstrain-style layout the evaluator expects:

    testset/
      images/  page001_l003.tif      one real document line per file
      gt/      page001_l003.gt.txt   its transcription, for you to fill in
      manifest.csv                   provenance for every line

Transcription is yours to do; nothing here invents ground truth. What it does
is remove the mechanical work so you are only typing Tamil, not cropping.

    # 1. segment pages into lines
    python3 make_testset.py segment --pages photos/ --out testset --stratum newsprint

    # 2. optionally pre-fill with a DIFFERENT engine's output to correct
    python3 make_testset.py bootstrap --out testset --model tam

    # 3. check your transcriptions are complete and well-formed
    python3 make_testset.py check --out testset

    # 4. measure your own transcription error floor (see --help for detail)
    python3 make_testset.py agreement --pass1 testset/gt --pass2 testset/gt_pass2

On bootstrapping: pre-filling with the model you intend to evaluate is
circular -- you would be scoring the model against its own output wherever
you failed to spot an error. Use stock `tam`, or better a different engine
entirely, and read every line against the image.
"""

import argparse
import csv
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

MIN_LINE_HEIGHT = 12        # px; below this a band is noise, not a line
MIN_INK_FRACTION = 0.002    # a band must carry at least this share of dark px
PAD = 6


def deskew(gray):
    """Rotate so text baselines are horizontal.

    Photographed pages are rarely square to the sensor, and a few degrees of
    skew smears the horizontal projection until adjacent lines merge.
    """
    inv = cv2.bitwise_not(gray)
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = cv2.findNonZero(bw)
    if coords is None:
        return gray, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    if abs(angle) < 0.15:
        return gray, 0.0
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    out = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, angle


def binarise(gray):
    """Adaptive threshold: photographs have uneven illumination, so a global
    cutoff loses one side of the page."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 15)


def find_lines(binary):
    """Horizontal projection with smoothing, then merge diacritic bands.

    Merging matters for Tamil: the pulli and vowel signs sit above the letter
    bodies and can form their own band, exactly the effect that costs the
    synthetic pipeline 0.075% of its pages.
    """
    proj = (binary > 0).sum(axis=1).astype(float)
    if proj.max() == 0:
        return []
    k = max(3, binary.shape[0] // 400)
    proj = np.convolve(proj, np.ones(k) / k, mode="same")

    thresh = max(1.0, proj.max() * 0.06)
    bands, inside, start = [], False, 0
    for i, v in enumerate(proj):
        if v > thresh and not inside:
            start, inside = i, True
        elif v <= thresh and inside:
            bands.append([start, i])
            inside = False
    if inside:
        bands.append([start, len(proj)])
    if not bands:
        return []

    heights = sorted(b[1] - b[0] for b in bands)
    median_h = heights[len(heights) // 2]
    merged = [bands[0]]
    for s, e in bands[1:]:
        if s - merged[-1][1] < median_h * 0.45:     # diacritic band
            merged[-1][1] = e
        else:
            merged.append([s, e])

    total_ink = (binary > 0).sum()
    keep = []
    for s, e in merged:
        if e - s < MIN_LINE_HEIGHT:
            continue
        if (binary[s:e] > 0).sum() < total_ink * MIN_INK_FRACTION:
            continue
        keep.append((s, e))
    return keep


def segment(args):
    pages = sorted(p for p in Path(args.pages).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    if not pages:
        sys.exit(f"no page images in {args.pages}")

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    new = not manifest.exists()
    fh = open(manifest, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["line_id", "source_page", "stratum", "skew_deg",
                    "y0", "y1", "height_px"])

    total = 0
    for page in pages:
        gray = cv2.imread(str(page), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[warn] unreadable {page.name}")
            continue
        gray, angle = deskew(gray)
        binary = binarise(gray)
        bands = find_lines(binary)
        stem = re.sub(r"[^A-Za-z0-9]+", "_", page.stem)[:28]

        for i, (y0, y1) in enumerate(bands, 1):
            crop = gray[max(0, y0 - PAD):min(gray.shape[0], y1 + PAD), :]
            cols = np.where((binarise(crop) > 0).sum(axis=0) > 0)[0]
            if cols.size:
                crop = crop[:, max(0, cols[0] - PAD):min(crop.shape[1], cols[-1] + PAD)]
            lid = f"{stem}_l{i:03d}"
            cv2.imwrite(str(out / "images" / f"{lid}.tif"), crop)
            gt = out / "gt" / f"{lid}.gt.txt"
            if not gt.exists():
                gt.write_text("", encoding="utf-8")
            w.writerow([lid, page.name, args.stratum, f"{angle:.2f}",
                        y0, y1, y1 - y0])
            total += 1
        print(f"  {page.name:<34} skew {angle:+5.2f}°  {len(bands):>3} lines")

    fh.close()
    print(f"\n{total} line images written to {out}/images")
    print(f"Empty transcriptions waiting in {out}/gt")
    print(f"Provenance appended to {manifest}")


def bootstrap(args):
    out = Path(args.out)
    imgs = sorted((out / "images").glob("*.tif"))
    filled = 0
    for img in imgs:
        gt = out / "gt" / f"{img.stem}.gt.txt"
        if gt.exists() and gt.read_text(encoding="utf-8").strip():
            continue
        r = subprocess.run(
            ["tesseract", str(img), "stdout", "-l", args.model, "--psm", "7"],
            capture_output=True, text=True)
        text = " ".join(r.stdout.split()) if r.returncode == 0 else ""
        gt.write_text(text, encoding="utf-8")
        filled += 1
    print(f"pre-filled {filled} of {len(imgs)} transcriptions using '{args.model}'")
    print("\nThese are MACHINE GUESSES, not ground truth. Read every one against\n"
          "its image and correct it. Anything you do not check is not data.")


def check(args):
    out = Path(args.out)
    imgs = {p.stem for p in (out / "images").glob("*.tif")}
    gts = {p.name[:-len(".gt.txt")] for p in (out / "gt").glob("*.gt.txt")}
    empty, nonfc, latin, ok = [], [], [], 0

    for stem in sorted(imgs & gts):
        t = (out / "gt" / f"{stem}.gt.txt").read_text(encoding="utf-8").strip()
        if not t:
            empty.append(stem); continue
        if unicodedata.normalize("NFC", t) != t:
            nonfc.append(stem)
        if not any("஀" <= c <= "௿" for c in t):
            latin.append(stem)
        ok += 1

    print(f"line images        {len(imgs):>5}")
    print(f"transcriptions     {len(gts):>5}")
    print(f"  complete         {ok:>5}")
    print(f"  still empty      {len(empty):>5}")
    print(f"  not NFC          {len(nonfc):>5}")
    print(f"  no Tamil at all  {len(latin):>5}")
    for label, xs in (("orphan images", imgs - gts), ("orphan transcriptions", gts - imgs)):
        if xs:
            print(f"  {label}: {sorted(xs)[:4]}")
    if nonfc:
        print(f"\n  Not NFC (re-save these): {nonfc[:5]}")
    if empty:
        print(f"\n  {len(empty)} lines still need transcribing, e.g. {empty[:4]}")
    else:
        print("\n  All transcribed. Run the agreement check next.")


def agreement(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tamil_ocr_eval import score_pair, aggregate

    p1, p2 = Path(args.pass1), Path(args.pass2)
    rows = []
    for f1 in sorted(p1.glob("*.gt.txt")):
        f2 = p2 / f1.name
        if not f2.exists():
            continue
        r = score_pair(f1.read_text(encoding="utf-8"), f2.read_text(encoding="utf-8"))
        r["name"] = f1.name
        rows.append(r)
    if not rows:
        sys.exit("no overlapping files between the two passes")

    agg = aggregate(rows)
    cer = agg["cer_grapheme_micro"] * 100
    print(f"lines compared           {agg['lines']:>6}")
    print(f"grapheme disagreement    {cer:>6.2f}%   <- transcription error floor")
    print(f"word disagreement        {agg['wer_micro'] * 100:>6.2f}%")
    print()
    if cer > 1.0:
        print("Above 1%: tighten the transcription rules and redo. A model\n"
              "difference smaller than this floor cannot be claimed.")
    else:
        print("Report this figure in the paper. It bounds the smallest model\n"
              "difference the test set can distinguish.")
    worst = sorted(rows, key=lambda r: -r["grapheme_edits"])[:8]
    print("\nMost disagreed lines (adjudicate these first):")
    for r in worst:
        if r["grapheme_edits"]:
            print(f"  {r['name']:<34} {r['grapheme_edits']:>3} edits")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("segment", help="pages -> line images + empty transcriptions")
    s.add_argument("--pages", required=True)
    s.add_argument("--out", default="testset")
    s.add_argument("--stratum", default="unspecified",
                   help="newsprint | book | form | signage | degraded")
    s.set_defaults(func=segment)

    b = sub.add_parser("bootstrap", help="pre-fill transcriptions for correction")
    b.add_argument("--out", default="testset")
    b.add_argument("--model", default="tam",
                   help="use stock 'tam', NOT the model you are evaluating")
    b.set_defaults(func=bootstrap)

    c = sub.add_parser("check", help="completeness and encoding check")
    c.add_argument("--out", default="testset")
    c.set_defaults(func=check)

    a = sub.add_parser("agreement", help="double-entry disagreement rate")
    a.add_argument("--pass1", required=True)
    a.add_argument("--pass2", required=True)
    a.set_defaults(func=agreement)

    args = ap.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
