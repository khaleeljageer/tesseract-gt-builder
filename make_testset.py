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
    #    (page images or PDFs; PDF pages are rasterised at 300 dpi)

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
import hashlib
import re
import shutil
import tempfile
import subprocess
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

MIN_LINE_HEIGHT = 8         # px; below this a band is noise, not a line
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
    cutoff loses one side of the page.

    The block size scales with the image, because this script sees both
    2300-px full-page scans and 160-px paragraph crops, and a fixed block is
    wrong for one of them. The median blur and opening remove the speckle that
    aged paper leaves behind, which would otherwise fill the inter-line
    valleys the projection profile depends on.
    """
    g = cv2.medianBlur(gray, 3)
    block = max(15, (min(gray.shape) // 30) | 1)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block, 15)
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _profile(binary):
    proj = (binary > 0).sum(axis=1).astype(float)
    k = max(3, binary.shape[0] // 150)
    return np.convolve(proj, np.ones(k) / k, mode="same")


def _otsu_level(proj):
    """Threshold the projection by Otsu rather than a fixed fraction of its
    peak. A fixed fraction fails on real pages: photographed paper leaves a
    non-zero floor of dark pixels in every row, so the valleys between lines
    never fall below a 6%-of-peak cut and the whole text block is returned as
    one band. Otsu finds the cut from the profile's own two modes."""
    if proj.max() <= 0:
        return None
    norm = (255 * proj / proj.max()).astype(np.uint8)
    t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t / 255.0 * proj.max()


def _extend(seeds, proj, low):
    """Grow Otsu seeds to the full extent of each line.

    Otsu isolates the dense core of a line -- roughly the x-height band -- and
    clips ascenders, descenders and the Tamil vowel signs that sit above and
    below it. Growing each seed down to a fixed lower level instead merges
    neighbours wherever the leading is tight. So the boundary between two
    consecutive seeds is placed at the valley floor of the projection, and
    each band is then trimmed inward past rows that carry no ink.
    """
    if not seeds:
        return []
    n = len(proj)
    cuts = [0]
    for i in range(len(seeds) - 1):
        a, b = seeds[i][1], seeds[i + 1][0]
        cuts.append(a + int(np.argmin(proj[a:b])) if b > a else a)
    cuts.append(n)
    out = []
    for i in range(len(seeds)):
        s, e = cuts[i], cuts[i + 1]
        while s < e and proj[s] <= low:
            s += 1
        while e > s and proj[e - 1] <= low:
            e -= 1
        if e > s:
            out.append([s, e])
    return out


def _runs(proj, thresh):
    out, inside, start = [], False, 0
    for i, v in enumerate(proj):
        if v > thresh and not inside:
            start, inside = i, True
        elif v <= thresh and inside:
            out.append([start, i])
            inside = False
    if inside:
        out.append([start, len(proj)])
    return out


def _split_tall(band, proj, median_h):
    """Cut a band that spans several lines at the projection minima."""
    s, e = band
    h = e - s
    if median_h <= 0 or h < median_h * 1.6:
        return [band]
    n = max(2, int(h / median_h + 0.35))
    seg = proj[s:e]
    cuts = []
    for j in range(1, n):
        c = int(round(j * h / n))
        w = max(2, int(median_h * 0.3))
        lo, hi = max(1, c - w), min(h - 1, c + w)
        if hi <= lo:
            continue
        cuts.append(s + lo + int(np.argmin(seg[lo:hi])))
    pts = [s] + sorted(set(cuts)) + [e]
    return [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)
            if pts[i + 1] - pts[i] > 2]


def find_lines(binary):
    """Horizontal projection profile -> one band per text line.

    Merging short bands into their neighbour matters for Tamil: the pulli and
    vowel signs sit above the letter bodies and can form a band of their own,
    exactly the effect that costs the synthetic pipeline 0.075% of its pages.
    """
    proj = _profile(binary)
    level = _otsu_level(proj)
    if level is None:
        return []
    seeds = _runs(proj, level)
    if not seeds:
        return []
    bands = _extend(seeds, proj, level * 0.20)
    if not bands:
        return []

    hs = sorted(b[1] - b[0] for b in bands)
    median_h = hs[len(hs) // 2]
    split = []
    for b in bands:
        split.extend(_split_tall(b, proj, median_h))
    bands = split
    hs = sorted(b[1] - b[0] for b in bands)
    median_h = hs[len(hs) // 2]

    merged = [bands[0]]
    for s, e in bands[1:]:
        gap = s - merged[-1][1]
        short = ((e - s) < median_h * 0.6
                 or (merged[-1][1] - merged[-1][0]) < median_h * 0.6)
        if gap < median_h * 0.5 and short:      # diacritic band
            merged[-1][1] = e
        else:
            merged.append([s, e])

    total_ink = (binary > 0).sum()
    keep = []
    for s, e in merged:
        if e - s < max(MIN_LINE_HEIGHT, median_h * 0.45):
            continue
        if (binary[s:e] > 0).sum() < total_ink * MIN_INK_FRACTION:
            continue
        keep.append((s, e))
    return keep


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
RENDER_DPI = 300


def _rasterise(pdf, workdir, dpi=RENDER_DPI):
    """Render each page of a PDF to a grayscale image.

    A born-digital PDF has no pixels until something renders it, and the DPI
    that render happens at decides the recognition problem. Rasterising at
    300 puts these pages on the same footing as the scans; taking whatever
    the embedded JPEG happens to be (150 dpi in this material) would not.
    """
    if not shutil.which("pdftoppm"):
        sys.exit("pdftoppm not found -- install poppler-utils to read PDF pages")
    stem = re.sub(r"[^A-Za-z0-9]+", "_", pdf.stem)[:28]
    prefix = Path(workdir) / stem
    r = subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-png",
                        str(pdf), str(prefix)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[warn] could not render {pdf.name}: {r.stderr.strip()[:80]}")
        return []
    return sorted(Path(workdir).glob(f"{stem}*.png"))


def _collect_pages(src, workdir, dpi=RENDER_DPI):
    """Every page to segment, as (image path, name to record, bytes to hash).

    PDF pages are hashed on their rendered pixels rather than the container,
    so the same page reaching us once as a PDF and once as a scan is still
    caught by the duplicate guard.
    """
    src = Path(src)
    out = []
    for p in sorted(src.iterdir()):
        if p.suffix.lower() in IMAGE_SUFFIXES:
            out.append((p, p.name, p.read_bytes()))
        elif p.suffix.lower() == ".pdf":
            rendered = _rasterise(p, workdir, dpi)
            multi = len(rendered) > 1
            for i, img in enumerate(rendered, 1):
                name = f"{p.name}#{i}" if multi else p.name
                out.append((img, name, img.read_bytes()))
    return out


def _normalise_polarity(crop):
    """Flip reverse-printed lines to dark-on-light.

    Display typography knocks text out of a coloured panel, and 17% of the
    lines in the signage material here are set that way. Every OCR engine
    expects dark ink on light ground, so leaving the polarity alone would
    measure the engine's handling of inverted images rather than its Tamil
    recognition. The flip is recorded per line in the manifest so the
    proportion can be reported rather than hidden.
    """
    if crop.size == 0:
        return crop, False
    level, _ = cv2.threshold(crop, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = float((crop < level).mean())
    # Ink covers roughly a fifth of a normal line crop and the ground covers
    # the rest, so the test is which class dominates -- not whether the crop
    # is dark overall. An underexposed photograph of light paper sits near
    # 0.55 and must not be flipped; knocked-out text sits above 0.80.
    if dark <= 0.65:
        return crop, False
    return cv2.bitwise_not(crop), True


def _seen_digests(out):
    """Map md5 -> source page name for every page already in the manifest."""
    manifest = out / "manifest.csv"
    if not manifest.exists():
        return {}
    seen = {}
    with open(manifest, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("page_md5"):
                seen.setdefault(row["page_md5"], row["source_page"])
    return seen


def segment(args):
    workdir = tempfile.mkdtemp(prefix="testset-render-")
    try:
        return _segment(args, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _segment(args, workdir):
    pages = _collect_pages(args.pages, workdir, getattr(args, "dpi", RENDER_DPI))
    if not pages:
        sys.exit(f"no page images or PDFs in {args.pages}")

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    new = not manifest.exists()
    fh = open(manifest, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["line_id", "source_page", "stratum", "skew_deg",
                    "y0", "y1", "height_px", "page_md5", "inverted"])

    # A page contributed twice -- the same file copied into two strata, a
    # duplicated capture -- would put identical lines in the test set twice and
    # silently double their weight in the aggregate CER. Refuse it.
    seen = _seen_digests(out)

    total = skipped = 0
    for page, page_name, raw in pages:
        digest = hashlib.md5(raw).hexdigest()
        if digest in seen:
            print(f"  {page_name:<34} duplicate of {seen[digest]}, skipped")
            skipped += 1
            continue
        seen[digest] = page_name

        gray = cv2.imread(str(page), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[warn] unreadable {page_name}")
            continue
        gray, angle = deskew(gray)
        binary = binarise(gray)
        bands = find_lines(binary)
        stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(page_name).stem)[:28]

        for i, (y0, y1) in enumerate(bands, 1):
            crop = gray[max(0, y0 - PAD):min(gray.shape[0], y1 + PAD), :]
            cols = np.where((binarise(crop) > 0).sum(axis=0) > 0)[0]
            if cols.size:
                crop = crop[:, max(0, cols[0] - PAD):min(crop.shape[1], cols[-1] + PAD)]
            crop, inverted = _normalise_polarity(crop)
            lid = f"{stem}_l{i:03d}"
            cv2.imwrite(str(out / "images" / f"{lid}.tif"), crop)
            gt = out / "gt" / f"{lid}.gt.txt"
            if not gt.exists():
                gt.write_text("", encoding="utf-8")
            w.writerow([lid, page_name, args.stratum, f"{angle:.2f}",
                        y0, y1, y1 - y0, digest, int(inverted)])
            total += 1
        print(f"  {page_name:<34} skew {angle:+5.2f}°  {len(bands):>3} lines")

    fh.close()
    if skipped:
        print(f"\n{skipped} duplicate page(s) skipped")
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
    s.add_argument("--dpi", type=int, default=RENDER_DPI,
                   help="resolution to rasterise PDF pages at (default 300)")
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
