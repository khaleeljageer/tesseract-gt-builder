"""Parameterised rendering and line segmentation.

A refactor of generate-gt.py that takes the font set, the text, and the output
directory as arguments so ablation variants can be generated independently.
Behaviour is otherwise identical, with two deliberate corrections:

1. FONT ASSIGNMENT. generate-gt.py computes a round-robin assignment in
   main() and then discards it: create_a4_tiff_image() calls
   random.choice(fonts) on the per-page window instead of using the pairing
   it was given. The module-level `random` is also never seeded. The corpus
   is therefore not reproducible, and font usage is not balanced (simulated
   spread across 27 fonts is roughly 8%).

   Here `assignment="round-robin"` uses the pairing directly and is exactly
   balanced and deterministic; `assignment="random"` reproduces the original
   behaviour but takes an explicit seed. Ablations must use round-robin —
   with random assignment two variants differ by more than the variable under
   test.

2. LINE HEIGHT PROBE. generate-gt.py measures line height from the Latin
   string "Sample", which does not exercise Tamil ascenders, descenders or
   the pulli. A Tamil probe is used instead so tall glyphs are not clipped.
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DefaultTamilConfig  # noqa: E402

# Exercises ascender, descender, pulli and a two-part vowel sign.
HEIGHT_PROBE = "ஆழ்ந்த கூஜா ஜஸ்ரீ"

BINARY_THRESHOLD = 200      # grayscale cutoff for inverse binarisation
PROJECTION_THRESHOLD = 10   # row-sum above which a row counts as ink
CROP_PADDING = 3


def load_fonts(font_dir, size, names=None):
    """Load fonts in deterministic (sorted) order.

    names, when given, selects a subset by filename stem. Sorting matters:
    os.listdir order is filesystem-dependent, so the original code's font
    indexing was not portable between machines.
    """
    paths = sorted(p for p in Path(font_dir).iterdir()
                   if p.suffix.lower() in (".ttf", ".otf"))
    if names is not None:
        wanted = set(names)
        paths = [p for p in paths if p.stem in wanted]
        missing = wanted - {p.stem for p in paths}
        if missing:
            raise FileNotFoundError(f"fonts not found in {font_dir}: {sorted(missing)}")

    fonts = []
    for path in paths:
        try:
            fonts.append((path.stem, ImageFont.truetype(str(path), size)))
        except OSError as exc:
            print(f"[warn] skipping unreadable font {path.name}: {exc}")
    if not fonts:
        raise RuntimeError(f"no usable fonts in {font_dir}")
    return fonts


def render_page(lines, page_fonts, out_path, cfg):
    """Draw one A4 page, one line per row, using the supplied font pairing."""
    image = Image.new("L", (cfg.A4_WIDTH, cfg.A4_HEIGHT), 255)
    draw = ImageDraw.Draw(image)

    probe = max(draw.textbbox((0, 0), HEIGHT_PROBE, font=f)[3] for _, f in page_fonts)
    line_height = probe + cfg.LINE_SPACING

    drawn = []
    for i, (line, (_, font)) in enumerate(zip(lines, page_fonts)):
        y = cfg.PADDING + i * line_height
        if y + probe + cfg.PADDING > cfg.A4_HEIGHT:
            break
        draw.text((cfg.PADDING, y), line, font=font, fill=0)
        drawn.append(line)

    image.save(out_path, "TIFF", dpi=(cfg.DPI, cfg.DPI))
    return drawn


def segment_page(image_path, out_dir, gt_lines, base_name, fonts_used):
    """Recover line crops from the rendered page by horizontal projection.

    Deliberately re-derives the lines from pixels rather than reusing the
    layout coordinates, so training crops carry the same geometry a segmenter
    produces at inference time. Returns the number of crops written.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[warn] unreadable page {image_path}")
        return 0
    height = image.shape[0]

    _, binary = cv2.threshold(image, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    projection = np.sum(binary, axis=1)

    bands, in_line, start = [], False, 0
    for i, val in enumerate(projection):
        if val > PROJECTION_THRESHOLD and not in_line:
            start, in_line = i, True
        elif val <= PROJECTION_THRESHOLD and in_line:
            bands.append((start, i))
            in_line = False
    if in_line:
        bands.append((start, height))

    if len(bands) != len(gt_lines):
        # Never silently misalign an image with the wrong transcription.
        print(f"[warn] {base_name}: {len(bands)} bands for {len(gt_lines)} lines; "
              f"skipping page")
        return 0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx, (y1, y2) in enumerate(bands):
        text = gt_lines[idx].strip()
        if not text:
            continue
        crop = image[max(0, y1 - CROP_PADDING):min(height, y2 + CROP_PADDING), :]
        _, thresh = cv2.threshold(crop, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(255 - thresh)
        if coords is not None:
            x, _, w, _ = cv2.boundingRect(coords)
            crop = crop[:, max(0, x - CROP_PADDING):min(crop.shape[1], x + w + CROP_PADDING)]

        stem = f"{base_name}_line_{idx + 1:03d}"
        cv2.imwrite(str(out_dir / f"{stem}.tif"), crop)
        (out_dir / f"{stem}.gt.txt").write_text(text, encoding="utf-8")
        written += 1

    return written


def generate(lines, out_dir, font_dir="fonts", font_names=None,
             assignment="round-robin", seed=0, cfg=None, keep_pages=False):
    """Render `lines` to paired .tif/.gt.txt crops under out_dir.

    Returns a manifest dict suitable for writing next to the corpus.
    """
    cfg = cfg or DefaultTamilConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page_dir = out_dir.parent / f"{out_dir.name}_pages"
    page_dir.mkdir(parents=True, exist_ok=True)

    fonts = load_fonts(font_dir, cfg.FONT_SIZE, font_names)
    n_fonts = len(fonts)

    if assignment == "round-robin":
        pick = lambda i: fonts[i % n_fonts]
    elif assignment == "random":
        import random as _random
        rng = _random.Random(seed)
        pick = lambda i: fonts[rng.randrange(n_fonts)]
    else:
        raise ValueError(f"unknown assignment {assignment!r}")

    lpp = cfg.LINES_PER_PAGE
    n_pages = (len(lines) + lpp - 1) // lpp
    total, font_counts = 0, {name: 0 for name, _ in fonts}

    for page in tqdm(range(n_pages), desc=f"render {out_dir.name}", unit="page"):
        start = page * lpp
        page_lines = lines[start:start + lpp]
        page_fonts = [pick(start + i) for i in range(len(page_lines))]

        base = f"page_{page + 1:06d}"
        img_path = page_dir / f"{base}.tif"

        drawn = render_page(page_lines, page_fonts, img_path, cfg)
        written = segment_page(img_path, out_dir, drawn, base, page_fonts)
        total += written
        for (name, _), line in zip(page_fonts, drawn):
            font_counts[name] += 1

        if not keep_pages:
            img_path.unlink(missing_ok=True)

    if not keep_pages:
        shutil.rmtree(page_dir, ignore_errors=True)

    manifest = {
        "output_dir": str(out_dir),
        "requested_lines": len(lines),
        "crops_written": total,
        "fonts": sorted(font_counts),
        "n_fonts": n_fonts,
        "font_line_counts": font_counts,
        "assignment": assignment,
        "seed": seed,
        "config": {
            "dpi": cfg.DPI, "font_size": cfg.FONT_SIZE,
            "line_spacing": cfg.LINE_SPACING, "lines_per_page": cfg.LINES_PER_PAGE,
            "padding": cfg.PADDING,
            "page_px": [cfg.A4_WIDTH, cfg.A4_HEIGHT],
        },
    }
    (out_dir.parent / f"{out_dir.name}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def validate(out_dir):
    """Every .tif has a .gt.txt and vice versa."""
    files = list(Path(out_dir).iterdir())
    imgs = {f.name[:-4] for f in files if f.name.endswith(".tif")}
    gts = {f.name[:-7] for f in files if f.name.endswith(".gt.txt")}
    return {"images": len(imgs), "transcriptions": len(gts),
            "orphan_images": sorted(imgs - gts)[:10],
            "orphan_transcriptions": sorted(gts - imgs)[:10]}
