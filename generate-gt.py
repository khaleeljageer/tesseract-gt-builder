import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from config import DefaultTamilConfig

# Configuration
FONT_DIR = "fonts1"
TEXT_FILE = "data/training-data.txt"
TMP_DIR = "tmp"
LINE_OUTPUT_DIR = "/home/khaleeljageer/tess_train/tesstrain/data/tam_new-ground-truth"
DPI = 300

font_config = DefaultTamilConfig()
PADDING = font_config.PADDING
FONT_SIZE = font_config.FONT_SIZE
A4_WIDTH_MM = font_config.A4_WIDTH_MM
A4_HEIGHT_MM = font_config.A4_HEIGHT_MM
MM_PER_INCH = font_config.MM_PER_INCH
A4_WIDTH = font_config.A4_WIDTH
A4_HEIGHT = font_config.A4_HEIGHT
LINE_SPACING = font_config.LINE_SPACING
LINES_PER_PAGE = font_config.LINES_PER_PAGE


def load_fonts(font_dir):
    fonts = []
    for file in os.listdir(font_dir):
        if file.endswith((".ttf", ".otf")):
            font_path = os.path.join(font_dir, file)
            try:
                font = ImageFont.truetype(font_path, FONT_SIZE)
                fonts.append(font)
            except Exception as e:
                print(f"Error loading font {file}: {e}")
    return fonts


def create_a4_tiff_image(lines, fonts, output_path):
    image = Image.new("L", (A4_WIDTH, A4_HEIGHT), 255)
    draw = ImageDraw.Draw(image)
    max_line_height = max([draw.textbbox((0, 0), "Sample", font=font)[3] for font in fonts])
    line_height = max_line_height + LINE_SPACING

    for i, line in enumerate(lines):
        font = random.choice(fonts)
        y = PADDING + i * line_height
        if y + max_line_height + PADDING > A4_HEIGHT:
            break
        draw.text((PADDING, y), line, font=font, fill=0)

    image.save(output_path, "TIFF", dpi=(DPI, DPI))


def create_ground_truth(lines, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def segment_lines_using_projection(image_path: str, output_dir: str, gt_lines, base_name, padding: int = 3):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape
        _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        projection = np.sum(binary, axis=1)
        in_line = False
        line_bounds = []
        threshold = 10

        for i, val in enumerate(projection):
            if val > threshold and not in_line:
                start = i
                in_line = True
            elif val <= threshold and in_line:
                end = i
                in_line = False
                line_bounds.append((start, end))

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for idx, (y1, y2) in enumerate(line_bounds):
            y1_pad = max(0, y1 - padding)
            y2_pad = min(height, y2 + padding)
            line_crop = image[y1_pad:y2_pad, :]
            _, line_thresh = cv2.threshold(line_crop, 200, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(255 - line_thresh)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                x1 = max(0, x - padding)
                x2 = min(line_crop.shape[1], x + w + padding)
                trimmed = line_crop[:, x1:x2]
            else:
                trimmed = line_crop

            line_text = gt_lines[idx] if idx < len(gt_lines) else ""
            if line_text.strip():
                image_out_path = Path(output_dir) / f"{base_name}_line_{idx + 1:03d}.tif"
                gt_out_path = Path(output_dir) / f"{base_name}_line_{idx + 1:03d}.gt.txt"
                cv2.imwrite(str(image_out_path), trimmed)
                with open(gt_out_path, "w", encoding="utf-8") as f:
                    f.write(line_text.strip())
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def validate_output(directory):
    all_files = os.listdir(directory)
    image_files = {f[:-5] for f in all_files if f.endswith(".tif")}
    gt_files = {f[:-7] for f in all_files if f.endswith(".gt.txt")}

    missing_images = gt_files - image_files
    missing_gts = image_files - gt_files

    if missing_images:
        print("Missing TIFF images for:")
        for name in sorted(missing_images):
            print(f"  {name}.gt.txt")

    if missing_gts:
        print("Missing GT files for:")
        for name in sorted(missing_gts):
            print(f"  {name}.tif")

    if not missing_images and not missing_gts:
        print("Validation successful: all GT and TIFF files matched.")


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(LINE_OUTPUT_DIR, exist_ok=True)

    fonts = load_fonts(FONT_DIR)
    if not fonts:
        print("No valid fonts found.")
        return

    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        all_lines = [line.strip() for line in f if line.strip()]

    total_lines = len(all_lines)
    total_fonts = len(fonts)

    print(f"[INFO] Total lines: {total_lines}, Fonts: {total_fonts}")
    total_pages = total_lines // LINES_PER_PAGE + (1 if total_lines % LINES_PER_PAGE else 0)

    for page_num in tqdm(range(total_pages), desc="Processing Pages"):
        start = page_num * LINES_PER_PAGE
        page_lines = all_lines[start:start + LINES_PER_PAGE]
        assigned_fonts = [fonts[i % total_fonts] for i in range(start, start + len(page_lines))]

        base_name = f"page_{page_num + 1:06d}"
        image_path = os.path.join(TMP_DIR, base_name + ".tif")
        gt_path = os.path.join(TMP_DIR, base_name + ".gt.txt")

        try:
            # Draw each line with its assigned font
            create_a4_tiff_image(page_lines, assigned_fonts, image_path)
            create_ground_truth(page_lines, gt_path)
            segment_lines_using_projection(image_path, LINE_OUTPUT_DIR, page_lines, base_name)
        except Exception as e:
            print(f"[ERROR] Failed processing {base_name}: {e}")

    shutil.rmtree(TMP_DIR)


if __name__ == "__main__":
    main()
