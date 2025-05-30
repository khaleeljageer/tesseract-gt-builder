import os
from multiprocessing import Pool, cpu_count
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Paths
input_text_file = "data/training-data.txt"  # Text file with 99,621 lines
fonts_base_dir = "fonts"  # Folder containing subdirectories 'Hangual_Fonts' and 'Printed_Fonts'
output_base_dir = "/home/khaleeljageer/tesseract_training/tesstrain/data/tamhng-ground-truth"  # Output base directory

# Ensure output base directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Load fonts directly from Hangual_Fonts and Printed_Fonts directories
font_categories = {"Hangual_Fonts": []}
for category in font_categories.keys():
    category_path = os.path.join(fonts_base_dir, category)
    if os.path.exists(category_path):
        font_files = [os.path.join(category_path, f) for f in os.listdir(category_path) if f.endswith(".ttf")]
        if font_files:
            font_categories[category] = font_files

# Ensure there are fonts available
if not any(font_categories.values()):
    raise FileNotFoundError("No TTF font files found in 'Hangual_Fonts' or 'Printed_Fonts' directories!")

# Read input text
print(f"Reading input text from {input_text_file}")
with open(input_text_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

print(f"Found {len(lines)} lines of text to process")


def create_tiff_image(args):
    text, font_path, image_path, gt_path = args
    try:
        font_size = 48
        font = ImageFont.truetype(font_path, font_size)

        # Determine text bounding box
        bbox = font.getbbox(text)
        text_width, text_height = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])

        # Create image with padding
        image = Image.new("L", (text_width + 20, text_height + 8), 255) # White background
        draw = ImageDraw.Draw(image)
        text_x = (image.width - text_width) // 2
        text_y = (image.height - text_height) // 2
        draw.text((text_x, text_y), text, font=font, fill=0)  # Black text

        # Save as TIFF
        image.save(image_path, format="TIFF", compression="tiff_lzw", dpi=(300, 300))

        # Create a GT file for this line
        with open(gt_path, "w", encoding="utf-8") as gt_file:
            gt_file.write(text)

        return True, None
    except Exception as e:
        return False, str(e)


# Prepare arguments for parallel processing
tasks = []
for category, font_paths in font_categories.items():
    # category_output_dir = os.path.join(output_base_dir, category)
    os.makedirs(output_base_dir, exist_ok=True)

    for font_path in font_paths:
        font_name = os.path.basename(font_path).replace(".ttf", "")
        # font_output_dir = os.path.join(category_output_dir, font_name)
        font_output_dir=output_base_dir
        # gt_dir = os.path.join(font_output_dir, "gt")
        gt_dir = font_output_dir
        # images_dir = os.path.join(font_output_dir, "images")
        images_dir = font_output_dir
        # os.makedirs(gt_dir, exist_ok=True)
        # os.makedirs(images_dir, exist_ok=True)

        for idx, line in enumerate(lines):
            line = line.strip()
            if line:
                unique_id = f"{font_name}_{idx + 1:05d}"
                gt_filename = f"{unique_id}.gt.txt"
                gt_path = os.path.join(gt_dir, gt_filename)
                image_filename = f"{unique_id}.tif"
                image_path = os.path.join(images_dir, image_filename)
                tasks.append((line, font_path, image_path, gt_path))

# Process tasks in parallel
core = cpu_count() - 1
print(f"Processing {len(tasks)} tasks using {core} CPU cores")
with Pool(core) as pool:
    results = list(
        tqdm(pool.imap(create_tiff_image, tasks), total=len(tasks), desc="Generating TIFF files", unit="images"))

# Summarize results
success_count = sum(1 for success, _ in results if success)
error_details = [(idx, font_path, error) for (success, error), (line, font_path, image_path, gt_path) in
                 zip(results, tasks) if not success]

print("\nTIFF and GT file generation completed!")
print(f"Successfully processed: {success_count} images")
if error_details:
    print(f"Errors encountered: {len(error_details)} images")
    print("First few errors:")
    for i, (line_idx, font_name, error) in enumerate(error_details[:5]):
        print(f"  Line {line_idx}, Font '{font_name}': {error}")
    if len(error_details) > 5:
        print(f"  ... and {len(error_details) - 5} more errors")
