import os
from multiprocessing import Pool, cpu_count
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Paths
input_text_file = "data/training-data.txt"  # Text file with 99,621 lines
fonts_dir = "fonts"  # Folder containing subdirectories 'print' and 'hangual'
output_base_dir = "tam-ground-truth"  # Output base directory

# Ensure output base directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Load fonts from subdirectories
font_categories = {"print": [], "hangual": []}
for category in font_categories.keys():
    category_path = os.path.join(fonts_dir, category)
    if os.path.exists(category_path):
        font_categories[category] = [os.path.join(category_path, f) for f in os.listdir(category_path) if
                                     f.endswith(".ttf")]

# Ensure there are fonts available
if not any(font_categories.values()):
    raise FileNotFoundError("No TTF font files found in 'print' or 'hangual' directories!")

# Read input text
print(f"Reading input text from {input_text_file}")
with open(input_text_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

print(f"Found {len(lines)} lines of text to process")


def create_tiff_image(args):
    text, font_path, image_path, gt_path = args
    try:
        font_size = 32  # Adjust font size as needed
        font = ImageFont.truetype(font_path, font_size)

        # Determine text bounding box
        bbox = font.getbbox(text)
        text_width, text_height = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])

        # Create image with initial padding
        image = Image.new("L", (text_width + 20, text_height + 10), 255)  # White background
        draw = ImageDraw.Draw(image)
        text_x = (image.width - text_width) // 2
        text_y = (image.height - text_height) // 2
        draw.text((text_x, text_y), text, font=font, fill=0)  # Black text

        # Save as TIFF
        image.save(image_path, format="TIFF")

        # Create a GT file for this line
        with open(gt_path, "w", encoding="utf-8") as gt_file:
            gt_file.write(text)

        return True, None
    except Exception as e:
        return False, str(e)


# Prepare arguments for parallel processing
tasks = []
for category, font_paths in font_categories.items():
    output_dir = os.path.join(output_base_dir, category)
    os.makedirs(output_dir, exist_ok=True)

    for idx, line in enumerate(lines):
        line = line.strip()
        if line:
            for font_idx, font_path in enumerate(font_paths):
                gt_filename = f"tam_{idx:06d}_font{font_idx:03d}.gt.txt"
                gt_path = os.path.join(output_dir, gt_filename)
                image_filename = f"tam_{idx:06d}_font{font_idx:03d}.tiff"
                image_path = os.path.join(output_dir, image_filename)
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
