import os
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Paths
input_text_file = "data/training-data.txt"  # Text file with 99,621 lines
fonts_dir = "fonts"  # Folder containing TTF font files
output_dir = "tam-ground-truth"  # Output directory for images and GT files

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)

# Load fonts
font_files = [f for f in os.listdir(fonts_dir) if f.endswith(".ttf")]
if not font_files:
    raise FileNotFoundError("No TTF font files found in the 'fonts' directory!")

print(f"Found {len(font_files)} font files to use")

# Read input text
print(f"Reading input text from {input_text_file}")
with open(input_text_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

print(f"Found {len(lines)} lines of text to process")


# Function to generate a TIFF image from text
def create_tiff_image(text, font_path, image_path):
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

    # Check if additional padding is needed
    bbox_after_draw = draw.textbbox((text_x, text_y), text, font=font)
    top_padding = max(10 - bbox_after_draw[1], 0)
    bottom_padding = max(10 - (image.height - bbox_after_draw[3]), 0)

    if top_padding > 0 or bottom_padding > 0:
        image_height = text_height + top_padding + bottom_padding
        image = Image.new("L", (text_width + 20, int(image_height)), 255)  # White background
        draw = ImageDraw.Draw(image)
        text_y = top_padding
        draw.text((text_x, text_y), text, font=font, fill=0)  # Black text

    # Save as TIFF
    image.save(image_path, format="TIFF")


# Process each line and create TIFF + GT files for each font
valid_lines = [line.strip() for line in lines if line.strip()]
print(f"Processing {len(valid_lines)} non-empty lines with {len(font_files)} fonts each")
print(f"Total images to generate: {len(valid_lines) * len(font_files)}")

# Initialize counters for statistics
success_count = 0
error_count = 0
error_details = []

# Create progress bar for the total number of combinations
total_combinations = len(valid_lines) * len(font_files)
pbar = tqdm(total=total_combinations, desc="Generating TIFF files", unit="images")

# Process each line with each font
for idx, line in enumerate(valid_lines[:800]):
    # Now create an image for each font
    for font_idx, font_file in enumerate(font_files):
        try:
            font_path = os.path.join(fonts_dir, font_file)
            # Create a GT file for this line
            gt_filename = f"tam_{idx:06d}_font{font_idx:03d}.gt.txt"
            gt_path = os.path.join(output_dir, gt_filename)
            with open(gt_path, "w", encoding="utf-8") as gt_file:
                gt_file.write(line)

            # Create a unique filename that includes both line and font index
            image_filename = f"tam_{idx:06d}_font{font_idx:03d}.tiff"
            image_path = os.path.join(output_dir, image_filename)

            # Generate TIFF image with this font
            create_tiff_image(line, font_path, image_path)

            success_count += 1
        except Exception as e:
            error_count += 1
            error_details.append((idx, font_file, str(e)))
        finally:
            # Update progress bar regardless of success/failure
            pbar.update(1)

# Close the progress bar
pbar.close()

# Print summary
print("\nTIFF and GT file generation completed!")
print(f"Successfully processed: {success_count} images")
if error_count > 0:
    print(f"Errors encountered: {error_count} images")
    print("First few errors:")
    for i, (line_idx, font_name, error) in enumerate(error_details[:5]):
        print(f"  Line {line_idx}, Font '{font_name}': {error}")
    if len(error_details) > 5:
        print(f"  ... and {len(error_details) - 5} more errors")
