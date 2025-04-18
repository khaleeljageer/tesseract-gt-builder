import os

# Base directory where generated dataset is stored
dataset_base_dir = "Zenode_DataSet"

# Counters for validation
total_checked = 0
missing_pairs = []
empty_gt_files = []

# Iterate over category directories (Hangual_Fonts, Printed_Fonts)
for category in os.listdir(dataset_base_dir):
    category_path = os.path.join(dataset_base_dir, category)

    if os.path.isdir(category_path):
        # Iterate over each font directory
        for font in os.listdir(category_path):
            font_path = os.path.join(category_path, font)
            gt_dir = os.path.join(font_path, "gt")
            images_dir = os.path.join(font_path, "images")

            if not os.path.exists(gt_dir) or not os.path.exists(images_dir):
                print(f"Warning: Missing gt or images folder in {font_path}")
                continue

            # Get lists of generated files
            gt_files = {f.replace(".gt.txt", "") for f in os.listdir(gt_dir) if f.endswith(".gt.txt")}
            image_files = {f.replace(".tiff", "") for f in os.listdir(images_dir) if f.endswith(".tiff")}

            # Check for missing pairs
            missing_in_gt = image_files - gt_files
            missing_in_images = gt_files - image_files

            for missing in missing_in_gt:
                missing_pairs.append(f"Missing GT file: {missing}.gt.txt in {gt_dir}")

            for missing in missing_in_images:
                missing_pairs.append(f"Missing TIFF file: {missing}.tiff in {images_dir}")

            # Check for empty GT files
            for gt_file in gt_files:
                gt_path = os.path.join(gt_dir, gt_file + ".gt.txt")
                if os.path.exists(gt_path) and os.path.getsize(gt_path) == 0:
                    empty_gt_files.append(f"Empty GT file: {gt_path}")

            total_checked += len(gt_files)

# Summary Report
print(f"\nVerification Completed! Total pairs checked: {total_checked}")
if missing_pairs:
    print(f"\n⚠️ Missing Pairs ({len(missing_pairs)}):")
    print("\n".join(missing_pairs[:10]))  # Show only first 10 for brevity
    if len(missing_pairs) > 10:
        print(f"... and {len(missing_pairs) - 10} more.")

if empty_gt_files:
    print(f"\n⚠️ Empty GT Files ({len(empty_gt_files)}):")
    print("\n".join(empty_gt_files[:10]))  # Show only first 10
    if len(empty_gt_files) > 10:
        print(f"... and {len(empty_gt_files) - 10} more.")

if not missing_pairs and not empty_gt_files:
    print("\n✅ All TIFF and GT files are correctly paired and valid!")
