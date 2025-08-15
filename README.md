# Tesseract GT Builder for Tamil OCR

This project provides a comprehensive suite of tools to generate ground truth (GT) data for Tesseract OCR for the Tamil language. It includes scripts for text normalization, image and GT file generation, data verification, and OCR evaluation.

## Features

-   **Text Normalization:** Pre-processes raw text files into a clean, trainable format.
-   **Ground Truth Generation:** Creates TIFF images and corresponding `.gt.txt` files for each line of text.
-   **Font Flexibility:** Uses a variety of Tamil fonts to generate diverse training data.
-   **Data Verification:** Includes a script to verify the integrity of the generated dataset.
-   **OCR Evaluation:** Provides tools to calculate Character Error Rate (CER) and Word Error Rate (WER).
-   **Frequency Analysis:** Scripts to analyze character and word frequencies in the dataset.

## Workflow

The project follows a clear workflow:

1.  **Data Preparation:** Raw text files (from `raw_data/` or other sources like JSON) are processed. `json2text.py` can be used to convert JSON data to text.
2.  **Text Normalization:** The `normalize-gt.py` script merges and normalizes the text data, creating `data/training-data.txt`.
3.  **GT Generation:** The `generate-gt.py` script takes the normalized text and generates `.tif` images and `.gt.txt` files in the `gt/` directory.
4.  **Verification:** The `verify.py` script can be adapted to check the consistency of the generated files in the `gt/` directory.
5.  **Evaluation:** After training a Tesseract model with the generated data, the `cer_wer_tamil.py` script can be used to evaluate its performance.
6.  **Analysis:** `find_cfr.py` can be used to analyze the character and word frequencies of the dataset and generate a graph of the character frequencies.

## Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/khaleeljageer/tesseract-gt-builder.git
    cd tesseract-gt-builder
    ```

2.  Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### 1. Prepare Your Data

-   Place your raw `.txt` files in the `raw_data/` directory.
-   If you have JSON data, you can use `json2text.py` to convert it to text. You might need to modify the script to fit your JSON structure.

### 2. Normalize the Text

Run the `normalize-gt.py` script to create the training data file:

```sh
python normalize-gt.py
```

This will create `data/training-data.txt`.

### 3. Generate Ground Truth Data

Run the `generate-gt.py` script to generate the images and GT files:

```sh
python generate-gt.py
```

The output will be saved in the `gt/` directory.

### 4. Verify the Generated Data

You can use the `verify.py` script to check for missing or empty files. You may need to modify the `dataset_base_dir` variable in the script to point to the `gt/` directory.

### 5. Evaluate Your OCR Model

After training your model, you can evaluate it using `cer_wer_tamil.py`:

```sh
python cer_wer_tamil.py --ground_truth /path/to/your/ground-truth.txt --prediction /path/to/your/ocr-output.txt
```

### 6. Analyze the Dataset

To analyze the character and word frequencies in your dataset, run:

```sh
python find_cfr.py
```

This will also generate a `char_freq_graph.png` file with a graph of the top 10 character frequencies.



## Project Structure

-   `config.py`: Configuration for the data generation process.
-   `generate-gt.py`: Main script to generate TIFF images and GT files.
-   `normalize-gt.py`: Script to normalize the text data.
-   `verify.py`: Script to verify the generated dataset.
-   `cer_wer_tamil.py`: Script to calculate CER and WER.
-   `find_cfr.py`: Script to find character and word frequencies and generate a frequency graph.
-   `json2text.py`: Utility to convert JSON to text.
-   `requirements.txt`: List of required Python packages.
-   `data/`: Directory for training data.
-   `raw_data/`: Directory for raw text files.
-   `fonts/`: Directory containing TTF font files.
-   `gt/`: Directory where the generated TIFF images and GT files are saved.
-   `cer_wer/`: Directory containing sample files for CER/WER calculation.
-   `model/`: Directory for trained models.

## Dependencies

-   Pillow
-   tqdm
-   open-tamil
-   jiwer
-   matplotlib
-   opencv-python
-   numpy

## Citation

If you use this repository or the associated dataset in your research, please cite:

**Dataset:**
```
@dataset{tamilocr_dataset_2025,
author = {Syedkhaleel Jageer},
title = {Synthetic OCR Dataset: 105,738 Tamil Text Lines Rendered in 27 Diverse Fonts with Corresponding Ground Truth Annotations},
year = {2025},
publisher = {Zenodo},
doi = {10.5281/zenodo.16881612},
url = {https://doi.org/10.5281/zenodo.16881612}
}
```

**Code Repository:**
```
@misc{jageer2025tesseractGTBuilder,
author = {Syedkhaleel Jageer},
title = {{Tesseract-GT-Builder: Tools to generate ground-truth data for Tesseract OCR (Tamil)}},
howpublished = {\url{https://github.com/khaleeljageer/tesseract-gt-builder}},
year = {2025},
note = {Accessed: August 15, 2025}
}
```

## License

This project is licensed under the MIT License.
