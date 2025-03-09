# Tamil Text Processor and Image Generator

This project processes Tamil text files and generates ground truth (GT) files and TIFF images for each line of text using various fonts.

## Requirements

- Python 3.x
- pip

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/khaleeljageer/tamil-text-processor.git
    cd tamil-text-processor
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Normalize Text File

The `normalize-gt.py` script reads a Tamil text file (`wikisource-ta.txt`) and processes it to ensure each line has no more than 7 words. The output is saved to `data/training-data.txt`.

To run the script:
```sh
python normalize-gt.py
```

### Generate Ground Truth and TIFF Images

The `generate-gt.py` script reads the processed text file (`data/training-data.txt`) and generates TIFF images and GT files for each line of text using various fonts from the `fonts` directory. The output is saved to the `tam-ground-truth` directory.

To run the script:
```sh
python generate-gt.py
```

## Project Structure

- `normalize-gt.py`: Script to normalize the text file.
- `generate-gt.py`: Script to generate TIFF images and GT files.
- `requirements.txt`: List of required Python packages.
- `data/`: Directory containing the processed text file.
- `fonts/`: Directory containing TTF font files.
- `tam-ground-truth/`: Directory where the generated TIFF images and GT files are saved.

## Dependencies

- `pillow`: Python Imaging Library (PIL) fork.
- `tqdm`: A fast, extensible progress bar for Python.
- `open-tamil`: Tamil text processing library.

## License

This project is licensed under the MIT License.