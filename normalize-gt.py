"""
Tamil Text Processor
This script reads wikisource-ta.txt from the current directory and creates
data/training-data.txt where each line has no more than 7 words.
"""

import os
from tqdm import tqdm
import time


def process_file(input_file="wikisource-ta.txt", output_file="data/training-data.txt", words_per_line=7):
    """
    Process a text file to ensure each line has no more than the specified number of words.

    Args:
        input_file (str): Path to the input text file
        output_file (str): Path to the output text file
        words_per_line (int): Maximum number of words per line
    """
    try:
        # Read the input file
        print(f"Reading file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split the content into words
        words = content.split()
        total_words = len(words)
        print(f"Total words found: {total_words}")

        # Process the words into lines with progress bar
        lines = []
        current_line = []

        print("Processing words...")
        for word in tqdm(words, desc="Formatting text", unit="words"):
            current_line.append(word)
            if len(current_line) >= words_per_line:
                lines.append(' '.join(current_line))
                current_line = []

        # Add the remaining words as the last line if any
        if current_line:
            lines.append(' '.join(current_line))

        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write to the output file with progress bar
        print(f"Writing to file: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in tqdm(lines, desc="Writing to file", unit="lines"):
                f.write(line + '\n')

        print(f"\nProcessing complete!")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Input file had {total_words} words, now formatted into {len(lines)} lines")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    start_time = time.time()
    process_file()
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
