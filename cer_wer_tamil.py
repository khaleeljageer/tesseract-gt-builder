import sys
import unicodedata
import argparse
from jiwer import wer
from difflib import SequenceMatcher


def normalize(text):
    """Normalize Unicode and remove unwanted characters"""
    text = unicodedata.normalize('NFKC', text)
    text = text.strip()
    return text


def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate (CER)"""
    reference = normalize(reference)
    hypothesis = normalize(hypothesis)

    matcher = SequenceMatcher(None, reference, hypothesis)
    distance = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            distance += (i2 - i1)  # number of characters changed/deleted/inserted

    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    return distance / len(reference)



def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate (WER)"""
    reference = normalize(reference)
    hypothesis = normalize(hypothesis)

    return wer(reference, hypothesis)


def read_file(file_path):
    """Read a UTF-8 file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="Calculate CER and WER for Tamil OCR evaluation.")
    parser.add_argument("--ground_truth", required=True, help="Path to the ground truth text file")
    parser.add_argument("--prediction", required=True, help="Path to the prediction text file")

    args = parser.parse_args()

    ground_truth = read_file(args.ground_truth)
    prediction = read_file(args.prediction)

    cer = calculate_cer(ground_truth, prediction)
    wer_score = calculate_wer(ground_truth, prediction)

    print(f"Character Error Rate (CER): {cer * 100:.2f}%")
    print(f"Word Error Rate (WER): {wer_score * 100:.2f}%")


if __name__ == "__main__":
    main()
