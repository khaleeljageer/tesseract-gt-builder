from tamil import utf8
from collections import Counter


def load_tamil_text(file_path):
    """Load text from a file and return it as a string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return ""
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""


def get_character_frequency(text):
    """Calculate frequency of Tamil characters, excluding spaces and special chars."""
    # Get Tamil letters only
    tamil_chars = utf8.get_letters(text)
    # Filter out spaces and special characters, keeping only Tamil letters
    filtered_chars = [char for char in tamil_chars if utf8.istamil(char) and not char.isspace()]
    return Counter(filtered_chars)


def get_word_frequency(text):
    """Calculate frequency of Tamil words using open-tamil."""
    # Split text into words using open-tamil's word splitter
    words = utf8.get_words(text)
    # Filter to ensure only Tamil words (optional, if mixed text)
    tamil_words = [word for word in words if any(utf8.istamil(c) for c in word)]
    return Counter(tamil_words)


def print_top_frequencies(counter, title, n=10):
    """Print the top n items from a Counter object."""
    print(f"\n{title}:")
    print("-" * 50)
    for item, freq in counter.most_common(n):
        print(f"{item}: {freq} occurrences")


def main():
    # Specify your Tamil dataset file path here
    file_path = "data/training-data.txt"

    # Load the text
    tamil_text = load_tamil_text(file_path)
    if not tamil_text:
        return

    # Calculate frequencies
    char_freq = get_character_frequency(tamil_text)
    word_freq = get_word_frequency(tamil_text)

    # Print results
    print_top_frequencies(char_freq, "Top 10 Character Frequencies")
    print_top_frequencies(word_freq, "Top 10 Word Frequencies")


if __name__ == "__main__":
    main()
