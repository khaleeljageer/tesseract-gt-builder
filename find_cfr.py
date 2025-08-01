from tamil import utf8
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import mplcairo

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

def plot_top_frequencies(counter, title, n=10, output_filename="char_freq_graph.png"):
    """Plot the top n items from a Counter object and save to a file."""
    top_items = counter.most_common(n)
    chars = [item[0] for item in top_items]
    freqs = [item[1] for item in top_items]
    
    # Path to a Tamil-supporting font
    font_path = 'fonts/NotoSerifTamil.ttf'
    try:
        tamil_font = fm.FontProperties(fname=font_path)
    except RuntimeError:
        print(f"Warning: Font not found at {font_path}. Using default font.")
        tamil_font = fm.FontProperties()


    mplcairo.set_options(raqm=True)

    plt.figure(figsize=(10, 6))
    plt.bar(chars, freqs, color='blue')

    plt.xlabel('Characters', fontproperties=tamil_font)
    plt.ylabel('Frequency')
    plt.title(title, fontproperties=tamil_font)

    for i, v in enumerate(freqs):
        plt.text(i, v + 0.05, str(v), ha='center', fontproperties=tamil_font)

    plt.xticks(chars, fontproperties=tamil_font)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"\nGraph saved to {output_filename}")


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

    # Plot character frequencies
    plot_top_frequencies(char_freq, "Top 10 Character Frequencies", n=10)


if __name__ == "__main__":
    main()
