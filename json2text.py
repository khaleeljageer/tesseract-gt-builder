import json
import re


# Function to remove English letters and keep Tamil and useful punctuation
def clean_tamil_text(text):
    # Remove English alphabets and digits
    text = re.sub(r'[a-zA-Z]', '', text)
    # Optionally remove special characters except basic punctuation
    text = re.sub(r'[^\u0B80-\u0BFF\s.,:;!?()\[\]\'"–-]', '', text)
    return text.strip()


# Read JSON data
with open("tamil-articles-from-wikinews.json", "r",
          encoding="utf-8") as json_file:
    data = json.load(json_file)

# Process and clean each item
cleaned_lines = []
for item in data:
    title = clean_tamil_text(str(item.get("title", "")))
    text = clean_tamil_text(str(item.get("text", "")))
    # target_text = clean_tamil_text(item.get("targets", ""))
    if title:
        # Format: input[TAB]target
        # cleaned_lines.append(input_text + " " + target_text)
        cleaned_lines.append(title+" " + text)

# Write to output file
with open("raw_data/wikinews-ta.txt", "w", encoding="utf-8") as output_file:
    for line in cleaned_lines:
        output_file.write(line)
