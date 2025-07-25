import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import mplcairo

# Path to a Tamil-supporting font (replace with your font path)
font_path = 'fonts/NotoSerifTamil.ttf'  # Example: Noto Sans Tamil
# Register the font
tamil_font = fm.FontProperties(fname=font_path)

# Data from your top 10 character frequencies
chars = ['க', 'ன்', 'ம்', 'க்', 'த', 'து', 'வ', 'ல்', 'அ', 'த்']
freqs = [2.67, 2.31, 2.16, 1.91, 1.69, 1.65, 1.63, 1.56, 1.46, 1.45]

mplcairo.set_options(raqm=True)

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(chars, freqs, color='blue')

# Set font for labels and title
plt.xlabel('Characters', fontproperties=tamil_font)
plt.ylabel('Frequency (%)')
plt.title('Top 10 Character Frequencies in Tamil OCR Dataset', fontproperties=tamil_font)

# Add percentage labels above bars
for i, v in enumerate(freqs):
    plt.text(i, v + 0.05, f'{v}%', ha='center', fontproperties=tamil_font)

# Apply Tamil font to x-axis labels
plt.xticks(chars, fontproperties=tamil_font)

# Save and display
plt.tight_layout()
plt.savefig('figure3_tamil_char_freq.png', dpi=300)  # Save for manuscript
plt.show()