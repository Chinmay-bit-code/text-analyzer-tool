from collections import Counter
import string

text = input("Enter a paragraph:\n")

stopwords = {
    "is", "am", "are", "the", "a", "an", "and", "to", "in", "of", "on",
    "for", "with", "as", "by", "at", "from", "this", "that", "it", "be"
}

lower_text = text.lower()

clean_text = ""
for ch in lower_text:
    if ch not in string.punctuation:
        clean_text += ch

words = clean_text.split()
filtered_words = [word for word in words if word not in stopwords]

word_count = len(filtered_words)

freq = Counter(filtered_words)
top_5_words = freq.most_common(5)

sentences = text.replace("!", ".").replace("?", ".").split(".")
sentence_lengths = []

for sentence in sentences:
    sentence_words = sentence.strip().split()
    if sentence_words:
        sentence_lengths.append(len(sentence_words))

if sentence_lengths:
    min_len = min(sentence_lengths)
    max_len = max(sentence_lengths)
    avg_len = sum(sentence_lengths) / len(sentence_lengths)
else:
    min_len = max_len = avg_len = 0

with open("text_analysis_output.txt", "w") as file:
    file.write(f"Total Word Count: {word_count}\n\n")
    file.write("Top 5 Frequent Words:\n")
    for word, count in top_5_words:
        file.write(f"{word}: {count}\n")
    file.write("\nSentence Length Statistics:\n")
    file.write(f"Minimum Length: {min_len}\n")
    file.write(f"Maximum Length: {max_len}\n")
    file.write(f"Average Length: {avg_len:.2f}\n")

print("Analysis complete. Results saved to text_analysis_output.txt")
