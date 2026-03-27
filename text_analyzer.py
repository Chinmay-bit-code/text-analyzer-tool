"""
Enhanced Text Analysis Program
================================
Features:
 - Input from terminal or .txt / .pdf file
 - Stopwords loaded from external file (with fallback defaults)
 - Configurable top-N frequent words + percentage + unique word count
 - Sentence stats: count, min/max/avg length, longest & shortest sentence
 - Flesch-Kincaid readability score
 - Output to terminal, timestamped .txt, .csv, and .json
 - Bar chart + histogram via matplotlib
 - POS tagging, Named Entity Recognition, Sentiment Analysis via nltk
 - Full error handling throughout
"""

import os
import re
import sys
import json
import string
import datetime
from collections import Counter

# ── Optional dependency imports ──────────────────────────────────────────────

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] matplotlib not found. Charts will be skipped. Install with: pip install matplotlib")

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk import pos_tag, ne_chunk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tree import Tree

    # Download required NLTK data quietly
    for pkg in ["punkt", "averaged_perceptron_tagger", "maxent_ne_chunker",
                "words", "vader_lexicon", "punkt_tab",
                "averaged_perceptron_tagger_eng"]:
        nltk.download(pkg, quiet=True)

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("[WARNING] nltk not found. Advanced NLP features will be skipped. Install with: pip install nltk")

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        import PyPDF2 as pypdf
        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_STOPWORDS = {
    "is", "am", "are", "the", "a", "an", "and", "to", "in", "of", "on",
    "for", "with", "as", "by", "at", "from", "this", "that", "it", "be",
    "was", "were", "has", "have", "had", "but", "or", "not", "so", "if",
    "do", "did", "does", "will", "would", "could", "should", "may", "might",
    "i", "you", "he", "she", "we", "they", "my", "your", "our", "their",
    "its", "me", "him", "her", "us", "them", "what", "which", "who", "how",
    "when", "where", "why", "all", "more", "also", "just", "then", "than",
    "there", "here", "can", "no", "up", "out", "about", "into", "after"
}


# ── Helper Functions ──────────────────────────────────────────────────────────

def load_stopwords(filepath="stopwords.txt"):
    """Load stopwords from an external file, fallback to defaults."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            words = {line.strip().lower() for line in f if line.strip()}
        print(f"[INFO] Loaded {len(words)} stopwords from '{filepath}'.")
        return words
    return DEFAULT_STOPWORDS


def get_input_text():
    """Get text from user: manual input or from a .txt / .pdf file."""
    print("\n=== Enhanced Text Analyzer ===")
    print("How would you like to provide the text?")
    print("  1. Type / paste text manually")
    print("  2. Load from a .txt file")
    print("  3. Load from a .pdf file")
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "2":
        path = input("Enter path to .txt file: ").strip()
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    elif choice == "3":
        if not PYPDF_AVAILABLE:
            raise ImportError("pypdf is not installed. Run: pip install pypdf")
        path = input("Enter path to .pdf file: ").strip()
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        reader = pypdf.PdfReader(path)
        return " ".join(page.extract_text() or "" for page in reader.pages)

    else:
        print("Enter your paragraph (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        return "\n".join(lines).strip()


def clean_and_tokenize(text, stopwords):
    """Lowercase, remove punctuation, split into words, filter stopwords."""
    lower = text.lower()
    clean = lower.translate(str.maketrans("", "", string.punctuation))
    words = clean.split()
    filtered = [w for w in words if w not in stopwords and w.isalpha()]
    return filtered


def sentence_statistics(text):
    """Return sentence-level stats and the longest / shortest sentences."""
    # Split on . ! ?
    raw_sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return 0, 0, 0, 0.0, "", ""

    lengths = [len(s.split()) for s in sentences]
    min_len = min(lengths)
    max_len = max(lengths)
    avg_len = sum(lengths) / len(lengths)
    count   = len(sentences)

    shortest = sentences[lengths.index(min_len)]
    longest  = sentences[lengths.index(max_len)]

    return count, min_len, max_len, avg_len, shortest, longest


def flesch_kincaid_score(text, word_count):
    """
    Approximate Flesch Reading Ease score.
    Score 90-100 = Very Easy, 0-30 = Very Difficult.
    """
    if word_count == 0:
        return 0.0

    # Count syllables (simple heuristic)
    def count_syllables(word):
        word = word.lower()
        vowels = "aeiouy"
        count = sum(1 for i, ch in enumerate(word)
                    if ch in vowels and (i == 0 or word[i-1] not in vowels))
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    sentences = re.split(r'[.!?]+', text)
    sentence_count = max(1, len([s for s in sentences if s.strip()]))
    all_words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    total_syllables = sum(count_syllables(w) for w in all_words)

    score = (206.835
             - 1.015  * (word_count / sentence_count)
             - 84.6   * (total_syllables / max(1, len(all_words))))
    return round(score, 2)


def readability_label(score):
    if score >= 90: return "Very Easy"
    if score >= 80: return "Easy"
    if score >= 70: return "Fairly Easy"
    if score >= 60: return "Standard"
    if score >= 50: return "Fairly Difficult"
    if score >= 30: return "Difficult"
    return "Very Confusing"


# ── Visualization ─────────────────────────────────────────────────────────────

def generate_charts(top_words, sentence_lengths, timestamp):
    """Generate bar chart for word frequency and histogram for sentence lengths."""
    if not MATPLOTLIB_AVAILABLE:
        print("[SKIP] Charts skipped — matplotlib not available.")
        return

    words  = [w for w, _ in top_words]
    counts = [c for _, c in top_words]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Text Analysis — Visual Summary", fontsize=14, fontweight="bold")

    # Bar chart
    axes[0].barh(words[::-1], counts[::-1], color="steelblue")
    axes[0].set_xlabel("Frequency")
    axes[0].set_title("Top Word Frequencies")
    for i, v in enumerate(counts[::-1]):
        axes[0].text(v + 0.1, i, str(v), va="center", fontsize=9)

    # Histogram
    axes[1].hist(sentence_lengths, bins=max(1, len(set(sentence_lengths))),
                 color="coral", edgecolor="black")
    axes[1].set_xlabel("Words per Sentence")
    axes[1].set_ylabel("Number of Sentences")
    axes[1].set_title("Sentence Length Distribution")

    plt.tight_layout()
    chart_file = f"text_analysis_chart_{timestamp}.png"
    plt.savefig(chart_file, dpi=120)
    plt.close()
    print(f"[INFO] Charts saved to '{chart_file}'")


# ── Advanced NLP ──────────────────────────────────────────────────────────────

def advanced_nlp(text):
    """POS tagging, Named Entity Recognition, Sentiment Analysis using NLTK."""
    if not NLTK_AVAILABLE:
        return None

    results = {}

    # Sentiment
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    results["sentiment"] = sentiment
    results["sentiment_scores"] = scores

    # POS tagging — count top POS categories
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    pos_counts = Counter(tag[:2] for _, tag in tagged)
    results["pos_summary"] = {
        "Nouns (NN)":      pos_counts.get("NN", 0),
        "Verbs (VB)":      pos_counts.get("VB", 0),
        "Adjectives (JJ)": pos_counts.get("JJ", 0),
        "Adverbs (RB)":    pos_counts.get("RB", 0),
    }

    # Named Entity Recognition
    chunked = ne_chunk(tagged)
    entities = []
    for subtree in chunked:
        if isinstance(subtree, Tree):
            entity_name = " ".join(word for word, tag in subtree.leaves())
            entity_type = subtree.label()
            entities.append((entity_name, entity_type))
    results["named_entities"] = entities

    return results


# ── Output Writers ────────────────────────────────────────────────────────────

def build_report(stats):
    """Build a human-readable report string from the stats dict."""
    s = stats
    lines = [
        "=" * 55,
        "          ENHANCED TEXT ANALYSIS REPORT",
        f"  Generated: {s['timestamp']}",
        "=" * 55,
        "",
        f"  Total Word Count (excl. stopwords) : {s['word_count']}",
        f"  Unique Words                       : {s['unique_words']}",
        f"  Total Sentences                    : {s['sentence_count']}",
        "",
        f"  Readability (Flesch Score)         : {s['flesch_score']} — {s['readability']}",
        "",
        "── Top Words ────────────────────────────────────────",
    ]
    for rank, (word, count) in enumerate(s["top_words"], 1):
        pct = (count / s["word_count"] * 100) if s["word_count"] else 0
        lines.append(f"  {rank}. {word:<20} {count:>4}  ({pct:.1f}%)")

    lines += [
        "",
        "── Sentence Length Statistics ───────────────────────",
        f"  Minimum : {s['min_len']} words",
        f"  Maximum : {s['max_len']} words",
        f"  Average : {s['avg_len']:.2f} words",
        "",
        f"  Shortest: \"{s['shortest_sentence'][:80]}\"",
        f"  Longest : \"{s['longest_sentence'][:80]}\"",
    ]

    if s.get("nlp"):
        nlp = s["nlp"]
        lines += [
            "",
            "── Sentiment Analysis ───────────────────────────────",
            f"  Overall Sentiment : {nlp['sentiment']}",
            f"  Scores            : Positive={nlp['sentiment_scores']['pos']:.2f}  "
            f"Neutral={nlp['sentiment_scores']['neu']:.2f}  "
            f"Negative={nlp['sentiment_scores']['neg']:.2f}",
            "",
            "── Parts of Speech ──────────────────────────────────",
        ]
        for pos_name, cnt in nlp["pos_summary"].items():
            lines.append(f"  {pos_name:<25} : {cnt}")

        if nlp["named_entities"]:
            lines += ["", "── Named Entities ───────────────────────────────────"]
            for name, etype in nlp["named_entities"]:
                lines.append(f"  [{etype}] {name}")

    lines += ["", "=" * 55]
    return "\n".join(lines)


def save_txt(report, timestamp):
    fname = f"text_analysis_{timestamp}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[INFO] Text report saved to '{fname}'")


def save_csv(stats, timestamp):
    import csv
    fname = f"text_analysis_{timestamp}.csv"
    with open(fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Words", stats["word_count"]])
        writer.writerow(["Unique Words", stats["unique_words"]])
        writer.writerow(["Sentences", stats["sentence_count"]])
        writer.writerow(["Flesch Score", stats["flesch_score"]])
        writer.writerow(["Readability", stats["readability"]])
        writer.writerow(["Min Sentence Length", stats["min_len"]])
        writer.writerow(["Max Sentence Length", stats["max_len"]])
        writer.writerow(["Avg Sentence Length", round(stats["avg_len"], 2)])
        writer.writerow([])
        writer.writerow(["Rank", "Word", "Count", "Percentage"])
        for rank, (word, count) in enumerate(stats["top_words"], 1):
            pct = round(count / stats["word_count"] * 100, 1) if stats["word_count"] else 0
            writer.writerow([rank, word, count, f"{pct}%"])
    print(f"[INFO] CSV saved to '{fname}'")


def save_json(stats, timestamp):
    fname = f"text_analysis_{timestamp}.json"
    exportable = {k: v for k, v in stats.items() if k != "sentence_lengths"}
    exportable["top_words"] = [{"word": w, "count": c} for w, c in stats["top_words"]]
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(exportable, f, indent=2)
    print(f"[INFO] JSON saved to '{fname}'")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Get text
    try:
        text = get_input_text()
    except (FileNotFoundError, ImportError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not text.strip():
        print("[ERROR] No text provided. Exiting.")
        sys.exit(1)

    # 2. Config
    try:
        top_n = int(input("\nHow many top words to show? (default 5): ").strip() or "5")
    except ValueError:
        top_n = 5

    # 3. Load stopwords
    stopwords = load_stopwords()

    # 4. Tokenize & analyse
    filtered_words = clean_and_tokenize(text, stopwords)

    if not filtered_words:
        print("[ERROR] No meaningful words found after filtering.")
        sys.exit(1)

    word_count   = len(filtered_words)
    unique_words = len(set(filtered_words))
    freq         = Counter(filtered_words)
    top_words    = freq.most_common(top_n)

    # 5. Sentence stats
    sent_count, min_len, max_len, avg_len, shortest, longest = sentence_statistics(text)

    # Raw sentence lengths for histogram
    raw_sentences   = re.split(r'[.!?]+', text)
    sentence_lengths = [len(s.split()) for s in raw_sentences if s.strip()]

    # 6. Readability
    flesch = flesch_kincaid_score(text, word_count)

    # 7. Timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 8. Advanced NLP
    nlp_results = advanced_nlp(text)

    # 9. Bundle stats
    stats = {
        "timestamp":         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "word_count":        word_count,
        "unique_words":      unique_words,
        "sentence_count":    sent_count,
        "top_words":         top_words,
        "min_len":           min_len,
        "max_len":           max_len,
        "avg_len":           avg_len,
        "shortest_sentence": shortest,
        "longest_sentence":  longest,
        "sentence_lengths":  sentence_lengths,
        "flesch_score":      flesch,
        "readability":       readability_label(flesch),
        "nlp":               nlp_results,
    }

    # 10. Build & print report
    report = build_report(stats)
    print("\n" + report)

    # 11. Save outputs
    print("\n[Saving outputs...]")
    save_txt(report, timestamp)
    save_csv(stats, timestamp)
    save_json(stats, timestamp)

    # 12. Charts
    generate_charts(top_words, sentence_lengths, timestamp)

    print("\n[✓] Analysis complete!")


if __name__ == "__main__":
    main()
