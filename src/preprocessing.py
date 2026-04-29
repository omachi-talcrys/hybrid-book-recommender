"""
preprocessing.py
----------------
Text cleaning utilities.
Loading is handled by data_loader.py.
"""

import string

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "i", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "what", "which", "who", "whom", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "about", "after", "as", "into", "also",
    "if", "then", "because", "while", "although", "though", "however",
    "one", "two", "first", "there", "their", "any", "up", "out"
}


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, remove stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def load_and_preprocess(books_path: str, reviews_path: str = None):
    """
    Convenience wrapper used by the Streamlit app and tfidf_model.
    Loads data via data_loader then applies clean_text.
    """
    from data_loader import load_books

    df = load_books(books_path, reviews_path=reviews_path)
    df["processed_text"] = df["combined_text"].apply(clean_text)
    return df
