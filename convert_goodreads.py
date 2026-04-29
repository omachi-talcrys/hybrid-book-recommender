"""
convert_goodreads.py
--------------------
Converts the Wan/McAuley Goodreads JSON.gz files into clean CSVs
with exactly the column names the book recommender expects.

Usage:
    python convert_goodreads.py

Put this script in the same folder as your downloaded .json.gz files,
or update the paths at the bottom to point to wherever they are.

Output:
    data/books.csv   — book_id, title, author, description, genres, avg_rating
    data/reviews.csv — book_id, review_text, rating
"""

import gzip
import json
import csv
import os
from pathlib import Path


# ── config — update these paths if needed ────────────────────────────────────
BOOKS_GZ   = "goodreads_books_young_adult.json"
REVIEWS_GZ = "goodreads_reviews_young_adult.json"
OUT_DIR    = "data"

# How many reviews to keep per book (keeps file size manageable)
# The data_loader will further filter to top 3 by rating anyway
MAX_REVIEWS_PER_BOOK = 10

# Minimum description length — filters out books with no real description
MIN_DESC_LENGTH = 50
# ─────────────────────────────────────────────────────────────────────────────


def extract_genres(genre_dict):
    """
    In this dataset genres is a dict like:
    {"children": 5, "young-adult": 120, "fiction": 30}
    We take the keys sorted by value (most tagged first), join as a string.
    """
    if not genre_dict or not isinstance(genre_dict, dict):
        return ""
    sorted_genres = sorted(genre_dict.items(), key=lambda x: x[1], reverse=True)
    return " ".join([g[0] for g in sorted_genres[:5]])  # top 5 genre tags


def convert_books(books_gz_path, out_path):
    print(f"Converting books: {books_gz_path}")
    written = 0
    skipped = 0

    with open(books_gz_path, "rt", encoding="utf-8") as f_in, \
         open(out_path, "w", newline="", encoding="utf-8") as f_out:

        writer = csv.DictWriter(f_out, fieldnames=[
            "book_id", "title", "author", "description", "genres", "avg_rating"
        ])
        writer.writeheader()

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                book = json.loads(line)
            except json.JSONDecodeError:
                continue

            book_id     = str(book.get("book_id", "")).strip()
            title       = book.get("title", "").strip()
            description = book.get("description", "").strip()
            avg_rating  = book.get("average_rating", "").strip()

            # Author: comes as a list of dicts [{author_id, role}]
            # We just grab the first author id — no names in this file
            # but we write the id so the join still works; name lookup
            # isn't needed for TF-IDF
            authors_raw = book.get("authors", [])
            if authors_raw and isinstance(authors_raw, list):
                author = authors_raw[0].get("author_id", "Unknown")
            else:
                author = "Unknown"

            genres = extract_genres(book.get("popular_shelves_genres", {}))
            if not genres:
                # fallback: use popular_shelves if genres dict missing
                shelves = book.get("popular_shelves", [])
                if shelves:
                    genres = " ".join([s.get("name", "") for s in shelves[:5]])

            # Skip books with no real description
            if not title or len(description) < MIN_DESC_LENGTH:
                skipped += 1
                continue

            writer.writerow({
                "book_id":     book_id,
                "title":       title,
                "author":      author,
                "description": description,
                "genres":      genres,
                "avg_rating":  avg_rating,
            })
            written += 1

            if written % 10000 == 0:
                print(f"  Books written: {written:,}")

    print(f"  Done. Written: {written:,}  |  Skipped (no desc): {skipped:,}")
    return written


def convert_reviews(reviews_gz_path, out_path, max_per_book=MAX_REVIEWS_PER_BOOK):
    print(f"Converting reviews: {reviews_gz_path}")
    from collections import defaultdict

    # First pass: collect reviews per book, keep highest rated ones
    # We store (rating, review_text) per book_id
    book_reviews = defaultdict(list)
    total = 0

    with open(reviews_gz_path, "rt", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                continue

            book_id     = str(review.get("book_id", "")).strip()
            review_text = review.get("review_text", "").strip()
            rating      = review.get("rating", 0)

            # Skip empty or very short reviews
            if not book_id or len(review_text) < 80:
                continue

            try:
                rating = int(rating)
            except (ValueError, TypeError):
                rating = 0

            book_reviews[book_id].append((rating, review_text))
            total += 1

            if total % 100000 == 0:
                print(f"  Reviews read: {total:,}")

    print(f"  Total valid reviews read: {total:,}")
    print(f"  Unique books with reviews: {len(book_reviews):,}")
    print(f"  Writing top {max_per_book} reviews per book...")

    written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=[
            "book_id", "review_text", "rating"
        ])
        writer.writeheader()

        for book_id, reviews in book_reviews.items():
            # Sort by rating descending, take top N
            top = sorted(reviews, key=lambda x: x[0], reverse=True)[:max_per_book]
            for rating, review_text in top:
                writer.writerow({
                    "book_id":     book_id,
                    "review_text": review_text,
                    "rating":      rating,
                })
                written += 1

    print(f"  Done. Review rows written: {written:,}")
    return written


def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUT_DIR, exist_ok=True)

    books_out   = os.path.join(OUT_DIR, "books.csv")
    reviews_out = os.path.join(OUT_DIR, "reviews.csv")

    # Check input files exist
    for path in [BOOKS_GZ, REVIEWS_GZ]:
        if not Path(path).exists():
            print(f"ERROR: File not found: {path}")
            print("Make sure the .json.gz files are in the same folder as this script,")
            print("or update the paths at the top of this file.")
            return

    print("=" * 60)
    print("Goodreads JSON.gz → CSV Converter")
    print("=" * 60)

    n_books   = convert_books(BOOKS_GZ, books_out)
    print()
    n_reviews = convert_reviews(REVIEWS_GZ, reviews_out)

    print()
    print("=" * 60)
    print("DONE")
    print(f"  Books CSV:   {books_out}  ({n_books:,} rows)")
    print(f"  Reviews CSV: {reviews_out}  ({n_reviews:,} rows)")
    print()
    print("Next steps:")
    print("  1. Move both CSV files into your book-recommender/data/ folder")
    print("  2. Run:  streamlit run app/streamlit_app.py")
    print("  3. In the sidebar, set books file to:   books.csv")
    print("  4. Check 'Include reviews file' and set: reviews.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
