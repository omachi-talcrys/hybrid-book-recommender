"""
data_loader.py
--------------
Handles loading books data in two formats:

  Format A — Single file (our included books.csv, or any CSV with
             title, author, description, genres, avg_rating columns)

  Format B — Two files from Kaggle Goodreads datasets:
             books.csv  (book metadata)
             reviews.csv (user reviews linked by book_id)

For Format B, we aggregate the top N reviews per book by rating
and append them to the description before preprocessing.

Usage:
    from data_loader import load_books

    # Single file (default)
    df = load_books("data/books.csv")

    # Two-file Goodreads format
    df = load_books("data/books.csv", reviews_path="data/reviews.csv")
"""

import pandas as pd


# How many reviews to include per book (keeps noise low)
MAX_REVIEWS_PER_BOOK = 3

# Minimum review length in characters (filters out "great!" type noise)
MIN_REVIEW_LENGTH = 80


# ── column name mapping ───────────────────────────────────────────────────────
# Kaggle Goodreads datasets use different column names depending on the version.
# We try each candidate and use the first one that exists.

TITLE_COLS        = ["title", "Title", "book_title", "name"]
AUTHOR_COLS       = ["author", "Author", "authors", "book_author"]
DESCRIPTION_COLS  = ["description", "Description", "desc", "summary", "book_desc"]
GENRE_COLS        = ["genres", "genre", "Genre", "categories", "shelves"]
RATING_COLS       = ["avg_rating", "average_rating", "rating", "book_rating"]
BOOK_ID_COLS      = ["book_id", "bookID", "id", "book_ID"]

REVIEW_BOOK_ID_COLS = ["book_id", "bookID", "id"]
REVIEW_TEXT_COLS    = ["review_text", "review", "text", "body", "comment"]
REVIEW_RATING_COLS  = ["rating", "review_rating", "score", "stars"]


def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    """Return the first candidate column name that exists in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalise_books(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to our standard names and drop rows missing key fields."""
    renames = {}

    title_col  = _find_col(df, TITLE_COLS)
    author_col = _find_col(df, AUTHOR_COLS)
    desc_col   = _find_col(df, DESCRIPTION_COLS)
    genre_col  = _find_col(df, GENRE_COLS)
    rating_col = _find_col(df, RATING_COLS)
    id_col     = _find_col(df, BOOK_ID_COLS)

    if not title_col or not desc_col:
        raise ValueError(
            f"Could not find title or description columns.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Expected title column named one of: {TITLE_COLS}\n"
            f"Expected description column named one of: {DESCRIPTION_COLS}"
        )

    if title_col  != "title":       renames[title_col]  = "title"
    if author_col and author_col != "author":   renames[author_col] = "author"
    if desc_col   != "description": renames[desc_col]   = "description"
    if genre_col  and genre_col != "genres":    renames[genre_col]  = "genres"
    if rating_col and rating_col != "avg_rating": renames[rating_col] = "avg_rating"
    if id_col     and id_col != "book_id":      renames[id_col]     = "book_id"

    df = df.rename(columns=renames)

    # Ensure all standard columns exist (fill missing ones with empty string)
    for col in ["author", "genres", "avg_rating", "book_id"]:
        if col not in df.columns:
            df[col] = ""

    df = df.dropna(subset=["title", "description"])
    df = df[df["description"].str.strip() != ""]
    df = df.reset_index(drop=True)

    return df


def _aggregate_reviews(reviews_df: pd.DataFrame, books_df: pd.DataFrame) -> pd.Series:
    """
    For each book in books_df, collect up to MAX_REVIEWS_PER_BOOK reviews
    sorted by rating (highest first). Return a Series indexed like books_df
    mapping book_id → aggregated review text.
    """
    book_id_col    = _find_col(reviews_df, REVIEW_BOOK_ID_COLS)
    review_text_col = _find_col(reviews_df, REVIEW_TEXT_COLS)
    review_rating_col = _find_col(reviews_df, REVIEW_RATING_COLS)

    if not book_id_col or not review_text_col:
        print("Warning: could not identify book_id or review text columns in reviews file.")
        print(f"Available columns: {list(reviews_df.columns)}")
        return pd.Series(dtype=str)

    rev = reviews_df[[book_id_col, review_text_col] +
                     ([review_rating_col] if review_rating_col else [])].copy()
    rev = rev.rename(columns={book_id_col: "book_id", review_text_col: "review_text"})
    if review_rating_col:
        rev = rev.rename(columns={review_rating_col: "rating"})
        rev["rating"] = pd.to_numeric(rev["rating"], errors="coerce").fillna(0)
    else:
        rev["rating"] = 0

    # Filter short/empty reviews
    rev = rev.dropna(subset=["review_text"])
    rev = rev[rev["review_text"].str.len() >= MIN_REVIEW_LENGTH]

    # Sort by rating descending, take top N per book
    rev = rev.sort_values("rating", ascending=False)
    top_reviews = (
        rev.groupby("book_id")["review_text"]
        .apply(lambda texts: " ".join(list(texts)[:MAX_REVIEWS_PER_BOOK]))
    )

    return top_reviews


def load_books(books_path: str, reviews_path: str = None) -> pd.DataFrame:
    """
    Load and normalise book data.

    Parameters
    ----------
    books_path   : Path to books CSV (required).
    reviews_path : Path to reviews CSV (optional). If provided, top reviews
                   are appended to each book's description.

    Returns
    -------
    Normalised DataFrame with columns:
        book_id, title, author, description, genres, avg_rating, combined_text
    """
    print(f"Loading books from: {books_path}")
    books_df = pd.read_csv(books_path, on_bad_lines="skip", low_memory=False)
    print(f"  Raw rows: {len(books_df)}")

    books_df = _normalise_books(books_df)
    print(f"  After cleaning: {len(books_df)} books")

    # Optionally merge reviews
    if reviews_path:
        print(f"Loading reviews from: {reviews_path}")
        try:
            reviews_df = pd.read_csv(reviews_path, on_bad_lines="skip", low_memory=False)
            print(f"  Raw review rows: {len(reviews_df)}")
            top_reviews = _aggregate_reviews(reviews_df, books_df)

            if len(top_reviews) > 0:
                books_df["book_id"] = books_df["book_id"].astype(str)
                top_reviews.index  = top_reviews.index.astype(str)
                books_df["review_text"] = books_df["book_id"].map(top_reviews).fillna("")
                matched = (books_df["review_text"] != "").sum()
                print(f"  Matched reviews for {matched} / {len(books_df)} books")
            else:
                books_df["review_text"] = ""
        except Exception as e:
            print(f"  Warning: could not load reviews — {e}")
            books_df["review_text"] = ""
    else:
        books_df["review_text"] = ""

    # Build combined_text: description + genres + reviews
    books_df["genres"]      = books_df["genres"].fillna("")
    books_df["review_text"] = books_df["review_text"].fillna("")
    books_df["combined_text"] = (
        books_df["description"] + " " +
        books_df["genres"]      + " " +
        books_df["review_text"]
    )

    return books_df


if __name__ == "__main__":
    # Quick test with the included dataset (no reviews)
    df = load_books("../data/books.csv")
    print(df[["title", "combined_text"]].head(2))
