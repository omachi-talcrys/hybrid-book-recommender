"""
tfidf_model.py
--------------
Baseline content-based recommender using TF-IDF + cosine similarity.

CANDIDATE_POOL: how many books are passed to the LLM re-ranker.
Keeping this at 20 gives the LLM enough diversity to surface books
that TF-IDF ranked 10-20 — making hybrid results visibly different
from the baseline top 5.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text

CANDIDATE_POOL = 20   # sent to LLM; increase if you have a large dataset


class TFIDFRecommender:
    """
    Fits a TF-IDF matrix on the book corpus and retrieves
    the most similar books to a given query via cosine similarity.
    """

    def __init__(self, df: pd.DataFrame, text_column: str = "processed_text"):
        self.df = df.reset_index(drop=True)
        self.text_column = text_column
        self.vectorizer = TfidfVectorizer(
            max_features=10000,       # bumped up for larger datasets
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,        # dampens very frequent terms
        )
        self._fit()

    def _fit(self):
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df[self.text_column]
        )

    def recommend(self, query: str, top_n: int = 5, exclude_exact: bool = True) -> pd.DataFrame:
        """
        Return top_n books most similar to query.
        Use top_n=5 for the displayed baseline results.
        Use top_n=CANDIDATE_POOL to get the pool sent to the LLM.
        """
        cleaned_query = clean_text(query)
        query_vec = self.vectorizer.transform([cleaned_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        result_df = self.df.copy()
        result_df["similarity_score"] = scores

        if exclude_exact:
            lower_query = query.lower().strip()
            result_df = result_df[result_df["title"].str.lower() != lower_query]

        result_df = (
            result_df
            .sort_values("similarity_score", ascending=False)
            .head(top_n)
            .copy()
            .reset_index(drop=True)
        )

        # Return only the columns we need (gracefully handle missing ones)
        cols = ["title", "author", "description", "genres", "avg_rating", "similarity_score"]
        cols = [c for c in cols if c in result_df.columns]
        return result_df[cols]
