# 📚 Hybrid Book Recommendation System

**TF-IDF Baseline vs. LLM-Enhanced Hybrid — A Comparative Study**

---

## Overview

This project builds and compares two book recommendation approaches:

| Model | Method | Strength |
|-------|--------|----------|
| **Baseline** | TF-IDF + Cosine Similarity | Fast, keyword-based matching |
| **Hybrid** | TF-IDF candidates → LLM re-ranking | Contextual, tone-aware, explainable |

The core hypothesis: **LLM re-ranking produces more relevant, context-aware recommendations than pure keyword similarity.**

---

## Project Structure

```
book-recommender/
│
├── data/
│   └── books.csv               # 50-book Goodreads-style dataset
│
├── src/
│   ├── preprocessing.py        # Text cleaning & feature engineering
│   ├── tfidf_model.py          # TF-IDF vectorizer + cosine similarity
│   └── llm_model.py            # Anthropic Claude re-ranking pipeline
│
├── app/
│   └── streamlit_app.py        # Main Streamlit UI
│
├── results/
│   └── sample_outputs.txt      # 3 example queries with analysis
│
├── README.md
└── requirements.txt
```

---

## Setup & Installation

### 1. Clone / download the project

```bash
cd book-recommender
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get an Anthropic API key

- Sign up at [console.anthropic.com](https://console.anthropic.com)
- The free tier is sufficient for this project

### 4. Run the app

```bash
streamlit run app/streamlit_app.py
```

---

## How It Works

### Baseline Model (TF-IDF)

1. Load and preprocess book descriptions + genres
2. Build a TF-IDF matrix (`max_features=5000`, bigrams)
3. For a user query: vectorize → compute cosine similarity → return top N

```python
from src.tfidf_model import build_recommender
rec = build_recommender("data/books.csv")
results = rec.recommend("slow emotional novel about grief", top_n=5)
```

### Hybrid Model (TF-IDF + LLM)

1. TF-IDF retrieves top 10 candidate books (the "recall" step)
2. Candidates + user query sent to Claude (the "precision" step)
3. Claude re-ranks by tone, theme, character similarity, and emotional resonance
4. Returns top 5 with human-readable explanations

```python
import anthropic
from src.llm_model import hybrid_recommend

client = anthropic.Anthropic(api_key="your-key-here")
results = hybrid_recommend("slow emotional novel about grief", rec, client)
```

---

## Example Queries

See `results/sample_outputs.txt` for full output. Summary:

| Query | Baseline Top Pick | Hybrid Top Pick | Why Hybrid Wins |
|-------|------------------|-----------------|-----------------|
| "slow emotional novel about grief" | The Remains of the Day | The Remains of the Day | Same pick, but hybrid explains WHY (Ishiguro's restrained prose, suppressed emotion) |
| "dark dystopian with strong female lead" | The Handmaid's Tale | The Handmaid's Tale | Hybrid correctly elevates Hunger Games above 1984 (female protagonist emphasis) |
| "epic fantasy heist with misfit crew" | Six of Crows | Six of Crows | Hybrid explains crew dynamics vs. solo protagonist distinction |

---

## Dataset

The dataset (`data/books.csv`) includes 50 books across genres:

- **Fields**: `book_id`, `title`, `author`, `description`, `genres`, `avg_rating`
- **Processing**: description + genres combined → lowercased → punctuation removed → stopwords filtered
- **Coverage**: Literary fiction, dystopia, fantasy, sci-fi, YA, mystery, historical fiction

---

## Key Design Decisions

- **No complex neural embeddings** — kept simple with sklearn TF-IDF
- **LLM as re-ranker only** — not a generative system; structured JSON output
- **50-book curated dataset** — large enough to demonstrate the comparison, small enough to run locally
- **Streamlit** — single-file UI, no Flask/Django overhead

---

## Evaluation: Why Hybrid is Better

The TF-IDF baseline is limited to surface-level keyword matching. It cannot:
- Understand **tone** ("slow" vs "fast-paced")
- Reason about **character archetypes** ("strong female lead" vs incidental female characters)
- Infer **reader experience** ("emotional resonance", "sense of dread")

The LLM hybrid addresses all three by reading book descriptions as a human critic would, then explaining its reasoning — making results both more accurate and more interpretable.

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
streamlit>=1.35.0
anthropic>=0.30.0
```

---

## Notes

- The Anthropic API key is entered in the Streamlit sidebar at runtime (not hardcoded)
- The app runs fully without the API key — baseline results are always shown
- LLM calls use `claude-sonnet-4-20250514` with structured JSON output for reliability
