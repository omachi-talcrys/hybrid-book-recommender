"""
llm_model.py
------------
Hybrid recommender: TF-IDF fetches a large candidate pool (20 books),
then Claude re-ranks them intelligently.

Using a pool of 20 instead of 10 means the LLM can surface books
that TF-IDF ranked 10-20 — producing results visibly different from
the baseline top 5.
"""

import json
import re
import anthropic
from tfidf_model import CANDIDATE_POOL


SYSTEM_PROMPT = """You are an expert literary critic and book recommendation specialist.
You have deep knowledge of narrative tone, character archetypes, emotional themes,
writing styles, and genre conventions.

Your job: re-rank a list of candidate books to find the BEST matches for the user's query.
Focus on:
- Thematic and tonal alignment (e.g. "slow" vs "fast-paced", "dark" vs "hopeful")
- Character similarity (ensemble cast, lone protagonist, morally complex lead, etc.)
- Emotional resonance the user is likely seeking
- Contextual nuance not captured by keyword matching

IMPORTANT: You are allowed — and encouraged — to pick books from lower in the candidate
list if they are a better match. Do not just return the top 5 in the same order.

Return ONLY valid JSON. No markdown, no explanation outside the JSON.
"""

RERANK_TEMPLATE = """User Query: "{query}"

Candidate books (ranked by keyword similarity — your job is to re-rank by actual fit):
{candidates}

Select the 5 best matches. Re-rank from best to worst. For each, write 1-2 sentences
explaining WHY it matches the query — focus on tone, characters, themes, not just plot.

Respond with ONLY this JSON structure:
{{
  "recommendations": [
    {{
      "rank": 1,
      "title": "Exact title from the list",
      "author": "Author name",
      "reason": "Why this matches the query tone, themes, or characters."
    }}
  ]
}}
"""


def format_candidates(df) -> str:
    lines = []
    for i, row in df.iterrows():
        desc = row["description"]
        if len(desc) > 220:
            desc = desc[:220] + "..."
        genres = row.get("genres", "")
        lines.append(
            f"{i+1}. \"{row['title']}\" by {row.get('author', 'Unknown')}\n"
            f"   Genres: {genres}\n"
            f"   Description: {desc}"
        )
    return "\n\n".join(lines)


def llm_rerank(query: str, candidates_df, client: anthropic.Anthropic) -> list[dict]:
    """Send candidates to Claude and get back a re-ranked list with reasons."""
    candidates_text = format_candidates(candidates_df)
    user_message = RERANK_TEMPLATE.format(
        query=query,
        candidates=candidates_text
    )

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if model adds them
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    parsed = json.loads(raw)
    return parsed.get("recommendations", [])


def hybrid_recommend(query: str, tfidf_recommender, client: anthropic.Anthropic) -> list[dict]:
    """
    Full hybrid pipeline:
      1. TF-IDF → top CANDIDATE_POOL books (default 20)
      2. LLM → re-rank and explain top 5

    The larger candidate pool is the key to getting different results
    from the baseline — the LLM can pull from books ranked 10-20 by TF-IDF.
    """
    candidates = tfidf_recommender.recommend(query, top_n=CANDIDATE_POOL)
    results = llm_rerank(query, candidates, client)
    return results
