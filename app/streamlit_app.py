"""
streamlit_app.py
----------------
Hybrid Book Recommendation System — Streamlit UI
Includes live Results Analysis Dashboard that updates with each query.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import anthropic
import pandas as pd
import numpy as np
from preprocessing import load_and_preprocess
from tfidf_model import TFIDFRecommender, CANDIDATE_POOL
from llm_model import hybrid_recommend

st.set_page_config(page_title="Hybrid Book Recommender", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Source+Sans+3:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }
.main-header { text-align:center; padding:24px 32px; background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460); border-radius:12px; margin-bottom:20px; color:white; }
.main-header h1 { font-size:2.4rem; margin:0; color:#e8c97e; }
.main-header p  { font-size:1rem; color:#a0b4cc; margin-top:6px; }
.col-head { padding:8px 14px; border-radius:8px; font-size:13px; font-weight:600; margin-bottom:10px; }
.ch-baseline { background:#e3e8f0; color:#1a2a3a; }
.ch-hybrid   { background:#d8eedf; color:#1a3a28; }
.book-card { background:white; border:0.5px solid #e0e6ef; border-radius:10px; padding:12px 14px; margin-bottom:8px; }
.rank   { font-family:'Playfair Display',serif; font-size:22px; font-weight:700; color:#c8a45a; float:left; margin-right:8px; line-height:1; }
.btitle { font-weight:700; font-size:14px; color:#1a1a2e; }
.author { color:#6b7a8d; font-size:12px; margin-top:1px; }
.score  { float:right; font-size:11px; background:#f0f4f8; padding:2px 8px; border-radius:10px; color:#5a6a7a; }
.desc   { font-size:12px; color:#5a6a7a; margin-top:4px; }
.reason { background:#f0f7f0; border-left:3px solid #4a9e6b; padding:6px 10px; border-radius:0 6px 6px 0; font-size:12px; color:#2d5a3d; margin-top:6px; line-height:1.45; }
.new-badge { display:inline-block; background:#fff3cd; color:#7a5a10; font-size:10px; font-weight:600; padding:1px 6px; border-radius:3px; margin-left:6px; vertical-align:middle; }
.metric-card { background:white; border:0.5px solid #e0e6ef; border-radius:10px; padding:16px; text-align:center; }
.metric-num  { font-size:2.2rem; font-weight:700; font-family:'Playfair Display',serif; }
.metric-label { font-size:12px; color:#6b7a8d; margin-top:4px; }
.dashboard-header { background:linear-gradient(135deg,#1a1a2e,#0f3460); border-radius:10px; padding:14px 20px; margin-bottom:16px; }
.dashboard-header h3 { color:#e8c97e; margin:0; font-size:1.1rem; }
.dashboard-header p  { color:#a0b4cc; margin:4px 0 0; font-size:12px; }
</style>
""", unsafe_allow_html=True)

if "query_history" not in st.session_state:
    st.session_state.query_history = []

st.markdown("""
<div class="main-header">
    <h1>📚 Hybrid Book Recommender</h1>
    <p>TF-IDF Baseline (top 5) vs LLM-Enhanced Hybrid (best 5 from top 20 candidates)</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Anthropic API Key", type="password", help="Required for hybrid recommendations.")
    st.markdown("---")
    st.subheader("📂 Data Source")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    books_file = st.text_input("Books CSV filename", value="books.csv")
    books_path = os.path.join(data_dir, books_file)
    use_reviews = st.checkbox("Include reviews file", value=False)
    reviews_path = None
    if use_reviews:
        reviews_file = st.text_input("Reviews CSV filename", value="reviews.csv")
        reviews_path = os.path.join(data_dir, reviews_file)
        if not os.path.exists(reviews_path):
            st.warning("reviews.csv not found in data/")
            reviews_path = None
    st.markdown("---")
    st.markdown(f"**How it works**\n1. **Baseline** — TF-IDF cosine similarity, returns top 5.\n2. **Hybrid** — TF-IDF fetches top {CANDIDATE_POOL} candidates, then Claude re-ranks.")
    st.markdown("---")
    if st.button("🗑️ Clear Results History", use_container_width=True):
        st.session_state.query_history = []
        st.rerun()

@st.cache_resource(show_spinner=False)
def load_models(bp, rp):
    df = load_and_preprocess(bp, reviews_path=rp)
    rec = TFIDFRecommender(df)
    return df, rec

if not os.path.exists(books_path):
    st.error(f"Books file not found: `{books_path}`")
    st.stop()

with st.spinner("Loading book database..."):
    try:
        df, tfidf_rec = load_models(books_path, reviews_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

review_note = " + reviews" if reviews_path else ""
st.success(f"✓ {len(df)} books loaded{review_note}", icon="📖")

tab1, tab2 = st.tabs(["🔍 Recommendations", "📊 Results Analysis Dashboard"])

# ── TAB 1: Recommendations ────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🔍 Enter your query")
    examples = [
        "slow emotional literary novel about grief",
        "dark dystopian society with strong female lead",
        "epic fantasy with magic and a heist",
        "coming of age story about identity and belonging",
        "emotional story about school and friendships",
        "dark coming of age with moral complexity",
    ]
    st.markdown("**Quick examples:**")
    ex_cols = st.columns(3)
    for i, ex in enumerate(examples):
        if ex_cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["query"] = ex

    query = st.text_input("Query", value=st.session_state.get("query",""),
        placeholder="e.g. 'A slow emotional novel about regret'",
        label_visibility="collapsed", key="query_input")
    run = st.button("📚 Get Recommendations", use_container_width=True)

    if run and query.strip():
        q = query.strip()
        baseline  = tfidf_rec.recommend(q, top_n=5)
        full_pool = tfidf_rec.recommend(q, top_n=CANDIDATE_POOL)

        hybrid = None
        hybrid_error = None
        if api_key:
            try:
                client = anthropic.Anthropic(api_key=api_key)
                with st.spinner(f"🤖 LLM reviewing top {CANDIDATE_POOL} candidates..."):
                    hybrid = hybrid_recommend(q, tfidf_rec, client)
            except Exception as e:
                hybrid_error = str(e)

        baseline_titles    = set(baseline["title"].tolist())
        baseline_avg_score = float(baseline["similarity_score"].mean())

        if hybrid:
            hybrid_titles  = [r["title"] for r in hybrid]
            non_overlap    = sum(1 for t in hybrid_titles if t not in baseline_titles)
            non_overlap_pct = int(round(non_overlap / len(hybrid_titles) * 100))
            pool_titles    = full_pool["title"].tolist()
            llm_top_title  = hybrid[0]["title"] if hybrid else None
            llm_top_rank   = (pool_titles.index(llm_top_title) + 1) if llm_top_title and llm_top_title in pool_titles else None

            st.session_state.query_history.append({
                "query": q[:40] + ("..." if len(q) > 40 else ""),
                "baseline_avg_score": round(baseline_avg_score, 3),
                "non_overlap": non_overlap,
                "non_overlap_pct": non_overlap_pct,
                "llm_top_tfidf_rank": llm_top_rank,
            })

        st.markdown("---")
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div class="col-head ch-baseline">📊 Baseline — TF-IDF Top 5</div>', unsafe_allow_html=True)
            st.caption("Pure keyword matching: returns the 5 books with highest cosine similarity score.")
            for i, row in baseline.iterrows():
                desc = row["description"][:140] + "..." if len(row["description"]) > 140 else row["description"]
                st.markdown(f"""<div class="book-card">
                    <span class="rank">{i+1}</span><span class="score">score: {row['similarity_score']:.3f}</span>
                    <div class="btitle">{row['title']}</div>
                    <div class="author">by {row.get('author','')}</div>
                    <div class="desc">{desc}</div></div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="col-head ch-hybrid">🤖 Hybrid — LLM Re-ranked (from top {CANDIDATE_POOL})</div>', unsafe_allow_html=True)
            st.caption(f"LLM selects the best 5 from {CANDIDATE_POOL} TF-IDF candidates, reasoning about tone and themes.")
            if hybrid:
                for rec in hybrid:
                    is_new = rec["title"] not in baseline_titles
                    new_badge = '<span class="new-badge">not in baseline</span>' if is_new else ""
                    st.markdown(f"""<div class="book-card">
                        <span class="rank">{rec['rank']}</span>
                        <div class="btitle">{rec['title']}{new_badge}</div>
                        <div class="author">by {rec.get('author','')}</div>
                        <div class="reason">💡 {rec['reason']}</div></div>""", unsafe_allow_html=True)
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("Non-overlap", f"{non_overlap}/5 ({non_overlap_pct}%)")
                m2.metric("Baseline avg score", f"{baseline_avg_score:.3f}")
                if llm_top_rank:
                    m3.metric("LLM #1 was TF-IDF rank", f"#{llm_top_rank}")
            elif hybrid_error:
                st.error(f"LLM error: {hybrid_error}")
            else:
                st.info("Add your Anthropic API key in the sidebar to enable hybrid recommendations.")

    elif run and not query.strip():
        st.warning("Please enter a query first.")

# ── TAB 2: Results Analysis Dashboard ────────────────────────────────────────
with tab2:
    st.markdown("""<div class="dashboard-header">
        <h3>📊 Live Results Analysis Dashboard</h3>
        <p>Updates automatically with each query you run. Tracks the metrics reported in the paper.</p>
    </div>""", unsafe_allow_html=True)

    history = st.session_state.query_history

    if not history:
        st.info("Run some queries in the Recommendations tab to see live analysis here.")
    else:
        hist_df = pd.DataFrame(history)

        avg_non_overlap    = hist_df["non_overlap_pct"].mean()
        avg_baseline_score = hist_df["baseline_avg_score"].mean()
        avg_llm_rank       = hist_df["llm_top_tfidf_rank"].dropna().mean()
        total_non_overlap  = hist_df["non_overlap"].sum()
        total_possible     = len(hist_df) * 5

        st.markdown("#### Aggregate Metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#028090">{avg_non_overlap:.0f}%</div><div class="metric-label">Avg hybrid non-overlap rate</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#1B3A6B">{avg_baseline_score:.3f}</div><div class="metric-label">Avg baseline cosine score</div></div>', unsafe_allow_html=True)
        with c3:
            rank_display = f"#{avg_llm_rank:.1f}" if not np.isnan(avg_llm_rank) else "N/A"
            st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#F4A621">{rank_display}</div><div class="metric-label">Avg TF-IDF rank of LLM top pick</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#2DB37A">{total_non_overlap}/{total_possible}</div><div class="metric-label">Total non-baseline picks</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### Non-Overlap Rate per Query (%)")
            st.bar_chart(hist_df[["query","non_overlap_pct"]].set_index("query"), color="#028090", height=280)
        with chart_col2:
            st.markdown("#### Baseline Avg Cosine Score per Query")
            st.bar_chart(hist_df[["query","baseline_avg_score"]].set_index("query"), color="#1B3A6B", height=280)

        rank_data = hist_df.dropna(subset=["llm_top_tfidf_rank"])
        if not rank_data.empty:
            st.markdown("#### TF-IDF Rank of LLM's Top Pick")
            st.caption("Higher rank number = LLM had to go further down the TF-IDF list to find the best match.")
            st.bar_chart(rank_data[["query","llm_top_tfidf_rank"]].set_index("query"), color="#F4A621", height=220)

        st.markdown("---")
        st.markdown("#### Per-Query Results Table")
        display_df = hist_df.copy()
        display_df["non_overlap_pct"] = display_df["non_overlap_pct"].astype(str) + "%"
        display_df["llm_top_tfidf_rank"] = display_df["llm_top_tfidf_rank"].apply(lambda x: f"#{int(x)}" if pd.notna(x) else "N/A")
        display_df = display_df.rename(columns={
            "query":"Query","baseline_avg_score":"Baseline Avg Score",
            "non_overlap":"Non-Overlap (count)","non_overlap_pct":"Non-Overlap %",
            "llm_top_tfidf_rank":"LLM #1 TF-IDF Rank"
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(f"Non-overlap rate = proportion of hybrid recommendations not in baseline top 5. TF-IDF rank shows how far the LLM reached into the {CANDIDATE_POOL}-candidate pool.")

st.markdown("---")
st.caption("Hybrid Book Recommendation System · TF-IDF Baseline vs LLM-Enhanced · Streamlit + Anthropic Claude")
