"""
Ripplet Interactive Dashboard

Integrates all four core analytical modules:
1. Sentiment & Emotion Timeline (modules/sentiment)
2. Trending Topics Detection (modules/trends)
3. Demographics & Community Profiling (modules/demographics)
4. Interactive Interaction Network & Key Opinion Leaders (modules/network)
Plus:
- Unified Time Range Slider driving all sections from a single DataFrame
- Influencer Spotlight Panel proving connected dataset flow
- Graceful empty-window handling
"""

from collections.abc import Iterator
from datetime import datetime, time
import base64
import html
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import networkx as nx
import pandas as pd
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components

# Import pipeline & shared analytics modules
from modules.demographics import profile_demographics
from modules.network import build_network
from modules.sentiment import analyze_sentiment
from modules.trends import detect_trends
from pipeline.loader import get_time_bounds, get_window

# Asset paths
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
FAVICON_PATH = ASSETS_DIR / "ripplet_favicon_64.png"
LOGO_PATH = ASSETS_DIR / "ripplet_logo.png"

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Theming
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ripplet",
    page_icon=Image.open(FAVICON_PATH) if FAVICON_PATH.exists() else "dashboard/assets/ripplet_favicon_64.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for rich aesthetics, cards, and badges
st.markdown(
    """
    <style>
    .header-container {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 0.3rem;
    }
    .header-logo {
        height: 56px;
        width: 56px;
        object-fit: contain;
        border-radius: 50%;
        display: inline-block;
        vertical-align: middle;
        box-shadow: 0 2px 10px rgba(29, 155, 240, 0.2);
    }
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1D9BF0, #8A2BE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin: 0;
        display: inline-block;
    }
    .subtitle {
        color: #6B7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .spotlight-card {
        background: linear-gradient(145deg, #F0F9FF, #E0F2FE);
        border: 1px solid #BAE6FD;
        border-radius: 14px;
        padding: 22px;
        margin-top: 15px;
        margin-bottom: 25px;
    }
    .trend-pill {
        display: inline-block;
        background-color: #EDE9FE;
        color: #6D28D9;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-sentiment {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-pos { background-color: #DCFCE7; color: #166534; }
    .badge-neg { background-color: #FEE2E2; color: #991B1B; }
    .badge-neu { background-color: #F1F5F9; color: #475569; }

    /* Low-confidence framing badge for irony detection */
    .badge-irony {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.82rem;
        background-color: #F8FAFC;
        color: #64748B;
        border: 1.5px dashed #94A3B8;
        cursor: help;
        transition: all 0.2s ease;
        margin-left: 6px;
        vertical-align: middle;
        text-decoration: none;
        box-shadow: none;
    }
    .badge-irony:hover {
        border-color: #475569;
        color: #334155;
        background-color: #F1F5F9;
    }
    .badge-irony-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        background-color: #E2E8F0;
        color: #475569;
        font-size: 0.72rem;
        font-weight: 700;
        line-height: 1;
    }
    .stream-tweet-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .bucket-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 8px;
        flex-wrap: wrap;
        gap: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Data Loader (Single Source of Truth)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_analyzed_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Fetches the slice from pipeline.loader.get_window() once, then
    passes it through analyze_sentiment() to enrich with sentiment and emotion.
    All dashboard sections derive strictly from this returned DataFrame.
    """
    raw_df = get_window(start_ts, end_ts)
    if raw_df.empty:
        return raw_df
    enriched_df = analyze_sentiment(raw_df)
    return enriched_df


@st.cache_data(show_spinner=False)
def get_demographics_cached(start, end, sample_size: int = 3000) -> dict:
    """
    Cached wrapper for demographics profiling over a sampled slice of the time window.
    Keyed on start/end timestamps to avoid cache serialization overhead on large DataFrames.
    """
    df = get_window(start, end)
    if df.empty:
        return profile_demographics(df)
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df
    return profile_demographics(sample_df)


# -----------------------------------------------------------------------------
# Reusable Card Rendering Function
# -----------------------------------------------------------------------------
def render_tweet_card(
    row: pd.Series | dict,
    context_label: str | None = None,
    matched_trends: list[str] | None = None,
) -> None:
    """
    Renders a unified, styled tweet card with sentiment, emotion, and
    optional lower-confidence irony indicator.

    Escapes HTML and converts newlines to <br> to prevent CommonMark from
    breaking out of HTML blocks into literal code blocks.
    Ensures flush-left formatting without leading indentation to avoid
    CommonMark indented code block parsing.
    """
    u_id = html.escape(str(row.get("user_id", "unknown")))
    raw_text = str(row.get("text", ""))
    clean_text = html.escape(raw_text).replace("\n", "<br>")
    ts_val = pd.to_datetime(row.get("timestamp"))
    ts_str = ts_val.strftime("%Y-%m-%d %H:%M") if pd.notnull(ts_val) else ""

    row_s_lbl = str(row.get("sentiment_label", "neutral")).lower()
    row_s_scr = float(row.get("sentiment_score", 0.50))
    row_e_lbl = str(row.get("emotion_label", "neutral")).lower()
    row_e_scr = float(row.get("emotion_score", 0.50))
    row_i_lbl = str(row.get("irony_label", "non_irony")).lower()

    row_b_class = (
        "badge-pos"
        if row_s_lbl == "positive"
        else ("badge-neg" if row_s_lbl == "negative" else "badge-neu")
    )

    # Lower-confidence irony badge (only when irony_label == "irony")
    row_irony_html = ""
    if row_i_lbl == "irony":
        row_irony_html = (
            '<span class="badge-irony" '
            'title="Irony detection model, ~70-80% accuracy on benchmark data — treat as a signal, not a fact.">'
            '<span class="badge-irony-icon">?</span> Possible irony/sarcasm</span>'
        )

    # Optional context label (e.g. role in Influencer Spotlight)
    context_html = ""
    if context_label:
        context_html = (
            f'  <div style="font-size: 0.88rem; color: #475569; font-weight: 600; margin-bottom: 6px;">\n'
            f'    {html.escape(context_label)}:\n'
            f'  </div>\n'
        )

    # Optional matched trending terms
    trends_html = ""
    if matched_trends is not None:
        if matched_trends:
            pills = "".join(f'<span class="trend-pill">#{html.escape(t)}</span> ' for t in matched_trends)
            trends_content = pills.strip()
        else:
            trends_content = '<span style="color: #94A3B8; font-size: 0.88rem;">None in this specific tweet</span>'
        trends_html = (
            f'  <div style="margin-top: 10px; padding-top: 8px; border-top: 1px dashed #E2E8F0;">\n'
            f'    <span style="font-size: 0.85rem; font-weight: 700; color: #334155; margin-right: 6px;">Matched Trending Terms:</span>\n'
            f'    {trends_content}\n'
            f'  </div>\n'
        )

    # Flush-left HTML string without leading indentation to prevent CommonMark indented code block parsing
    card_html = (
        f'<div class="stream-tweet-card">\n'
        f'{context_html}'
        f'  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 6px;">\n'
        f'    <span style="font-weight: 700; color: #0284C7; font-size: 0.95rem;">\n'
        f'      @{u_id} <span style="color: #64748B; font-weight: 400; font-size: 0.85rem;">• {ts_str}</span>\n'
        f'    </span>\n'
        f'    <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 4px;">\n'
        f'      <span class="badge-sentiment {row_b_class}">{row_s_lbl.upper()} ({row_s_scr:.2f})</span>\n'
        f'      <span class="badge-sentiment badge-neu" style="margin-left: 4px;">Emotion: {row_e_lbl.upper()} ({row_e_scr:.2f})</span>\n'
        f'      {row_irony_html}\n'
        f'    </div>\n'
        f'  </div>\n'
        f'  <div style="color: #1E293B; font-size: 0.95rem; line-height: 1.45; background: #FFFFFF; padding: 10px 14px; border-radius: 8px; border: 1px solid #E2E8F0;">\n'
        f'    "{clean_text}"\n'
        f'  </div>\n'
        f'{trends_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main() -> None:
    if LOGO_PATH.exists():
        with open(LOGO_PATH, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f"""
            <div class="header-container">
                <img src="data:image/png;base64,{logo_b64}" alt="Ripplet Logo" class="header-logo" />
                <span class="main-title">Ripplet</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="main-title">Ripplet</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="subtitle">Audience intelligence from sentiment, trends, and influence, combined.</div>',
        unsafe_allow_html=True,
    )

    # 1. Bounded Time Range Slider
    earliest_dt, latest_dt = get_time_bounds()
    earliest_date = earliest_dt.date()
    latest_date = latest_dt.date()

    # Default window: 3-day sample around early August 2020
    default_start_date = earliest_date
    default_end_date = min(earliest_date + pd.Timedelta(days=3), latest_date)

    st.markdown("### ⏱️ Select Time Window")
    slider_dates = st.slider(
        "Time Range Slider (Pipeline Bounds)",
        min_value=earliest_date,
        max_value=latest_date,
        value=(default_start_date, default_end_date),
        format="YYYY-MM-DD",
        help="Select start and end dates. All sections below update from this single window.",
    )

    start_date, end_date = slider_dates
    start_ts = pd.Timestamp.combine(start_date, time.min)
    end_ts = pd.Timestamp.combine(end_date, time.max)

    # Fetch and analyze window once
    with st.spinner("Loading window and analyzing sentiment..."):
        df_window = load_analyzed_window(start_ts, end_ts)

    # Top summary metrics
    total_tweets = len(df_window)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Tweets", f"{total_tweets:,}")
    with col_m2:
        unique_authors = df_window["user_id"].nunique() if not df_window.empty else 0
        st.metric("Unique Authors", f"{unique_authors:,}")
    with col_m3:
        has_mentions = (
            df_window["text"].str.contains("@", na=False).sum()
            if not df_window.empty
            else 0
        )
        pct_mentions = (has_mentions / max(total_tweets, 1)) * 100
        st.metric("Mentions / Interactions", f"{pct_mentions:.1f}%")
    with col_m4:
        span_days = max((end_date - start_date).days, 1)
        st.metric("Window Span", f"{span_days} days")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Empty Window Guard
    # -------------------------------------------------------------------------
    if df_window.empty:
        st.warning(
            f"⚠️ No tweets found in the selected time window [{start_date} to {end_date}]. "
            "Please expand the time range slider above."
        )
        st.info("All downstream sections are waiting for a non-empty time window.")
        return

    # -------------------------------------------------------------------------
    # Section (a): Sentiment & Emotion Timeline
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">📈 (a) Sentiment & Emotion Timeline</div>', unsafe_allow_html=True)
    st.caption("Distribution of tweet polarity confidence and discrete emotion categories across the selected range.")

    col_s1, col_s2 = st.columns([3, 2])

    with col_s1:
        # Resample sentiment polarity confidence over time
        resample_freq = "1D" if span_days > 2 else "1h"
        try:
            ts_indexed = df_window.copy()
            ts_indexed["timestamp"] = pd.to_datetime(ts_indexed["timestamp"])
            ts_indexed = ts_indexed.set_index("timestamp")

            sentiment_resampled = (
                ts_indexed.groupby([pd.Grouper(freq=resample_freq), "sentiment_label"])
                .size()
                .unstack(fill_value=0)
            )
            # Ensure standard columns exist
            for col in ["positive", "neutral", "negative"]:
                if col not in sentiment_resampled.columns:
                    sentiment_resampled[col] = 0

            st.markdown(f"**Tweet Volume by Sentiment ({resample_freq} buckets)**")
            st.area_chart(sentiment_resampled[["positive", "neutral", "negative"]])
        except Exception as e:
            st.info(f"Timeline generation preview: {e}")

    with col_s2:
        st.markdown("**Emotion Category Share**")
        if "emotion_label" in df_window.columns:
            emotion_counts = df_window["emotion_label"].value_counts()
            st.bar_chart(emotion_counts)
        else:
            st.info("Emotion scores not available.")

    # -------------------------------------------------------------------------
    # Tweet Stream & Time Bucket Breakdown for Section (a)
    # -------------------------------------------------------------------------
    st.markdown(
        '<div style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-top: 1.4rem; margin-bottom: 0.3rem;">'
        '🔎 Timeline Tweet & Bucket Explorer'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Inspect individual tweets or resampled time buckets with sentiment, discrete emotion, and signal-framed irony flags.")

    tab_tweets, tab_buckets = st.tabs(["💬 Window Tweet Feed (Sample)", "📅 Time Bucket Breakdown"])

    with tab_tweets:
        col_f1, col_f2 = st.columns([2, 3])
        with col_f1:
            feed_filter = st.selectbox(
                "Filter tweets in stream:",
                ["All Sample Tweets", "Flagged Irony Only (Signal)", "Positive Sentiment", "Negative Sentiment"],
                key="sec_a_feed_filter",
            )

        # Filter the DataFrame
        if feed_filter == "Flagged Irony Only (Signal)":
            display_tweets_df = df_window[df_window["irony_label"].astype(str).str.lower() == "irony"]
        elif feed_filter == "Positive Sentiment":
            display_tweets_df = df_window[df_window["sentiment_label"].astype(str).str.lower() == "positive"]
        elif feed_filter == "Negative Sentiment":
            display_tweets_df = df_window[df_window["sentiment_label"].astype(str).str.lower() == "negative"]
        else:
            # For "All Sample Tweets", create a representative slice that includes both irony and non-irony if present
            irony_subset = df_window[df_window["irony_label"].astype(str).str.lower() == "irony"].head(3)
            non_irony_subset = df_window[df_window["irony_label"].astype(str).str.lower() != "irony"].head(5)
            display_tweets_df = pd.concat([irony_subset, non_irony_subset]).sort_values(by="timestamp", ascending=False)

        if not display_tweets_df.empty:
            for _, row in display_tweets_df.head(8).iterrows():
                render_tweet_card(row)
        else:
            st.info("No tweets found matching the selected stream filter.")

    with tab_buckets:
        try:
            ts_indexed = df_window.copy()
            ts_indexed["timestamp"] = pd.to_datetime(ts_indexed["timestamp"])
            ts_indexed = ts_indexed.set_index("timestamp")

            bucket_groups = ts_indexed.groupby(pd.Grouper(freq=resample_freq))
            has_buckets = False

            for bucket_time, bucket_df in bucket_groups:
                if bucket_df.empty:
                    continue
                has_buckets = True
                b_total = len(bucket_df)
                b_date_str = bucket_time.strftime("%Y-%m-%d" if resample_freq == "1D" else "%Y-%m-%d %H:%M")
                
                # Sentiment breakdown in this bucket
                pos_c = (bucket_df["sentiment_label"] == "positive").sum()
                neg_c = (bucket_df["sentiment_label"] == "negative").sum()
                neu_c = (bucket_df["sentiment_label"] == "neutral").sum()
                
                # Dominant sentiment
                dom_s = "positive" if pos_c >= max(neg_c, neu_c) else ("negative" if neg_c >= neu_c else "neutral")
                b_class = "badge-pos" if dom_s == "positive" else ("badge-neg" if dom_s == "negative" else "badge-neu")

                # Check irony count in this time bucket
                irony_count = (bucket_df["irony_label"].astype(str).str.lower() == "irony").sum()
                bucket_irony_html = ""
                if irony_count > 0:
                    bucket_irony_html = (
                        f'<span class="badge-irony" '
                        f'title="Irony detection model, ~70-80% accuracy on benchmark data — treat as a signal, not a fact.">'
                        f'<span class="badge-irony-icon">?</span> Possible irony ({irony_count} flagged)</span>'
                    )

                bucket_html = (
                    f'<div class="bucket-row">\n'
                    f'  <div>\n'
                    f'    <span style="font-weight: 700; color: #0F172A; font-size: 0.95rem;">📅 {b_date_str}</span>\n'
                    f'    <span style="color: #64748B; font-size: 0.88rem; margin-left: 10px;">Total: {b_total:,} tweets</span>\n'
                    f'  </div>\n'
                    f'  <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 6px;">\n'
                    f'    <span class="badge-sentiment {b_class}">Dominant: {dom_s.upper()}</span>\n'
                    f'    <span style="font-size: 0.85rem; color: #475569; margin-left: 4px;">(+{pos_c} / -{neg_c} / ~{neu_c})</span>\n'
                    f'    {bucket_irony_html}\n'
                    f'  </div>\n'
                    f'</div>'
                )
                st.markdown(bucket_html, unsafe_allow_html=True)
            if not has_buckets:
                st.info("No time bucket data available.")
        except Exception as e:
            st.info(f"Time bucket breakdown preview: {e}")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Section (b): Trending Topics
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">🔥 (b) Trending Topics (detect_trends)</div>', unsafe_allow_html=True)
    st.caption("Breakout hashtags and keywords exceeding the rolling average by 2.0x within the window.")

    # Partition the single df_window into daily batches for detect_trends
    def generate_window_batches() -> Iterator[pd.DataFrame]:
        for _, day_batch in df_window.groupby(pd.Grouper(key="timestamp", freq="1D")):
            if not day_batch.empty:
                yield day_batch

    with st.spinner("Detecting trends across window batches..."):
        trends_df = detect_trends(
            generate_window_batches(),
            window_size=7,
            threshold=2.0,
            min_frequency=3,
        )

    col_t1, col_t2 = st.columns([3, 2])

    trending_terms_set: set[str] = set()

    with col_t1:
        if not trends_df.empty and trends_df["is_trending"].any():
            trending_only = trends_df[trends_df["is_trending"]].copy()
            # Group by term to find total frequency among trending occurrences
            top_trending = (
                trending_only.groupby("term")["frequency"]
                .max()
                .sort_values(ascending=False)
                .head(15)
            )
            trending_terms_set = set(top_trending.index.tolist())

            st.markdown("**Top Trending Terms (Spike Ratio >= 2.0x)**")
            st.bar_chart(top_trending)
        else:
            st.info("No terms exceeded the 2.0x trend spike threshold in this window. (Requires multiple successive batches to establish a baseline).")

    with col_t2:
        st.markdown("**Recent Trend Feed**")
        if not trends_df.empty:
            recent_display = trends_df.tail(10)[["term", "frequency", "is_trending"]].copy()
            st.dataframe(recent_display, use_container_width=True, hide_index=True)
        else:
            st.write("No term activity recorded.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Section (c): Demographics Breakdown
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">🌍 (c) Demographics Breakdown (profile_demographics)</div>', unsafe_allow_html=True)
    st.caption("Strictly aggregated, anonymized community breakdown by bio language, geography, and profession.")

    with st.spinner("Profiling demographics..."):
        demo_summary = get_demographics_cached(start_ts, end_ts)

    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        st.markdown("**Top Languages (langdetect)**")
        lang_data = demo_summary.get("languages", {})
        if lang_data:
            df_lang = pd.Series(lang_data).head(8)
            st.bar_chart(df_lang)
        else:
            st.write("No language data available.")

    with col_d2:
        st.markdown("**Top Regions (pycountry)**")
        region_data = demo_summary.get("regions", {})
        if region_data:
            # Filter out or place unknown at the end for clean view
            top_regions = pd.Series(region_data).head(8)
            st.bar_chart(top_regions)
        else:
            st.write("No location data available.")

    with col_d3:
        st.markdown("**Interest & Profession Domains**")
        interest_data = demo_summary.get("interest_categories", {})
        if interest_data:
            df_interest = pd.Series(interest_data).head(8)
            st.bar_chart(df_interest)
        else:
            st.write("No bio interest data available.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Section (d): Interactive Network Graph
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">🕸️ (d) Interaction Network Graph (build_network)</div>', unsafe_allow_html=True)
    st.caption("Directed graph where nodes are user accounts and edges represent @mentions and retweets.")

    with st.spinner("Building interaction network & computing PageRank..."):
        G, top_10_kols = build_network(df_window, return_tuple=True)

    if len(G.nodes) > 0:
        col_net1, col_net2 = st.columns([3, 1])

        with col_net2:
            st.markdown(f"**Network Summary**")
            st.write(f"- **Total Nodes**: {len(G.nodes):,}")
            st.write(f"- **Total Edges**: {len(G.edges):,}")
            max_render = st.slider("Max Nodes in View", min_value=15, max_value=80, value=35, step=5)
            st.caption("Pruned to highest-PageRank sub-network for smooth interactive rendering.")

        with col_net1:
            # Construct interactive PyVis graph for top sub-network
            top_nodes = [k["user_id"] for k in top_10_kols[:max_render]]
            # Expand to include immediate neighbors of top nodes up to max_render
            sub_nodes_set = set(top_nodes)
            for n in top_nodes:
                if len(sub_nodes_set) >= max_render:
                    break
                sub_nodes_set.update(list(G.successors(n))[:3])
                sub_nodes_set.update(list(G.predecessors(n))[:3])

            subG = G.subgraph(sub_nodes_set)

            net = Network(height="460px", width="100%", directed=True, cdn_resources="remote")

            # Color palette based on sentiment score
            for node in subG.nodes():
                pr = G.nodes[node].get("pagerank", 0.0)
                sent = G.nodes[node].get("sentiment_score")

                # Node size scaled by PageRank
                size = 12 + int(pr * 1200)

                # Node color mapped to sentiment score
                if sent is not None:
                    if sent >= 0.70:
                        color = "#10B981"  # Positive green
                    elif sent <= 0.40:
                        color = "#EF4444"  # Negative red
                    else:
                        color = "#3B82F6"  # Neutral blue
                    s_str = f"{sent:.2f}"
                else:
                    color = "#94A3B8"  # Slate gray
                    s_str = "N/A (Mentioned only)"

                tooltip = (
                    f"User: @{node}\n"
                    f"PageRank: {pr:.5f}\n"
                    f"Betweenness: {G.nodes[node].get('betweenness', 0.0):.5f}\n"
                    f"Avg Sentiment: {s_str}\n"
                    f"In-Degree: {G.in_degree(node)}"
                )

                net.add_node(node, label=node, size=size, color=color, title=tooltip)

            for u, v, data in subG.edges(data=True):
                w = data.get("weight", 1)
                net.add_edge(u, v, value=w, color="#CBD5E1", arrows="to")

            net.set_options(
                """
                {
                    "physics": {
                        "forceAtlas2Based": {
                            "gravitationalConstant": -50,
                            "centralGravity": 0.01,
                            "springLength": 100,
                            "springConstant": 0.08
                        },
                        "maxVelocity": 30,
                        "solver": "forceAtlas2Based",
                        "timestep": 0.35,
                        "stabilization": {"iterations": 80}
                    }
                }
                """
            )

            html_content = net.generate_html()
            components.html(html_content, height=480, scrolling=False)
    else:
        st.info("No interaction edges (@mentions or retweets) found in the selected window.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Influencer Spotlight Panel
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">🌟 Ripplet Influencer Spotlight Panel</div>', unsafe_allow_html=True)
    st.caption(
        "Demonstrates Ripplet's end-to-end dataset unity: pick any Key Opinion Leader from Section (d), "
        "inspect their highest-engagement tweet from the shared DataFrame, its sentiment labels from Section (a), "
        "and cross-referenced trending topics from Section (b)."
    )

    if top_10_kols:
        kol_options = [f"#{k['rank']} @{k['user_id']} (PageRank: {k['pagerank']:.4f})" for k in top_10_kols]
        selected_option = st.selectbox("Select Key Opinion Leader (Top 10 Nodes):", kol_options)
        selected_index = kol_options.index(selected_option)
        selected_kol = top_10_kols[selected_index]
        selected_user = selected_kol["user_id"]

        # Pull tweets from the EXACT SAME df_window
        # 1. First look for tweets authored by this user
        authored_tweets = df_window[df_window["user_id"].astype(str).str.lower() == selected_user.lower()]

        if not authored_tweets.empty:
            # Pick highest engagement tweet (by followers or length)
            best_tweet = (
                authored_tweets.sort_values(by="followers", ascending=False).iloc[0]
                if "followers" in authored_tweets.columns
                else authored_tweets.iloc[0]
            )
            tweet_role = "Authored Tweet"
        else:
            # If user was mentioned, find the tweet mentioning them
            mention_pattern = re.compile(rf"@{re.escape(selected_user)}\b", re.IGNORECASE)
            mention_mask = df_window["text"].astype(str).str.contains(mention_pattern)
            mentioning_tweets = df_window[mention_mask]

            if not mentioning_tweets.empty:
                best_tweet = (
                    mentioning_tweets.sort_values(by="followers", ascending=False).iloc[0]
                    if "followers" in mentioning_tweets.columns
                    else mentioning_tweets.iloc[0]
                )
                tweet_role = f"Top Tweet Mentioning @{selected_user} (Authored by @{best_tweet.get('user_id', 'unknown')})"
            else:
                best_tweet = None
                tweet_role = "No tweet text available"

        # Render Spotlight
        kol_summary_html = (
            f'<div class="spotlight-card">\n'
            f'  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">\n'
            f'    <div>\n'
            f'      <span style="font-size: 1.4rem; font-weight: 800; color: #0369A1;">@{html.escape(selected_user)}</span>\n'
            f'      <span style="color: #64748B; font-size: 0.95rem; margin-left: 10px;">Rank #{selected_kol["rank"]} | In-Degree: {selected_kol["in_degree"]} | PageRank: {selected_kol["pagerank"]:.5f}</span>\n'
            f'    </div>\n'
            f'  </div>\n'
            f'</div>'
        )
        st.markdown(kol_summary_html, unsafe_allow_html=True)

        if best_tweet is not None:
            # Cross-reference with Section (b) trending terms
            text_lower = str(best_tweet.get("text", "")).lower()
            tweet_hashtags = str(best_tweet.get("hashtags", "")).lower()
            matched_trends = [
                term
                for term in trending_terms_set
                if (term in text_lower or term in tweet_hashtags)
            ]

            render_tweet_card(
                best_tweet,
                context_label=tweet_role,
                matched_trends=matched_trends,
            )
        else:
            st.info(f"No tweets directly matching @{selected_user} in this time window.")
    else:
        st.info("No Key Opinion Leaders identified in this window.")


if __name__ == "__main__":
    main()
