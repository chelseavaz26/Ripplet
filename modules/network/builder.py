"""
Twitter Network Analysis & Key Opinion Leader (KOL) Detection Module

Parses @mentions and 'RT @username' interactions from tweet text,
builds a directed weighted social graph, computes PageRank and betweenness
centrality, attaches average user sentiment, and extracts top 10 Key Opinion Leaders.
"""

from collections import Counter
import re
from typing import Any
import networkx as nx
import pandas as pd

# Regular expressions for interaction extraction
_RT_RE = re.compile(r"\bRT\s+@([A-Za-z0-9_]+)", re.IGNORECASE)
_MENTION_RE = re.compile(r"@([A-Za-z0-9_]+)")


def extract_interactions(text: Any) -> tuple[list[str], list[str]]:
    """
    Parses @mentions and 'RT @username' retweet patterns out of tweet text.

    Returns:
        tuple[list[str], list[str]]: (retweet_targets, mention_targets)
    """
    if text is None or pd.isna(text) or not isinstance(text, str):
        return [], []

    # 1. Retweet targets: 'RT @username'
    rt_matches = _RT_RE.findall(text)
    rt_targets = [m.strip() for m in rt_matches if m.strip()]

    # 2. All @mentions in text
    mention_matches = _MENTION_RE.findall(text)
    mention_targets = [m.strip() for m in mention_matches if m.strip()]

    return rt_targets, mention_targets


def build_network(
    df: pd.DataFrame,
    max_betweenness_nodes: int = 1500,
    return_tuple: bool = True,
) -> tuple[nx.DiGraph, list[dict[str, Any]]] | nx.DiGraph:
    """
    Builds a directed interaction network from a DataFrame of tweets.

    Args:
        df: Input DataFrame from pipeline.get_window() (with optional sentiment_score).
        max_betweenness_nodes: Threshold above which approximate betweenness
            centrality is computed (k=500) for fast live-demo performance.
        return_tuple: If True, returns (G, top_10_kols). If False, returns G.
            In both cases, G.graph["key_opinion_leaders"] is also populated.

    Returns:
        tuple[nx.DiGraph, list[dict[str, Any]]] or nx.DiGraph:
            The directed graph where nodes are user_id and edges represent mentions/RTs,
            plus the ranked list of top 10 nodes by PageRank.
    """
    G = nx.DiGraph()
    G.graph["key_opinion_leaders"] = []

    if df is None or df.empty:
        return (G, []) if return_tuple else G

    # 1. Precompute author sentiment averages if sentiment_score column exists
    author_sentiment: dict[Any, float] = {}
    if "sentiment_score" in df.columns and "user_id" in df.columns:
        valid_sentiment = df.dropna(subset=["sentiment_score"])
        if not valid_sentiment.empty:
            mean_series = valid_sentiment.groupby("user_id")["sentiment_score"].mean()
            author_sentiment = {k: float(v) for k, v in mean_series.items()}

    # 2. Extract interaction edges
    edge_counter: Counter[tuple[str, str]] = Counter()
    edge_types: dict[tuple[str, str], set[str]] = {}

    has_user_id = "user_id" in df.columns
    has_text = "text" in df.columns

    if not has_text:
        return (G, []) if return_tuple else G

    for idx, row in df.iterrows():
        # Determine source user
        if has_user_id and pd.notna(row["user_id"]) and str(row["user_id"]).strip():
            author = str(row["user_id"]).strip()
        else:
            # Fallback surrogate to prevent collapsing all unassigned authors into a single node
            author = f"user_{idx}"

        text = row["text"]
        rt_targets, mention_targets = extract_interactions(text)

        # Process retweet interactions
        for rt in rt_targets:
            if rt.lower() != author.lower():
                pair = (author, rt)
                edge_counter[pair] += 1
                edge_types.setdefault(pair, set()).add("retweet")

        # Process mention interactions
        for mention in mention_targets:
            if mention.lower() != author.lower():
                pair = (author, mention)
                edge_counter[pair] += 1
                edge_types.setdefault(pair, set()).add("mention")

    # If no interaction edges found, return empty graph
    if not edge_counter:
        return (G, []) if return_tuple else G

    # 3. Add weighted edges to directed graph
    for (src, dst), weight in edge_counter.items():
        types = sorted(list(edge_types.get((src, dst), ["mention"])))
        G.add_edge(src, dst, weight=weight, interaction_types=types)

    # 4. Compute PageRank
    try:
        pagerank_scores = nx.pagerank(G, alpha=0.85, weight="weight")
    except Exception:
        # Fallback if graph is empty or un-converged
        pagerank_scores = {n: 0.0 for n in G.nodes()}

    # 5. Compute Betweenness Centrality
    node_count = len(G)
    try:
        if node_count > max_betweenness_nodes:
            # Fast approximation on large graphs for live demos
            k_sample = min(500, node_count)
            betweenness_scores = nx.betweenness_centrality(
                G, k=k_sample, weight="weight", seed=42
            )
        else:
            betweenness_scores = nx.betweenness_centrality(G, weight="weight")
    except Exception:
        betweenness_scores = {n: 0.0 for n in G.nodes()}

    # 6. Attach node attributes: sentiment_score, pagerank, betweenness, degrees
    sentiment_attrs: dict[str, float | None] = {}
    for node in G.nodes():
        # Match case-insensitively with author sentiment if needed
        s_score = author_sentiment.get(node)
        sentiment_attrs[node] = s_score

    nx.set_node_attributes(G, pagerank_scores, "pagerank")
    nx.set_node_attributes(G, betweenness_scores, "betweenness")
    nx.set_node_attributes(G, sentiment_attrs, "sentiment_score")

    # 7. Extract Top 10 Key Opinion Leaders (KOLs) by PageRank
    sorted_kols = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10_kols: list[dict[str, Any]] = []

    for rank, (node, pr) in enumerate(sorted_kols, start=1):
        top_10_kols.append({
            "rank": rank,
            "user_id": node,
            "pagerank": float(pr),
            "betweenness": float(betweenness_scores.get(node, 0.0)),
            "sentiment_score": sentiment_attrs.get(node),
            "in_degree": int(G.in_degree(node)),
            "out_degree": int(G.out_degree(node)),
        })

    G.graph["key_opinion_leaders"] = top_10_kols

    if return_tuple:
        return G, top_10_kols
    return G
