"""
Sanity-Check and Verification Test Script for Twitter Network Module

Verifies:
1. Synthetic controlled unit tests:
   - Interaction regex extraction (@mentions and RT @username)
   - Directed graph construction (edges, weights, interaction types)
   - PageRank and betweenness centrality computation
   - Node sentiment_score attachment from DataFrame
   - Top 10 Key Opinion Leader (KOL) ranking
   - Edge cases: empty DataFrame, tweets without mentions
2. End-to-end integration test:
   - Consumes real window DataFrame from pipeline.loader.get_window()
   - Simulates incoming sentiment_score column
   - Verifies network construction, PageRank, and top 10 KOL extraction
   - Enforces Rule One compliance (only receives DataFrame from pipeline)
"""

from datetime import timedelta
from pathlib import Path
import sys

# Configure stdout for utf-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import networkx as nx
import pandas as pd
from modules.network import build_network, extract_interactions
from pipeline.loader import get_time_bounds, get_window


def run_unit_tests() -> None:
    print("=" * 65)
    print(" RUNNING NETWORK MODULE UNIT TESTS (SYNTHETIC DATA)")
    print("=" * 65)

    # 1. Test interaction extraction
    print("\n[TEST 1] Testing interaction regex parsing...")
    rts, mentions = extract_interactions("RT @BorisJohnson: Important announcement with @BBCNews and @RishiSunak")
    assert "BorisJohnson" in rts, "Expected BorisJohnson in RT targets"
    assert "BorisJohnson" in mentions
    assert "BBCNews" in mentions
    assert "RishiSunak" in mentions
    print(f"  [PASS] Extracted RT: {rts}, Mentions: {mentions}")

    # 2. Test controlled graph construction
    print("\n[TEST 2] Testing graph construction and metrics...")
    synthetic_df = pd.DataFrame([
        # user1 mentions leaderA twice and leaderB once
        {"user_id": "user1", "text": "Talking to @leaderA and again to @leaderA", "sentiment_score": 0.8},
        {"user_id": "user1", "text": "Now saying hi to @leaderB", "sentiment_score": 0.6},
        # user2 mentions leaderA
        {"user_id": "user2", "text": "RT @leaderA: Breaking news update", "sentiment_score": 0.1},
        # user3 mentions leaderA
        {"user_id": "user3", "text": "Great point by @leaderA", "sentiment_score": -0.2},
        # leaderA mentions leaderB
        {"user_id": "leaderA", "text": "Agreed with @leaderB on this policy", "sentiment_score": 0.5},
    ])

    G, kols = build_network(synthetic_df)

    assert isinstance(G, nx.DiGraph), "Output must be a networkx.DiGraph"
    assert len(G.nodes) > 0, "Graph must contain nodes"
    assert len(G.edges) > 0, "Graph must contain edges"

    print(f"  Graph size: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Check edge weight: user1 -> leaderA had 2 mentions in first tweet
    assert G.has_edge("user1", "leaderA"), "Edge user1 -> leaderA must exist"
    assert G["user1"]["leaderA"]["weight"] >= 2, "Weight should reflect multiple mentions"
    print(f"  [PASS] Edge user1 -> leaderA weight = {G['user1']['leaderA']['weight']}")

    # Check node attributes
    for node in ["user1", "leaderA", "leaderB"]:
        assert "pagerank" in G.nodes[node], f"Node {node} must have pagerank attribute"
        assert "betweenness" in G.nodes[node], f"Node {node} must have betweenness attribute"
        assert "sentiment_score" in G.nodes[node], f"Node {node} must have sentiment_score attribute"

    # leaderA sentiment should be 0.5 (authored by leaderA)
    assert G.nodes["leaderA"]["sentiment_score"] == 0.5
    # user1 sentiment should be average of (0.8 + 0.6) / 2 = 0.7
    assert round(G.nodes["user1"]["sentiment_score"], 2) == 0.70
    print("  [PASS] Node sentiment scores accurately averaged and attached.")

    # Check Key Opinion Leaders
    assert len(kols) > 0, "KOL list must not be empty"
    top_leader = kols[0]["user_id"]
    print(f"  Top KOL detected: '{top_leader}' (PageRank={kols[0]['pagerank']:.4f})")
    assert top_leader in ["leaderA", "leaderB"], "leaderA or leaderB should have highest PageRank"
    assert G.graph["key_opinion_leaders"] == kols, "KOLs must also be attached to G.graph"
    print("  [PASS] Key Opinion Leaders ranking verified.")

    # 3. Test edge cases
    print("\n[TEST 3] Testing edge cases (empty and non-mention DataFrames)...")
    empty_G, empty_kols = build_network(pd.DataFrame())
    assert isinstance(empty_G, nx.DiGraph) and len(empty_G) == 0
    assert empty_kols == []

    no_mentions_df = pd.DataFrame([
        {"user_id": "u1", "text": "Just enjoying the sunshine.", "sentiment_score": 0.5}
    ])
    nom_G, nom_kols = build_network(no_mentions_df)
    assert isinstance(nom_G, nx.DiGraph) and len(nom_G) == 0
    assert nom_kols == []
    print("  [PASS] Edge cases handled gracefully.")


def run_integration_test() -> None:
    print("\n" + "=" * 65)
    print(" RUNNING INTEGRATION TEST (PIPELINE.GET_WINDOW)")
    print("=" * 65)

    earliest, _ = get_time_bounds()
    # Pull a 12-hour window
    window_start = earliest
    window_end = earliest + timedelta(hours=12)

    print(f"Retrieving window [{window_start} -> {window_end}) via pipeline.get_window()...")
    df = get_window(window_start, window_end).copy()
    print(f"Retrieved {len(df):,} tweets.")

    # Simulate sentiment scores as if passed through modules/sentiment/
    # Assigning reproducible synthetic scores based on hash for testing
    df["sentiment_score"] = [((hash(t) % 100) / 100.0) for t in df["text"]]

    print("Building interaction network...")
    G, kols = build_network(df)

    print(f"Constructed directed network:")
    print(f"  Nodes (Accounts): {len(G.nodes):,}")
    print(f"  Edges (Interactions): {len(G.edges):,}")
    assert len(G.nodes) > 0, "Network should have nodes"
    assert len(G.edges) > 0, "Network should have edges"

    print(f"\nTop {len(kols)} Key Opinion Leaders (KOLs) by PageRank:")
    for kol in kols:
        s_display = f"{kol['sentiment_score']:.2f}" if kol["sentiment_score"] is not None else "N/A"
        print(
            f"  #{kol['rank']:<2} @{kol['user_id']:<22} "
            f"| PageRank: {kol['pagerank']:.5f} "
            f"| In-Degree: {kol['in_degree']:>3} "
            f"| Betweenness: {kol['betweenness']:.5f} "
            f"| Avg Sentiment: {s_display}"
        )

    assert len(kols) == min(10, len(G.nodes))
    assert all("user_id" in k and "pagerank" in k for k in kols)

    print("\n" + "=" * 65)
    print(" ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_unit_tests()
    run_integration_test()
