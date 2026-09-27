"""
Sanity-Check and Verification Test Script for Twitter Trends Detection Module

Verifies:
1. Synthetic controlled unit tests:
   - Cold start baseline handling (batch 0)
   - Rolling average tracking over window_size batches
   - Spike threshold detection (e.g. 2.0x spike flagged as is_trending=True)
   - Sub-threshold frequency non-trending behavior
   - Sparse hashtag fallback with keyword extraction from text
2. End-to-end integration test:
   - Consumes batches from pipeline.loader.stream_batches()
   - Verifies target DataFrame schema:
     ['window_start', 'window_end', 'term', 'frequency', 'is_trending']
   - Asserts pipeline Rule One compliance (derives strictly from pipeline.stream_batches)
"""

from collections.abc import Iterator
from datetime import datetime, timedelta
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

import pandas as pd
from modules.trends import TREND_COLUMNS, detect_trends
from pipeline.loader import stream_batches


def generate_synthetic_batches() -> Iterator[pd.DataFrame]:
    """
    Yields 9 synthetic daily batches to precisely test rolling average and spike logic.
    - 'steady_topic': freq=10 across all batches (should never be trending after batch 0)
    - 'spike_topic': freq=10 in batches 0-7, spikes to 30 in batch 8 (3.0x > 2.0x -> trending)
    - 'emerging_topic': absent in batches 0-7, appears with freq=8 in batch 8 (breakout -> trending)
    - 'low_freq_topic': freq=1 across batches (should not exceed min_frequency)
    """
    base_time = datetime(2020, 8, 1, 0, 0, 0)

    for day in range(9):
        ts_start = base_time + timedelta(days=day)
        ts_end = ts_start + timedelta(hours=23, minutes=59)

        rows = []
        # Steady topic: 10 tweets every day
        for i in range(10):
            rows.append({
                "tweet_id": f"t_steady_{day}_{i}",
                "timestamp": ts_start + timedelta(minutes=i),
                "text": "Discussing steady topic in this tweet.",
                "hashtags": "['steady_topic']",
            })

        # Spike topic: 10 tweets on days 0-7, 30 tweets on day 8
        spike_count = 30 if day == 8 else 10
        for i in range(spike_count):
            rows.append({
                "tweet_id": f"t_spike_{day}_{i}",
                "timestamp": ts_start + timedelta(minutes=i + 15),
                "text": "Discussing spike topic in this tweet.",
                "hashtags": "['spike_topic']",
            })

        # Emerging topic: only appears on day 8 (8 tweets)
        if day == 8:
            for i in range(8):
                rows.append({
                    "tweet_id": f"t_emerging_{day}_{i}",
                    "timestamp": ts_start + timedelta(minutes=i + 30),
                    "text": "Brand new breaking news event!",
                    "hashtags": "['emerging_topic']",
                })

        # Low frequency topic: 1 tweet per day
        rows.append({
            "tweet_id": f"t_low_{day}",
            "timestamp": ts_start + timedelta(minutes=50),
            "text": "Single tweet with rare topic.",
            "hashtags": "['rare_tag']",
        })

        yield pd.DataFrame(rows)


def generate_sparse_hashtag_batches() -> Iterator[pd.DataFrame]:
    """
    Yields batches where the 'hashtags' column is completely NaN / empty,
    testing automated keyword fallback from 'text'.
    """
    base_time = datetime(2020, 8, 1, 0, 0, 0)

    for day in range(3):
        ts_start = base_time + timedelta(days=day)
        rows = []
        kw = "vaccine" if day < 2 else "lockdown"
        count = 15 if day < 2 else 45

        for i in range(count):
            rows.append({
                "tweet_id": f"kw_{day}_{i}",
                "timestamp": ts_start + timedelta(minutes=i),
                "text": f"Breaking medical announcement regarding {kw} trials update today.",
                "hashtags": None,  # No hashtags supplied
            })

        yield pd.DataFrame(rows)


def run_unit_tests() -> None:
    print("=" * 65)
    print(" RUNNING TRENDS MODULE UNIT TESTS (SYNTHETIC BATCHES)")
    print("=" * 65)

    # 1. Test schema & spike detection
    print("\n[TEST 1] Testing rolling spike detection and schema compliance...")
    df_trends = detect_trends(
        generate_synthetic_batches(),
        window_size=7,
        threshold=2.0,
        min_frequency=2,
    )

    print(f"  Total trend records produced: {len(df_trends)}")
    print(f"  Output DataFrame columns: {list(df_trends.columns)}")
    assert list(df_trends.columns) == TREND_COLUMNS, f"Columns must match {TREND_COLUMNS}"
    assert not df_trends.empty, "DataFrame should contain records"

    # Batch 0 check (cold start)
    first_window = df_trends["window_start"].min()
    batch0_df = df_trends[df_trends["window_start"] == first_window]
    assert not batch0_df["is_trending"].any(), "No term should be trending in the first baseline batch."
    print("  [PASS] Cold start baseline verified (no false trends on day 0).")

    # Day 8 check
    latest_window = df_trends["window_start"].max()
    day8_df = df_trends[df_trends["window_start"] == latest_window]

    spike_record = day8_df[day8_df["term"] == "spike_topic"]
    assert not spike_record.empty, "spike_topic should be in day 8 output"
    assert bool(spike_record.iloc[0]["is_trending"]) is True, "spike_topic (30 vs 10 avg) must be flagged is_trending=True"
    print(f"  [PASS] 'spike_topic' detected as trending on day 8 (freq={spike_record.iloc[0]['frequency']}).")

    steady_record = day8_df[day8_df["term"] == "steady_topic"]
    assert not steady_record.empty, "steady_topic should be in day 8 output"
    assert bool(steady_record.iloc[0]["is_trending"]) is False, "steady_topic (10 vs 10 avg) must NOT be trending"
    print(f"  [PASS] 'steady_topic' correctly flagged as non-trending (freq={steady_record.iloc[0]['frequency']}).")

    emerging_record = day8_df[day8_df["term"] == "emerging_topic"]
    assert not emerging_record.empty, "emerging_topic should be in day 8 output"
    assert bool(emerging_record.iloc[0]["is_trending"]) is True, "emerging_topic must be flagged is_trending=True"
    print(f"  [PASS] 'emerging_topic' correctly flagged as trending (breakout).")

    # Verify min_frequency filtered out 'rare_tag'
    rare_record = day8_df[day8_df["term"] == "rare_tag"]
    assert rare_record.empty, "rare_tag with frequency=1 should be filtered by min_frequency=2"
    print("  [PASS] min_frequency filtering verified.")

    # 2. Test sparse hashtag fallback
    print("\n[TEST 2] Testing keyword extraction fallback on sparse hashtags...")
    df_kw_trends = detect_trends(
        generate_sparse_hashtag_batches(),
        window_size=2,
        threshold=2.0,
        min_frequency=3,
        sparsity_threshold=20,
    )
    print(f"  Total records extracted without hashtags: {len(df_kw_trends)}")
    terms = set(df_kw_trends["term"])
    print(f"  Extracted text keywords: {list(terms)[:5]}")
    assert any("vaccine" in t or "lockdown" in t for t in terms), "Expected 'vaccine' or 'lockdown' in extracted keywords"
    print("  [PASS] Sparse hashtag fallback with keyword extraction passed.")


def run_integration_test(max_batches: int = 10) -> None:
    print("\n" + "=" * 65)
    print(" RUNNING INTEGRATION TEST (PIPELINE.STREAM_BATCHES)")
    print("=" * 65)

    print(f"Streaming first {max_batches} daily batches from pipeline/data.parquet...")

    def batch_slice() -> Iterator[pd.DataFrame]:
        for i, b in enumerate(stream_batches(freq="1D")):
            if i >= max_batches:
                break
            yield b

    df_trends = detect_trends(
        batch_slice(),
        window_size=7,
        threshold=2.0,
        min_frequency=5,
        top_terms_per_batch=50,
    )

    print(f"\nIntegration run completed:")
    print(f"  Total records generated: {len(df_trends):,}")
    print(f"  Columns: {list(df_trends.columns)}")
    assert list(df_trends.columns) == TREND_COLUMNS, f"Columns must match {TREND_COLUMNS}"

    trending_only = df_trends[df_trends["is_trending"]]
    print(f"  Trending term count: {len(trending_only):,}")

    assert len(df_trends) > 0, "Expected trend results from real dataset"
    assert len(trending_only) > 0, "Expected some terms to be flagged as trending"

    print("\nSample Trending Terms (Top 10):")
    for _, row in trending_only.head(10).iterrows():
        print(
            f"  Window [{str(row['window_start'])[:10]} -> {str(row['window_end'])[:10]}] "
            f"Term: {row['term']:<20} Freq: {row['frequency']:>5} | Trending: {row['is_trending']}"
        )

    print("\n" + "=" * 65)
    print(" ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_unit_tests()
    run_integration_test(max_batches=8)
