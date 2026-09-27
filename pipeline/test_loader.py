"""
Sanity-Check Test Script for Timeline-Replay Data Loader

Verifies:
1. get_time_bounds() returns valid chronological boundaries.
2. get_window(start, end) extracts the correct subset within [start, end).
3. stream_batches(freq='1D') generates successive time-windowed batches.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Reconfigure stdout for utf-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from pipeline.loader import get_time_bounds, get_window, stream_batches


def run_sanity_checks() -> None:
    print("=" * 65)
    print(" RUNNING TIMELINE-REPLAY LOADER SANITY CHECKS")
    print("=" * 65)

    # 1. Test get_time_bounds()
    print("\n[TEST 1] Testing get_time_bounds()...")
    earliest, latest = get_time_bounds()
    print(f"  Earliest Timestamp: {earliest}")
    print(f"  Latest Timestamp:   {latest}")
    assert earliest < latest, f"Earliest ({earliest}) must precede latest ({latest})"
    assert isinstance(earliest, datetime), "Earliest bound must be a datetime instance"
    assert isinstance(latest, datetime), "Latest bound must be a datetime instance"
    print("  [PASS] get_time_bounds() passed.")

    # 2. Test get_window()
    print("\n[TEST 2] Testing get_window()...")
    # Pull a 24-hour window starting 5 days after earliest
    window_start = pd.Timestamp(earliest).floor("D") + pd.Timedelta(days=5)
    window_end = window_start + pd.Timedelta(days=1)

    print(f"  Requesting window: [{window_start}  to  {window_end})")
    window_df = get_window(window_start, window_end)
    row_count = len(window_df)
    print(f"  Rows retrieved: {row_count:,}")
    print(f"  DataFrame columns: {list(window_df.columns)}")

    assert row_count > 0, "Expected non-empty window for sample date."
    min_in_window = window_df["timestamp"].min()
    max_in_window = window_df["timestamp"].max()
    print(f"  Timestamp range in window: {min_in_window}  to  {max_in_window}")

    assert min_in_window >= window_start, f"Min in window {min_in_window} must be >= {window_start}"
    assert max_in_window < window_end, f"Max in window {max_in_window} must be < {window_end}"

    expected_base_cols = [
        "tweet_id",
        "timestamp",
        "user_id",
        "text",
        "bio",
        "location",
        "followers",
        "hashtags",
    ]
    for col in expected_base_cols:
        assert col in window_df.columns, f"Missing expected column '{col}' in DataFrame"
    print("  [PASS] get_window() passed.")

    # 3. Test stream_batches()
    print("\n[TEST 3] Testing stream_batches(freq='1D')...")
    total_streamed_rows = 0
    batch_count = 0
    preview_limit = 3

    print(f"  Streaming batches (previewing first {preview_limit}):")
    for i, batch in enumerate(stream_batches(freq="1D")):
        batch_count += 1
        total_streamed_rows += len(batch)
        if i < preview_limit:
            ts_summary = (
                f"{batch['timestamp'].min()} -> {batch['timestamp'].max()}"
                if not batch.empty
                else "empty batch"
            )
            print(f"    Batch {i + 1:02d}: {len(batch):>5,} rows | Time: {ts_summary}")

    print(f"  Total daily batches yielded: {batch_count}")
    print(f"  Total rows across all batches: {total_streamed_rows:,}")

    # Verify total matches whole dataset
    full_df = get_window(earliest, latest + timedelta(seconds=1))
    assert total_streamed_rows == len(full_df), (
        f"Streamed row count ({total_streamed_rows}) should equal full dataset ({len(full_df)})"
    )
    print("  [PASS] stream_batches() passed.")

    # 4. Sample Row Inspection
    print("\n[TEST 4] Inspecting sample cleaned row:")
    sample_row = window_df.iloc[0].to_dict()
    for k, v in sample_row.items():
        v_str = str(v)
        if len(v_str) > 70:
            v_str = v_str[:67] + "..."
        print(f"    {k:>10}: {v_str}")

    print("\n" + "=" * 65)
    print(" ALL SANITY CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_sanity_checks()
