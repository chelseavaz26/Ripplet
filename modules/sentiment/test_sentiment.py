"""
Test Script for Twitter Sentiment Module
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

import pandas as pd
from modules.sentiment import analyze_sentiment
from modules.sentiment.analyzer import REQUIRED_COLUMNS
from pipeline.loader import get_time_bounds, get_window


def run_tests() -> None:
    print("=" * 65)
    print(" RUNNING SENTIMENT MODULE TESTS")
    print("=" * 65)

    # 1. Test missing precomputed columns raises RuntimeError
    print("\n[TEST 1] Testing missing columns raises descriptive RuntimeError...")
    raw_sample_df = pd.DataFrame([
        {"user_id": "u1", "text": "I love this amazing progress and vaccine recovery!"},
        {"user_id": "u2", "text": "Terrible death and tragedy during the crisis."},
    ])

    try:
        analyze_sentiment(raw_sample_df)
        assert False, "Expected RuntimeError when passing DataFrame without precomputed sentiment columns"
    except RuntimeError as exc:
        print(f"  Caught expected error: {exc}")
        assert "enrich_sentiment.py" in str(exc), "Error message must guide user to run enrich_sentiment.py"
        assert "sentiment_label" in str(exc), "Error message must mention missing columns"
        print("  [PASS] Missing precomputed columns correctly raises descriptive RuntimeError.")

    # 2. Test valid precomputed DataFrame pass-through
    print("\n[TEST 2] Testing valid precomputed DataFrame pass-through...")
    enriched_sample_df = pd.DataFrame([
        {
            "user_id": "u1",
            "text": "I love this amazing progress!",
            "sentiment_label": "positive",
            "sentiment_score": 0.9850,
            "emotion_label": "joy",
            "emotion_score": 0.9520,
        },
        {
            "user_id": "u2",
            "text": "Terrible death and tragedy.",
            "sentiment_label": "negative",
            "sentiment_score": 0.9780,
            "emotion_label": "sadness",
            "emotion_score": 0.9100,
        },
    ])

    result_df = analyze_sentiment(enriched_sample_df)
    for col in REQUIRED_COLUMNS:
        assert col in result_df.columns, f"Missing expected column {col}"
    assert result_df.iloc[0]["sentiment_label"] == "positive"
    assert result_df.iloc[1]["sentiment_label"] == "negative"
    print("  [PASS] Precomputed DataFrame correctly validated and returned.")

    # 3. Test empty DataFrame handling
    print("\n[TEST 3] Testing empty DataFrame handling...")
    empty_df = pd.DataFrame()
    res_empty = analyze_sentiment(empty_df)
    assert res_empty.empty
    for col in REQUIRED_COLUMNS:
        assert col in res_empty.columns
    print("  [PASS] Empty DataFrame handled gracefully with expected columns.")

    # 4. Pipeline integration check
    print("\n[TEST 4] Testing pipeline window integration...")
    earliest, _ = get_time_bounds()
    window_df = get_window(earliest, earliest + timedelta(hours=2))
    has_enriched_cols = all(col in window_df.columns for col in REQUIRED_COLUMNS)

    if has_enriched_cols:
        print(f"  Dataset is enriched ({len(window_df)} rows in sample window). Testing analyze_sentiment()...")
        res_df = analyze_sentiment(window_df)
        assert len(res_df) == len(window_df)
        for col in REQUIRED_COLUMNS:
            assert col in res_df.columns
        print("  [PASS] Pipeline integration passed with enriched dataset.")
    else:
        print("  Dataset is not yet enriched. Verifying analyze_sentiment() rejects unenriched pipeline window...")
        try:
            analyze_sentiment(window_df)
            assert False, "Expected unenriched window_df to raise RuntimeError"
        except RuntimeError as exc:
            assert "enrich_sentiment.py" in str(exc)
            print("  [PASS] Correctly rejected unenriched pipeline window.")

    print("\n" + "=" * 65)
    print(" ALL SENTIMENT TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
