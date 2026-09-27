"""
Sanity-Check and Verification Test Script for Twitter Demographics Module

Verifies:
1. Synthetic controlled unit tests:
   - User deduplication by user_id
   - Language detection accuracy with langdetect
   - Geographic matching against pycountry (with messy locations and 'unknown')
   - Profession/interest taxonomy classification
   - Strict privacy assertion: output contains ONLY aggregate counts,
     zero per-user records or identifiers.
2. End-to-end integration test:
   - Pulls real window from pipeline.loader.get_window()
   - Evaluates speed and verifies aggregate distributions
   - Complies strictly with pipeline Rule One
"""

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
from modules.demographics import (
    GeographicResolver,
    classify_bio_interests,
    detect_bio_language,
    profile_demographics,
)
from pipeline.loader import get_time_bounds, get_window


def run_unit_tests() -> None:
    print("=" * 65)
    print(" RUNNING DEMOGRAPHICS MODULE UNIT TESTS (SYNTHETIC DATA)")
    print("=" * 65)

    # 1. Test Language Detection
    print("\n[TEST 1] Testing language detection...")
    assert detect_bio_language("Software engineer and open source hacker.") == "en"
    assert detect_bio_language("Médecin à Paris et chercheur en virologie.") == "fr"
    assert detect_bio_language("Profesor de matemáticas y apasionado del fútbol.") == "es"
    assert detect_bio_language("😊✨💖") == "unknown"
    assert detect_bio_language("") == "unknown"
    assert detect_bio_language(None) == "unknown"
    print("  [PASS] detect_bio_language() passed.")

    # 2. Test Geographic Resolution
    print("\n[TEST 2] Testing geographic resolution via pycountry...")
    resolver = GeographicResolver()
    assert resolver.resolve("Seattle, WA") == "United States"
    assert resolver.resolve("New York, USA") == "United States"
    assert resolver.resolve("Austin, Texas") == "United States"
    assert resolver.resolve("London, UK") == "United Kingdom"
    assert resolver.resolve("Toronto, Canada") == "Canada"
    assert resolver.resolve("British Columbia") == "Canada"
    assert resolver.resolve("Galway") == "Ireland"
    assert resolver.resolve("Paris, France") == "France"
    assert resolver.resolve("Tokyo, Japan") == "Japan"
    assert resolver.resolve("Somewhere in the multiverse") == "unknown"
    assert resolver.resolve(None) == "unknown"
    print("  [PASS] GeographicResolver passed.")

    # 3. Test Interest Classification
    print("\n[TEST 3] Testing interest and profession taxonomy...")
    nurse_cats = classify_bio_interests("ICU nurse fighting COVID. Mother of two.")
    assert "Healthcare & Medicine" in nurse_cats
    tech_cats = classify_bio_interests("Senior Python developer & DevOps architect.")
    assert "Technology & Engineering" in tech_cats
    founder_cats = classify_bio_interests("Founder & CEO @ FinTech startup.")
    assert "Business & Finance" in founder_cats
    phd_cats = classify_bio_interests("PhD student & university researcher in physics.")
    assert "Education & Academia" in phd_cats
    other_cats = classify_bio_interests("Just loving the weekend coffee!")
    assert other_cats == ["Other / Unclassified"]
    print("  [PASS] classify_bio_interests() passed.")

    # 4. Test Deduplication & Privacy Assertion
    print("\n[TEST 4] Testing user deduplication and strict privacy assertion...")
    synthetic_tweets = pd.DataFrame([
        # User 1 tweets twice
        {
            "tweet_id": "t1",
            "user_id": "usr_1",
            "bio": "ICU nurse fighting COVID.",
            "location": "Seattle, WA",
            "text": "Wear a mask please.",
        },
        {
            "tweet_id": "t2",
            "user_id": "usr_1",
            "bio": "ICU nurse fighting COVID.",
            "location": "Seattle, WA",
            "text": "Another mask tweet.",
        },
        # User 2
        {
            "tweet_id": "t3",
            "user_id": "usr_2",
            "bio": "Software developer building AI apps.",
            "location": "London, UK",
            "text": "Hello world.",
        },
        # User 3 with unparseable location and empty bio
        {
            "tweet_id": "t4",
            "user_id": "usr_3",
            "bio": None,
            "location": "Somewhere over the rainbow",
            "text": "Just looking around.",
        },
    ])

    result = profile_demographics(synthetic_tweets)

    print("  Result dictionary keys:", list(result.keys()))
    expected_keys = {"languages", "regions", "interest_categories"}
    assert set(result.keys()) == expected_keys, f"Result keys must be exactly {expected_keys}"

    # Verify deduplication: User 1 tweeted twice but should only be counted once
    # Total unique users = 3
    total_users_by_regions = sum(result["regions"].values())
    assert total_users_by_regions == 3, f"Expected 3 unique users after deduplication, got {total_users_by_regions}"
    print(f"  [PASS] Deduplication verified: 4 tweets condensed to {total_users_by_regions} unique profiles.")

    # Check privacy: verify no user_id, tweet_id, or string values exist in the top level
    for k, v in result.items():
        assert isinstance(v, dict), f"Sub-field {k} must be a dict"
        for sub_k, sub_v in v.items():
            assert isinstance(sub_k, str), f"Key {sub_k} must be string"
            assert isinstance(sub_v, int), f"Value {sub_v} must be int count"
    print("  [PASS] Strict privacy assertion verified (aggregate counts only, 0 identifiers).")

    # 5. Test Empty DataFrame Handling
    empty_result = profile_demographics(pd.DataFrame())
    assert set(empty_result.keys()) == expected_keys
    assert empty_result["languages"] == {}
    print("  [PASS] Empty DataFrame handling verified.")


def run_integration_test() -> None:
    print("\n" + "=" * 65)
    print(" RUNNING INTEGRATION TEST (PIPELINE.GET_WINDOW)")
    print("=" * 65)

    earliest, _ = get_time_bounds()
    window_start = earliest
    window_end = earliest + timedelta(days=2)

    print(f"Retrieving window [{window_start} -> {window_end}) via pipeline.get_window()...")
    df = get_window(window_start, window_end)
    print(f"Retrieved {len(df):,} tweets.")

    print("Profiling demographics...")
    t0 = datetime.now()
    summary = profile_demographics(df)
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"Demographics profile generated in {elapsed:.2f} seconds.")

    print("\nAggregate Profile Highlights:")
    print("  Top Languages:")
    for lang, cnt in list(summary["languages"].items())[:5]:
        print(f"    {lang:>10}: {cnt:>5,}")

    print("  Top Regions:")
    for region, cnt in list(summary["regions"].items())[:5]:
        print(f"    {region:>18}: {cnt:>5,}")

    print("  Top Interest / Profession Categories:")
    for cat, cnt in list(summary["interest_categories"].items())[:5]:
        print(f"    {cat:>35}: {cnt:>5,}")

    # Verify counts are non-zero
    assert sum(summary["languages"].values()) > 0
    assert sum(summary["regions"].values()) > 0
    assert sum(summary["interest_categories"].values()) > 0

    print("\n" + "=" * 65)
    print(" ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_unit_tests()
    run_integration_test()
