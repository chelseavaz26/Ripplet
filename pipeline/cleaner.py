"""
Timeline-Replay Data Pipeline Cleaner

Loads raw tweet CSV from data/raw/ (or data/), parses timestamps,
normalizes columns, sorts ascending by timestamp, and writes to pipeline/data.parquet.
"""

import glob
import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline.cleaner")

# Target normalized schema
TARGET_COLUMNS = [
    "tweet_id",
    "timestamp",
    "user_id",
    "text",
    "bio",
    "location",
    "followers",
    "hashtags",
]

# Candidate mapping definitions from raw source columns to target columns
CANDIDATE_MAPPINGS = {
    "tweet_id": ["tweet_id", "id", "id_str", "status_id"],
    "timestamp": ["date", "created_at", "timestamp", "created_time", "datetime"],
    "user_id": ["user_id", "author_id", "user_name", "username"],
    "text": ["text", "tweet_text", "full_text", "content"],
    "bio": ["user_description", "bio", "description", "user_bio"],
    "location": ["user_location", "location", "place"],
    "followers": ["user_followers", "followers_count", "followers"],
    "hashtags": ["hashtags", "tags", "hash_tags"],
}


def find_raw_csv(base_dir: str = ".") -> Path:
    """
    Search for the source CSV in data/raw/ first, then fall back to data/.
    Treats the data directory as strictly read-only.
    """
    base = Path(base_dir)
    search_paths = [
        base / "data" / "raw" / "*.csv",
        base / "data" / "*.csv",
    ]

    for pattern in search_paths:
        csv_files = glob.glob(str(pattern))
        if csv_files:
            csv_path = Path(csv_files[0])
            logger.info("Discovered source CSV at: %s", csv_path)
            return csv_path

    raise FileNotFoundError(
        f"No CSV file found in '{base / 'data' / 'raw'}' or '{base / 'data'}'"
    )


def identify_columns(df_columns: list[str]) -> dict[str, str | None]:
    """
    Identifies best matching source column for each target field.
    Returns mapping {target_field: source_column_name_or_None}.
    """
    col_lookup = {c.lower().strip(): c for c in df_columns}
    matched_mapping: dict[str, str | None] = {}

    for target_field, candidates in CANDIDATE_MAPPINGS.items():
        matched = None
        for cand in candidates:
            if cand.lower() in col_lookup:
                matched = col_lookup[cand.lower()]
                break
        matched_mapping[target_field] = matched

    return matched_mapping


def clean_dataset(
    csv_path: Path | None = None,
    output_path: str = "pipeline/data.parquet",
) -> pd.DataFrame:
    """
    Executes the cleaning, parsing, normalization, sorting, and parquet export.
    """
    if csv_path is None:
        csv_path = find_raw_csv()

    logger.info("Loading raw dataset from %s ...", csv_path)
    df_raw = pd.read_csv(csv_path)
    initial_row_count = len(df_raw)
    logger.info("Loaded %d rows with %d columns.", initial_row_count, len(df_raw.columns))

    # Identify matching columns
    col_mapping = identify_columns(df_raw.columns.tolist())
    logger.info("Detected column mappings:")
    for target, src in col_mapping.items():
        logger.info("  %s -> %s", target, f"'{src}'" if src else "NOT FOUND (will fill null)")

    # 1. Parse timestamp column
    timestamp_src = col_mapping["timestamp"]
    if timestamp_src is None:
        raise ValueError(
            "Could not identify a timestamp column. Checked candidates: "
            + str(CANDIDATE_MAPPINGS["timestamp"])
        )

    logger.info("Parsing timestamp from column '%s'...", timestamp_src)
    parsed_timestamps = pd.to_datetime(df_raw[timestamp_src], errors="coerce")
    invalid_mask = parsed_timestamps.isna()
    dropped_count = int(invalid_mask.sum())

    if dropped_count > 0:
        logger.warning(
            "Dropped %d rows (%.2f%%) due to unparseable timestamps.",
            dropped_count,
            (dropped_count / initial_row_count) * 100,
        )
        df_clean = df_raw.loc[~invalid_mask].copy()
        df_clean["timestamp"] = parsed_timestamps.loc[~invalid_mask]
    else:
        logger.info("All %d rows parsed with valid timestamps (0 dropped).", initial_row_count)
        df_clean = df_raw.copy()
        df_clean["timestamp"] = parsed_timestamps

    # 2. Normalize schema
    clean_dict: dict[str, pd.Series | np.ndarray] = {
        "timestamp": df_clean["timestamp"]
    }

    for target_col in TARGET_COLUMNS:
        if target_col == "timestamp":
            continue
        src_col = col_mapping.get(target_col)
        if src_col is not None and src_col in df_clean.columns:
            clean_dict[target_col] = df_clean[src_col]
        else:
            # Create column filled with null
            clean_dict[target_col] = pd.Series([None] * len(df_clean), index=df_clean.index, dtype="object")

    df_normalized = pd.DataFrame(clean_dict, columns=TARGET_COLUMNS)

    # 3. Sort by timestamp ascending
    logger.info("Sorting dataset by timestamp ascending...")
    df_sorted = df_normalized.sort_values(by="timestamp", ascending=True).reset_index(drop=True)

    earliest = df_sorted["timestamp"].min()
    latest = df_sorted["timestamp"].max()
    logger.info("Dataset timestamp bounds: %s to %s", earliest, latest)

    # 4. Save to Parquet
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving cleaned dataset to %s ...", out_file)
    df_sorted.to_parquet(out_file, index=False, engine="pyarrow")
    logger.info(
        "Parquet file written successfully. File size: %.2f MB across %d rows.",
        out_file.stat().st_size / (1024 * 1024),
        len(df_sorted),
    )

    return df_sorted


if __name__ == "__main__":
    clean_dataset()
