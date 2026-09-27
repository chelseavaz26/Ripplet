"""
Timeline-Replay Data Pipeline Loader

Provides time-window querying, batch streaming, and time bounds extraction
over the cleaned Parquet tweet dataset.
"""

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data.parquet"

# Cached dataframe instance
_CACHED_DF: pd.DataFrame | None = None
_CACHED_PATH: Path | None = None


def load_dataset(parquet_path: str | Path = DEFAULT_DATA_PATH, reload: bool = False) -> pd.DataFrame:
    """
    Loads and caches the cleaned Parquet dataset.
    Ensures dataset is sorted by timestamp for fast range slicing.
    """
    global _CACHED_DF, _CACHED_PATH
    path = Path(parquet_path)

    if _CACHED_DF is not None and not reload and _CACHED_PATH == path:
        return _CACHED_DF

    if not path.exists():
        raise FileNotFoundError(
            f"Parquet dataset not found at '{path}'. "
            "Run 'python pipeline/cleaner.py' first to build the pipeline dataset."
        )

    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError("Parquet file must contain a 'timestamp' column.")

    # Ensure timestamp is datetime64 and dataset is sorted ascending
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values(by="timestamp", ascending=True).reset_index(drop=True)

    _CACHED_DF = df
    _CACHED_PATH = path
    return _CACHED_DF


def get_time_bounds(parquet_path: str | Path = DEFAULT_DATA_PATH) -> tuple[datetime, datetime]:
    """
    Returns the earliest and latest timestamp in the dataset.
    Used by dashboards to size the time slider accurately.

    Returns:
        tuple[datetime, datetime]: (earliest_timestamp, latest_timestamp)
    """
    df = load_dataset(parquet_path)
    if df.empty:
        raise ValueError("Dataset is empty; cannot determine time bounds.")

    earliest: datetime = df["timestamp"].iloc[0].to_pydatetime()
    latest: datetime = df["timestamp"].iloc[-1].to_pydatetime()
    return earliest, latest


def get_window(
    start: datetime | str | pd.Timestamp,
    end: datetime | str | pd.Timestamp,
    parquet_path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """
    Returns all rows with timestamp in [start, end).

    Uses O(log N) binary search on the sorted timestamp column for
    near-instantaneous range extraction.

    Args:
        start: Inclusive start of the time window.
        end: Exclusive end of the time window.
        parquet_path: Optional custom path to parquet file.

    Returns:
        pd.DataFrame: Sliced dataframe containing tweets within [start, end).
    """
    df = load_dataset(parquet_path)
    if df.empty:
        return df.copy()

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    # Align timezone naive/aware formatting with dataframe
    if start_ts.tzinfo is not None and df["timestamp"].dt.tz is None:
        start_ts = start_ts.tz_localize(None)
    if end_ts.tzinfo is not None and df["timestamp"].dt.tz is None:
        end_ts = end_ts.tz_localize(None)

    timestamps = df["timestamp"]
    # Binary search bounds: [start, end)
    idx_start = int(timestamps.searchsorted(start_ts, side="left"))
    idx_end = int(timestamps.searchsorted(end_ts, side="left"))

    if idx_start >= idx_end:
        return df.iloc[0:0].copy()

    return df.iloc[idx_start:idx_end].copy()


def stream_batches(
    freq: str = "1D",
    parquet_path: str | Path = DEFAULT_DATA_PATH,
) -> Iterator[pd.DataFrame]:
    """
    Yields successive time-windowed batches across the full dataset in
    chronological order, one batch per day (or per given freq), to
    simulate a live feed for demo purposes.

    Args:
        freq: Frequency window string (e.g. '1D', '12h', '1h'). Default '1D'.
        parquet_path: Optional custom path to parquet file.

    Yields:
        pd.DataFrame: Batches of tweets partitioned into successive time windows.
    """
    df = load_dataset(parquet_path)
    if df.empty:
        return

    earliest, latest = get_time_bounds(parquet_path)
    offset = pd.tseries.frequencies.to_offset(freq)
    if offset is None:
        raise ValueError(f"Invalid frequency string: '{freq}'. Example valid values: '1D', '6h', '1h'.")

    try:
        curr_start = pd.Timestamp(earliest).floor(freq)
    except Exception:
        curr_start = pd.Timestamp(earliest)

    latest_ts = pd.Timestamp(latest)

    while curr_start <= latest_ts:
        curr_end = curr_start + offset
        batch = get_window(curr_start, curr_end, parquet_path=parquet_path)
        yield batch
        curr_start = curr_end
