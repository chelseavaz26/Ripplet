"""
Twitter Timeline Trends Detection Module

Identifies breakout hashtags and keywords across successive time windows
using a rolling historical average and configurable spike threshold.
"""

from collections import Counter, deque
from collections.abc import Iterator
import ast
import re
from typing import Any
import pandas as pd

# Comprehensive English stop words tailored for Twitter / conversational text
STOP_WORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
    # Common Twitter noise and conversational filler tokens
    "amp", "rt", "http", "https", "co", "via", "get", "got", "just", "like",
    "one", "two", "also", "new", "now", "today", "day", "will", "us", "people"
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")
_WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")

# Target output columns for trend analysis
TREND_COLUMNS = ["window_start", "window_end", "term", "frequency", "is_trending"]


def parse_hashtags_cell(val: Any) -> list[str]:
    """
    Parses a single cell's hashtag entry into a list of cleaned lowercase hashtags.
    Handles python-string lists ("['covid19']"), raw strings, lists, or NaN.
    """
    if val is None or pd.isna(val):
        return []

    if isinstance(val, list):
        return [str(t).lower().strip("#").strip() for t in val if str(t).strip()]

    if isinstance(val, str):
        val_str = val.strip()
        if not val_str or val_str.lower() == "nan":
            return []

        # Attempt to parse stringified list e.g. "['tag1', 'tag2']"
        if val_str.startswith("[") and val_str.endswith("]"):
            try:
                parsed = ast.literal_eval(val_str)
                if isinstance(parsed, list):
                    return [
                        str(t).lower().strip("#").strip()
                        for t in parsed
                        if str(t).strip()
                    ]
            except Exception:
                pass

        # Fallback to regex extraction of alphanumeric hashtags
        found = _HASHTAG_RE.findall(val_str)
        if found:
            return [t.lower().strip() for t in found if t.strip()]

        # Comma/space separated words
        tokens = re.split(r"[,\s]+", val_str)
        return [t.lower().strip("#").strip() for t in tokens if len(t.strip("#").strip()) > 1]

    return []


def extract_keywords_from_text(texts: pd.Series | list[str], top_n: int = 50) -> Counter[str]:
    """
    Extracts top keywords from tweet texts using regex tokenization and stopword removal.
    Used when hashtags in a batch are sparse.
    """
    keyword_counter: Counter[str] = Counter()

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue

        # Strip URLs and mentions before tokenizing
        cleaned = _URL_RE.sub(" ", text)
        cleaned = _MENTION_RE.sub(" ", cleaned)

        # Extract words with >= 3 characters
        words = _WORD_RE.findall(cleaned.lower())
        meaningful = [w for w in words if w not in STOP_WORDS]
        keyword_counter.update(meaningful)

    if top_n is not None and top_n > 0:
        return Counter(dict(keyword_counter.most_common(top_n)))
    return keyword_counter


def extract_batch_terms(
    batch: pd.DataFrame,
    sparsity_threshold: int = 20,
    sparsity_ratio: float = 0.05,
    top_keywords: int = 50,
) -> Counter[str]:
    """
    Extracts hashtag term frequencies from a batch DataFrame.
    If hashtags are sparse (fewer than sparsity_threshold instances or
    less than sparsity_ratio occurrences per tweet), extracts top keywords
    from the text column and merges them.
    """
    term_counts: Counter[str] = Counter()

    # 1. Extract hashtags from 'hashtags' column if present
    if "hashtags" in batch.columns:
        for val in batch["hashtags"]:
            tags = parse_hashtags_cell(val)
            for t in tags:
                if len(t) >= 2:
                    term_counts[t] += 1

    # 2. Extract inline hashtags from 'text' column if present
    if "text" in batch.columns:
        for txt in batch["text"]:
            if isinstance(txt, str) and "#" in txt:
                inline_tags = _HASHTAG_RE.findall(txt)
                for it in inline_tags:
                    cleaned_it = it.lower().strip()
                    if len(cleaned_it) >= 2:
                        term_counts[cleaned_it] += 1

    total_hashtags = sum(term_counts.values())
    batch_len = len(batch)

    # 3. Check sparsity: if hashtags are sparse, supplement with top keywords from text
    is_sparse = (
        total_hashtags < sparsity_threshold
        or (batch_len > 0 and (total_hashtags / batch_len) < sparsity_ratio)
    )

    if is_sparse and "text" in batch.columns:
        kw_counts = extract_keywords_from_text(batch["text"], top_n=top_keywords)
        term_counts.update(kw_counts)

    return term_counts


class RollingTrendTracker:
    """
    Manages sliding window frequency history across successive batches
    and calculates whether terms exceed the rolling average baseline.
    """

    def __init__(self, window_size: int = 7, threshold: float = 2.0, min_frequency: int = 2):
        self.window_size = window_size
        self.threshold = threshold
        self.min_frequency = min_frequency
        self.history: deque[Counter[str]] = deque(maxlen=window_size)

    def evaluate_and_update(
        self,
        current_counts: Counter[str],
        top_n: int | None = None,
    ) -> list[tuple[str, int, bool]]:
        """
        Evaluates current term frequencies against the historical rolling average,
        then pushes current_counts into the sliding history window.

        Returns:
            list of (term, frequency, is_trending) tuples.
        """
        results: list[tuple[str, int, bool]] = []
        k = len(self.history)

        # Candidate terms filtered by min_frequency
        candidates = [
            (term, freq)
            for term, freq in current_counts.items()
            if freq >= self.min_frequency
        ]

        # Sort by frequency descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        if top_n is not None and top_n > 0:
            candidates = candidates[:top_n]

        for term, freq in candidates:
            if k == 0:
                # First window: baseline establishment, not yet trending
                is_trending = False
            else:
                past_sum = sum(h.get(term, 0) for h in self.history)
                rolling_avg = past_sum / k

                if past_sum == 0:
                    # Emerging breakout term not seen in recent history
                    is_trending = freq >= self.min_frequency
                else:
                    spike_ratio = freq / rolling_avg
                    is_trending = (spike_ratio >= self.threshold) and (freq >= self.min_frequency)

            results.append((term, freq, is_trending))

        # Update history queue after evaluation
        self.history.append(current_counts)
        return results


def detect_trends(
    batches: Iterator[pd.DataFrame],
    window_size: int = 7,
    threshold: float = 2.0,
    min_frequency: int = 2,
    top_terms_per_batch: int | None = None,
    sparsity_threshold: int = 20,
    sparsity_ratio: float = 0.05,
) -> pd.DataFrame:
    """
    Consumes batches of tweets from pipeline.stream_batches(), extracts hashtags
    (and text keywords if hashtags are sparse), tracks a rolling average frequency
    over the last `window_size` batches, and flags terms exceeding the rolling average
    by `threshold` (default: 2.0x) as trending.

    Args:
        batches: Iterator yielding DataFrame batches in chronological order.
        window_size: Number of past batches to compute the rolling average baseline (default: 7).
        threshold: Multiplier above rolling average to flag a term as trending (default: 2.0).
        min_frequency: Minimum occurrence count in current batch to qualify as trending (default: 2).
        top_terms_per_batch: Optional limit on terms returned per batch (default: None, returns all).
        sparsity_threshold: Min hashtag count in a batch before triggering keyword fallback (default: 20).
        sparsity_ratio: Min hashtag-to-tweet ratio before triggering keyword fallback (default: 0.05).

    Returns:
        pd.DataFrame with columns:
            ['window_start', 'window_end', 'term', 'frequency', 'is_trending']
    """
    tracker = RollingTrendTracker(
        window_size=window_size,
        threshold=threshold,
        min_frequency=min_frequency,
    )

    rows: list[dict[str, Any]] = []

    for batch in batches:
        if batch is None or batch.empty:
            continue

        # Extract time window bounds
        if "timestamp" in batch.columns and not batch["timestamp"].empty:
            window_start = batch["timestamp"].min()
            window_end = batch["timestamp"].max()
        else:
            window_start = None
            window_end = None

        # Extract terms (hashtags + keywords fallback)
        batch_counts = extract_batch_terms(
            batch,
            sparsity_threshold=sparsity_threshold,
            sparsity_ratio=sparsity_ratio,
        )

        evaluated = tracker.evaluate_and_update(
            batch_counts,
            top_n=top_terms_per_batch,
        )

        for term, freq, is_trending in evaluated:
            rows.append({
                "window_start": window_start,
                "window_end": window_end,
                "term": term,
                "frequency": freq,
                "is_trending": is_trending,
            })

    if not rows:
        return pd.DataFrame(columns=TREND_COLUMNS)

    return pd.DataFrame(rows, columns=TREND_COLUMNS)
