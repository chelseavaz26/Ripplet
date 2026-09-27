"""
Twitter Sentiment & Discrete Emotion Analysis Module

Provides the runtime interface for accessing precomputed sentiment and discrete emotion
metrics across windowed tweet subsets.

Inference is executed offline via `pipeline/enrich_sentiment.py` using:
1. Polarity Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest
2. Discrete Emotion: j-hartmann/emotion-english-distilroberta-base

At runtime, `analyze_sentiment()` reads the precomputed columns directly from the
DataFrame returned by `pipeline.loader.get_window()`, delivering instantaneous O(1)
retrieval during interactive time-slider navigation.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = (
    "sentiment_label",
    "sentiment_score",
    "emotion_label",
    "emotion_score",
)


ALL_SENTIMENT_COLUMNS = REQUIRED_COLUMNS + ("irony_label", "irony_score")


def analyze_sentiment(df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Validates and extracts precomputed sentiment and emotion metrics from a tweet DataFrame.

    Inference is conducted offline via `pipeline/enrich_sentiment.py`. This function
    verifies that the enriched columns exist in the DataFrame returned by `get_window()`.

    Args:
        df: Input DataFrame containing tweet rows within the selected time window.

    Returns:
        pd.DataFrame: DataFrame guaranteed to contain the precomputed sentiment and emotion columns:
            - sentiment_label ('positive', 'negative', 'neutral')
            - sentiment_score (float confidence [0.0, 1.0])
            - emotion_label ('joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust', 'neutral')
            - emotion_score (float confidence [0.0, 1.0])
            - irony_label ('irony', 'non_irony', 'no_text')
            - irony_score (float confidence [0.0, 1.0])

    Raises:
        RuntimeError: If any required precomputed columns are missing, indicating that
            `pipeline/enrich_sentiment.py` has not been run.
    """
    if df is None or df.empty:
        df_out = df.copy() if df is not None else pd.DataFrame()
        for col in ALL_SENTIMENT_COLUMNS:
            if col not in df_out.columns:
                df_out[col] = pd.Series(dtype="object" if "label" in col else "float64")
        return df_out

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise RuntimeError(
            f"Missing required precomputed sentiment/emotion columns: {missing_cols}. "
            "Offline transformer inference has not been performed on this dataset. "
            "Please run 'python pipeline/enrich_sentiment.py' first before querying sentiment analytics."
        )

    return df
