"""
Offline Sentiment, Discrete Emotion, and Irony Pipeline Enrichment

Executes offline transformer inference over the primary dataset (pipeline/data.parquet)
using:
1. Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest
2. Discrete Emotion: j-hartmann/emotion-english-distilroberta-base
3. Irony: cardiffnlp/twitter-roberta-base-irony

Adds six precomputed columns:
- sentiment_label ('positive', 'negative', 'neutral', 'no_text')
- sentiment_score (float confidence [0.0, 1.0], or NaN if no_text)
- emotion_label ('joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust', 'neutral', 'no_text')
- emotion_score (float confidence [0.0, 1.0], or NaN if no_text)
- irony_label ('irony', 'non_irony', 'no_text')
- irony_score (float confidence [0.0, 1.0], or NaN if no_text)

Maintains the single-shared-dataset contract by atomically updating pipeline/data.parquet
with automatic backup, per-row null detection, and checkpoint/resume capabilities.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline.enrich_sentiment")

SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
IRONY_MODEL_ID = "cardiffnlp/twitter-roberta-base-irony"

ENRICHED_COLUMNS = [
    "sentiment_label",
    "sentiment_score",
    "emotion_label",
    "emotion_score",
    "irony_label",
    "irony_score",
]

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data.parquet"
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent / "enrichment_checkpoint.parquet"


def setup_inference_device(device_arg: int | str = -1) -> int:
    """Configures device and CPU threading for optimal inference throughput."""
    if device_arg == -1:
        threads = os.cpu_count() or 4
        torch.set_num_threads(threads)
        logger.info("Using CPU with %d threads for offline inference.", threads)
        return -1

    if torch.cuda.is_available():
        logger.info("Using CUDA GPU (%s) for offline inference.", torch.cuda.get_device_name(0))
        return 0

    logger.warning("CUDA requested but not available. Falling back to CPU.")
    threads = os.cpu_count() or 4
    torch.set_num_threads(threads)
    return -1


def load_transformer_pipelines(device: int) -> tuple[pipeline, pipeline, pipeline]:
    """Initializes and returns the sentiment, emotion, and irony transformers pipelines."""
    logger.info("Loading sentiment model: %s ...", SENTIMENT_MODEL_ID)
    try:
        sent_pipe = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL_ID,
            tokenizer=SENTIMENT_MODEL_ID,
            truncation=True,
            max_length=128,
            device=device,
            model_kwargs={"local_files_only": True},
        )
    except Exception:
        sent_pipe = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL_ID,
            tokenizer=SENTIMENT_MODEL_ID,
            truncation=True,
            max_length=128,
            device=device,
        )

    logger.info("Loading emotion model: %s ...", EMOTION_MODEL_ID)
    try:
        emo_pipe = pipeline(
            "text-classification",
            model=EMOTION_MODEL_ID,
            tokenizer=EMOTION_MODEL_ID,
            truncation=True,
            max_length=128,
            device=device,
            model_kwargs={"local_files_only": True},
        )
    except Exception:
        emo_pipe = pipeline(
            "text-classification",
            model=EMOTION_MODEL_ID,
            tokenizer=EMOTION_MODEL_ID,
            truncation=True,
            max_length=128,
            device=device,
        )

    logger.info("Loading irony model: %s ...", IRONY_MODEL_ID)
    try:
        irony_pipe = pipeline(
            "text-classification",
            model=IRONY_MODEL_ID,
            tokenizer=IRONY_MODEL_ID,
            truncation=True,
            max_length=128,
            device=device,
            model_kwargs={"local_files_only": True},
        )
    except Exception:
        irony_pipe = pipeline(
            "text-classification",
            model=IRONY_MODEL_ID,
            tokenizer=IRONY_MODEL_ID,
            truncation=True,
            max_length=128,
            device=device,
        )

    return sent_pipe, emo_pipe, irony_pipe


def is_missing_value(val: object) -> bool:
    """Returns True if val is None, NaN, pd.NA, empty string, or whitespace."""
    if val is None or pd.isna(val):
        return True
    s = str(val).strip()
    return s == "" or s.lower() in ("none", "nan", "<na>")


def row_needs_inference(label_val: object, score_val: object, force_recompute: bool) -> bool:
    """
    Determines whether a row requires model inference.
    If force_recompute is True, always returns True.
    If label_val is 'no_text', the row was marked as having no text and requires no inference.
    Otherwise, returns True if label_val or score_val is missing.
    """
    if force_recompute:
        return True
    if is_missing_value(label_val):
        return True
    if str(label_val).lower() == "no_text":
        return False
    return is_missing_value(score_val)


def enrich_dataset(
    parquet_path: Path = DEFAULT_DATA_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    batch_size: int = 64,
    checkpoint_interval: int = 2560,
    resume: bool = True,
    force_recompute: bool = False,
    limit: int | None = None,
    device: int | str = -1,
) -> pd.DataFrame:
    """
    Enriches pipeline/data.parquet with sentiment, emotion, and irony labels and scores.
    Supports per-row null detection to avoid recomputing already-populated fields,
    distinguishes blank text rows with 'no_text' and null scores, provides periodic
    checkpoint saving, and atomically updates the dataset.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Source parquet file not found at '{parquet_path}'")

    logger.info("Loading source dataset from %s ...", parquet_path)
    df = pd.read_parquet(parquet_path)
    total_source_rows = len(df)
    logger.info("Source dataset loaded with %d rows and columns: %s", total_source_rows, list(df.columns))

    if limit is not None and limit > 0:
        logger.info("Limiting processing to the first %d rows as requested.", limit)
        df = df.iloc[:limit].copy()
    else:
        df = df.copy()

    total_rows = len(df)

    # Ensure placeholder columns exist
    for col in ENRICHED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Load existing column values into working lists
    sentiment_labels: list[object] = df["sentiment_label"].tolist()
    sentiment_scores: list[object] = df["sentiment_score"].tolist()
    emotion_labels: list[object] = df["emotion_label"].tolist()
    emotion_scores: list[object] = df["emotion_score"].tolist()
    irony_labels: list[object] = df["irony_label"].tolist()
    irony_scores: list[object] = df["irony_score"].tolist()

    # Merge non-null values from existing checkpoint if resuming
    if resume and checkpoint_path.exists():
        try:
            df_chk = pd.read_parquet(checkpoint_path)
            chk_rows = min(len(df_chk), total_rows)
            if chk_rows > 0:
                sample_n = min(5, chk_rows)
                if (df["text"].iloc[:sample_n].values == df_chk["text"].iloc[:sample_n].values).all():
                    restored = 0
                    for col in ENRICHED_COLUMNS:
                        if col in df_chk.columns:
                            chk_col_vals = df_chk[col].iloc[:chk_rows].tolist()
                            for r_i in range(chk_rows):
                                c_val = chk_col_vals[r_i]
                                if not is_missing_value(c_val) or str(c_val).lower() == "no_text":
                                    if col == "sentiment_label":
                                        sentiment_labels[r_i] = c_val
                                    elif col == "sentiment_score":
                                        sentiment_scores[r_i] = c_val
                                    elif col == "emotion_label":
                                        emotion_labels[r_i] = c_val
                                    elif col == "emotion_score":
                                        emotion_scores[r_i] = c_val
                                    elif col == "irony_label":
                                        irony_labels[r_i] = c_val
                                    elif col == "irony_score":
                                        irony_scores[r_i] = c_val
                                    restored += 1
                    logger.info("Restored non-null entries from checkpoint at %s.", checkpoint_path)
                else:
                    logger.warning("Checkpoint text mismatch with current dataset. Ignoring checkpoint.")
        except Exception as e:
            logger.warning("Could not load checkpoint (%s).", e)

    # Determine first unprocessed row by checking per-row status across all models
    start_idx = 0
    if not force_recompute:
        for r_i in range(total_rows):
            if (
                row_needs_inference(sentiment_labels[r_i], sentiment_scores[r_i], False)
                or row_needs_inference(emotion_labels[r_i], emotion_scores[r_i], False)
                or row_needs_inference(irony_labels[r_i], irony_scores[r_i], False)
            ):
                start_idx = r_i
                break
        else:
            start_idx = total_rows

    if start_idx >= total_rows:
        logger.info("All %d rows are already fully enriched with valid values for all columns.", total_rows)
        # Ensure proper data types
        df["sentiment_label"] = sentiment_labels
        df["sentiment_score"] = pd.to_numeric(sentiment_scores, downcast="float")
        df["emotion_label"] = emotion_labels
        df["emotion_score"] = pd.to_numeric(emotion_scores, downcast="float")
        df["irony_label"] = irony_labels
        df["irony_score"] = pd.to_numeric(irony_scores, downcast="float")
        return df

    # Configure device and load pipelines
    dev = setup_inference_device(device)
    sent_pipe, emo_pipe, irony_pipe = load_transformer_pipelines(dev)

    logger.info(
        "Beginning batch inference: starting at row %d of %d (batch_size=%d, checkpoint_interval=%d, force_recompute=%s)",
        start_idx,
        total_rows,
        batch_size,
        checkpoint_interval,
        force_recompute,
    )

    t0_all = time.time()
    last_checkpoint_idx = start_idx
    all_texts = df["text"].fillna("").astype(str).tolist()

    with torch.inference_mode():
        pbar = tqdm(
            total=total_rows,
            initial=start_idx,
            desc="Enriching Sentiment, Emotion & Irony",
            unit="tweets",
            dynamic_ncols=True,
        )

        for i in range(start_idx, total_rows, batch_size):
            batch_end = min(i + batch_size, total_rows)
            current_batch_len = batch_end - i

            sent_work: list[tuple[int, str]] = []
            emo_work: list[tuple[int, str]] = []
            irony_work: list[tuple[int, str]] = []

            # Per-row null inspection
            for sub_idx in range(current_batch_len):
                g_idx = i + sub_idx
                raw_text = all_texts[g_idx]
                stripped = raw_text.strip()
                is_empty = (stripped == "")

                need_sent = row_needs_inference(sentiment_labels[g_idx], sentiment_scores[g_idx], force_recompute)
                need_emo = row_needs_inference(emotion_labels[g_idx], emotion_scores[g_idx], force_recompute)
                need_irony = row_needs_inference(irony_labels[g_idx], irony_scores[g_idx], force_recompute)

                if is_empty:
                    # Distinguish empty rows: set label to 'no_text' and score to None (null float)
                    if need_sent:
                        sentiment_labels[g_idx] = "no_text"
                        sentiment_scores[g_idx] = None
                    if need_emo:
                        emotion_labels[g_idx] = "no_text"
                        emotion_scores[g_idx] = None
                    if need_irony:
                        irony_labels[g_idx] = "no_text"
                        irony_scores[g_idx] = None
                else:
                    if need_sent:
                        sent_work.append((g_idx, stripped))
                    if need_emo:
                        emo_work.append((g_idx, stripped))
                    if need_irony:
                        irony_work.append((g_idx, stripped))

            # Execute pipeline passes only for rows that actually require inference
            if sent_work:
                sent_results = sent_pipe([t for _, t in sent_work], batch_size=batch_size)
                for (g_idx, _), res in zip(sent_work, sent_results):
                    sentiment_labels[g_idx] = str(res["label"]).lower()
                    sentiment_scores[g_idx] = round(float(res["score"]), 4)

            if emo_work:
                emo_results = emo_pipe([t for _, t in emo_work], batch_size=batch_size)
                for (g_idx, _), res in zip(emo_work, emo_results):
                    emotion_labels[g_idx] = str(res["label"]).lower()
                    emotion_scores[g_idx] = round(float(res["score"]), 4)

            if irony_work:
                irony_results = irony_pipe([t for _, t in irony_work], batch_size=batch_size)
                for (g_idx, _), res in zip(irony_work, irony_results):
                    irony_labels[g_idx] = str(res["label"]).lower()
                    irony_scores[g_idx] = round(float(res["score"]), 4)

            pbar.update(current_batch_len)
            processed_count = batch_end

            # Periodic checkpoint saving
            if processed_count - last_checkpoint_idx >= checkpoint_interval or processed_count == total_rows:
                df["sentiment_label"] = sentiment_labels
                df["sentiment_score"] = pd.to_numeric(sentiment_scores, downcast="float")
                df["emotion_label"] = emotion_labels
                df["emotion_score"] = pd.to_numeric(emotion_scores, downcast="float")
                df["irony_label"] = irony_labels
                df["irony_score"] = pd.to_numeric(irony_scores, downcast="float")

                chk_df = df.iloc[:processed_count].copy()
                chk_df.to_parquet(checkpoint_path, index=False)
                last_checkpoint_idx = processed_count
                logger.debug("Saved intermediate checkpoint (%d rows) to %s", processed_count, checkpoint_path)

        pbar.close()

    total_elapsed = time.time() - t0_all
    throughput = (total_rows - start_idx) / max(total_elapsed, 0.001)
    logger.info(
        "Enrichment inference complete: %d rows processed in %.2fs (%.2f tweets/sec).",
        total_rows - start_idx,
        total_elapsed,
        throughput,
    )

    df["sentiment_label"] = sentiment_labels
    df["sentiment_score"] = pd.to_numeric(sentiment_scores, downcast="float")
    df["emotion_label"] = emotion_labels
    df["emotion_score"] = pd.to_numeric(emotion_scores, downcast="float")
    df["irony_label"] = irony_labels
    df["irony_score"] = pd.to_numeric(irony_scores, downcast="float")

    # Safety: backup original dataset if backup does not already exist
    backup_path = parquet_path.with_suffix(".parquet.bak")
    if not backup_path.exists():
        logger.info("Creating initial backup at %s ...", backup_path)
        shutil.copy2(parquet_path, backup_path)

    # Atomic write to temporary file before replacing pipeline/data.parquet
    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    logger.info("Writing enriched dataset to temporary file %s ...", tmp_path)
    df.to_parquet(tmp_path, index=False)

    logger.info("Atomically updating %s ...", parquet_path)
    if parquet_path.exists():
        parquet_path.unlink()
    tmp_path.rename(parquet_path)
    logger.info("Successfully replaced %s with enriched dataset.", parquet_path)

    # If full dataset was enriched, remove checkpoint file
    if limit is None and checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed temporary checkpoint file %s.", checkpoint_path)

    # Print summary of enriched distributions
    print("\n" + "=" * 60)
    print(" ENRICHMENT SUMMARY STATISTICS")
    print("=" * 60)
    print("Sentiment distribution:")
    print(df["sentiment_label"].value_counts(dropna=False))
    print("\nDiscrete emotion distribution:")
    print(df["emotion_label"].value_counts(dropna=False))
    print("\nIrony distribution:")
    print(df["irony_label"].value_counts(dropna=False))
    print("=" * 60 + "\n")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline transformer sentiment, emotion & irony enrichment for pipeline/data.parquet"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to pipeline/data.parquet",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to save intermediate checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for pipeline inference (default: 64)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=2560,
        help="Number of rows between checkpoint writes (default: 2560)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoint and restart from row 0",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of all models even if per-row values are already populated",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit to process (useful for rapid dry-runs)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device to run on (-1 for CPU, 0 for first GPU)",
    )

    args = parser.parse_args()

    enrich_dataset(
        parquet_path=Path(args.data_path),
        checkpoint_path=Path(args.checkpoint_path),
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
        force_recompute=args.force_recompute,
        limit=args.limit,
        device=args.device,
    )


if __name__ == "__main__":
    main()
