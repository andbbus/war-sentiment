"""
Batch sentiment inference pipeline.

Reads:  data/processed/articles_clean.parquet
Writes: data/processed/articles_with_sentiment.parquet
        data/processed/inference_meta.json  (model, library versions, device)

Usage:
    python -m src.sentiment.inference [--batch-size 16] [--limit N] [--threads N]

Checkpoints every 500 articles so the run can be safely interrupted and resumed.
Resuming matches on article_id (a hash of the URL): after changing the cleaning
step, delete the old output first, or stale scores are kept for changed texts.
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import transformers
from tqdm import tqdm

from src.sentiment.model import SentimentModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "articles_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "articles_with_sentiment.parquet"
META_PATH = PROJECT_ROOT / "data" / "processed" / "inference_meta.json"

CHECKPOINT_EVERY = 500
DEFAULT_BATCH_SIZE = 16


def load_done_ids(output_path: Path) -> set[str]:
    """Load article_ids already processed (for resume support)."""
    if not output_path.exists():
        return set()
    try:
        df = pd.read_parquet(output_path, columns=["article_id"])
        return set(df["article_id"].tolist())
    except Exception:
        return set()


def save_checkpoint(records: list[dict], output_path: Path):
    """Append batch of results to the checkpoint parquet."""
    df_new = pd.DataFrame(records)
    if output_path.exists():
        df_existing = pd.read_parquet(output_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_parquet(output_path, index=False)


def main(batch_size: int = DEFAULT_BATCH_SIZE, limit: int | None = None,
         threads: int | None = None, allow_fallback: bool = False):
    if threads:
        torch.set_num_threads(threads)
    print(f"Loading clean articles from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Total articles: {len(df):,}")

    # Resume: skip already-processed articles
    done_ids = load_done_ids(OUTPUT_PATH)
    if done_ids:
        df = df[~df["article_id"].isin(done_ids)]
        print(f"Resuming: {len(done_ids):,} already done, {len(df):,} remaining")

    if limit:
        df = df.head(limit)
        print(f"Limiting to {limit} articles")

    if df.empty:
        print("Nothing to process.")
        return

    # Load model
    model = SentimentModel(allow_fallback=allow_fallback)
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps({
        "model": model.model_name,
        "label_index": model.label_index,
        "device": str(model.device),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))

    buffer: list[dict] = []
    rows = df.to_dict("records")

    for i in tqdm(range(0, len(rows), batch_size), desc="Sentiment inference"):
        batch_rows = rows[i : i + batch_size]
        texts = [r.get("full_text", "") or "" for r in batch_rows]

        predictions = model.predict(texts)

        for row, pred in zip(batch_rows, predictions):
            record = {**row, **pred}
            buffer.append(record)

        if len(buffer) >= CHECKPOINT_EVERY:
            save_checkpoint(buffer, OUTPUT_PATH)
            buffer = []

    if buffer:
        save_checkpoint(buffer, OUTPUT_PATH)

    print(f"\nInference complete. Results saved to {OUTPUT_PATH}")

    # Quick summary
    result_df = pd.read_parquet(OUTPUT_PATH)
    print("\n--- Sentiment distribution by country ---")
    summary = (
        result_df.groupby("country")["bert_sentiment"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .rename("pct")
        .reset_index()
        .pivot(index="country", columns="bert_sentiment", values="pct")
        .fillna(0)
    )
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch sentiment inference.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="Max articles (for testing)")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for torch")
    parser.add_argument("--allow-fallback", action="store_true",
                        help="Use the fallback model if the main one fails to load")
    args = parser.parse_args()
    main(batch_size=args.batch_size, limit=args.limit, threads=args.threads,
         allow_fallback=args.allow_fallback)
