"""
Preprocessing pipeline: cleans and validates scraped articles.

Reads:  data/scraped/articles_text.parquet
Writes: data/processed/articles_clean.parquet

Usage:
    python -m src.preprocess.pipeline
"""

import hashlib
from pathlib import Path

import pandas as pd

from src.collect.config import domain_to_language
from src.preprocess.cleaner import clean_text, deduplicate, is_valid_article

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "scraped" / "articles_text.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "articles_clean.parquet"


def make_article_id(url: str) -> str:
    """Stable hash of URL as article identifier."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def validate_language(row: pd.Series) -> bool:
    """
    Return True if detected language matches the expected language for this outlet.
    We allow a 2-char prefix match (e.g., 'pt-br' matches 'pt').
    Articles with unknown detected language are kept (None → True).
    """
    detected = row.get("detected_language")
    expected = row.get("language")
    if not detected or not expected:
        return True
    return detected[:2] == expected[:2]


def print_summary(df: pd.DataFrame, label: str):
    print(f"\n{'='*50}")
    print(f"  {label}: {len(df):,} articles")
    print(f"{'='*50}")
    summary = (
        df.groupby("country")
        .agg(
            articles=("url", "count"),
            outlets=("outlet_name", "nunique"),
            avg_words=("word_count", "mean"),
        )
        .sort_values("articles", ascending=False)
    )
    summary["avg_words"] = summary["avg_words"].round(0).astype(int)
    print(summary.to_string())


def main():
    print(f"Loading scraped articles from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded: {len(df):,} rows")

    # ------------------------------------------------------------------
    # Step 1: Keep only successfully scraped articles
    # ------------------------------------------------------------------
    df = df[df["scrape_status"] == "ok"].copy()
    print(f"After status filter (ok only): {len(df):,}")

    # ------------------------------------------------------------------
    # Step 2: Clean text
    # ------------------------------------------------------------------
    print("Cleaning text...")
    df["full_text"] = df["full_text"].fillna("").apply(clean_text)
    df["word_count"] = df["full_text"].apply(lambda t: len(t.split()))

    # ------------------------------------------------------------------
    # Step 3: Filter by minimum length
    # ------------------------------------------------------------------
    mask = df["full_text"].apply(is_valid_article)
    df = df[mask].copy()
    print(f"After min-length filter (≥100 words): {len(df):,}")

    # ------------------------------------------------------------------
    # Step 4: Deduplicate
    # ------------------------------------------------------------------
    df = deduplicate(df)

    # ------------------------------------------------------------------
    # Step 5: Language validation
    # ------------------------------------------------------------------
    lang_mask = df.apply(validate_language, axis=1)
    n_dropped = (~lang_mask).sum()
    df = df[lang_mask].copy()
    print(f"Language validation: dropped {n_dropped:,} language-mismatched articles")

    # ------------------------------------------------------------------
    # Step 6: Generate stable article IDs
    # ------------------------------------------------------------------
    df["article_id"] = df["url"].apply(make_article_id)

    # ------------------------------------------------------------------
    # Step 7: Select and order final columns
    # ------------------------------------------------------------------
    final_cols = [
        "article_id",
        "url",
        "published_date",
        "source_domain",
        "outlet_name",
        "country",
        "language",
        "detected_language",
        "title",
        "full_text",
        "word_count",
        "gdelt_tone",
        "themes",
        "scrape_status",
    ]
    # Only keep columns that exist
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].copy()

    # ------------------------------------------------------------------
    # Step 8: Summary stats & save
    # ------------------------------------------------------------------
    print_summary(df, "Final clean dataset")

    # Weekly coverage
    df["published_date"] = pd.to_datetime(df["published_date"])
    weekly = df.groupby(df["published_date"].dt.to_period("W")).size()
    print("\n--- Articles per week ---")
    print(weekly.to_string())

    # Check for thin countries (< 50 articles)
    country_counts = df.groupby("country").size()
    thin = country_counts[country_counts < 50]
    if not thin.empty:
        print(f"\nWARNING: Low article counts for: {thin.to_dict()}")
        print("Consider relaxing theme filters or extending date range.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} clean articles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
