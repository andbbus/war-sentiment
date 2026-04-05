"""
Final export and validation of sentiment results.

Reads:  data/processed/articles_with_sentiment.parquet
Writes: data/processed/summary_by_country.csv  (for quick inspection)

Usage:
    python -m src.sentiment.export
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "articles_with_sentiment.parquet"
SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "summary_by_country.csv"

REQUIRED_COLUMNS = [
    "article_id", "url", "published_date", "source_domain", "outlet_name",
    "country", "language", "title", "full_text", "word_count",
    "gdelt_tone", "bert_positive", "bert_neutral", "bert_negative", "bert_sentiment",
]


def validate(df: pd.DataFrame):
    """Check schema and flag any issues."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
    else:
        print("Schema OK: all required columns present")

    # Check for nulls in key columns
    key_cols = ["article_id", "country", "bert_sentiment"]
    for col in key_cols:
        if col in df.columns:
            n_null = df[col].isna().sum()
            if n_null > 0:
                print(f"WARNING: {n_null:,} nulls in '{col}'")

    # Check bert probability columns sum to ~1
    prob_cols = ["bert_positive", "bert_neutral", "bert_negative"]
    if all(c in df.columns for c in prob_cols):
        prob_sum = df[prob_cols].sum(axis=1)
        bad = ((prob_sum < 0.98) | (prob_sum > 1.02)).sum()
        if bad > 0:
            print(f"WARNING: {bad:,} rows where bert probabilities don't sum to 1")
        else:
            print("Bert probabilities: sum check OK")


def main():
    print(f"Loading sentiment results from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Total articles: {len(df):,}")

    validate(df)

    # Per-country summary
    summary = (
        df.groupby(["country", "outlet_name"])
        .agg(
            n_articles=("article_id", "count"),
            avg_gdelt_tone=("gdelt_tone", "mean"),
            avg_bert_positive=("bert_positive", "mean"),
            avg_bert_neutral=("bert_neutral", "mean"),
            avg_bert_negative=("bert_negative", "mean"),
            pct_positive=("bert_sentiment", lambda x: (x == "positive").mean() * 100),
            pct_neutral=("bert_sentiment", lambda x: (x == "neutral").mean() * 100),
            pct_negative=("bert_sentiment", lambda x: (x == "negative").mean() * 100),
        )
        .round(3)
        .sort_values(["country", "n_articles"], ascending=[True, False])
        .reset_index()
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSummary saved to {SUMMARY_PATH}")

    print("\n--- Sentiment distribution by country ---")
    country_summary = (
        df.groupby("country")
        .agg(
            n=("article_id", "count"),
            pct_pos=("bert_sentiment", lambda x: (x == "positive").mean() * 100),
            pct_neu=("bert_sentiment", lambda x: (x == "neutral").mean() * 100),
            pct_neg=("bert_sentiment", lambda x: (x == "negative").mean() * 100),
            avg_tone=("gdelt_tone", "mean"),
        )
        .round(1)
        .sort_values("n", ascending=False)
    )
    print(country_summary.to_string())

    # GDELT tone vs BERT correlation (only if tone is available)
    if "gdelt_tone" in df.columns and df["gdelt_tone"].notna().any() and "bert_positive" in df.columns:
        df["bert_net"] = df["bert_positive"] - df["bert_negative"]
        corr = df[["gdelt_tone", "bert_net"]].dropna().corr().iloc[0, 1]
        print(f"\nGDELT tone vs BERT net sentiment (Pearson r): {corr:.3f}")
        if abs(corr) < 0.3:
            print("WARNING: Low correlation — inspect model quality per language")
    else:
        print("\nNote: GDELT tone not available (Doc API doesn't provide tone scores).")


if __name__ == "__main__":
    main()
