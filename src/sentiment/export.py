"""
Final export and validation of sentiment results.

Reads:  data/processed/articles_with_sentiment.parquet
Writes (text-free, safe to publish):
        data/processed/summary_by_country.csv     one row per country
        data/processed/summary_by_outlet.csv      one row per outlet
        data/processed/article_sentiment_scores.csv  one row per article, no article text
        data/processed/outlet_coverage.csv        every configured outlet, GDELT hits vs final

Usage:
    python -m src.sentiment.export
"""

from pathlib import Path

import pandas as pd

from src.collect.config import OUTLETS, domain_for_url

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED / "articles_with_sentiment.parquet"
COUNTRY_PATH = PROCESSED / "summary_by_country.csv"
OUTLET_PATH = PROCESSED / "summary_by_outlet.csv"
SCORES_PATH = PROCESSED / "article_sentiment_scores.csv"
COVERAGE_PATH = PROCESSED / "outlet_coverage.csv"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "gdelt_articles.parquet"

REQUIRED_COLUMNS = [
    "article_id", "url", "published_date", "source_domain", "outlet_name",
    "country", "language", "title", "full_text", "word_count",
    "bert_positive", "bert_neutral", "bert_negative", "bert_sentiment",
]

# Columns of the public, text-free per-article file
SCORE_COLUMNS = [
    "article_id", "url", "published_date", "source_domain", "outlet_name", "country",
    "language", "detected_language", "title", "word_count",
    "bert_negative", "bert_neutral", "bert_positive", "bert_sentiment",
]


def validate(df: pd.DataFrame):
    """Check schema and flag any issues."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
    else:
        print("Schema OK: all required columns present")

    for col in ["article_id", "country", "bert_sentiment"]:
        if col in df.columns:
            n_null = df[col].isna().sum()
            if n_null > 0:
                print(f"WARNING: {n_null:,} nulls in '{col}'")

    if df["article_id"].duplicated().any():
        print(f"WARNING: {df['article_id'].duplicated().sum():,} duplicate article_ids")

    prob_cols = ["bert_positive", "bert_neutral", "bert_negative"]
    prob_sum = df[prob_cols].sum(axis=1)
    bad = ((prob_sum < 0.98) | (prob_sum > 1.02)).sum()
    print(f"WARNING: {bad:,} rows where probabilities don't sum to 1" if bad
          else "Probabilities: sum check OK")


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = (
        df.assign(bert_net=df["bert_positive"] - df["bert_negative"])
        .groupby(by)
        .agg(
            n_articles=("article_id", "count"),
            n_outlets=("outlet_name", "nunique"),
            mean_negative=("bert_negative", "mean"),
            mean_neutral=("bert_neutral", "mean"),
            mean_positive=("bert_positive", "mean"),
            mean_net=("bert_net", "mean"),
            pct_negative=("bert_sentiment", lambda x: (x == "negative").mean() * 100),
            pct_neutral=("bert_sentiment", lambda x: (x == "neutral").mean() * 100),
            pct_positive=("bert_sentiment", lambda x: (x == "positive").mean() * 100),
        )
        .round(3)
        .reset_index()
    )
    if "outlet_name" in by:
        out = out.drop(columns="n_outlets")
    return out.sort_values("n_articles", ascending=False)


def outlet_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """All configured outlets with GDELT hits (own site only) and final article counts."""
    raw = pd.read_parquet(RAW_PATH, columns=["url", "source_domain"])
    raw = raw[raw["url"].map(domain_for_url) == raw["source_domain"]]
    hits = raw.groupby("source_domain").size()
    final = df.groupby("source_domain").size()
    rows = [
        {"country": country, "outlet_name": name, "source_domain": dom, "language": lang,
         "gdelt_urls": int(hits.get(dom, 0)), "final_articles": int(final.get(dom, 0))}
        for country, entries in OUTLETS.items() for name, dom, lang in entries
    ]
    return pd.DataFrame(rows)


def main():
    print(f"Loading sentiment results from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Total articles: {len(df):,}")
    validate(df)

    country = summarise(df, ["country"])
    outlet = summarise(df, ["country", "outlet_name"])
    country.to_csv(COUNTRY_PATH, index=False)
    outlet.to_csv(OUTLET_PATH, index=False)
    outlet_coverage(df).to_csv(COVERAGE_PATH, index=False)

    scores = df[[c for c in SCORE_COLUMNS if c in df.columns]].copy()
    scores["published_date"] = pd.to_datetime(scores["published_date"]).dt.date
    scores.sort_values(["country", "outlet_name", "published_date", "article_id"]).to_csv(
        SCORES_PATH, index=False)

    print(f"\nSaved {COUNTRY_PATH.name}, {OUTLET_PATH.name}, {SCORES_PATH.name}, "
          f"{COVERAGE_PATH.name} (no article text)")
    print("\n--- Sentiment by country ---")
    print(country.to_string(index=False))


if __name__ == "__main__":
    main()
