"""
Article scraper: fetches full text from URLs in data/raw/gdelt_articles.parquet.

Uses trafilatura for robust text extraction. Implements:
- Domain-randomized request order (avoids hammering one outlet)
- Per-domain rate limiting (~2 req/sec)
- Exponential backoff via tenacity
- Checkpointing every 100 articles (safe to restart)
- Language detection with langdetect

Usage:
    python -m src.collect.scraper [--limit N] [--workers N]

Output:
    data/scraped/articles_text.parquet
"""

import hashlib
import random
import time
from collections import defaultdict
from pathlib import Path

import langdetect
import pandas as pd
import tenacity
import trafilatura
from tqdm import tqdm

from src.collect.config import domain_to_language

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "gdelt_articles.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "scraped" / "articles_text.parquet"

# ---------------------------------------------------------------------------
# Scraping config
# ---------------------------------------------------------------------------
MIN_TEXT_CHARS = 200       # Below this → "paywall"
CHECKPOINT_EVERY = 100     # Save progress after every N articles
PER_DOMAIN_DELAY = 0.5     # Seconds between requests to the same domain


# ---------------------------------------------------------------------------
# Retry decorator for fetch
# ---------------------------------------------------------------------------
@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
    stop=tenacity.stop_after_attempt(3),
    retry=tenacity.retry_if_exception_type((Exception,)),
    reraise=False,
)
def _fetch_with_retry(url: str) -> str | None:
    """Fetch raw HTML from a URL with retries. Returns HTML string or None."""
    return trafilatura.fetch_url(url)


def scrape_url(url: str) -> dict:
    """
    Scrape a single URL. Returns a dict with title, full_text, and scrape_status.
    """
    result = {
        "title": None,
        "full_text": None,
        "word_count": 0,
        "detected_language": None,
        "scrape_status": "error",
    }

    try:
        html = _fetch_with_retry(url)
    except Exception as e:
        result["scrape_status"] = f"error:{type(e).__name__}"
        return result

    if not html:
        result["scrape_status"] = "empty"
        return result

    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    title = trafilatura.extract(html, output_format="xml")
    # Try to parse title from XML metadata
    if title:
        import re
        m = re.search(r"<title>(.*?)</title>", title)
        result["title"] = m.group(1).strip() if m else None

    if not text:
        result["scrape_status"] = "empty"
        return result

    text = text.strip()
    word_count = len(text.split())
    result["full_text"] = text
    result["word_count"] = word_count

    if len(text) < MIN_TEXT_CHARS:
        result["scrape_status"] = "paywall"
    else:
        result["scrape_status"] = "ok"

    # Language detection
    try:
        result["detected_language"] = langdetect.detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        result["detected_language"] = None

    return result


def load_existing_urls(output_path: Path) -> set[str]:
    """Load already-scraped URLs from checkpoint file to support resuming."""
    if not output_path.exists():
        return set()
    try:
        df = pd.read_parquet(output_path, columns=["url"])
        return set(df["url"].tolist())
    except Exception:
        return set()


def save_checkpoint(records: list[dict], output_path: Path):
    """Append records to parquet checkpoint file."""
    df_new = pd.DataFrame(records)
    if output_path.exists():
        df_existing = pd.read_parquet(output_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_parquet(output_path, index=False)


def build_domain_order(df: pd.DataFrame) -> list[int]:
    """
    Return row indices in a domain-interleaved order.
    Instead of processing all NYT articles then all BBC articles,
    we round-robin across domains to avoid hammering a single outlet.
    """
    domain_buckets: dict[str, list[int]] = defaultdict(list)
    for idx, row in df.iterrows():
        domain_buckets[row["source_domain"]].append(idx)

    # Shuffle within each domain for good measure
    for bucket in domain_buckets.values():
        random.shuffle(bucket)

    # Round-robin interleave
    order = []
    buckets = list(domain_buckets.values())
    while any(buckets):
        for bucket in buckets:
            if bucket:
                order.append(bucket.pop(0))
    return order


def main(limit: int | None = None):
    print(f"Loading GDELT articles from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Total articles in GDELT data: {len(df):,}")

    # Resume: skip already-scraped URLs
    done_urls = load_existing_urls(OUTPUT_PATH)
    if done_urls:
        df = df[~df["url"].isin(done_urls)]
        print(f"Resuming: {len(done_urls):,} already scraped, {len(df):,} remaining")

    if limit:
        df = df.head(limit)
        print(f"Limiting to {limit} articles for this run")

    # Domain-interleaved order
    indices = build_domain_order(df)
    df_ordered = df.loc[indices].reset_index(drop=True)

    # Per-domain last-request timestamps for rate limiting
    domain_last_request: dict[str, float] = {}

    buffer: list[dict] = []
    stats = defaultdict(int)

    for i, (_, row) in enumerate(tqdm(df_ordered.iterrows(), total=len(df_ordered), desc="Scraping")):
        url = row["url"]
        domain = row["source_domain"]

        # Per-domain rate limiting
        last = domain_last_request.get(domain, 0.0)
        elapsed = time.time() - last
        if elapsed < PER_DOMAIN_DELAY:
            time.sleep(PER_DOMAIN_DELAY - elapsed)
        domain_last_request[domain] = time.time()

        scraped = scrape_url(url)
        stats[scraped["scrape_status"]] += 1

        record = {
            "url": url,
            "published_date": row["published_date"],
            "source_domain": domain,
            "outlet_name": row["outlet_name"],
            "country": row["country"],
            "language": row["language"],
            "gdelt_tone": row.get("gdelt_tone"),
            "themes": row.get("themes"),
            **scraped,
        }
        buffer.append(record)

        # Checkpoint
        if len(buffer) >= CHECKPOINT_EVERY:
            save_checkpoint(buffer, OUTPUT_PATH)
            buffer = []

    # Final flush
    if buffer:
        save_checkpoint(buffer, OUTPUT_PATH)

    print(f"\nScraping complete. Saved to {OUTPUT_PATH}")
    print("\n--- Scrape status summary ---")
    for status, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count:,}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape article text from GDELT URLs.")
    parser.add_argument("--limit", type=int, default=None, help="Max articles to scrape (for testing)")
    args = parser.parse_args()
    main(limit=args.limit)
