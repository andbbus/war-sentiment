"""
GDELT data collection via the GDELT Doc API 2.0 (free, no BigQuery required).

Strategy:
- One query per outlet over the full 3-month window (max 250 articles per outlet)
- If a domain hits the 250 limit, a second query is made sorted ascending
  to capture earlier articles that the descending query missed
- 5 parallel threads to keep total runtime under ~10 minutes

Output: data/raw/gdelt_articles.parquet

Usage:
    python -m src.collect.gdelt_query [--dry-run] [--country USA]
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

import pandas as pd
import requests
from tqdm import tqdm

from src.collect.config import (
    DATE_START,
    DATE_END,
    OUTLETS,
    domain_to_country,
    domain_to_outlet,
    domain_to_language,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "gdelt_articles.parquet"

# ---------------------------------------------------------------------------
# GDELT Doc API config
# ---------------------------------------------------------------------------
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250
WORKERS = 5             # Parallel threads
REQUEST_DELAY = 1.0     # Seconds between requests per thread

# Iran-Israel conflict keywords (space = AND in GDELT)
KEYWORDS = "Iran Israel war military"


def _fetch(params: dict, domain: str, attempt: int = 0) -> list[dict]:
    """Single GDELT Doc API call with retry on 429."""
    try:
        resp = requests.get(GDELT_DOC_API, params=params, timeout=60)
        if resp.status_code == 429:
            wait = 30 * (attempt + 1)
            time.sleep(wait)
            if attempt < 2:
                return _fetch(params, domain, attempt + 1)
            return []
        resp.raise_for_status()
        return resp.json().get("articles", [])
    except Exception:
        if attempt < 2:
            time.sleep(10)
            return _fetch(params, domain, attempt + 1)
        return []


def parse_article(art: dict, domain: str) -> dict | None:
    url = art.get("url", "")
    if not url:
        return None
    raw_date = art.get("seendate", "")
    try:
        pub_date = datetime.strptime(raw_date[:8], "%Y%m%d").date()
    except (ValueError, TypeError):
        pub_date = None
    return {
        "url":            url,
        "title":          art.get("title", ""),
        "published_date": pub_date,
        "source_domain":  domain,
        "outlet_name":    domain_to_outlet(domain),
        "country":        domain_to_country(domain),
        "language":       domain_to_language(domain),
        "gdelt_tone":     None,  # Not available in Doc API
    }


def collect_domain(domain: str, start: str, end: str, dry_run: bool = False) -> list[dict]:
    """
    Collect articles for one domain over the full date range.
    Makes a descending query (most recent first), and if we hit 250 records,
    also makes an ascending query to capture earlier articles.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d%H%M%S")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").strftime("%Y%m%d%H%M%S")

    base_params = {
        "query":         f"{KEYWORDS} domain:{domain}",
        "mode":          "artlist",
        "maxrecords":    3 if dry_run else MAX_RECORDS,
        "startdatetime": start_dt,
        "enddatetime":   end_dt,
        "format":        "json",
    }

    # Query 1: newest first
    params_desc = {**base_params, "sort": "DateDesc"}
    articles_desc = _fetch(params_desc, domain)
    time.sleep(REQUEST_DELAY)

    # Query 2: oldest first — only if we hit the cap (and not dry run)
    articles_asc = []
    if not dry_run and len(articles_desc) >= MAX_RECORDS:
        params_asc = {**base_params, "sort": "DateAsc"}
        articles_asc = _fetch(params_asc, domain)
        time.sleep(REQUEST_DELAY)

    seen_urls: set[str] = set()
    results = []
    for art in articles_desc + articles_asc:
        parsed = parse_article(art, domain)
        if parsed and parsed["url"] not in seen_urls:
            seen_urls.add(parsed["url"])
            results.append(parsed)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Fetch 3 articles per domain (quick test)")
    parser.add_argument("--country", default=None, help="Limit to one country (e.g. USA)")
    args = parser.parse_args()

    if args.country:
        if args.country not in OUTLETS:
            print(f"Unknown country. Available: {list(OUTLETS.keys())}")
            return
        domains = [dom for _, dom, _ in OUTLETS[args.country]]
        print(f"Country: {args.country} ({len(domains)} outlets)")
    else:
        domains = [dom for entries in OUTLETS.values() for _, dom, _ in entries]
        print(f"All {len(domains)} outlets across {len(OUTLETS)} countries")

    print(f"Date range: {DATE_START} → {DATE_END}")
    if args.dry_run:
        print("DRY RUN: 3 articles per domain")

    all_records: list[dict] = []
    seen_urls: set[str] = set()
    lock = Lock()

    def task(domain):
        records = collect_domain(domain, DATE_START, DATE_END, dry_run=args.dry_run)
        return domain, records

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(task, d): d for d in domains}
        with tqdm(total=len(domains), desc="Outlets") as pbar:
            for future in as_completed(futures):
                domain, records = future.result()
                with lock:
                    for r in records:
                        if r["url"] not in seen_urls:
                            seen_urls.add(r["url"])
                            all_records.append(r)
                pbar.update(1)
                pbar.set_postfix({"articles": len(all_records), "last": domain})

    df = pd.DataFrame(all_records)
    print(f"\nTotal unique articles: {len(df):,}")

    if df.empty:
        print("No articles found.")
        return

    print("\n--- Articles per country ---")
    print(df.groupby("country").size().sort_values(ascending=False).to_string())

    if not args.dry_run:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"\nSaved to {OUTPUT_PATH}")
    else:
        print("\nSample:")
        print(df[["country", "outlet_name", "published_date", "title"]].head(15).to_string())


if __name__ == "__main__":
    main()
