"""
GDELT data collection via the GDELT Doc API 2.0 (free, no BigQuery required).

Strategy:
- The date window is split into weekly slices, one query per outlet and slice.
  A slice that returns the API cap (250 records) is split in half again, down
  to single days, so busy outlets are not silently truncated.
- Requests are sequential with at least 5.5 s between them: the API allows
  one request every 5 seconds and answers faster clients with HTTP 429 and a
  plain-text body.
- Failed requests are retried, then logged to data/raw/gdelt_failures.csv and
  reported at the end. An outlet with no articles is therefore distinguishable
  from an outlet whose queries failed.
- Only URLs whose host belongs to the queried outlet are kept: the API's
  `domain:` operator matches suffixes (spiegel.de also returns tagesspiegel.de).

`published_date` is GDELT's `seendate`, the date GDELT first saw the URL. It is
usually the publication day, but not always.

Output: data/raw/gdelt_articles.parquet (+ gdelt_failures.csv if any query failed)

Usage:
    python -m src.collect.gdelt_query [--dry-run] [--country USA]
"""

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.collect.config import (
    DATE_START,
    DATE_END,
    OUTLETS,
    domain_for_url,
    domain_to_country,
    domain_to_outlet,
    domain_to_language,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "gdelt_articles.parquet"
FAILURES_PATH = PROJECT_ROOT / "data" / "raw" / "gdelt_failures.csv"

# ---------------------------------------------------------------------------
# GDELT Doc API config
# ---------------------------------------------------------------------------
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250
MIN_INTERVAL = 5.5      # Seconds between any two requests (API limit: 1 per 5 s)
MAX_ATTEMPTS = 4
SLICE_DAYS = 7

# Iran-Israel conflict keywords (space = AND in GDELT: all four must appear)
KEYWORDS = "Iran Israel war military"

_last_request = 0.0


class GdeltError(Exception):
    pass


def _get(params: dict) -> list[dict]:
    """One rate-limited Doc API call. Raises GdeltError on any failure."""
    global _last_request
    wait = MIN_INTERVAL - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()

    try:
        resp = requests.get(GDELT_DOC_API, params=params, timeout=60)
    except requests.RequestException as e:
        raise GdeltError(f"network: {type(e).__name__}") from e
    if resp.status_code == 429:
        raise GdeltError("rate-limited (429)")
    if resp.status_code != 200:
        raise GdeltError(f"HTTP {resp.status_code}")
    try:
        payload = resp.json()
    except ValueError:
        # Invalid queries and throttling notices come back as plain text
        raise GdeltError("non-JSON response: " + resp.text[:120].replace("\n", " "))
    return payload.get("articles", [])


def _fetch(params: dict) -> list[dict]:
    """Call the API with retries and growing back-off; re-raise after the last attempt."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return _get(params)
        except GdeltError:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(15 * (attempt + 1))
    return []


def parse_article(art: dict, domain: str) -> dict | None:
    url = art.get("url", "")
    if not url or domain_for_url(url) != domain:
        return None
    raw_date = art.get("seendate", "")
    try:
        seen_date = datetime.strptime(raw_date[:8], "%Y%m%d").date()
    except (ValueError, TypeError):
        seen_date = None
    return {
        "url":            url,
        "title":          art.get("title", ""),
        "published_date": seen_date,   # GDELT seendate (first seen), see module docstring
        "source_domain":  domain,
        "outlet_name":    domain_to_outlet(domain),
        "country":        domain_to_country(domain),
        "language":       domain_to_language(domain),
        "gdelt_tone":     None,  # Not available in Doc API
    }


def _slices(start: datetime, end: datetime, days: int):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def collect_window(domain: str, start: datetime, end: datetime, failures: list, max_records: int) -> list[dict]:
    """Collect one outlet over [start, end); split the window if it hits the cap."""
    params = {
        "query":         f"{KEYWORDS} domain:{domain}",
        "mode":          "artlist",
        "maxrecords":    max_records,
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime":   (end - timedelta(seconds=1)).strftime("%Y%m%d%H%M%S"),
        "format":        "json",
        "sort":          "DateDesc",
    }
    try:
        articles = _fetch(params)
    except GdeltError as e:
        failures.append({"domain": domain, "start": start.date(), "end": end.date(), "error": str(e)})
        return []

    if len(articles) >= MAX_RECORDS:
        if end - start > timedelta(days=1):
            mid = start + (end - start) / 2
            return (collect_window(domain, start, mid, failures, max_records)
                    + collect_window(domain, mid, end, failures, max_records))
        failures.append({"domain": domain, "start": start.date(), "end": end.date(),
                         "error": f"cap of {MAX_RECORDS} reached within one day (truncated)"})
    return [a for a in (parse_article(x, domain) for x in articles) if a]


def collect_domain(domain: str, start: str, end: str, failures: list, dry_run: bool = False) -> list[dict]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)   # DATE_END is inclusive
    if dry_run:
        return collect_window(domain, start_dt, end_dt, failures, max_records=3)
    out = []
    for s, e in _slices(start_dt, end_dt, SLICE_DAYS):
        out.extend(collect_window(domain, s, e, failures, MAX_RECORDS))
    return out


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

    print(f"Date range: {DATE_START} → {DATE_END} (inclusive), {SLICE_DAYS}-day slices")
    if args.dry_run:
        print("DRY RUN: 3 articles per domain, one query each")

    all_records: list[dict] = []
    seen_urls: set[str] = set()
    failures: list[dict] = []

    with tqdm(domains, desc="Outlets") as pbar:
        for domain in pbar:
            for r in collect_domain(domain, DATE_START, DATE_END, failures, dry_run=args.dry_run):
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_records.append(r)
            pbar.set_postfix({"articles": len(all_records), "failed": len(failures)})

    df = pd.DataFrame(all_records)
    print(f"\nTotal unique articles: {len(df):,}")

    if failures:
        fail_df = pd.DataFrame(failures)
        print(f"\nWARNING: {len(fail_df)} queries failed or were truncated "
              f"({fail_df['domain'].nunique()} outlets). Counts for these outlets are incomplete:")
        print(fail_df.groupby("domain").size().to_string())
        if not args.dry_run:
            FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
            fail_df.to_csv(FAILURES_PATH, index=False)
            print(f"Details: {FAILURES_PATH}")

    if df.empty:
        print("No articles found.")
        return

    print("\n--- Articles per country ---")
    print(df.groupby("country").size().sort_values(ascending=False).to_string())
    empty = [d for d in domains if d not in set(df["source_domain"])]
    if empty:
        print(f"\nOutlets with no articles: {empty}")

    if not args.dry_run:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"\nSaved to {OUTPUT_PATH}")
    else:
        print("\nSample:")
        print(df[["country", "outlet_name", "published_date", "title"]].head(15).to_string())


if __name__ == "__main__":
    main()
