"""
Pure text cleaning functions (no I/O).
Applied during the preprocessing pipeline to normalize article text.
"""

import hashlib
import re
import unicodedata

import pandas as pd


# ---------------------------------------------------------------------------
# Boilerplate phrases to strip (common across news sites)
# ---------------------------------------------------------------------------
_BOILERPLATE_PATTERNS = [
    r"subscribe to (read|continue|access|unlock).*",
    r"already a subscriber\?.*",
    r"sign (in|up) to read.*",
    r"this article is for subscribers only.*",
    r"get unlimited access.*",
    r"continue reading.*",
    r"read more.*",
    r"advertisement\s*$",
    r"©\s*\d{4}.*",
    r"all rights reserved.*",
    r"follow us on (twitter|facebook|instagram|social media).*",
    r"share this article.*",
    r"comments? \(\d+\).*",
]
_BOILERPLATE_RE = re.compile(
    "|".join(_BOILERPLATE_PATTERNS),
    flags=re.IGNORECASE | re.MULTILINE,
)


def clean_text(text: str) -> str:
    """
    Normalize article text:
    1. NFKC Unicode normalization
    2. Strip HTML entities (residual &amp; etc.)
    3. Remove boilerplate phrases
    4. Collapse excessive whitespace and newlines
    """
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Decode common HTML entities that trafilatura may leave
    html_entities = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&apos;": "'",
        "&#39;": "'",
        "&nbsp;": " ",
    }
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)

    # Strip boilerplate
    text = _BOILERPLATE_RE.sub("", text)

    # Collapse multiple blank lines to single newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces/tabs to single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return text.strip()


def is_valid_article(text: str, min_words: int = 100) -> bool:
    """
    Return True if the article text has enough content for analysis.
    Filters out stubs, error pages, and paywall snippets.
    """
    if not text:
        return False
    word_count = len(text.split())
    return word_count >= min_words


def _text_fingerprint(text: str) -> str:
    """MD5 of first 200 characters of text (for near-duplicate detection)."""
    snippet = text[:200].strip().lower()
    return hashlib.md5(snippet.encode("utf-8")).digest().hex()


def deduplicate(df: pd.DataFrame, text_col: str = "full_text", url_col: str = "url") -> pd.DataFrame:
    """
    Remove duplicate articles from a DataFrame.

    Two passes:
    1. Exact URL dedup (keeps first occurrence).
    2. Near-duplicate dedup via first-200-char text fingerprint
       (catches syndicated copies of the same article on different URLs).

    Args:
        df:       DataFrame with at least `url_col` and `text_col` columns.
        text_col: Name of the column containing article text.
        url_col:  Name of the column containing article URLs.

    Returns:
        Deduplicated DataFrame.
    """
    before = len(df)

    # Pass 1: exact URL dedup
    df = df.drop_duplicates(subset=[url_col], keep="first")
    after_url = len(df)

    # Pass 2: near-duplicate text fingerprint
    df = df.copy()
    df["_fingerprint"] = df[text_col].fillna("").apply(_text_fingerprint)
    df = df.drop_duplicates(subset=["_fingerprint"], keep="first")
    df = df.drop(columns=["_fingerprint"])

    after_text = len(df)
    print(
        f"Dedup: {before:,} → {after_url:,} (URL) → {after_text:,} (near-dup) "
        f"[removed {before - after_text:,} total]"
    )
    return df
