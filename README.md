# war-sentiment

Cross-country sentiment analysis of media coverage of the **Iran–Israel conflict**
(4 January – 4 April 2026). Compares framing, word usage, and sentiment across news
outlets in 10 countries, plus Al Jazeera as a separate source, using a multilingual
XLM-RoBERTa sentiment model. 51 outlets were queried; 35 contribute articles to the
final sample of 3,138.

**[View the report →](https://andbbus.github.io/war-sentiment/analysis/report.html)** 

## Results at a glance

| Country / source | Articles | Outlets | % Negative | % Neutral | % Positive | Mean net |
|---|---|---|---|---|---|---|
| Spain | 24 | 5 | 95.8% | 0.0% | 4.2% | −0.62 |
| Al Jazeera | 329 | 1 | 94.8% | 4.9% | 0.3% | −0.65 |
| USA | 333 | 3 | 94.0% | 5.4% | 0.6% | −0.64 |
| UK | 869 | 4 | 93.8% | 5.3% | 0.9% | −0.65 |
| Italy | 240 | 5 | 93.3% | 5.4% | 1.2% | −0.62 |
| France | 262 | 3 | 92.7% | 3.8% | 3.4% | −0.62 |
| Canada | 128 | 1 | 88.3% | 10.9% | 0.8% | −0.58 |
| Argentina | 30 | 3 | 86.7% | 13.3% | 0.0% | −0.53 |
| Israel | 816 | 4 | 81.5% | 14.5% | 4.0% | −0.53 |
| Germany | 103 | 4 | 62.1% | 37.9% | 0.0% | −0.47 |
| Brazil | 4 | 2 | 50.0% | 50.0% | 0.0% | −0.43 |

Mean net = mean of P(positive) − P(negative). Spain, Argentina and Brazil are too small for
country-level conclusions. Cross-language comparisons are affected by how the model treats
each language; see the report's conclusions.

## Outlets queried

The report's coverage table shows which of these contribute articles.

| Region | Country | Language | Outlets |
|--------|---------|----------|---------|
| North America | USA | English | NYT, Washington Post, Fox News, CNN, WSJ |
| North America | Canada | English | Globe and Mail, Toronto Star, National Post, CBC, CTV |
| Europe | UK | English | The Guardian, BBC, The Times, Daily Mail, The Independent |
| Europe | Italy | Italian | Corriere della Sera, La Repubblica, La Stampa, Il Sole 24 Ore, ANSA |
| Europe | Spain | Spanish | El País, El Mundo, ABC, La Vanguardia, El Confidencial |
| Europe | France | French | Le Monde, Le Figaro, Libération, Le Parisien, BFM TV |
| Europe | Germany | German | Der Spiegel, Die Zeit, FAZ, Süddeutsche Zeitung, Bild |
| South America | Brazil | Portuguese | Folha, O Globo, Estadão, UOL, Veja |
| South America | Argentina | Spanish | La Nación, Clarín, Infobae, Página 12, Perfil |
| Middle East | Al Jazeera | English | aljazeera.com |
| Middle East | Israel | English | Jerusalem Post, Haaretz, Times of Israel, Ynet, Israel National News |

## Methodology

1. **GDELT**: Articles collected via the [GDELT Doc API 2.0](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) (free, no credentials required) with the query `Iran Israel war military` (all four words) per outlet. The collector queries week by week, splits any window that hits the 250-record cap, keeps to the API's limit of one request every 5 seconds, and logs failed queries instead of treating them as empty. The article date is GDELT's `seendate`, the day GDELT first saw the URL.
2. **Scraping**: Full article text extracted with [trafilatura](https://trafilatura.readthedocs.io/). 3,928 of 4,105 URLs (96%) returned text; 177 returned nothing (bot-blocked outlets).
3. **Cleaning**: URLs from other sites are dropped (GDELT's `domain:` filter matches suffixes, so `spiegel.de` also returned `tagesspiegel.de`). Consent banners, paywall notices and page furniture are removed, articles need at least 100 words, duplicates are removed, and the detected language must match the outlet's language.
4. **Sentiment**: [`cardiffnlp/xlm-roberta-base-sentiment-multilingual`](https://huggingface.co/cardiffnlp/xlm-roberta-base-sentiment-multilingual), XLM-RoBERTa fine-tuned on multilingual tweets for 3-class sentiment (negative / neutral / positive). Articles longer than 512 tokens are split into overlapping chunks; probabilities are aggregated by token-weighted average.
5. **Analysis**: R + ggplot2 + ggridges + tidytext, rendered as a [Quarto](https://quarto.org/) report.

## Requirements

- Python 3.11+
- R 4.3+
- [Quarto](https://quarto.org/docs/get-started/)

No API keys or cloud credentials needed.

## Setup

### Python

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### R

```r
install.packages(c(
  "tidyverse", "arrow", "ggplot2", "patchwork", "viridis",
  "tidytext", "ggwordcloud", "ggridges", "knitr", "kableExtra",
  "scales", "lubridate", "zoo", "stopwords", "ggrepel"
))
```

## Running the pipeline

```bash
# Step 1: Collect GDELT metadata (~1 hour: weekly queries, one request every 5.5 s)
python -m src.collect.gdelt_query

# Dry run (3 articles per outlet, quick check):
python -m src.collect.gdelt_query --dry-run

# Step 2: Scrape article text (~3 hours; safe to interrupt — checkpoints every 100 articles)
python -m src.collect.scraper

# Step 3: Preprocess and clean
python -m src.preprocess.pipeline

# Step 4: Sentiment inference (~1 hour on Apple Silicon MPS or a 16-core CPU)
# After changing the cleaning step, delete data/processed/articles_with_sentiment.parquet first
python -m src.sentiment.inference

# Step 5: Export and validate
python -m src.sentiment.export

# Step 6: Render the site (output in docs/, served by GitHub Pages)
quarto render
```

## Data in this repository

Full article text is **not** redistributed: it belongs to the publishers. The repository
ships only what can be shared:

- `data/raw/gdelt_articles.parquet`: GDELT metadata (URL, headline, outlet, date) for all 4,105 URLs.
- `data/processed/article_sentiment_scores.csv`: the 3,138 analysed articles with outlet,
  country, headline, word count and XLM-RoBERTa class probabilities, without the text.
- `data/processed/summary_by_country.csv`, `summary_by_outlet.csv`: the tables above, by country and by outlet.
- `data/processed/outlet_coverage.csv`: all 51 queried outlets, GDELT URLs and final articles.
- `data/processed/pipeline_attrition.csv`: articles per country after each cleaning step.
- `data/processed/inference_meta.json`: model, label mapping and library versions of the scoring run.

Every section of the report except *Word Use* renders from these files alone. *Word Use*
needs `data/processed/articles_with_sentiment.parquet`, which steps 2–5 rebuild locally; the
scraped text will differ slightly where outlets have since changed or paywalled pages.

## Project structure

```
war-sentiment/
├── data/
│   ├── raw/                    # GDELT Doc API results (4,105 articles)
│   ├── scraped/                # Full article text (local only, not committed)
│   └── processed/              # Scores without text (3,138), summaries, coverage, attrition
├── src/
│   ├── collect/
│   │   ├── config.py           # All outlet domains, languages, date range
│   │   ├── gdelt_query.py      # GDELT Doc API collection (rate-limited, weekly slices)
│   │   └── scraper.py          # trafilatura scraper with checkpointing
│   ├── preprocess/
│   │   ├── cleaner.py          # Text cleaning functions
│   │   └── pipeline.py         # Preprocessing orchestration
│   └── sentiment/
│       ├── model.py            # XLM-RoBERTa wrapper + 512-token chunking
│       ├── inference.py        # Batch inference pipeline
│       └── export.py           # Summary statistics and validation
├── analysis/
│   ├── report.qmd              # Main Quarto report (includes the four sections below)
│   ├── _01_overview.qmd        # Coverage and pipeline attrition
│   ├── _02_sentiment.qmd       # Sentiment by country and outlet
│   ├── _03_words.qmd           # Word clouds, TF-IDF, bigrams (needs local article text)
│   └── _04_timeline.qmd        # Trends over time
├── docs/                       # Rendered site (GitHub Pages)
├── _quarto.yml
└── requirements.txt
```

## Notes on data quality

- **11 outlets returned no URLs**, among them the Washington Post, the WSJ and the Times of Israel. The likely cause is GDELT's rate limit: during collection, throttled requests were counted as empty results. The collector in this repository logs failed requests to `data/raw/gdelt_failures.csv`, so a new run shows which outlets are genuinely empty.
- **Wrong-site URLs:** GDELT's `domain:` operator matches suffixes, so the queries for The Independent, Der Spiegel, Die Zeit and the Daily Mail also returned 170 URLs from other papers (Tottenham and Enfield Independent, Tagesspiegel, Unsere Zeit, Hull Daily Mail). The cleaning step removes them; Der Spiegel has no articles left.
- **Canada** rests on one outlet (Globe and Mail): CBC, National Post and Toronto Star were blocked by anti-scraping protections and CTV returned no URLs. **Le Monde** pages were JavaScript walls with no text.
- **Germany** shows unusually high neutral coverage (38% of articles), for the tabloid Bild as much as for Die Zeit and the Süddeutsche Zeitung. Part of this is genre (liveblogs and agency copy read as less negative everywhere); among German articles that are neither, a smaller gap remains, which suggests the model's handling of German rather than editorial style.
- **Brazil, Spain and Argentina** have 4, 24 and 30 articles: indicative only.
- Dates are GDELT's `seendate`, the day GDELT first saw the article.
- GDELT tone scores are not available via the Doc API, so cross-validation against GDELT tone is not included. Scores computed on CPU and on Apple silicon agree to within 0.00001.

## License

Code: [MIT](LICENSE)
Analysis and report: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
