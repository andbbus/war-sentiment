# war-sentiment

Cross-country sentiment analysis of media coverage of the **Iran–Israel conflict**
(January–April 2026). Compares framing, word usage, and sentiment across newspapers
in 11 countries using multilingual BERT.

**[View the report →](https://andbbus.github.io/war-sentiment/analysis/report.html)** *(GitHub Pages — enable under Settings → Pages → Branch: main / docs)*

## Results at a glance

| Country | Articles | % Negative | % Neutral | % Positive |
|---------|----------|-----------|----------|-----------|
| Al-Jazeera | 329 | 94.8% | 5.2% | 0.0% |
| USA | 333 | 94.0% | 5.4% | 0.6% |
| UK | 973 | 93.2% | 6.1% | 0.7% |
| Italy | 232 | 93.5% | 5.2% | 1.3% |
| France | 262 | 92.7% | 3.8% | 3.4% |
| Canada | 128 | 88.3% | 10.9% | 0.8% |
| Argentina | 30 | 86.7% | 13.3% | 0.0% |
| Israel | 817 | 81.2% | 14.8% | 4.0% |
| Germany | 126 | 62.7% | 37.3% | 0.0% |
| Spain | 24 | 91.7% | 4.2% | 4.2% |
| Brazil | 4 | 50.0% | 50.0% | 0.0% |

## Coverage

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
| Middle East | Al-Jazeera | English | aljazeera.com |
| Middle East | Israel | English | Jerusalem Post, Haaretz, Times of Israel, Ynet, Israel National News |

## Methodology

1. **GDELT**: Articles collected via the [GDELT Doc API 2.0](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/) (free, no credentials required). One query per outlet over the full 3-month window, with a second ascending query if the 250-article cap is hit.
2. **Scraping**: Full article text extracted with [trafilatura](https://trafilatura.readthedocs.io/). 3,928 of 4,105 articles (96%) scraped successfully; 177 returned empty (bot-blocked outlets).
3. **Sentiment**: [`cardiffnlp/xlm-roberta-base-sentiment-multilingual`](https://huggingface.co/cardiffnlp/xlm-roberta-base-sentiment-multilingual) — multilingual XLM-RoBERTa fine-tuned for 3-class sentiment (positive / neutral / negative). Articles longer than 512 tokens are split into overlapping chunks; probabilities are aggregated by token-weighted average.
4. **Analysis**: R + ggplot2 + ggridges + tidytext, rendered as a [Quarto](https://quarto.org/) report.

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
# Step 1: Collect GDELT metadata (~6 min for all outlets)
python -m src.collect.gdelt_query

# Dry run (3 articles per outlet, quick check):
python -m src.collect.gdelt_query --dry-run

# Step 2: Scrape article text (~3 hours; safe to interrupt — checkpoints every 100 articles)
python -m src.collect.scraper

# Step 3: Preprocess and clean
python -m src.preprocess.pipeline

# Step 4: Sentiment inference (~1 hour on Apple Silicon MPS)
python -m src.sentiment.inference

# Step 5: Export and validate
python -m src.sentiment.export

# Step 6: Render report
quarto render analysis/report.qmd --output-dir output/
```

## Project structure

```
war-sentiment/
├── data/
│   ├── raw/                    # GDELT Doc API results (4,105 articles)
│   ├── scraped/                # Full article text (3,928 ok, 177 empty)
│   └── processed/              # Clean + sentiment-scored articles (3,258)
├── src/
│   ├── collect/
│   │   ├── config.py           # All outlet domains, languages, date range
│   │   ├── gdelt_query.py      # GDELT Doc API collection (parallel, 5 threads)
│   │   └── scraper.py          # trafilatura scraper with checkpointing
│   ├── preprocess/
│   │   ├── cleaner.py          # Text cleaning functions
│   │   └── pipeline.py         # Preprocessing orchestration
│   └── sentiment/
│       ├── model.py            # XLM-RoBERTa wrapper + 512-token chunking
│       ├── inference.py        # Batch inference pipeline
│       └── export.py           # Summary statistics and validation
├── analysis/
│   ├── report.qmd              # Main Quarto report
│   ├── 01_overview.qmd         # Data overview
│   ├── 02_sentiment.qmd        # Sentiment comparison
│   ├── 03_words.qmd            # Word clouds + TF-IDF
│   └── 04_timeline.qmd         # Temporal trends
├── output/                     # Rendered HTML report (GitHub Pages)
├── _quarto.yml
└── requirements.txt
```

## Notes on data quality

- **Canada** is effectively based on 2 outlets (Globe and Mail, CTV News) — CBC, National Post, and Toronto Star were fully blocked by anti-scraping protections.
- **Germany** shows unusually high neutral coverage (37%) compared to other countries; Der Spiegel had a 30% empty rate.
- **Brazil and Spain** have low article counts (4 and 24 respectively) due to fewer GDELT matches for those outlet domains — results for these two countries should be treated as indicative only.
- GDELT tone scores are not available via the Doc API (only in the BigQuery GKG pipeline), so cross-validation against GDELT tone is not included.

## License

Code: [MIT](LICENSE)
Analysis and report: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
