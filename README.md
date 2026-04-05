# war-sentiment

Cross-country sentiment analysis of media coverage of the **Iran–Israel conflict**
(January–April 2026). Compares framing, word usage, and sentiment across 55 newspapers
in 11 countries using multilingual BERT.

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

1. **GDELT**: Articles collected via [GDELT 2.0 GKG](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/) queried through Google BigQuery.
2. **Scraping**: Full article text extracted with [trafilatura](https://trafilatura.readthedocs.io/).
3. **Sentiment**: [`cardiffnlp/xlm-roberta-base-sentiment-multilingual`](https://huggingface.co/cardiffnlp/xlm-roberta-base-sentiment-multilingual) — multilingual XLM-RoBERTa fine-tuned for sentiment (positive / neutral / negative).
4. **Analysis**: R + ggplot2, rendered as a [Quarto](https://quarto.org/) report.

## Requirements

- Python 3.11+
- R 4.3+
- [Quarto](https://quarto.org/docs/get-started/)
- Google Cloud account with BigQuery access

## Setup

### Python

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### BigQuery credentials

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account.json"
# Optionally set billing project:
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
```

### R

```r
install.packages("renv")
renv::restore()   # Installs all R packages from renv.lock
```

## Running the pipeline

```bash
# Step 1: Collect GDELT metadata (dry run to check cost first)
python -m src.collect.gdelt_query --dry-run

# Step 1b: Full collection
python -m src.collect.gdelt_query

# Step 2: Scrape article text (safe to interrupt — checkpoints every 100 articles)
python -m src.collect.scraper

# Step 3: Preprocess and clean
python -m src.preprocess.pipeline

# Step 4: Sentiment inference (safe to interrupt — checkpoints every 500 articles)
python -m src.sentiment.inference

# Step 5: Export and validate
python -m src.sentiment.export

# Step 6: Render report
quarto render analysis/report.qmd --output-dir output/
```

## Project structure

```
war-sentiment/
├── data/               # Not committed — reproduce via pipeline
│   ├── raw/            # GDELT BigQuery results
│   ├── scraped/        # Full article text
│   └── processed/      # Clean + sentiment-scored articles
├── src/
│   ├── collect/
│   │   ├── config.py       # All outlet domains, languages, date range
│   │   ├── gdelt_query.py  # BigQuery GDELT queries
│   │   └── scraper.py      # trafilatura scraper with checkpointing
│   ├── preprocess/
│   │   ├── cleaner.py      # Text cleaning functions
│   │   └── pipeline.py     # Preprocessing orchestration
│   └── sentiment/
│       ├── model.py        # XLM-RoBERTa wrapper + chunking
│       ├── inference.py    # Batch inference pipeline
│       └── export.py       # Summary statistics and validation
├── analysis/
│   ├── report.qmd          # Main Quarto report
│   ├── 01_overview.qmd     # Data overview
│   ├── 02_sentiment.qmd    # Sentiment comparison
│   ├── 03_words.qmd        # Word clouds + TF-IDF
│   └── 04_timeline.qmd     # Temporal trends
├── output/             # Rendered HTML report (can be served via GitHub Pages)
├── _quarto.yml
├── requirements.txt
└── renv.lock
```

## Notes on paywalls

Some outlets (NYT, WSJ, The Times) enforce strict paywalls. `trafilatura` typically
extracts the article title and first 1-3 paragraphs before the paywall. These short
snippets are marked `scrape_status = "paywall"` and excluded from BERT analysis.
The GDELT tone score (computed by GDELT's own pipeline from the full article) is
still available for all articles regardless of scrape success.

## License

Code: [MIT](LICENSE)
Analysis and report: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
