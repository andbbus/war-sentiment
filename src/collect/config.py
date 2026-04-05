"""
Central configuration for war-sentiment data collection.
All outlet domains, languages, date range, and search keywords live here.
Every other module imports from this file.
"""

# ---------------------------------------------------------------------------
# Date range: last 3 months as of April 4, 2026
# ---------------------------------------------------------------------------
DATE_START = "2026-01-04"
DATE_END = "2026-04-04"
DATE_RANGE = (DATE_START, DATE_END)

# ---------------------------------------------------------------------------
# GDELT theme keywords used to filter articles (matched against V2Themes)
# ---------------------------------------------------------------------------
GDELT_THEME_KEYWORDS = ["IRAN", "ISRAEL", "MILITARY", "WAR"]

# ---------------------------------------------------------------------------
# Outlet registry
# Format per entry: (outlet_name, domain, language_iso639_1)
# ---------------------------------------------------------------------------
OUTLETS: dict[str, list[tuple[str, str, str]]] = {
    "USA": [
        ("New York Times",      "nytimes.com",          "en"),
        ("Washington Post",     "washingtonpost.com",   "en"),
        ("Fox News",            "foxnews.com",          "en"),
        ("CNN",                 "cnn.com",              "en"),
        ("Wall Street Journal", "wsj.com",              "en"),
    ],
    "Canada": [
        ("Globe and Mail",  "theglobeandmail.com",  "en"),
        ("Toronto Star",    "thestar.com",          "en"),
        ("National Post",   "nationalpost.com",     "en"),
        ("CBC News",        "cbc.ca",               "en"),
        ("CTV News",        "ctvnews.ca",           "en"),
    ],
    "UK": [
        ("The Guardian",    "theguardian.com",      "en"),
        ("BBC News",        "bbc.co.uk",            "en"),
        ("The Times",       "thetimes.co.uk",       "en"),
        ("Daily Mail",      "dailymail.co.uk",      "en"),
        ("The Independent", "independent.co.uk",    "en"),
    ],
    "Italy": [
        ("Corriere della Sera", "corriere.it",      "it"),
        ("La Repubblica",       "repubblica.it",    "it"),
        ("La Stampa",           "lastampa.it",      "it"),
        ("Il Sole 24 Ore",      "ilsole24ore.com",  "it"),
        ("ANSA",                "ansa.it",          "it"),
    ],
    "Spain": [
        ("El País",         "elpais.com",           "es"),
        ("El Mundo",        "elmundo.es",           "es"),
        ("ABC",             "abc.es",               "es"),
        ("La Vanguardia",   "lavanguardia.com",     "es"),
        ("El Confidencial", "elconfidencial.com",   "es"),
    ],
    "France": [
        ("Le Monde",    "lemonde.fr",       "fr"),
        ("Le Figaro",   "lefigaro.fr",      "fr"),
        ("Libération",  "liberation.fr",    "fr"),
        ("Le Parisien", "leparisien.fr",    "fr"),
        ("BFM TV",      "bfmtv.com",        "fr"),
    ],
    "Germany": [
        ("Der Spiegel",         "spiegel.de",           "de"),
        ("Die Zeit",            "zeit.de",              "de"),
        ("FAZ",                 "faz.net",              "de"),
        ("Süddeutsche Zeitung", "sueddeutsche.de",      "de"),
        ("Bild",                "bild.de",              "de"),
    ],
    "Brazil": [
        ("Folha de S.Paulo",    "folha.uol.com.br",     "pt"),
        ("O Globo",             "oglobo.globo.com",     "pt"),
        ("Estadão",             "estadao.com.br",       "pt"),
        ("UOL",                 "uol.com.br",           "pt"),
        ("Veja",                "veja.abril.com.br",    "pt"),
    ],
    "Argentina": [
        ("La Nación",   "lanacion.com.ar",      "es"),
        ("Clarín",      "clarin.com",           "es"),
        ("Infobae",     "infobae.com",          "es"),
        ("Página 12",   "pagina12.com.ar",      "es"),
        ("Perfil",      "perfil.com",           "es"),
    ],
    "AlJazeera": [
        ("Al-Jazeera English", "aljazeera.com", "en"),
    ],
    "Israel": [
        ("Jerusalem Post",       "jpost.com",                "en"),
        ("Haaretz",              "haaretz.com",              "en"),
        ("Times of Israel",      "timesofisrael.com",        "en"),
        ("Ynet News",            "ynetnews.com",             "en"),
        ("Israel National News", "israelnationalnews.com",   "en"),
    ],
}

# ---------------------------------------------------------------------------
# Derived lookups (built once at import time)
# ---------------------------------------------------------------------------

# Map domain → country
_DOMAIN_TO_COUNTRY: dict[str, str] = {
    domain: country
    for country, entries in OUTLETS.items()
    for _, domain, _ in entries
}

# Map domain → outlet name
_DOMAIN_TO_OUTLET: dict[str, str] = {
    domain: name
    for entries in OUTLETS.values()
    for name, domain, _ in entries
}

# Map domain → language
_DOMAIN_TO_LANGUAGE: dict[str, str] = {
    domain: lang
    for entries in OUTLETS.values()
    for _, domain, lang in entries
}

# Flat list of all domains
ALL_DOMAINS: list[str] = list(_DOMAIN_TO_COUNTRY.keys())


def domain_to_country(domain: str) -> str | None:
    """Return country for a given domain, or None if not found."""
    return _DOMAIN_TO_COUNTRY.get(domain)


def domain_to_outlet(domain: str) -> str | None:
    """Return outlet name for a given domain, or None if not found."""
    return _DOMAIN_TO_OUTLET.get(domain)


def domain_to_language(domain: str) -> str | None:
    """Return expected ISO 639-1 language code for a given domain."""
    return _DOMAIN_TO_LANGUAGE.get(domain)


def get_all_domains() -> list[str]:
    """Return all registered source domains."""
    return ALL_DOMAINS


def build_domain_regex() -> str:
    """
    Build a regex pattern matching any registered domain in a URL.
    Escapes dots; used in GDELT BigQuery REGEXP_CONTAINS queries.
    Example output: '(nytimes\\.com|washingtonpost\\.com|...)'
    """
    escaped = [d.replace(".", "\\.") for d in ALL_DOMAINS]
    return "(" + "|".join(escaped) + ")"


if __name__ == "__main__":
    print(f"Total outlets: {len(ALL_DOMAINS)}")
    for country, entries in OUTLETS.items():
        print(f"  {country}: {[name for name, _, _ in entries]}")
    print(f"\nDomain regex (truncated): {build_domain_regex()[:120]}...")
