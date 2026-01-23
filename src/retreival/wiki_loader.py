import wikipediaapi
from pathlib import Path

CACHE_DIR = Path("data/wiki")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

wiki = wikipediaapi.Wikipedia(
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

def fetch_wikipedia_page(title: str) -> str | None:
    cache_file = CACHE_DIR / f"{title.replace(' ', '_')}.txt"

    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    page = wiki.page(title)
    if not page.exists():
        return None

    text = page.text
    cache_file.write_text(text, encoding="utf-8")
    return text
