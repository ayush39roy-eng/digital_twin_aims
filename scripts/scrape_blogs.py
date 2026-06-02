import requests
from bs4 import BeautifulSoup
from pathlib import Path
import re
import time

BASE_URL = "https://karpathy.github.io"
OUT_DIR = Path(__file__).parent.parent / "data" / "blogs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch(url: str) -> str | None:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  FAILED {url}: {e}")
        return None


def extract_post_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.match(r"^/\d{4}/", href):
            links.append(BASE_URL + href)
    return list(dict.fromkeys(links))


def extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article") or soup.find("div", class_="post-content") or soup.find("div", class_="entry-content")
    if article:
        text = article.get_text(separator="\n")
    else:
        text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def slug(url: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", url.lower().split("karpathy.github.io/")[-1].strip("/")) or "post"


def main():
    print(f"Fetching index from {BASE_URL}")
    html = fetch(BASE_URL)
    if not html:
        print("Could not fetch main page.")
        return

    links = extract_post_links(html)
    print(f"Found {len(links)} post links")

    for url in links:
        print(f"Scraping {url}")
        html = fetch(url)
        if not html:
            continue
        text = extract_article_text(html)
        filename = OUT_DIR / f"{slug(url)}.txt"
        filename.write_text(text, encoding="utf-8")
        time.sleep(0.5)

    print(f"Done — saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
