import requests
from bs4 import BeautifulSoup
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "data" / "interviews"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPISODES = [
    ("https://lexfridman.com/andrej-karpathy-3", "lex_333.txt"),
    ("https://lexfridman.com/andrej-karpathy-2", "lex_2.txt"),
    ("https://lexfridman.com/andrej-karpathy", "lex_1.txt"),
]


def fetch(url: str) -> str | None:
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  FAILED {url}: {e}")
        return None


def extract_transcript(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    transcript_div = (
        soup.find("div", class_="transcript")
        or soup.find("div", id="transcript")
        or soup.find("div", class_="entry-content")
        or soup.find("article")
    )
    if transcript_div:
        text = transcript_div.get_text(separator="\n")
    else:
        text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def main():
    for url, filename in EPISODES:
        print(f"Fetching {url}")
        html = fetch(url)
        if not html:
            print(f"  Skipping {filename}")
            continue
        text = extract_transcript(html)
        out_path = OUT_DIR / filename
        out_path.write_text(text, encoding="utf-8")
        print(f"  Saved {len(text)} chars to {out_path}")

    print(f"Done — saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
