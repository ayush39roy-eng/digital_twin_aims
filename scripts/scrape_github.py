import requests
from pathlib import Path

REPOS = [
    ("karpathy", "nanoGPT"),
    ("karpathy", "micrograd"),
    ("karpathy", "makemore"),
    ("karpathy", "minGPT"),
    ("karpathy", "llm.c"),
]

OUT_DIR = Path(__file__).parent.parent / "data" / "github"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_BASE = "https://raw.githubusercontent.com"


def fetch_readme(owner: str, repo: str) -> str | None:
    for branch in ["master", "main"]:
        url = f"{RAW_BASE}/{owner}/{repo}/{branch}/README.md"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                print(f"  Fetched {url}")
                return r.text
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
    return None


def main():
    for owner, repo in REPOS:
        print(f"Fetching {owner}/{repo}")
        content = fetch_readme(owner, repo)
        if content is None:
            print(f"  Not found — skipping")
            continue
        out_path = OUT_DIR / f"{repo.replace('.', '_')}_readme.txt"
        out_path.write_text(content, encoding="utf-8")
        print(f"  Saved to {out_path}")

    print(f"Done — saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
