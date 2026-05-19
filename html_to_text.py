"""
Usage: python3 scraper.py
Re-downloads and cleans all Wikipedia docs into docs/
"""

import re
import urllib.request

PAGES = {
    "transformer.txt": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
    "large_language_model.txt": "https://en.wikipedia.org/wiki/Large_language_model",
    "cognitive_psychology.txt": "https://en.wikipedia.org/wiki/Cognitive_psychology",
}

def fetch_and_clean(url):
    from html.parser import HTMLParser

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text = []
            self.skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ('script', 'style', 'nav', 'footer', 'table'):
                self.skip = True

        def handle_endtag(self, tag):
            if tag in ('script', 'style', 'nav', 'footer', 'table'):
                self.skip = False

        def handle_data(self, data):
            if not self.skip:
                self.text.append(data)

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8')

    parser = TextExtractor()
    parser.feed(html)
    text = ' '.join(parser.text)

    # remove citation numbers like [ 31 ] or [note 1]
    text = re.sub(r'\[\s*\d+\s*\]', '', text)
    text = re.sub(r'\[note\s*\d+\]', '', text)
    text = re.sub(r'\[edit\]', '', text)

    # remove LaTeX math blocks
    text = re.sub(r'\{\\displaystyle[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)

    # remove leftover symbols from math
    text = re.sub(r'[{}\|\\]', '', text)

    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text[:50000]


if __name__ == "__main__":
    import os
    os.makedirs("docs", exist_ok=True)

    for filename, url in PAGES.items():
        print(f"Downloading {filename}...")
        text = fetch_and_clean(url)
        with open(f"docs/{filename}", "w") as f:
            f.write(text)
        print(f"  saved {len(text)} chars")

    print("\nDone. Delete chroma_db/ and rerun app.py to rebuild the vector store.")
