"""
Test script for fetching real academic metadata for an arXiv paper.

Lesson learned: searching OpenAlex by title alone can match the WRONG paper
(title collisions, paper-mill entries using famous titles). This version
extracts the exact arXiv ID from the URL and queries arXiv's own API, which
matches by ID and cannot be fooled by a duplicate title.
"""

import re
import requests
import xml.etree.ElementTree as ET


def extract_arxiv_id(url: str) -> str | None:
    """
    Pull the arXiv ID out of a URL like https://arxiv.org/abs/1706.03762
    Returns None if no arXiv-style ID is found.
    """
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", url)
    return match.group(1) if match else None


def query_arxiv_api(arxiv_id: str) -> dict:
    """
    Query arXiv's own API by exact ID. This cannot collide with an unrelated
    paper that happens to share a title, unlike a text search.
    """
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    # arXiv's API returns Atom XML, not JSON.
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(response.text)
    entry = root.find("atom:entry", ns)

    if entry is None:
        return {}

    return {
        "title": entry.find("atom:title", ns).text.strip(),
        "published": entry.find("atom:published", ns).text,
        "updated": entry.find("atom:updated", ns).text,
        "authors": [
            a.find("atom:name", ns).text
            for a in entry.findall("atom:author", ns)
        ],
        "summary": entry.find("atom:summary", ns).text.strip()[:200],
    }


if __name__ == "__main__":
    test_url = "https://arxiv.org/abs/1706.03762"
    arxiv_id = extract_arxiv_id(test_url)
    print(f"Extracted arXiv ID: {arxiv_id}")

    if arxiv_id:
        data = query_arxiv_api(arxiv_id)
        print("\nReal metadata from arXiv's own API:")
        for key, value in data.items():
            print(f"  {key}: {value}")
    else:
        print("Could not extract an arXiv ID from that URL.")

#run with uv run python test_openalex.py

#What we did (Claude):

# Tried OpenAlex's title search → got a wrong/fake match (title collision), which taught us searching by title is unreliable.
# Built an arXiv-specific fix: extract the arXiv ID directly from the URL, query arXiv's own API by that exact ID → got the correct, verified record.

# Why we're not continuing down the arXiv-specific path:

# It only applies to arXiv (and by extension, maybe bioRxiv with similar logic) — just 2 of your 24 test URLs.
# arXiv's own API gives title/date/authors, but not the actual signal you need (peer-review status, citation count) — you'd still need a second lookup (OpenAlex or Crossref) to get that, and doing it correctly (avoiding the same collision problem) adds more complexity for a narrow payoff.