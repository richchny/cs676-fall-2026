"""
Test script for fetching real publication metadata from Crossref, using an
exact DOI extracted from a URL.

Same lesson as the arXiv test: query by the exact identifier already present
in the URL, never by a fuzzy title search.
"""

import re
import requests


def extract_doi(url: str) -> str | None:
    """
    Pull a DOI out of a URL path, e.g.
    https://www.pnas.org/doi/10.1073/pnas.2020123118 -> "10.1073/pnas.2020123118"

    DOIs always start with "10." followed by a registrant code, a slash, and
    a suffix. This mirrors the regex already used in credibility.py's
    rule_based_signals() to detect that a DOI is present at all.
    """
    match = re.search(r"(10\.\d{4,9}/[^\s/]+)", url)
    return match.group(1) if match else None


def query_crossref(doi: str) -> dict:
    """
    Query Crossref by exact DOI. This is an exact-match lookup keyed on the
    DOI itself, so it cannot collide with an unrelated work the way a title
    search can.
    """
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()["message"]


if __name__ == "__main__":
    test_url = "https://www.nejm.org/doi/full/10.1056/NEJMoa2034577"
    doi = extract_doi(test_url)
    print(f"Extracted DOI: {doi}")

    if doi:
        data = query_crossref(doi)
        print("\nReal metadata from Crossref:")
        print(f"  title: {data.get('title')}")
        print(f"  container-title (venue): {data.get('container-title')}")
        print(f"  publisher: {data.get('publisher')}")
        print(f"  type: {data.get('type')}")
        print(f"  published: {data.get('published')}")
        print(f"  is-referenced-by-count (citations): {data.get('is-referenced-by-count')}")
        # Crossref exposes update notices (including retractions) here:
        print(f"  update-to: {data.get('update-to')}")
    else:
        print("No DOI found in that URL.")