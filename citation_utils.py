# citation_utils.py
#
# Shared citation-extraction logic used by both build_faiss_index.py
# (to build citation_index.pkl) and hallucination_detector.py (to check
# generated answers against it). Kept in one place deliberately — these
# two use sites need to agree on exactly what counts as a citation, or
# the dataset-wide index and the per-answer check will drift apart.

import re

# Each pattern captures the section/article NUMBER so we can compare numbers,
# not exact phrasing (since "Section 302" vs "धारा 302" should be treated
# as the same citation when checked against a bilingual reference entry).
CITATION_PATTERNS = [
    # English: "Section 302", "Article 21", "Sec. 420" (number after keyword)
    re.compile(r"\b(?:Section|Sec\.?|Article|Art\.?)\s*(\d+(?:\.\d+)?[A-Za-z]?)\b", re.IGNORECASE),
    # Hindi / Nepali (Devanagari), number AFTER keyword: "धारा 302", "अनुच्छेद 21"
    re.compile(r"(?:धारा|अनुच्छेद)\s*[:\-]?\s*(\d+(?:\.\d+)?[A-Za-z]?)"),
    # Hindi / Nepali (Devanagari), number BEFORE keyword: "302 धारा"
    re.compile(r"\b(\d+(?:\.\d+)?[A-Za-z]?)\s*(?:धारा|अनुच्छेद)"),
    # Punjabi (Gurmukhi), number after: "ਧਾਰਾ 302"
    re.compile(r"ਧਾਰਾ\s*[:\-]?\s*(\d+(?:\.\d+)?[A-Za-z]?)"),
    # Punjabi (Gurmukhi), number before: "302 ਧਾਰਾ"
    re.compile(r"\b(\d+(?:\.\d+)?[A-Za-z]?)\s*ਧਾਰਾ"),
]


def extract_citations(text: str) -> set:
    """Pull out every section/article number mentioned in a piece of text."""
    citations = set()
    for pattern in CITATION_PATTERNS:
        for match in pattern.finditer(text):
            citations.add(match.group(1).strip())
    return citations