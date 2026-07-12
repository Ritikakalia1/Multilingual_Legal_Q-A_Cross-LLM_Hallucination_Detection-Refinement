# retrieval_verifier.py
#
# Cheap, deterministic check that runs BEFORE the meta-critic scores an
# answer: if the question explicitly names a section/article number ("What
# is the punishment under Section 303?"), does any retrieved candidate
# actually cite that same number anywhere in its own text?
#
# This is the "does the section number match?" step from the improvement
# roadmap — retrieval mistakes (asking about 303, getting 304B back)
# currently cascade silently through the whole pipeline. This module
# doesn't replace retrieval; it re-ranks the existing shortlist using an
# exact citation match as a hard signal, so a correct-but-lower-cosine-
# similarity candidate can still win over a higher-similarity but
# wrong-section one.

from citation_utils import extract_citations


def verify_and_reorder(question: str, candidates: list) -> tuple:
    """
    question: the resolved question text.
    candidates: list of retrieval result dicts (must have 'question',
        'answer', 'question_en', 'answer_en' keys — same shape produced by
        HallucinationDetector.retrieve_reference / DocumentIndex.retrieve).

    Returns (reordered_candidates, verification_info) where
    verification_info = {
        "citation_queried": set of citation numbers found in the question,
        "matched_index": index into the ORIGINAL candidates list that was
            promoted to top (or None if no citation was queried, or none
            of the candidates matched it),
    }

    If the question cites no section/article number, this is a no-op —
    candidates are returned unchanged, since there's nothing to verify.
    """
    queried_citations = extract_citations(question)
    if not queried_citations or not candidates:
        return candidates, {"citation_queried": queried_citations, "matched_index": None}

    def candidate_citations(c):
        combined = " ".join([
            c.get("question", ""), c.get("answer", ""),
            c.get("question_en", ""), c.get("answer_en", ""),
        ])
        return extract_citations(combined)

    for i, cand in enumerate(candidates):
        if queried_citations & candidate_citations(cand):
            if i == 0:
                return candidates, {"citation_queried": queried_citations, "matched_index": 0}
            # Promote the matching candidate to the front; keep the rest in
            # their existing relative order behind it.
            reordered = [candidates[i]] + candidates[:i] + candidates[i + 1:]
            return reordered, {"citation_queried": queried_citations, "matched_index": i}

    # None of the retrieved candidates cite the number the question asked
    # about — retrieval likely missed entirely. Caller uses matched_index
    # is None as a signal to cap grounding_score / lean toward abstention.
    return candidates, {"citation_queried": queried_citations, "matched_index": None}