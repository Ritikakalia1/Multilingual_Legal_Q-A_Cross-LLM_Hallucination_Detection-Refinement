# reranker.py
#
# Cross-encoder reranking stage: FAISS/embedding retrieval (bi-encoder) is
# fast but comparing two INDEPENDENTLY-embedded vectors is a coarser
# similarity signal than a cross-encoder that reads the query and each
# candidate TOGETHER. This stage takes the top-k bi-encoder candidates and
# re-scores/re-orders just that shortlist — cheap, since k is small (e.g.
# 10), not the whole corpus.
#
# Uses a multilingual cross-encoder since the dataset spans en/hi/pa/ne.

import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

try:
    # Add RERANKER_MODEL = "..." to config.py to override.
    from config import RERANKER_MODEL
except ImportError:
    # Multilingual MiniLM cross-encoder — covers Hindi/Punjabi/Nepali/English
    # reasonably well, and is small enough to run alongside the embedder +
    # the two LLMs without blowing the VRAM budget.
    RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Reranker:
    def __init__(self):
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        logger.info(f"Loading reranker: {RERANKER_MODEL}")
        self.model = CrossEncoder(RERANKER_MODEL)
        logger.info("✅ Reranker ready")

    def rerank(self, query: str, candidates: list, top_k: int = None) -> list:
        """
        candidates: list of dicts with at least a 'question' and 'answer'
        key (same shape produced by retrieve_reference() / DocumentIndex.
        retrieve()). Reranks against each candidate's ANSWER text (or chunk
        text, for uploaded documents) — that's what actually needs to
        ground the generated answer, not just the matched question.

        Adds a 'rerank_score' key to each candidate dict (mutates in
        place) and returns the list re-sorted by it, descending.
        'retrieval_score' (the original bi-encoder cosine score) is left
        untouched alongside it, so both signals stay visible downstream.
        """
        if not candidates:
            return candidates
        if len(candidates) == 1:
            candidates[0]["rerank_score"] = candidates[0].get("retrieval_score", 0.0)
            return candidates

        self._load()

        pairs = [(query, c.get("answer", "")) for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return reranked[:top_k] if top_k else reranked

    def unload(self):
        self.model = None


# ── Singleton, mirroring model_loader's / critic_llm's pattern ──
reranker = Reranker()