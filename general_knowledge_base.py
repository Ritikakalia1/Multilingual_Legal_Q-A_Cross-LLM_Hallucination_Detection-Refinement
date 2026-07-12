# general_knowledge_base.py
#
# A SECOND, broader grounding source alongside the curated FAISS dataset
# (build_faiss_index.py) and per-upload DocumentIndex (document_index.py).
#
# ── DUAL-EMBEDDER EXTENSION ──
# Mirrors the split used everywhere else in the pipeline: English content
# is embedded with BAAI/bge-base-en-v1.5 (better retrieval quality on
# English-only text), Hindi/Punjabi/Nepali content stays on LaBSE. Since
# vectors from the two models live in different, incompatible spaces, this
# means TWO Chroma collections instead of one:
#
#   legal_general_kb_en    — English PDF chunks + English QA entries, BGE
#   legal_general_kb_multi — hi/pa/ne PDF chunks + QA entries, LaBSE
#
# retrieve(lang=...) selects which collection to query — same as
# hallucination_detector.retrieve_reference()'s preferred_lang routing to
# one of two FAISS indices. Citation index stays a SINGLE combined set
# across both collections (regex-based, language-agnostic).

import os
import pickle
import chromadb
from sentence_transformers import SentenceTransformer

from citation_utils import extract_citations
from reranker import reranker
from config import BGE_QUERY_INSTRUCTION

CHROMA_DIR = "chroma_store"
COLLECTION_NAME_EN = "legal_general_kb_en"
COLLECTION_NAME_MULTI = "legal_general_kb_multi"
CITATION_INDEX_PATH = os.path.join(CHROMA_DIR, "general_kb_citation_index.pkl")

RETRIEVAL_POOL_SIZE = 15   # wide pool for the bi-encoder pass, narrowed by
                           # reranker.rerank() afterward.


class GeneralKnowledgeBase:
    """Thin wrapper around TWO persistent Chroma collections — one per
    embedder. Embeddings for each collection are computed with whichever
    embedder matches that collection's language group; never let Chroma
    use its own default embedding function, or vectors won't live in the
    same space as everything else similarity_critic compares against."""

    def __init__(self, embedder_en: SentenceTransformer, embedder_multi: SentenceTransformer,
                 persist_dir: str = CHROMA_DIR):
        self.embedder_en = embedder_en
        self.embedder_multi = embedder_multi
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)

        # embedding_function=None on both: we always supply our own
        # embeddings via `embeddings=` on add()/query() rather than letting
        # Chroma embed text itself with its default (MiniLM, English-only)
        # model.
        self.collection_en = self.client.get_or_create_collection(
            name=COLLECTION_NAME_EN, embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_multi = self.client.get_or_create_collection(
            name=COLLECTION_NAME_MULTI, embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        self.citation_universe = self._load_citation_index()

    def _collection_and_embedder_for_lang(self, lang: str):
        if lang == "en":
            return self.collection_en, self.embedder_en, True
        return self.collection_multi, self.embedder_multi, False

    def _load_citation_index(self) -> set:
        if os.path.exists(CITATION_INDEX_PATH):
            with open(CITATION_INDEX_PATH, "rb") as f:
                return pickle.load(f)
        return set()

    def citation_exists(self, citation_number: str) -> bool:
        return citation_number in self.citation_universe

    # ── Ingestion (called by build_chroma_index.py, not per-request) ──
    def add_entries(self, entries: list, lang_group: str, batch_size: int = 64):
        """entries: list of dicts with at least 'text' + a 'lang' field.
        lang_group: "en" or "multi" — selects which collection + embedder
        this batch is added to. Caller (build_chroma_index.py) is
        responsible for splitting entries by language BEFORE calling this,
        since a single call always targets one collection.

        IDs are assigned sequentially against that collection's current
        size so re-running ingestion on a GROWING corpus just appends."""
        collection, embedder, _ = self._collection_and_embedder_for_lang(
            "en" if lang_group == "en" else "multi"
        )
        start_id = collection.count()
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            texts = [e["text"] for e in batch]
            # Passages/documents get NO BGE instruction prefix — only
            # queries do (see retrieve() below).
            embeddings = embedder.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
            ids = [f"{lang_group}_{start_id + i + j}" for j in range(len(batch))]
            metadatas = [{k: v for k, v in e.items() if k != "text"} for e in batch]
            collection.add(
                ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas,
            )

    def rebuild_citation_index(self, all_entries: list):
        """Run once at the end of ingestion, over ALL entries from BOTH
        language groups combined — the citation set stays single/combined
        regardless of the embedder split, same as build_faiss_index.py."""
        citation_index = set()
        for e in all_entries:
            citation_index |= extract_citations(e.get("text", ""))
        with open(CITATION_INDEX_PATH, "wb") as f:
            pickle.dump(citation_index, f)
        self.citation_universe = citation_index
        return citation_index

    # ── Retrieval (called per-request) ──
    def retrieve(self, question: str, top_k: int = 3, lang: str = None) -> list:
        """Routes to the English (BGE) or multilingual (LaBSE) collection
        based on `lang` — this SELECTS which corpus is searched, same as
        hallucination_detector.retrieve_reference()'s preferred_lang.
        Defaults to the multilingual collection if lang is None."""
        collection, embedder, is_bge = self._collection_and_embedder_for_lang(lang or "multi")

        if collection.count() == 0:
            return []

        query_text = (BGE_QUERY_INSTRUCTION + question) if is_bge else question
        query_vec = embedder.encode(
            [query_text], normalize_embeddings=True, convert_to_numpy=True
        ).tolist()

        pool_size = min(RETRIEVAL_POOL_SIZE, collection.count())

        raw = collection.query(query_embeddings=query_vec, n_results=pool_size)
        if not raw["ids"] or not raw["ids"][0]:
            return []

        results = []
        for doc, meta, dist in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
            # Chroma with hnsw:space=cosine returns cosine DISTANCE
            # (0 = identical); convert to the similarity convention
            # (1 = identical) the rest of the pipeline expects.
            score = 1.0 - dist
            results.append({
                "question": meta.get("question", f"[{meta.get('source_file', 'corpus')} excerpt]"),
                "answer": doc,
                "answer_en": meta.get("answer_en", doc if meta.get("lang", "en") == "en" else ""),
                "question_en": meta.get("question_en", ""),
                "act": meta.get("act", meta.get("source_file", "UNKNOWN")),
                "lang": meta.get("lang", "en"),
                "retrieval_score": float(score),
            })

        return reranker.rerank(question, results, top_k=top_k)


# ── Singleton — construct with BOTH embedders app.py/detector already
# loaded. Wire this in app.py:
#   general_kb = init_general_kb(detector.embedder_en, detector.embedder_multi) ──
general_kb = None


def init_general_kb(embedder_en: SentenceTransformer, embedder_multi: SentenceTransformer):
    global general_kb
    if general_kb is None:
        general_kb = GeneralKnowledgeBase(embedder_en, embedder_multi)
    return general_kb