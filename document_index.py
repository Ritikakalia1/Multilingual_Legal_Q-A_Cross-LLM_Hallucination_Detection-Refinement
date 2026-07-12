# document_index.py
#
# On-the-fly retrieval for user-uploaded documents (FIRs, contracts,
# judgment excerpts). NOW routes to the same dual embedder setup as the
# static dataset: English documents use BAAI/bge-base-en-v1.5, hi/pa/ne
# documents use LaBSE — DocumentIndexManager picks the right one at
# upload time based on the declared language, and DocumentIndex stores
# whether it's using BGE (for the query-instruction-prefix convention).
#
# Retrieval remains two-stage: FAISS pulls a wide candidate pool of
# chunks, then a cross-encoder (reranker.py) reranks that pool.

import time
import uuid
import faiss
from sentence_transformers import SentenceTransformer

from citation_utils import extract_citations
from reranker import reranker
from config import BGE_QUERY_INSTRUCTION

CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40
DOC_TTL_SECONDS = 3600
DOC_RETRIEVAL_POOL_SIZE = 10


def chunk_document(text: str, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS) -> list:
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


class DocumentIndex:
    """One FAISS index + chunk store for a single uploaded document.

    is_bge: whether `embedder` is the BGE model — determines whether
    queries at retrieval time need the BGE_QUERY_INSTRUCTION prefix.
    Chunks (passages) never get this prefix, per BGE's own convention."""

    def __init__(self, doc_id, doc_name, lang, embedder: SentenceTransformer, is_bge: bool = False):
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.lang = lang
        self.embedder = embedder
        self.is_bge = is_bge
        self.chunks = []
        self.index = None
        self.citation_universe = set()
        self.last_used = time.time()

    def build(self, raw_text: str):
        self.chunks = chunk_document(raw_text)
        if not self.chunks:
            raise ValueError("Document is empty after chunking.")
        # Passages get NO instruction prefix under BGE's convention.
        embeddings = self.embedder.encode(
            self.chunks, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.citation_universe = extract_citations(raw_text)

    def retrieve(self, question: str, top_k: int = 3) -> list:
        self.last_used = time.time()
        query_text = (BGE_QUERY_INSTRUCTION + question) if self.is_bge else question
        query_vec = self.embedder.encode(
            [query_text], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        pool_size = min(max(top_k, DOC_RETRIEVAL_POOL_SIZE), len(self.chunks))
        scores, indices = self.index.search(query_vec, pool_size)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk_text = self.chunks[idx]
            results.append({
                "question": f"[chunk {idx} of {self.doc_name}]",
                "answer": chunk_text, "answer_en": chunk_text, "question_en": "",
                "act": self.doc_name, "lang": self.lang,
                "retrieval_score": float(score),
            })

        if not results:
            return results

        return reranker.rerank(question, results, top_k=top_k)

    def citation_exists(self, citation_number: str) -> bool:
        return citation_number in self.citation_universe


class DocumentIndexManager:
    """NOW holds BOTH embedders and picks the right one per document at
    upload time based on its declared language, instead of one shared
    embedder passed in at construction."""

    def __init__(self, embedder_en: SentenceTransformer, embedder_multi: SentenceTransformer):
        self.embedder_en = embedder_en
        self.embedder_multi = embedder_multi
        self._store = {}

    def create(self, doc_name: str, lang: str, raw_text: str) -> str:
        self._evict_stale()
        doc_id = uuid.uuid4().hex[:12]
        if lang == "en":
            embedder, is_bge = self.embedder_en, True
        else:
            embedder, is_bge = self.embedder_multi, False
        doc_index = DocumentIndex(doc_id, doc_name, lang, embedder, is_bge=is_bge)
        doc_index.build(raw_text)
        self._store[doc_id] = doc_index
        return doc_id

    def get(self, doc_id: str):
        doc_index = self._store.get(doc_id)
        if doc_index:
            doc_index.last_used = time.time()
        return doc_index

    def _evict_stale(self):
        now = time.time()
        for k in [k for k, v in self._store.items() if now - v.last_used > DOC_TTL_SECONDS]:
            del self._store[k]