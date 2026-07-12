# hallucination_detector.py
#
# Given a generated legal answer, this module:
#   1. Retrieves a shortlist of candidate gold reference QA pairs via
#      FAISS (build_faiss_index.py) OR, for user-uploaded documents, the
#      closest chunks from an in-memory DocumentIndex (document_index.py),
#      OR, for the broader legal_corpus (PDFs + 10K QA json), the closest
#      matches from GeneralKnowledgeBase (general_knowledge_base.py) —
#      then RERANKS that shortlist with a cross-encoder (reranker.py).
#   1b. VERIFIES retrieval against any section/article number explicitly
#      named in the question (retrieval_verifier.py).
#   2. Runs a Citation Grounding Critic.
#   3. Runs a Semantic Similarity Critic — cosine similarity between the
#      generated answer and the retrieved reference. NOW uses a
#      DUAL-EMBEDDER setup: English text is embedded with BAAI/bge-base-
#      en-v1.5 (better retrieval quality on English-only corpora than a
#      multilingual model); Hindi/Punjabi/Nepali text stays on LaBSE
#      (the one model here that actually understands those languages).
#      Every embed call routes to the correct model based on the
#      relevant entry's `lang` field — mixing vectors from the two
#      models would be meaningless, since they live in different,
#      incompatible vector spaces.
#   4. Runs an Entailment/Contradiction Critic.
#   5. Meta-critic combines all of the above into a single grounding score
#      + verdict — including an ABSTAIN verdict when grounding is too low.
#
# ── DUAL-EMBEDDER EXTENSION (NEW) ──
# self.embedder_en / self.embedder_multi replace the single self.embedder.
# self.embedder is KEPT as an alias to embedder_multi for backward
# compatibility with anything still referencing detector.embedder directly
# (e.g. document_index.py's/general_knowledge_base.py's default args) —
# LaBSE is the safer general fallback across all four languages if a
# caller doesn't specify.
#
# Two static FAISS indices now exist instead of one:
#   legal_qa_en.index / legal_qa_en_metadata.pkl       (English, BGE)
#   legal_qa_multi.index / legal_qa_multi_metadata.pkl (hi/pa/ne, LaBSE)
# citation_index.pkl is unchanged — one combined set, regex-based.
#
# BGE models are trained asymmetrically: the query side needs an
# instruction prefix for good retrieval quality; the passage/document side
# does not. See BGE_QUERY_INSTRUCTION in config.py and _encode_query below.

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from citation_utils import extract_citations
from entailment_checker import entailment_checker
from reranker import reranker
from retrieval_verifier import verify_and_reorder
from config import EMBED_MODEL_EN, EMBED_MODEL_MULTI, BGE_QUERY_INSTRUCTION, FAISS_DIR

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def simple_tokenize(text: str) -> list:
    return text.strip().split()


def bleu_score(reference: str, generated: str) -> float:
    try:
        ref_tokens = [simple_tokenize(reference)]
        gen_tokens = simple_tokenize(generated)
        smoothie = SmoothingFunction().method4
        return float(sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothie))
    except Exception:
        return 0.0


def rouge_l_score(reference: str, generated: str) -> float:
    try:
        scores = _rouge.score(reference, generated)
        return float(scores["rougeL"].fmeasure)
    except Exception:
        return 0.0

EN_INDEX_PATH = f"{FAISS_DIR}/legal_qa_en.index"
EN_METADATA_PATH = f"{FAISS_DIR}/legal_qa_en_metadata.pkl"
MULTI_INDEX_PATH = f"{FAISS_DIR}/legal_qa_multi.index"
MULTI_METADATA_PATH = f"{FAISS_DIR}/legal_qa_multi_metadata.pkl"
CITATION_INDEX_PATH = f"{FAISS_DIR}/citation_index.pkl"

# ── Thresholds (tune these against a labeled sample once you have one) ──
SIMILARITY_HALLUCINATION_THRESHOLD = 0.55
CITATION_MISMATCH_WEIGHT = 0.5
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.60       # dataset path (question-to-question, LaBSE-era value —
                                             # RETUNE for the English/BGE path once you have a few
                                             # real examples; BGE's cosine-similarity distribution
                                             # can sit at different absolute values than LaBSE's.
                                             # This constant currently applies to BOTH the en and
                                             # multi dataset paths — split it into
                                             # RETRIEVAL_CONFIDENCE_THRESHOLD_EN /
                                             # _MULTI if calibration shows they need different bars.)
DOC_RETRIEVAL_CONFIDENCE_THRESHOLD = 0.15
GENERAL_KB_CONFIDENCE_THRESHOLD = 0.20

GROUNDING_SCORE_CONTRADICTION_CAP = 0.15
RETRIEVAL_MISMATCH_GROUNDING_CAP = 0.20
ABSTENTION_GROUNDING_THRESHOLD = 0.40
RETRIEVAL_POOL_SIZE = 10


class HallucinationDetector:
    def __init__(self):
        print(f"Loading English embedding model: {EMBED_MODEL_EN} ...")
        self.embedder_en = SentenceTransformer(EMBED_MODEL_EN)

        print(f"Loading multilingual embedding model: {EMBED_MODEL_MULTI} ...")
        self.embedder_multi = SentenceTransformer(EMBED_MODEL_MULTI)

        # Backward-compat alias — points at the multilingual model since
        # it's the safer default across all four languages for any caller
        # that doesn't specify which embedder it wants.
        self.embedder = self.embedder_multi

        print("Loading FAISS indices and metadata...")
        self.index_en = faiss.read_index(EN_INDEX_PATH)
        with open(EN_METADATA_PATH, "rb") as f:
            self.metadata_en = pickle.load(f)

        self.index_multi = faiss.read_index(MULTI_INDEX_PATH)
        with open(MULTI_METADATA_PATH, "rb") as f:
            self.metadata_multi = pickle.load(f)

        with open(CITATION_INDEX_PATH, "rb") as f:
            self.citation_index = pickle.load(f)

        print(f"✅ Detector ready (en: {self.index_en.ntotal} entries, "
              f"multi: {self.index_multi.ntotal} entries, "
              f"{len(self.citation_index)} unique citation numbers loaded)")

    def _embedder_and_index_for_lang(self, lang: str):
        """Routes to the English (BGE) or multilingual (LaBSE) embedder +
        index pair based on language. Defaults to multilingual if lang is
        None/unrecognized — LaBSE is the safer general fallback."""
        if lang == "en":
            return self.embedder_en, self.index_en, self.metadata_en, True
        return self.embedder_multi, self.index_multi, self.metadata_multi, False

    def _encode_query(self, embedder, text: str, is_bge: bool):
        """BGE needs a query-side instruction prefix for good retrieval
        quality; the passage/document side does not. LaBSE has no such
        requirement, so is_bge=False is a no-op prefix."""
        prefixed = (BGE_QUERY_INSTRUCTION + text) if is_bge else text
        return embedder.encode(
            [prefixed], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

    def citation_exists_in_dataset(self, citation_number: str) -> bool:
        return citation_number in self.citation_index

    def retrieve_reference(self, question: str, top_k: int = 1, preferred_lang: str = None):
        """Find the closest gold QA entries to the INPUT QUESTION.

        NOW routes to the English (BGE) or multilingual (LaBSE) index
        based on preferred_lang — since the two indices live in different
        embedding spaces, there is no single "search everything" call
        anymore; preferred_lang effectively selects WHICH corpus you're
        searching, not just a re-ranking preference within one corpus.
        Defaults to the multilingual index if preferred_lang is None.

        Two-stage within whichever index is selected: FAISS bi-encoder
        pulls a wide candidate pool (RETRIEVAL_POOL_SIZE), then a
        cross-encoder reranks that pool.
        """
        lang_for_routing = preferred_lang or "multi"
        embedder, index, metadata, is_bge = self._embedder_and_index_for_lang(
            "en" if lang_for_routing == "en" else "multi"
        )
        query_vec = self._encode_query(embedder, question, is_bge)

        pool_size = max(top_k, RETRIEVAL_POOL_SIZE)
        pool_size = min(pool_size, index.ntotal) if index.ntotal > 0 else pool_size
        scores, indices = index.search(query_vec, pool_size)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = metadata[idx]
            results.append({**entry, "retrieval_score": float(score)})

        if not results:
            return results

        reranked = reranker.rerank(question, results, top_k=None)
        return reranked[:top_k]

    # ── Critic 1: Citation Grounding ── (UNCHANGED — no embedder dependency)
    def citation_grounding_critic(self, generated_answer: str, reference_entry: dict,
                                   citation_exists_fn=None) -> dict:
        citation_exists_fn = citation_exists_fn or self.citation_exists_in_dataset

        generated_citations = extract_citations(generated_answer)

        reference_text = (
            reference_entry.get("answer", "")
            + " " + reference_entry.get("answer_en", "")
            + " " + reference_entry.get("question", "")
            + " " + reference_entry.get("question_en", "")
        )
        reference_citations = extract_citations(reference_text)

        if not generated_citations:
            return {
                "citations_found": [], "unverified_citations": [], "grounded_citations": [],
                "verdict": "no_citations_claimed", "penalty": 0.0,
            }

        wide_unverified = {c for c in generated_citations if not citation_exists_fn(c)}
        exists_but_wrong_reference = (
            (generated_citations - wide_unverified) - reference_citations
        )
        unverified = wide_unverified | exists_but_wrong_reference
        grounded = generated_citations & reference_citations
        penalty = CITATION_MISMATCH_WEIGHT * (len(unverified) / len(generated_citations))

        return {
            "citations_found": sorted(generated_citations),
            "unverified_citations": sorted(unverified),
            "grounded_citations": sorted(grounded),
            "verdict": "fabricated_citation" if unverified else "all_citations_grounded",
            "penalty": penalty,
        }

    # ── Critic 2: Semantic Similarity (+ supplementary lexical metrics) ──
    def similarity_critic(self, generated_answer: str, reference_entry: dict) -> dict:
        """NOW routes to the embedder that matches reference_entry['lang']
        — comparing vectors from two different embedding spaces (BGE vs
        LaBSE) would be meaningless. Both generated_answer and the
        reference answer are PASSAGES being compared to each other here
        (not a query-to-passage comparison), so under BGE's convention
        neither side gets the query instruction prefix."""
        lang = reference_entry.get("lang", "en")
        embedder, _, _, _ = self._embedder_and_index_for_lang(lang)

        gen_vec = embedder.encode([generated_answer], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        ref_vec = embedder.encode([reference_entry["answer"]], normalize_embeddings=True, convert_to_numpy=True).astype("float32")

        similarity = float(np.dot(gen_vec[0], ref_vec[0]))
        bleu = bleu_score(reference_entry["answer"], generated_answer)
        rouge_l = rouge_l_score(reference_entry["answer"], generated_answer)

        return {
            "similarity_score": similarity,
            "bleu_score": round(bleu, 3),
            "rouge_l_score": round(rouge_l, 3),
            "verdict": "low_similarity" if similarity < SIMILARITY_HALLUCINATION_THRESHOLD else "acceptable_similarity",
        }

    # ── Critic 3: Entailment / Contradiction ── (UNCHANGED — separate NLI model)
    def entailment_critic(self, generated_answer: str, reference_entry: dict) -> dict:
        result = entailment_checker.check(
            reference_answer=reference_entry["answer"],
            draft_answer=generated_answer,
            lang=reference_entry.get("lang", "en"),
        )
        return result

    # ── Shared meta-critic core ── (UNCHANGED from your current version)
    def _run_meta_critic(self, question: str, generated_answer: str,
                          top_matches: list, citation_exists_fn,
                          retrieval_threshold: float = RETRIEVAL_CONFIDENCE_THRESHOLD) -> dict:
        if not top_matches:
            return {"error": "No reference entries retrieved."}

        top_matches, verification_info = verify_and_reorder(question, top_matches)
        citation_verified = verification_info["matched_index"] is not None
        verification_info_out = {
            "citation_queried": sorted(verification_info["citation_queried"]),
            "matched_index": verification_info["matched_index"],
        }

        best_match = top_matches[0]

        if not citation_verified and best_match["retrieval_score"] < retrieval_threshold:
            return {
                "question": question,
                "generated_answer": generated_answer,
                "retrieved_reference": {
                    "question": best_match["question"], "answer": best_match["answer"],
                    "act": best_match["act"], "lang": best_match["lang"],
                },
                "retrieval_confidence": best_match["retrieval_score"],
                "verdict": "NO CONFIDENT REFERENCE FOUND (source may not cover this topic)",
                "grounding_score": None, "hallucination_score": None,
                "citation_check": None, "similarity_check": None, "entailment_check": None,
                "verification_info": verification_info_out,
                "top_matches": [
                    {"question": m["question"], "answer": m["answer"], "score": m["retrieval_score"]}
                    for m in top_matches
                ],
            }

        citation_result = self.citation_grounding_critic(
            generated_answer, best_match, citation_exists_fn=citation_exists_fn
        )
        similarity_result = self.similarity_critic(generated_answer, best_match)
        entailment_result = self.entailment_critic(generated_answer, best_match)

        grounding_score = max(0.0, similarity_result["similarity_score"] - citation_result["penalty"])

        is_contradiction = entailment_result["is_contradiction"]
        if is_contradiction:
            grounding_score = min(grounding_score, GROUNDING_SCORE_CONTRADICTION_CAP)

        retrieval_mismatch = bool(verification_info["citation_queried"]) and not citation_verified
        if retrieval_mismatch:
            grounding_score = min(grounding_score, RETRIEVAL_MISMATCH_GROUNDING_CAP)

        hallucination_score = round(1.0 - grounding_score, 3)

        if is_contradiction:
            final_verdict = "HALLUCINATION DETECTED (contradicts reference)"
        elif retrieval_mismatch:
            final_verdict = "HALLUCINATION DETECTED (retrieved reference doesn't cover the cited section)"
        elif citation_result["verdict"] == "fabricated_citation":
            final_verdict = "HALLUCINATION DETECTED (fabricated citation)"
        elif grounding_score < ABSTENTION_GROUNDING_THRESHOLD:
            final_verdict = "ABSTAIN (insufficient evidence to answer reliably)"
        elif similarity_result["verdict"] == "low_similarity":
            final_verdict = "HALLUCINATION DETECTED (low grounding to reference)"
        else:
            final_verdict = "LIKELY GROUNDED"

        return {
            "question": question,
            "generated_answer": generated_answer,
            "retrieved_reference": {
                "question": best_match["question"], "answer": best_match["answer"],
                "act": best_match["act"], "lang": best_match["lang"],
            },
            "citation_check": citation_result,
            "similarity_check": similarity_result,
            "entailment_check": entailment_result,
            "verification_info": verification_info_out,
            "grounding_score": round(grounding_score, 3),
            "hallucination_score": hallucination_score,
            "verdict": final_verdict,
            "top_matches": [
                {"question": m["question"], "answer": m["answer"], "score": m["retrieval_score"]}
                for m in top_matches
            ],
        }

    # ── Meta-critic: static dataset path ──
    def evaluate(self, question: str, generated_answer: str, lang: str = None) -> dict:
        """lang now determines WHICH index (English/BGE vs multilingual/
        LaBSE) is searched, not just a re-ranking preference — see
        retrieve_reference()."""
        top_matches = self.retrieve_reference(question, top_k=3, preferred_lang=lang)
        if not top_matches:
            return {"error": "No reference entries retrieved — check FAISS index."}
        return self._run_meta_critic(
            question, generated_answer, top_matches,
            citation_exists_fn=self.citation_exists_in_dataset,
        )

    # ── Meta-critic: user-uploaded document path ── (UNCHANGED — DocumentIndex
    # already picks its own embedder at creation time in document_index.py)
    def evaluate_against_document(self, question: str, generated_answer: str, doc_index) -> dict:
        top_matches = doc_index.retrieve(question, top_k=3)
        if not top_matches:
            return {"error": "Document produced no retrievable chunks."}
        return self._run_meta_critic(
            question, generated_answer, top_matches,
            citation_exists_fn=doc_index.citation_exists,
            retrieval_threshold=DOC_RETRIEVAL_CONFIDENCE_THRESHOLD,
        )

    # ── Meta-critic: broader legal_corpus knowledge base path ──
    def evaluate_against_general_kb(self, question: str, generated_answer: str,
                                     general_kb, lang: str = None) -> dict:
        """UNCHANGED in signature — general_kb.retrieve() is responsible
        for its own embedder routing (see the note in the "still need to
        update" section below: general_knowledge_base.py likely needs the
        same dual-embedder treatment, but I haven't seen its current code)."""
        top_matches = general_kb.retrieve(question, top_k=3, lang=lang)
        if not top_matches:
            return {"error": "General knowledge base returned no candidates."}
        return self._run_meta_critic(
            question, generated_answer, top_matches,
            citation_exists_fn=general_kb.citation_exists,
            retrieval_threshold=GENERAL_KB_CONFIDENCE_THRESHOLD,
        )

    # ── Refinement loop ── (UNCHANGED)
    def refine_and_reevaluate(self, lang: str, question: str, initial_result: dict) -> dict:
        if initial_result.get("verdict") == "LIKELY GROUNDED":
            return {"refinement_needed": False, "initial_result": initial_result}

        if initial_result.get("verdict", "").startswith("NO CONFIDENT REFERENCE"):
            return {
                "refinement_needed": False, "initial_result": initial_result,
                "note": "No reliable reference to ground refinement in — source likely doesn't cover this topic.",
            }

        if initial_result.get("verdict", "").startswith("ABSTAIN"):
            return {
                "refinement_needed": False, "initial_result": initial_result,
                "note": "Grounding was too low to responsibly refine into a confident answer — abstaining instead.",
            }

        from model_loader import model_loader as loader

        reference = initial_result["retrieved_reference"]
        grounding_context = f"{reference['question']}\n{reference['answer']}"

        refined_answer = loader.generate(lang=lang, question=question, context=grounding_context)
        refined_result = self.evaluate(question, refined_answer, lang=lang)

        return {
            "refinement_needed": True,
            "initial_result": initial_result,
            "refined_answer": refined_answer,
            "refined_result": refined_result,
            "improved": (
                (refined_result.get("grounding_score") or 0)
                > (initial_result.get("grounding_score") or 0)
            ),
        }


detector = HallucinationDetector()