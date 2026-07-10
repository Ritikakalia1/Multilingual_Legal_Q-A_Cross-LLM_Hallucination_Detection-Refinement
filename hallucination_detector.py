# hallucination_detector.py
#
# Given a generated legal answer, this module:
#   1. Retrieves the closest gold reference QA pair via FAISS (build_faiss_index.py)
#   2. Runs a Citation Grounding Critic — extracts "Section X" / "धारा X" /
#      "ਧਾਰਾ X" style citations from the generated answer and checks whether
#      each one actually appears in the retrieved reference. This is the
#      critic that directly targets the fabricated-citation failure mode
#      you saw (e.g. "BNS 1987 धारा 7.3", "Constitution Sixty-third Amendment
#      Act, 1992" — neither exists in the reference).
#   3. Runs a Semantic Similarity Critic — cosine similarity between the
#      generated answer and the retrieved reference, using the same
#      multilingual embedding space used to build the index.
#   4. Runs an Entailment/Contradiction Critic — cosine similarity cannot
#      tell "Yes, State includes a Union territory" apart from "State does
#      NOT include a Union territory" (both share nearly all the same
#      words, so similarity stays high ~0.82 despite being opposites). This
#      critic explicitly checks logical polarity against the reference,
#      independent of topical similarity.
#   5. Meta-critic combines all three into a single grounding score + verdict.
#
# Deliberately NOT using an LLM to judge citation grounding — that check is
# regex + string match against retrieved ground truth, so it itself can't
# hallucinate. The entailment critic below IS a model, but a small
# discriminative NLI classifier constrained to 3 labels, not a free-text
# generator — it can't fabricate new claims, only classify the relationship
# between two texts it's shown directly.
#
# ── NAMING FIX (see hallucination_score / grounding_score below) ──
# Previously this module computed a single field called "hallucination_score"
# that actually behaved as a GROUNDING score: it started near 0 for bad
# answers and rose toward 1 for well-grounded ones (LIKELY GROUNDED verdicts
# always had the *highest* values of this field). That's the opposite of
# what "hallucination score" implies to a reader, and it's exactly the
# 0.451 -> 1.000 confusion seen in the dashboard. The underlying math was
# fine — only the name was wrong. Fix: the core metric is now called
# `grounding_score` (0 = ungrounded, 1 = fully grounded), and a genuine
# `hallucination_score = 1 - grounding_score` (0 = clean, 1 = hallucinated)
# is also computed and returned, so the dashboard can show either framing
# with an explicit "higher/lower is better" label instead of one field
# straddling both meanings.

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from citation_utils import extract_citations
from entailment_checker import entailment_checker

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
# use_stemmer=False deliberately — Porter stemming is English-only and would
# silently do nothing (or something wrong) on Devanagari/Gurmukhi text.


def simple_tokenize(text: str) -> list:
    """Whitespace tokenization — works reasonably across en/hi/pa/ne scripts.
    NLTK's word_tokenize is English-tuned (handles English contractions/
    punctuation rules) and isn't a good fit for Devanagari/Gurmukhi text."""
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

FAISS_DIR = "faiss_index"
INDEX_PATH = f"{FAISS_DIR}/legal_qa.index"
METADATA_PATH = f"{FAISS_DIR}/legal_qa_metadata.pkl"
CITATION_INDEX_PATH = f"{FAISS_DIR}/citation_index.pkl"
EMBED_MODEL_NAME = "sentence-transformers/LaBSE"

# ── Thresholds (tune these against a labeled sample once you have one) ──
SIMILARITY_HALLUCINATION_THRESHOLD = 0.55   # below this → likely off-topic/fabricated
CITATION_MISMATCH_WEIGHT = 0.5              # how much one bad citation drags the score down
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.60       # below this → no reliable reference exists at all
                                             # (question-to-question match, not answer similarity —
                                             # tune this against a labeled sample of "should have
                                             # matched" vs "genuinely has no match" questions.
                                             # NOTE: the known BNS test case below scores ~0.616 —
                                             # if that case should be flagged as "no confident
                                             # reference," this threshold needs raising above that,
                                             # e.g. to 0.65-0.70. Recheck after rerunning the
                                             # smoke test and calibrate against more examples.)
GROUNDING_SCORE_CONTRADICTION_CAP = 0.15    # grounding_score is forced at or below this
                                             # whenever the entailment critic flags a contradiction,
                                             # regardless of how high similarity/citation scores are.
                                             # This is what fixes the "Yes" vs "does not" bug — high
                                             # embedding similarity used to override everything.
                                             # (Previously named CONTRADICTION_SCORE_CAP — renamed
                                             # to match the grounding_score naming fix.)


class HallucinationDetector:
    def __init__(self):
        print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

        print("Loading FAISS index and metadata...")
        self.index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)

        with open(CITATION_INDEX_PATH, "rb") as f:
            self.citation_index = pickle.load(f)

        print(f"✅ Detector ready ({self.index.ntotal} reference entries, "
              f"{len(self.citation_index)} unique citation numbers loaded)")

    def citation_exists_in_dataset(self, citation_number: str) -> bool:
        """Precise check: does ANY entry in the whole dataset genuinely cite
        this section/article number? This is a harder, more reliable signal
        than semantic similarity for numeric citations — semantic retrieval
        can't tell 'this section is really discussed here' from 'this text
        happens to share legal vocabulary' (as we saw with the Article 307
        false match for a Section 302 question)."""
        return citation_number in self.citation_index

    def retrieve_reference(self, question: str, top_k: int = 1):
        """Find the closest gold QA entries to the INPUT QUESTION.

        Deliberately retrieves by question, not by the generated answer —
        if the generated answer is itself hallucinated/off-topic, embedding
        it for retrieval would pull an unrelated reference (this is exactly
        what happened when a rambling BNS hallucination retrieved an
        unrelated Article 297 territorial-waters entry). Retrieving by the
        original question keeps retrieval independent of whether the answer
        is trustworthy.
        """
        query_vec = self.embedder.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = self.metadata[idx]
            results.append({**entry, "retrieval_score": float(score)})
        return results

    # ── Critic 1: Citation Grounding ──
    def citation_grounding_critic(self, generated_answer: str, reference_entry: dict) -> dict:
        """
        Checks every section/article number mentioned in the generated answer
        against the retrieved reference (both native-language and English
        versions, since a citation might appear in either).
        """
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
                "citations_found": [],
                "unverified_citations": [],
                "grounded_citations": [],
                "verdict": "no_citations_claimed",
                "penalty": 0.0,
            }

        # First check: does this citation exist ANYWHERE in the whole
        # dataset (precise, deterministic)? This catches cases like Section
        # 302 where the retrieved reference is a coincidental/weak match —
        # checking the full citation index is more reliable than trusting
        # just the one retrieved entry.
        dataset_wide_unverified = {
            c for c in generated_citations if not self.citation_exists_in_dataset(c)
        }

        # Second check: of the citations that DO exist somewhere in the
        # dataset, are they specifically grounded in THIS retrieved reference?
        exists_but_wrong_reference = (
            (generated_citations - dataset_wide_unverified) - reference_citations
        )

        unverified = dataset_wide_unverified | exists_but_wrong_reference
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
        gen_vec = self.embedder.encode(
            [generated_answer], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        ref_vec = self.embedder.encode(
            [reference_entry["answer"]], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        similarity = float(np.dot(gen_vec[0], ref_vec[0]))

        # BLEU/ROUGE-L are SUPPLEMENTARY here, not primary — they're lexical
        # n-gram overlap metrics, and Hindi/Punjabi/Nepali have far more
        # flexible word order than English, so a genuinely correct paraphrase
        # can legitimately score low on these. Report them, but let cosine
        # similarity + citation-grounding drive the actual verdict.
        bleu = bleu_score(reference_entry["answer"], generated_answer)
        rouge_l = rouge_l_score(reference_entry["answer"], generated_answer)

        return {
            "similarity_score": similarity,
            "bleu_score": round(bleu, 3),
            "rouge_l_score": round(rouge_l, 3),
            "verdict": "low_similarity" if similarity < SIMILARITY_HALLUCINATION_THRESHOLD else "acceptable_similarity",
        }

    # ── Critic 3: Entailment / Contradiction ──
    def entailment_critic(self, generated_answer: str, reference_entry: dict) -> dict:
        """
        Cosine similarity answers "is this the same topic?" — it cannot
        answer "is this the same claim?". "Yes, X includes Y" and "X does
        NOT include Y" share almost every word and score ~0.82 similarity
        despite being direct opposites. This critic checks polarity
        explicitly: does the generated answer entail, contradict, or sit
        neutral relative to the retrieved reference (the premise)?
        """
        result = entailment_checker.check(
            reference_answer=reference_entry["answer"],
            draft_answer=generated_answer,
            lang=reference_entry.get("lang", "en"),
        )
        return result

    # ── Meta-critic: combine into one verdict ──
    def evaluate(self, question: str, generated_answer: str) -> dict:
        top_matches = self.retrieve_reference(question, top_k=3)
        if not top_matches:
            return {"error": "No reference entries retrieved — check FAISS index."}

        best_match = top_matches[0]

        # ── Retrieval-confidence gate ──
        # best_match["retrieval_score"] measures how close the INPUT QUESTION
        # is to the closest reference question (not the answer). If even the
        # best match is a weak/coincidental hit — e.g. the input question was
        # about IPC 302 murder, but the closest thing in the dataset is an
        # unrelated Constitution article that happens to mention "302" as
        # part of a list — we should say so honestly rather than run
        # citation/similarity checks against a reference that isn't actually
        # relevant. Forcing a verdict off a bad match produces misleading
        # "grounded" or "hallucinated" labels either way.
        if best_match["retrieval_score"] < RETRIEVAL_CONFIDENCE_THRESHOLD:
            return {
                "question": question,
                "generated_answer": generated_answer,
                "retrieved_reference": {
                    "question": best_match["question"],
                    "answer": best_match["answer"],
                    "act": best_match["act"],
                    "lang": best_match["lang"],
                },
                "retrieval_confidence": best_match["retrieval_score"],
                "verdict": "NO CONFIDENT REFERENCE FOUND (dataset may not cover this topic)",
                "grounding_score": None,
                "hallucination_score": None,
                "citation_check": None,
                "similarity_check": None,
                "entailment_check": None,
                "top_matches": [
                    {"question": m["question"], "answer": m["answer"], "score": m["retrieval_score"]}
                    for m in top_matches
                ],
            }

        citation_result = self.citation_grounding_critic(generated_answer, best_match)
        similarity_result = self.similarity_critic(generated_answer, best_match)
        entailment_result = self.entailment_critic(generated_answer, best_match)

        # Combine: start from similarity score, subtract citation penalty.
        # This means a fluent-but-fabricated answer (high similarity, bad
        # citations) still gets caught, which cosine similarity alone would miss.
        # NOTE: this is a GROUNDING score — 0 = ungrounded, 1 = fully
        # grounded. It was previously (mis)named "hallucination_score" even
        # though higher values always meant LESS hallucination.
        grounding_score = max(
            0.0, similarity_result["similarity_score"] - citation_result["penalty"]
        )

        # Contradiction overrides everything else, no matter how high
        # similarity or how clean the citations are — this is the fix for
        # the "Yes" vs "does not" case, where similarity=0.825 and citation
        # grounding was clean, but the answer directly negates the reference.
        is_contradiction = entailment_result["is_contradiction"]
        if is_contradiction:
            grounding_score = min(grounding_score, GROUNDING_SCORE_CONTRADICTION_CAP)

        # True hallucination framing (0 = clean, 1 = fully hallucinated),
        # derived from grounding_score so the two can never disagree.
        hallucination_score = round(1.0 - grounding_score, 3)

        if is_contradiction:
            final_verdict = "HALLUCINATION DETECTED (contradicts reference)"
        elif citation_result["verdict"] == "fabricated_citation":
            final_verdict = "HALLUCINATION DETECTED (fabricated citation)"
        elif similarity_result["verdict"] == "low_similarity":
            final_verdict = "HALLUCINATION DETECTED (low grounding to reference)"
        else:
            final_verdict = "LIKELY GROUNDED"

        return {
            "question": question,
            "generated_answer": generated_answer,
            "retrieved_reference": {
                "question": best_match["question"],
                "answer": best_match["answer"],
                "act": best_match["act"],
                "lang": best_match["lang"],
            },
            "citation_check": citation_result,
            "similarity_check": similarity_result,
            "entailment_check": entailment_result,
            "grounding_score": round(grounding_score, 3),      # higher is better (0-1)
            "hallucination_score": hallucination_score,        # lower is better (0-1)
            "verdict": final_verdict,
            "top_matches": [
                {"question": m["question"], "answer": m["answer"], "score": m["retrieval_score"]}
                for m in top_matches
            ],
        }

    # ── Refinement loop: re-prompt with grounding context, then re-check ──
    def refine_and_reevaluate(self, lang: str, question: str, initial_result: dict) -> dict:
        """
        If evaluate() flagged a hallucination, re-prompt the SAME fine-tuned
        model (not a swap to a different model) with the retrieved reference
        text injected as grounding context, then re-run the full evaluation
        on the new answer. Returns both results so you can report a direct
        before/after comparison.

        Lazy-imports model_loader so this module can be used standalone
        (e.g. in the smoke test) without loading the 3B model unless refinement
        is actually needed.
        """
        if initial_result.get("verdict") == "LIKELY GROUNDED":
            return {"refinement_needed": False, "initial_result": initial_result}

        if initial_result.get("verdict", "").startswith("NO CONFIDENT REFERENCE"):
            # Nothing reliable to ground a refinement in — refining against a
            # weak/irrelevant reference would just teach the model to repeat
            # wrong context. Don't refine; flag as out-of-scope instead.
            return {
                "refinement_needed": False,
                "initial_result": initial_result,
                "note": "No reliable reference to ground refinement in — dataset likely doesn't cover this topic.",
            }

        from model_loader import model_loader as loader  # lazy import

        reference = initial_result["retrieved_reference"]
        grounding_context = (
            f"{reference['question']}\n{reference['answer']}"
        )

        refined_answer = loader.generate(lang=lang, question=question, context=grounding_context)
        refined_result = self.evaluate(question, refined_answer)

        # grounding_score is higher-is-better, so "improved" is a plain
        # increase. (Equivalently: refined hallucination_score < initial.)
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


# ── Singleton, mirroring model_loader's pattern ──
detector = HallucinationDetector()


if __name__ == "__main__":
    # Test 1: the actual hallucinated output already seen from the hi
    # adapter's BNS control question — fabricated citation case.
    test_question = "IPC की धारा 302 के अनुरूप BNS की कौन सी धारा है?"
    test_answer = (
        "IPC की 302 धारा को BNS 1987 की धारा 7.3 (क) के अनुसार प्रतिष्ठित करती है। "
        "यहाँ 'BNS' या 'Bombay Nazareen Society's Journal' का उल्लेख है।"
    )

    result = detector.evaluate(test_question, test_answer)

    print("\n=== Hallucination Detector Test 1: fabricated citation ===")
    print("Question:", result["question"])
    print("Generated Answer:", result["generated_answer"])

    print("\n--- Retrieval scores for top 3 matches ---")
    for m in result["top_matches"]:
        print(f"  score={m['score']:.4f}  question={m['question'][:80]}")

    if result.get("citation_check") is not None:
        print("\nCitation Check:", result["citation_check"])
        print("\nSimilarity Check:", result["similarity_check"])
        print("\nEntailment Check:", result["entailment_check"])
        print("\nGrounding Score (higher = better):", result["grounding_score"])
        print("Hallucination Score (lower = better):", result["hallucination_score"])
    print("Verdict:", result["verdict"])
    print("\nTop Retrieved Reference:", result["retrieved_reference"])

    # Test 2: the negation/contradiction case from the UI screenshot —
    # this is what GROUNDING_SCORE_CONTRADICTION_CAP + entailment_critic are for.
    # Reference answer (from dataset): "In the proviso, 'State' does not
    # include a Union territory." Generated answer directly negates it.
    test_question_2 = "Does 'State' include a Union territory in the proviso?"
    test_answer_2 = "Yes, 'State' includes a Union territory in the proviso."

    result_2 = detector.evaluate(test_question_2, test_answer_2)

    print("\n=== Hallucination Detector Test 2: negation/contradiction ===")
    print("Question:", result_2["question"])
    print("Generated Answer:", result_2["generated_answer"])
    if result_2.get("similarity_check") is not None:
        print("\nSimilarity Check:", result_2["similarity_check"])
        print("Entailment Check:", result_2["entailment_check"])
        print("Grounding Score (higher = better):", result_2["grounding_score"])
        print("Hallucination Score (lower = better):", result_2["hallucination_score"])
    print("Verdict:", result_2["verdict"])
    print("Expected: HALLUCINATION DETECTED (contradicts reference), grounding_score <=",
          GROUNDING_SCORE_CONTRADICTION_CAP)