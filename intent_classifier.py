# intent_classifier.py
#
# Routes an incoming question to one of four pipeline branches, BEFORE any
# retrieval or generation happens:
#
#   legal_knowledge  -> cross_llm_refine()            (static FAISS dataset)
#   document_qa      -> cross_llm_refine_document()   (uploaded doc, targeted question)
#   explanation      -> explain_document()            (uploaded doc, "explain/summarize this")
#   drafting         -> draft_generator.draft()       (template-based generation, no retrieval)
#
# Two-stage, same pattern as conversation_memory.looks_like_followup ->
# critic_llm.rewrite_followup: a cheap deterministic gate handles the
# obvious cases so the LLM classifier is only invoked when genuinely
# ambiguous (keeps latency/cost down, and keeps the failure mode "falls
# back to legal_knowledge" rather than "silently misroutes").
#
# IMPORTANT: this module does NOT decide whether a doc_id is present —
# app.py already knows that from the request. What it decides is INTENT
# given that context: e.g. "who is the investigating officer" with a doc_id
# present is document_qa; "explain this judgment in simple terms" with the
# same doc_id present is explanation, not document_qa, because the desired
# output shape is a summary, not a chunk-grounded direct-quote answer.

import re

VALID_INTENTS = {"legal_knowledge", "document_qa", "explanation", "drafting"}

# ── Heuristic keyword cues ──
# Deliberately conservative: only fire on strong, unambiguous signals.
# Anything that doesn't match a cue falls through to the LLM classifier
# (or, if no doc_id, defaults straight to legal_knowledge — see classify()).

DRAFTING_CUES = re.compile(
    r"\b(draft|write|generate|prepare|create)\b.{0,30}\b(agreement|affidavit|notice|"
    r"complaint|application|contract|lease|will|rti|fir draft)\b",
    re.IGNORECASE,
)

EXPLANATION_CUES = re.compile(
    r"\b(explain|summari[sz]e|simplify|break down|what does this (?:mean|say)|"
    r"in simple (?:terms|english|words))\b",
    re.IGNORECASE,
)

# Question words/patterns that strongly suggest a targeted factual lookup
# against a specific uploaded document (as opposed to a general explanation).
DOCUMENT_QA_CUES = re.compile(
    r"\b(who is|what is the|when|where|which section|how much|what date)\b",
    re.IGNORECASE,
)


def _heuristic_classify(question: str, doc_present: bool):
    """Returns a confident intent string, or None if ambiguous (caller
    should fall back to the LLM classifier)."""
    if DRAFTING_CUES.search(question):
        return "drafting"

    if doc_present:
        if EXPLANATION_CUES.search(question):
            return "explanation"
        if DOCUMENT_QA_CUES.search(question):
            return "document_qa"
        return None  # ambiguous — could be either, ask the LLM

    # No document uploaded: explanation/document_qa aren't reachable targets
    # (nothing to explain/query), so anything that isn't clearly drafting
    # is legal_knowledge. No LLM call needed.
    return "legal_knowledge"


def classify(question: str, doc_present: bool, classifier_fn=None) -> str:
    """
    question: raw or resolved question text (resolved is safer if
        conversation memory already ran — see app.py wiring).
    doc_present: whether a doc_id is attached to this request.
    classifier_fn: callable(question, doc_present) -> str, supplied by the
        caller. app.py wires this to critic_llm.classify_intent so we reuse
        the already-loaded critic model rather than loading a new one —
        same pattern as conversation_memory.resolve_followup_question.

    Falls back to "legal_knowledge" (the safest, most general branch) if
    the LLM call fails or returns something outside VALID_INTENTS — a
    misroute into legal_knowledge just means a possibly-irrelevant RAG
    lookup, whereas misrouting INTO drafting or explanation could mean
    generating a legal document off a question that was never asking for one.
    """
    intent = _heuristic_classify(question, doc_present)
    if intent is not None:
        return intent

    if classifier_fn is None:
        return "document_qa" if doc_present else "legal_knowledge"

    try:
        intent = classifier_fn(question, doc_present).strip().lower()
        return intent if intent in VALID_INTENTS else "legal_knowledge"
    except Exception:
        return "legal_knowledge"