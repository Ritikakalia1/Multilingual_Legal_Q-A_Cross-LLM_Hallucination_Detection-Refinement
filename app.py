# app.py
#
# Flask backend running the full cross-LLM refinement pipeline, with THREE
# grounding sources, multi-turn conversation memory, AND intent-based
# routing to four branches:
#
#   (a) Static dataset (intent=legal_knowledge, no doc_id sent):
#       LLM1 drafts -> hallucination_detector retrieves from FAISS dataset
#       -> LLM2 critiques -> LLM1 regenerates -> re-evaluate. FALLS BACK to
#       the general knowledge base (see (a-fallback) below) if the curated
#       dataset has no confident reference for the question.
#
#   (a-fallback) Broader legal corpus (still intent=legal_knowledge, no
#       doc_id sent, curated dataset came up empty):
#       general_knowledge_base.py — a PERSISTENT ChromaDB collection built
#       once via build_chroma_index.py from legal_corpus/ PDFs
#       (commercial_court_act&rules, guideline,
#       Legal_Services_Authorities_Act_1987, Rules, schemes, women&law) +
#       the 10K IndicLegalQA JSON. Same three-step LLM1 -> critic -> LLM1
#       cycle, just grounded in this broader (noisier) source instead.
#       Entirely opt-in: if the Chroma store hasn't been built yet,
#       general_kb below is still constructed (cheap — it just opens/
#       creates an empty persistent collection) but retrieve() on an empty
#       collection returns no candidates, so cross_llm_refine() behaves
#       exactly as it did before this feature existed.
#
#   (b) User-uploaded document, targeted question (intent=document_qa,
#       doc_id present):
#       /upload_document chunks + embeds a pasted document into an
#       in-memory DocumentIndex (document_index.py), returning a doc_id.
#       Subsequent /ask calls with that doc_id run the SAME three-step
#       pipeline, but grounded in that document's chunks instead of the
#       static corpus.
#
#   (c) User-uploaded document, "explain/summarize this" (intent=explanation,
#       doc_id present):
#       Different generation shape than (b) — a reduce-over-chunks summary
#       rather than a retrieval-then-answer. See explain_document() in
#       cross_llm_refinement.py.
#
#   (d) Drafting (intent=drafting, doc_id irrelevant):
#       Template-retrieval + slot-filling via draft_generator.py — NOT
#       free-generation. No hallucination-detector pass needed here since
#       nothing in the output is LLM-invented text; it's a substitution
#       into a fixed template.
#
#   Routing itself happens via intent_classifier.py: a cheap keyword
#   heuristic handles the obvious cases, falling back to an LLM call
#   (critic_llm.classify_intent, reusing the already-loaded critic model)
#   only when ambiguous.
#
#   Multi-turn memory: /ask accepts an optional session_id (client-
#   generated, e.g. crypto.randomUUID() in the browser). When present,
#   follow-up questions ("what about the punishment for that?") are
#   rewritten into standalone questions using recent turns as context
#   before the pipeline runs — see conversation_memory.py. NOTE: this
#   still only applies to branches (a)/(b)/(c) — drafting has its own,
#   separate multi-turn mechanism (known_slots, see below), since
#   "resolving a follow-up" and "filling in a missing form field" are
#   different problems.
#
# Both the initial and refined results are returned so the frontend can
# show a direct before/after comparison, matching the calibration data in
# results_batch.csv.

from flask import Flask, request, jsonify, render_template
from cross_llm_refinement import cross_llm_refine, cross_llm_refine_document, explain_document
from hallucination_detector import detector
from document_index import DocumentIndexManager
from general_knowledge_base import init_general_kb
from conversation_memory import conversation_manager
from critic_llm import critic_llm
from intent_classifier import classify
import draft_generator
import csv
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

SUPPORTED_LANGS = ["en", "hi", "pa", "ne"]

# NEW:
doc_manager = DocumentIndexManager(
    embedder_en=detector.embedder_en,
    embedder_multi=detector.embedder_multi,
)
general_kb = init_general_kb(detector.embedder_en, detector.embedder_multi)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Chunks + embeds a pasted legal document (FIR, contract, judgment
    excerpt) into an in-memory FAISS index scoped to that document only.
    Returns a doc_id to pass to /ask for document-grounded Q&A."""
    payload = request.json or {}
    text = payload.get('text', '').strip()
    doc_name = payload.get('doc_name', 'uploaded_document').strip()
    lang = payload.get('lang', 'en').strip().lower()

    if not text:
        return jsonify({"error": "No document text provided."}), 400
    if lang not in SUPPORTED_LANGS:
        return jsonify({"error": f"Unsupported language '{lang}'. Choose from {SUPPORTED_LANGS}."}), 400
    if len(text.split()) < 20:
        return jsonify({"error": "Document too short to index meaningfully (need at least ~20 words)."}), 400

    try:
        doc_id = doc_manager.create(doc_name, lang, text)
    except Exception as e:
        return jsonify({"error": f"Failed to index document: {e}"}), 500

    doc_index = doc_manager.get(doc_id)
    return jsonify({
        "doc_id": doc_id,
        "doc_name": doc_name,
        "chunk_count": len(doc_index.chunks),
    })


@app.route('/ask', methods=['POST'])
def ask():
    payload = request.json or {}
    question = payload.get('question', '').strip()
    lang = payload.get('lang', 'en').strip().lower()
    doc_id = payload.get('doc_id')          # optional — present when grounding against an uploaded document
    session_id = payload.get('session_id')  # optional — present for multi-turn follow-up resolution
    known_slots = payload.get('known_slots', {})  # optional — accumulated drafting form fields from prior turns

    if not question:
        return jsonify({"error": "Please enter a question."}), 400
    if lang not in SUPPORTED_LANGS:
        return jsonify({"error": f"Unsupported language '{lang}'. Choose from {SUPPORTED_LANGS}."}), 400

    doc_index = None
    if doc_id:
        doc_index = doc_manager.get(doc_id)
        if doc_index is None:
            return jsonify({"error": "Document session expired or not found — please re-upload."}), 404

    # ── Classify intent before doing anything else ──
    # doc_present just means "is a doc_id attached" — classify() uses that
    # plus the question text to decide the branch (see intent_classifier.py
    # docstring for why "explain this" vs "who is the IO" route differently
    # even with the same doc_id attached).
    intent = classify(question, doc_present=bool(doc_index), classifier_fn=critic_llm.classify_intent)

    # ── Branch: drafting ──
    # No retrieval, no hallucination-detector pass — the output is a
    # template substitution, not LLM-generated legal content, so there's
    # nothing here for the detector to check.
    if intent == "drafting":
        result = draft_generator.draft(question, critic_llm.extract_slots, known_slots)
        result["intent"] = "drafting"
        result["question"] = question
        result["lang"] = lang
        return jsonify(result)

    # ── Branch: explanation (uploaded doc, "explain/summarize this") ──
    elif intent == "explanation" and doc_index is not None:
        result = explain_document(lang, question, doc_index)

    # ── Branch: document_qa (uploaded doc, targeted question) ──
    elif doc_index is not None:
        result = cross_llm_refine_document(lang, question, doc_index, session_id=session_id)

    # ── Branch: legal_knowledge (static dataset, default) ──
    # general_kb is passed through so cross_llm_refine() can fall back to
    # the broader legal_corpus if the curated FAISS dataset has no
    # confident reference for this question — see cross_llm_refinement.py.
    else:
        result = cross_llm_refine(lang, question, session_id=session_id, general_kb=general_kb)

    result["intent"] = intent
    result["question"] = question
    result["lang"] = lang
    result["doc_id"] = doc_id

    return jsonify(result)


@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Clears conversation memory for a session_id — called when the user
    hits 'New Conversation' in the UI. Does NOT affect any active uploaded
    document (doc_id lifecycle is independent of conversation memory)."""
    payload = request.json or {}
    session_id = payload.get('session_id')
    if session_id:
        conversation_manager.reset(session_id)
    return jsonify({"ok": True})


@app.route('/batch_results')
def batch_results():
    """Serves results_batch.csv (your calibration set) as JSON, so the
    frontend can render the summary table showing detector performance
    across the real test cases."""
    path = "results_batch.csv"
    if not os.path.exists(path):
        return jsonify({"error": "results_batch.csv not found — run run_batch_eval.py first."}), 404

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return jsonify(rows)


if __name__ == '__main__':
    app.run(debug=True, port=5000)