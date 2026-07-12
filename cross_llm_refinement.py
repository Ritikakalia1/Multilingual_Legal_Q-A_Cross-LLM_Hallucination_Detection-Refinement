# cross_llm_refinement.py
#
# True cross-LLM refinement: LLM1 = your fine-tuned per-language adapter
# (model_loader), LLM2 = a separate, stronger general critic (critic_llm).
#
# Four orchestration functions:
#   cross_llm_refine()          — grounds against your static FAISS dataset
#                                  (curated, citation-verified, narrow
#                                  coverage). FALLS BACK to
#                                  cross_llm_refine_general() when the
#                                  dataset reports no confident reference —
#                                  see the "GENERAL KB FALLBACK" section
#                                  below.
#   cross_llm_refine_general()  — NEW: grounds against GeneralKnowledgeBase
#                                  (general_knowledge_base.py, a persistent
#                                  ChromaDB collection built from your
#                                  legal_corpus/ PDFs + the 10K IndicLegalQA
#                                  JSON). Broader coverage, noisier source —
#                                  only ever reached as a fallback, never
#                                  the first thing tried.
#   cross_llm_refine_document() — grounds against a single user-uploaded
#                                  document (document_index.DocumentIndex),
#                                  retrieved BEFORE the first generation
#                                  since the adapter has no fine-tuning-time
#                                  knowledge of an arbitrary uploaded FIR/
#                                  contract/judgment.
#   explain_document()          — "explain/summarize this document", a
#                                  map-reduce summarization over ALL of the
#                                  document's chunks, not a targeted
#                                  retrieval-then-answer. See its own
#                                  docstring below for why this doesn't run
#                                  through the hallucination detector the
#                                  way the other three do.
#
# ── GENERAL KB FALLBACK ──
# cross_llm_refine() now accepts an optional `general_kb` argument
# (app.py wires this to general_knowledge_base.general_kb). When the
# curated FAISS dataset's initial verdict is "NO CONFIDENT REFERENCE FOUND"
# AND a general_kb was supplied, control passes to
# cross_llm_refine_general() instead of immediately giving up — the
# curated dataset simply not covering a topic (e.g. "anticipatory bail" if
# it's thin in ipc_qa.json) doesn't mean the broader legal_corpus doesn't
# cover it too.
#
# If general_kb is None (not wired up / not yet built), behavior is
# UNCHANGED from before this feature existed — cross_llm_refine() reports
# "no confident reference" exactly as it always did. This makes the
# feature fully opt-in: existing deployments without a Chroma corpus built
# yet keep working exactly as before.
#
# ── MULTI-TURN MEMORY (source-scoped) ──
# cross_llm_refine(), cross_llm_refine_general(), and
# cross_llm_refine_document() all accept an optional session_id. When
# present:
#   1. Prior turns for that session are pulled from conversation_memory,
#      SCOPED TO THIS FUNCTION'S OWN SOURCE ("static_dataset", "general_kb",
#      or "document"). A question asked against the uploaded document never
#      sees context from an unrelated general-KB or static-dataset question
#      asked earlier in the same session, and vice versa — this is what
#      fixes the topic-bleeding bug where a follow-up FIR question was
#      getting resolved using leftover context from an unrelated
#      "Para Legal Volunteers Scheme" question asked in between.
#   2. If the new question looks like a follow-up ("what about the
#      punishment for that?"), it's rewritten into a standalone question
#      BEFORE retrieval/generation — retrieval embeds literal text, so an
#      unresolved "that" would retrieve nothing useful.
#   3. The pipeline below is otherwise completely unchanged — it always
#      operates on `resolved_question`, never on the raw follow-up text.
#   4. The final answer is stored back into the session, tagged with this
#      function's source, for future same-source turns.
# Both the raw `question` (for display) and `resolved_question` (what was
# actually asked to the pipeline) are returned so the frontend can show
# "Interpreted as: ..." when they differ.
#
# NOTE on double-resolution: when cross_llm_refine() falls back into
# cross_llm_refine_general(), the question has ALREADY been resolved once
# (scoped to source="static_dataset"). cross_llm_refine_general() is called
# with the ALREADY-resolved question and session_id=None for its own
# internal resolution step, so it doesn't try to re-resolve an
# already-standalone question against the session a second time (which
# would be a harmless no-op via looks_like_followup, but skipping it is
# cheaper and clearer). The session turn is still recorded exactly once —
# under source="static_dataset", by cross_llm_refine() itself, since that's
# the entry point the user actually hit — by whichever function's result is
# ultimately returned.
#
# The critic (LLM2) is grounded in whichever reference the detector
# retrieved — real ground truth pulled from FAISS, the general KB, or the
# uploaded document, not the critic's own parametric knowledge. So a
# hallucinating critic can't inject new fabricated law into the "refined"
# answer; at worst it just fails to catch something, which the re-run of
# the detector on the refined answer will still surface.
#
# ── ABSTENTION ──
# hallucination_detector returns an "ABSTAIN (insufficient evidence to
# answer reliably)" verdict when grounding is too low to responsibly
# refine into a confident answer. All three retrieval-grounded functions
# below treat this the same way they treat "NO CONFIDENT REFERENCE FOUND"
# (aside from cross_llm_refine()'s general_kb fallback, which is specific
# to the latter): skip refinement, surface the reason to the frontend, and
# don't attempt to force a "refined" answer out of a draft that was
# already too ungrounded to trust.
#
# ── CITATION-AWARE GENERATION ──
# grounding_context passed into the refinement regeneration step asks LLM1
# to attribute any section/article it states directly to the retrieved
# reference/document excerpt (e.g. "According to the retrieved reference:
# ..."), rather than stating it as a bare, unattributed fact. This only
# shapes the PROMPT — it doesn't force LLM1's output format, and how well
# it's followed depends on model_loader.generate()'s own prompt template,
# which isn't part of this file. If instructions here aren't being
# respected in practice, the fix likely belongs in model_loader.py's
# system prompt instead.

from model_loader import model_loader as loader
from hallucination_detector import detector
from critic_llm import critic_llm
from conversation_memory import conversation_manager, resolve_followup_question

CITATION_AWARE_INSTRUCTION = (
    "When you state a specific section, article, or fact taken directly "
    "from the reference below, attribute it explicitly — e.g. 'According "
    "to the retrieved reference: ...' — rather than stating it as a bare "
    "fact with no attribution."
)


def cross_llm_refine(lang: str, question: str, session_id: str = None, general_kb=None) -> dict:
    """
    general_kb: optional GeneralKnowledgeBase instance (see
    general_knowledge_base.py). app.py wires this to
    general_knowledge_base.general_kb, constructed once at startup. Pass
    None (the default) to keep this function's behavior identical to
    before the general-KB fallback existed.
    """
    resolved_question = question
    session = None
    if session_id:
        session = conversation_manager.get_or_create(session_id)
        resolved_question = resolve_followup_question(
            question, session, critic_llm.rewrite_followup, source="static_dataset"
        )

    # ── Step 1: LLM1 (fine-tuned adapter) generates the initial draft ──
    initial_answer = loader.generate(lang=lang, question=resolved_question)
    initial_result = detector.evaluate(resolved_question, initial_answer, lang=lang)

    verdict = initial_result.get("verdict", "")

    if verdict.startswith("NO CONFIDENT REFERENCE"):
        if general_kb is not None:
            # Curated dataset doesn't cover this topic — try the broader
            # legal_corpus (PDFs + 10K QA json) before giving up entirely.
            # Note: passes the ALREADY-resolved question and session_id=None
            # — see "NOTE on double-resolution" in the module docstring.
            fallback_result = cross_llm_refine_general(
                lang, resolved_question, general_kb, session_id=None
            )
            fallback_result["resolved_question"] = resolved_question
            fallback_result["fell_back_from_dataset"] = True
            fallback_result["dataset_initial_result"] = initial_result
            if session is not None:
                final_answer = fallback_result.get(
                    "refined_answer", fallback_result.get("initial_answer")
                )
                session.add_turn(question, resolved_question, final_answer, source="static_dataset")
            return fallback_result

        result = {
            "refinement_ran": False,
            "reason": "No confident reference found — dataset likely doesn't cover this topic.",
            "initial_answer": initial_answer,
            "initial_result": initial_result,
        }
    elif verdict.startswith("ABSTAIN"):
        result = {
            "refinement_ran": False,
            "reason": "Grounding was too low to answer reliably — abstaining rather than asserting a possibly-wrong answer.",
            "initial_answer": initial_answer,
            "initial_result": initial_result,
        }
    elif "doesn't cover the cited section" in verdict:
        # Retrieval verification already confirmed the cited section isn't
        # anywhere in the reference pool — this is a wrong-source problem,
        # not a wording problem. LLM2 critiquing/LLM1 regenerating against
        # the SAME wrong reference can't fix that, so skip the cycle
        # rather than burning a refinement pass that's predictably a no-op.
        result = {
            "refinement_ran": False,
            "reason": "The question named a section the dataset doesn't appear to cover — refinement can't fix a retrieval miss, so skipping it. Treat the draft answer with extra caution.",
            "initial_answer": initial_answer,
            "initial_result": initial_result,
        }
    elif verdict == "LIKELY GROUNDED":
        result = {
            "refinement_ran": False,
            "reason": "Initial draft already grounded — no refinement needed.",
            "initial_answer": initial_answer,
            "initial_result": initial_result,
        }
    else:
        # ── Step 2: LLM2 (critic) compares draft vs retrieved reference ──
        reference = initial_result["retrieved_reference"]
        citation_check = initial_result.get("citation_check")
        critique = critic_llm.critique(resolved_question, initial_answer, reference, citation_check)

        # ── Step 3: LLM1 regenerates using reference + critique as context ──
        grounding_context = (
            f"Reference answer:\n{reference['answer']}\n\n"
            f"Reviewer feedback on the previous draft:\n{critique}"
        )
        refined_answer = loader.generate(
            lang=lang, question=resolved_question,
            context=grounding_context, instruction=CITATION_AWARE_INSTRUCTION,
        )
        refined_result = detector.evaluate(resolved_question, refined_answer, lang=lang)

        initial_grounding = initial_result.get("grounding_score")
        refined_grounding = refined_result.get("grounding_score")
        initial_grounding = initial_grounding if initial_grounding is not None else 0.0
        refined_grounding = refined_grounding if refined_grounding is not None else 0.0

        result = {
            "refinement_ran": True,
            "initial_answer": initial_answer,
            "initial_result": initial_result,
            "critique": critique,
            "refined_answer": refined_answer,
            "refined_result": refined_result,
            "improved": refined_grounding > initial_grounding,
            "score_delta": round(refined_grounding - initial_grounding, 3),
        }

    result["resolved_question"] = resolved_question
    result["source"] = "static_dataset"

    if session is not None:
        final_answer = result.get("refined_answer", result.get("initial_answer"))
        session.add_turn(question, resolved_question, final_answer, source="static_dataset")

    return result


def cross_llm_refine_general(lang: str, question: str, general_kb, session_id: str = None) -> dict:
    """
    Same three-step pipeline as cross_llm_refine(), but grounded in the
    broader GeneralKnowledgeBase (legal_corpus/ PDFs + 10K IndicLegalQA
    JSON, via ChromaDB — see general_knowledge_base.py) instead of the
    curated FAISS dataset.

    Reached in two ways:
      1. Directly, if you want to always query the general corpus (not
         wired up by default — app.py currently only reaches this via
         cross_llm_refine()'s fallback).
      2. As a FALLBACK from cross_llm_refine() when the curated dataset
         has no confident reference for the question — the normal path.

    Mirrors cross_llm_refine_document() structurally (retrieval happens
    before the first generation isn't needed here the way it is for
    documents, since — like the static dataset — LLM1's fine-tuning may
    already cover some of this corpus's topics; but the metric thresholds
    and citation universe both come from general_kb, not the dataset).
    """
    resolved_question = question
    session = None
    if session_id:
        session = conversation_manager.get_or_create(session_id)
        resolved_question = resolve_followup_question(
            question, session, critic_llm.rewrite_followup, source="general_kb"
        )

    # ── Step 1: LLM1 generates the initial draft ──
    initial_answer = loader.generate(lang=lang, question=resolved_question)
    initial_result = detector.evaluate_against_general_kb(
        resolved_question, initial_answer, general_kb, lang=lang
    )

    verdict = initial_result.get("verdict", "")

    if verdict.startswith("NO CONFIDENT REFERENCE"):
        result = {
            "refinement_ran": False,
            "reason": "No confident reference found in the broader legal corpus either — likely outside coverage entirely.",
            "initial_answer": initial_answer,
            "initial_result": initial_result,
        }
    elif verdict.startswith("ABSTAIN"):
        result = {
            "refinement_ran": False,
            "reason": "Grounding in the general legal corpus was too low to answer reliably — abstaining rather than asserting a possibly-wrong answer.",
            "initial_answer": initial_answer,
            "initial_result": initial_result,
        }
    elif "doesn't cover the cited section" in verdict:
        result = {
            "refinement_ran": False,
            "reason": "The question named a section the general legal corpus doesn't appear to cover — refinement can't fix a retrieval miss, so skipping it. Treat the draft answer with extra caution.",
            "initial_answer": initial_answer,
            "initial_result": initial_result,
        }
    elif verdict == "LIKELY GROUNDED":
        result = {
            "refinement_ran": False,
            "reason": "Initial draft already grounded in the general legal corpus — no refinement needed.",
            "initial_answer": initial_answer,
            "initial_result": initial_result,
        }
    else:
        # ── Step 2: LLM2 critiques draft vs the retrieved general-KB reference ──
        reference = initial_result["retrieved_reference"]
        citation_check = initial_result.get("citation_check")
        critique = critic_llm.critique(resolved_question, initial_answer, reference, citation_check)

        # ── Step 3: LLM1 regenerates using the reference + critique ──
        grounding_context = (
            f"Reference (from broader legal corpus):\n{reference['answer']}\n\n"
            f"Reviewer feedback on the previous draft:\n{critique}"
        )
        refined_answer = loader.generate(
            lang=lang, question=resolved_question,
            context=grounding_context, instruction=CITATION_AWARE_INSTRUCTION,
        )
        refined_result = detector.evaluate_against_general_kb(
            resolved_question, refined_answer, general_kb, lang=lang
        )

        initial_grounding = initial_result.get("grounding_score")
        refined_grounding = refined_result.get("grounding_score")
        initial_grounding = initial_grounding if initial_grounding is not None else 0.0
        refined_grounding = refined_grounding if refined_grounding is not None else 0.0

        result = {
            "refinement_ran": True,
            "initial_answer": initial_answer,
            "initial_result": initial_result,
            "critique": critique,
            "refined_answer": refined_answer,
            "refined_result": refined_result,
            "improved": refined_grounding > initial_grounding,
            "score_delta": round(refined_grounding - initial_grounding, 3),
        }

    result["resolved_question"] = resolved_question
    result["source"] = "general_kb"

    if session is not None:
        final_answer = result.get("refined_answer", result.get("initial_answer"))
        session.add_turn(question, resolved_question, final_answer, source="general_kb")

    return result


def cross_llm_refine_document(lang: str, question: str, doc_index, session_id: str = None, top_k: int = 3) -> dict:
    """Same three-step pipeline as cross_llm_refine(), but grounded in a
    single uploaded document instead of the static dataset.

    Key difference: retrieval happens BEFORE the first generation (not just
    at refinement time), because the fine-tuned adapter has zero knowledge
    of an arbitrary uploaded document's contents — it needs the relevant
    chunks injected as context just to draft a sensible first answer.

    Follow-up resolution works identically to cross_llm_refine() — see the
    module docstring above — except scoped to source="document", so a
    document follow-up is only ever resolved against prior DOCUMENT turns
    in this session, never against an unrelated static-dataset or
    general-KB question asked in between.
    """
    resolved_question = question
    session = None
    if session_id:
        session = conversation_manager.get_or_create(session_id)
        resolved_question = resolve_followup_question(
            question, session, critic_llm.rewrite_followup, source="document"
        )

    top_chunks = doc_index.retrieve(resolved_question, top_k=top_k)
    context = "\n\n".join(c["answer"] for c in top_chunks) if top_chunks else ""

    # ── Step 1: LLM1 drafts, grounded in the retrieved document chunks ──
    initial_answer = loader.generate(
        lang=lang, question=resolved_question,
        context=context, instruction=CITATION_AWARE_INSTRUCTION,
    )
    initial_result = detector.evaluate_against_document(resolved_question, initial_answer, doc_index)

    verdict = initial_result.get("verdict", "")

    if verdict.startswith("NO CONFIDENT REFERENCE"):
        result = {
            "refinement_ran": False,
            "reason": "No relevant passage found in the uploaded document for this question.",
            "initial_answer": initial_answer,
            "initial_result": initial_result,
        }
    elif verdict.startswith("ABSTAIN"):
        result = {
            "refinement_ran": False,
            "reason": "Grounding in the uploaded document was too low to answer reliably — abstaining rather than asserting a possibly-wrong answer.",
            "initial_answer": initial_answer,
            "initial_result": initial_result,
        }
    elif "doesn't cover the cited section" in verdict:
        # Mirrors the equivalent skip in cross_llm_refine() (static dataset
        # path). If the question named a specific section/article and
        # retrieval_verifier confirmed none of the retrieved document
        # chunks actually contain it, the retrieved chunk is the wrong
        # passage — critiquing/regenerating against that SAME wrong chunk
        # can't fix a retrieval miss, it can only restate it. Without this
        # branch, the refinement cycle runs anyway, LLM1 regenerates an
        # answer nearly identical to the first draft (nothing new to
        # ground it in), and the grounding score doesn't move — which is
        # exactly what was observed in testing: a "refined" answer
        # identical to the draft with a 0.000 score delta, after spending
        # a full LLM1 + LLM2 + LLM1 cycle for no benefit.
        result = {
            "refinement_ran": False,
            "reason": "The question named a section the uploaded document doesn't appear to cover — refinement can't fix a retrieval miss, so skipping it. Treat the draft answer with extra caution.",
            "initial_answer": initial_answer,
            "initial_result": initial_result,
        }
    elif verdict == "LIKELY GROUNDED":
        result = {
            "refinement_ran": False,
            "reason": "Initial draft already grounded in the document — no refinement needed.",
            "initial_answer": initial_answer,
            "initial_result": initial_result,
        }
    else:
        # ── Step 2: LLM2 critiques draft vs the retrieved document chunk ──
        reference = initial_result["retrieved_reference"]
        citation_check = initial_result.get("citation_check")
        critique = critic_llm.critique(resolved_question, initial_answer, reference, citation_check)

        # ── Step 3: LLM1 regenerates using the document excerpt + critique ──
        grounding_context = (
            f"Document excerpt:\n{reference['answer']}\n\n"
            f"Reviewer feedback on the previous draft:\n{critique}"
        )
        refined_answer = loader.generate(
            lang=lang, question=resolved_question,
            context=grounding_context, instruction=CITATION_AWARE_INSTRUCTION,
        )
        refined_result = detector.evaluate_against_document(resolved_question, refined_answer, doc_index)

        initial_grounding = initial_result.get("grounding_score")
        refined_grounding = refined_result.get("grounding_score")
        initial_grounding = initial_grounding if initial_grounding is not None else 0.0
        refined_grounding = refined_grounding if refined_grounding is not None else 0.0

        result = {
            "refinement_ran": True,
            "initial_answer": initial_answer,
            "initial_result": initial_result,
            "critique": critique,
            "refined_answer": refined_answer,
            "refined_result": refined_result,
            "improved": refined_grounding > initial_grounding,
            "score_delta": round(refined_grounding - initial_grounding, 3),
        }

    result["resolved_question"] = resolved_question
    result["source"] = "document"

    if session is not None:
        final_answer = result.get("refined_answer", result.get("initial_answer"))
        session.add_turn(question, resolved_question, final_answer, source="document")

    return result


def explain_document(lang: str, question: str, doc_index) -> dict:
    """
    Handles intent=explanation: "explain this in simple terms" / "summarize
    this document" — structurally different from cross_llm_refine_document(),
    which retrieves the TOP-K chunks relevant to a SPECIFIC question and
    answers directly from them.

    This function instead walks ALL of the document's chunks (map step)
    and asks LLM1 to produce one coherent plain-language explanation
    across the whole document (reduce step) — a summarization task, not a
    retrieval-then-answer task.

    NOT run through hallucination_detector / the critic cycle: the
    detector's grounding score and the critic's critique() are both built
    around "does this answer match ONE retrieved reference passage",
    which doesn't fit "does this summary faithfully compress the WHOLE
    document." That's a different consistency question (entailment
    against a synthesized reference, or a dedicated summary-faithfulness
    check) that isn't built yet — flagged in the returned dict via
    "verified": False so the frontend can visually distinguish this
    branch's output (no verdict banner, no grounding score) rather than
    implying it received the same scrutiny as the other branches.
    """
    from config import MAX_INPUT_LENGTH

    chunks = doc_index.chunks
    if not chunks:
        return {
            "refinement_ran": False,
            "verified": False,
            "reason": "Document has no indexed content to explain.",
            "answer": "",
        }

    full_text = "\n\n".join(chunks)
    # Rough word-based proxy for whether the whole document fits in one
    # generation call — good enough here since we only need a fits/doesn't
    # binary, not an exact token count. 1.3x multiplier gives headroom for
    # non-English scripts, which often tokenize less efficiently than
    # whitespace-delimited word counts would suggest.
    approx_tokens = len(full_text.split()) * 1.3

    if approx_tokens <= MAX_INPUT_LENGTH:
        # Fits in one shot — single explain pass over the whole document.
        summary = loader.generate(
            lang=lang, question=question,
            context=full_text,
            instruction=(
                "Explain the document above in simple, plain language a "
                "non-lawyer could understand. Do not invent any fact, "
                "name, date, or number that isn't present in the "
                "document text."
            ),
        )
    else:
        # Reduce: summarize each chunk first, then combine the summaries.
        # Each chunk-level call is independently constrained the same way
        # (no invented facts) so a long document can't smuggle a
        # fabrication through the map step unnoticed.
        chunk_summaries = []
        for chunk in chunks:
            chunk_summary = loader.generate(
                lang=lang, question="Summarize this excerpt in plain language.",
                context=chunk,
                instruction="Do not invent any fact not present in this excerpt.",
            )
            chunk_summaries.append(chunk_summary)

        combined = "\n\n".join(chunk_summaries)
        summary = loader.generate(
            lang=lang, question=question,
            context=combined,
            instruction=(
                "These are section-by-section summaries of one document. "
                "Combine them into a single coherent plain-language "
                "explanation, removing redundancy. Do not invent any fact "
                "not present in the summaries above."
            ),
        )

    return {
        "refinement_ran": False,
        "verified": False,
        "reason": (
            "Explanation/summary — not scored by the hallucination detector "
            "(a whole-document summary has no single reference passage to "
            "grade against)."
        ),
        "answer": summary,
        "chunk_count": len(chunks),
        "resolved_question": question,
    }