# cross_llm_refinement.py
#
# True cross-LLM refinement: LLM1 = your fine-tuned per-language adapter
# (model_loader), LLM2 = a separate, stronger general critic (critic_llm).
#
# Unlike the old flan-t5-small/base prototype, the critic here is grounded
# in the SAME retrieved reference the hallucination detector already
# validated — it critiques against real ground truth pulled from your
# FAISS index, not its own general/parametric knowledge of Indian law. So
# a hallucinating critic can't inject new fabricated law into the
# "refined" answer; at worst it just fails to catch something, which the
# re-run of detector.evaluate() on the refined answer will still surface.

from model_loader import model_loader as loader
from hallucination_detector import detector
from critic_llm import critic_llm


def cross_llm_refine(lang: str, question: str) -> dict:
    # ── Step 1: LLM1 (fine-tuned adapter) generates the initial draft ──
    initial_answer = loader.generate(lang=lang, question=question)
    initial_result = detector.evaluate(question, initial_answer)

    verdict = initial_result.get("verdict", "")

    # No usable reference at all — nothing reliable to critique or refine
    # against. Refining here would just be the critic inventing an opinion
    # with no ground truth behind it, which is exactly what we don't want.
    if verdict.startswith("NO CONFIDENT REFERENCE"):
        return {
            "refinement_ran": False,
            "reason": "No confident reference found — dataset likely doesn't cover this topic.",
            "initial_answer": initial_answer,
            "initial_result": initial_result,
        }

    # Already grounded — no refinement needed. Still returned in the same
    # shape so the frontend doesn't need special-casing.
    if verdict == "LIKELY GROUNDED":
        return {
            "refinement_ran": False,
            "reason": "Initial draft already grounded — no refinement needed.",
            "initial_answer": initial_answer,
            "initial_result": initial_result,
        }

    # ── Step 2: LLM2 (critic) compares draft vs retrieved reference ──
    reference = initial_result["retrieved_reference"]
    citation_check = initial_result.get("citation_check")
    critique = critic_llm.critique(question, initial_answer, reference, citation_check)

    # ── Step 3: LLM1 regenerates using reference + critique as context ──
    grounding_context = (
        f"Reference answer:\n{reference['answer']}\n\n"
        f"Reviewer feedback on the previous draft:\n{critique}"
    )
    refined_answer = loader.generate(lang=lang, question=question, context=grounding_context)
    refined_result = detector.evaluate(question, refined_answer)

    # ── "Improved" / delta comparison ──
    # NOTE: this must compare grounding_score (higher = better), not
    # hallucination_score (lower = better). Prior to the naming fix in
    # hallucination_detector.py, "hallucination_score" WAS the grounding
    # score, so `refined > initial` happened to be correct. Now that
    # hallucination_score genuinely means "higher = more hallucinated",
    # using it here would flip the verdict: a refinement that successfully
    # drops hallucination_score from 0.55 -> 0.05 would incorrectly report
    # improved=False. grounding_score is the field designed for this
    # comparison — it's mutually consistent with hallucination_score
    # (grounding_score == 1 - hallucination_score) but reads correctly here.
    initial_grounding = initial_result.get("grounding_score")
    refined_grounding = refined_result.get("grounding_score")
    initial_grounding = initial_grounding if initial_grounding is not None else 0.0
    refined_grounding = refined_grounding if refined_grounding is not None else 0.0

    return {
        "refinement_ran": True,
        "initial_answer": initial_answer,
        "initial_result": initial_result,
        "critique": critique,
        "refined_answer": refined_answer,
        "refined_result": refined_result,
        "improved": refined_grounding > initial_grounding,
        "score_delta": round(refined_grounding - initial_grounding, 3),  # grounding delta: positive = improved
    }