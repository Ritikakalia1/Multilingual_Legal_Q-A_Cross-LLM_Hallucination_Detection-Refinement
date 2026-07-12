# critic_llm.py
#
# LLM2 in the cross-LLM refinement pipeline: a separate, general-purpose
# instruct model that critiques the fine-tuned adapter's draft answer
# STRICTLY against the retrieved reference text already validated by
# hallucination_detector.retrieve_reference(). It is explicitly instructed
# NOT to introduce any legal fact, section, or article that isn't present
# in that reference — its job is to compare, not to add new law. This is
# what keeps a free-text critic model from becoming its own hallucination
# source (the failure mode your old flan-t5-small/base prototype didn't
# guard against at all).
#
# Also doubles as the small-task LLM for two other constrained jobs that
# don't warrant their own model load — same pattern as rewrite_followup:
#   - classify_intent(): routes a question to a pipeline branch (see
#     intent_classifier.py, which calls this only when its cheap keyword
#     heuristic can't decide on its own).
#   - extract_slots(): pulls stated form-field values out of a drafting
#     request (see draft_generator.py) without inventing missing ones.

import torch
import gc
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Add CRITIC_MODEL = "..." to config.py to override.
    from config import CRITIC_MODEL
except ImportError:
    # Pick something that fits your GPU and is stronger / more general than
    # your fine-tuned per-language adapters. It does NOT need per-language
    # fine-tuning — it only ever reasons over the draft + reference text
    # it's shown directly, in English-transliterated comparison mode.
    CRITIC_MODEL = "Qwen/Qwen2.5-3B-Instruct"

MAX_CRITIQUE_TOKENS = 220


class CriticLLM:
    """LLM2 — loaded lazily and separately from model_loader's base model,
    so the two can coexist on GPU (or you can point CRITIC_MODEL at
    something small if VRAM is tight)."""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def _load(self):
        if self.model is not None:
            return
        logger.info(f"Loading critic LLM: {CRITIC_MODEL}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(CRITIC_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            CRITIC_MODEL,
            quantization_config=bnb_config,
            device_map={"": 0},
        )
        self.model.eval()
        logger.info("✅ Critic LLM loaded")

    def _generate(self, messages, max_new_tokens):
        self._load()
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def critique(self, question: str, draft_answer: str, reference: dict, citation_check: dict) -> str:
        """
        Compares draft_answer against the retrieved reference ONLY.
        Passes along which citations the detector already flagged as
        unverified, so the critic doesn't have to (mis-)re-derive that
        itself — it just has to explain the implication.
        """
        unverified = (citation_check or {}).get("unverified_citations", [])
        unverified_note = (
            f"The automated citation checker already flagged these citations in the "
            f"draft as UNVERIFIED against the reference/dataset: {', '.join(unverified)}. "
            f"Treat these as confirmed errors — call them out."
            if unverified else
            "The automated citation checker found no citation issues in the draft."
        )

        system_prompt = (
            "You are a strict legal-answer reviewer. You will be shown a DRAFT answer "
            "and a REFERENCE answer that is known to be correct. Your ONLY job is to "
            "compare the draft against the reference and list concrete discrepancies: "
            "points present in the reference but missing from the draft, contradictions, "
            "or unsupported claims in the draft. "
            "Do NOT introduce any legal fact, section number, or article that is not "
            "present in the reference text below. If you are unsure whether something "
            "is correct, say it needs verification rather than asserting it as fact. "
            "Be concise and specific — a short bullet list is fine."
        )
        user_prompt = (
            f"Question: {question}\n\n"
            f"REFERENCE (ground truth, retrieved from verified dataset):\n{reference.get('answer', '')}\n\n"
            f"DRAFT ANSWER TO REVIEW:\n{draft_answer}\n\n"
            f"{unverified_note}\n\n"
            "List the specific problems with the draft compared to the reference:"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._generate(messages, MAX_CRITIQUE_TOKENS)

    
    def rewrite_followup(self, context_text: str, followup_question: str) -> str:
        """Rewrites a follow-up question ('what about the punishment for
        that?') into a standalone question, using recent Q/A turns as
        context. Deliberately a SMALL, constrained task — rewrite only,
        no new legal facts — so this can't become its own hallucination
        source. Reuses the same loaded critic model as critique()."""
        system_prompt = (
            "You rewrite a follow-up question into a standalone question, "
            "using the conversation so far ONLY to resolve pronouns and "
            "references (e.g. 'that', 'it', 'the same act'). Do not answer "
            "the question. Do not add any new legal fact. Output ONLY the "
            "rewritten standalone question, nothing else."
        )
        user_prompt = (
            f"Conversation so far:\n{context_text}\n\n"
            f"Follow-up question: {followup_question}\n\n"
            f"Standalone question:"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._generate(messages, max_new_tokens=60)

    def classify_intent(self, question: str, doc_present: bool) -> str:
        """Routes a question to one of four pipeline branches. Called by
        intent_classifier.classify() ONLY when its cheap keyword heuristic
        couldn't decide on its own — see that module's docstring. Deliberately
        constrained to output a single category token, nothing else, so a
        malformed/rambling response can't accidentally slip past the caller's
        VALID_INTENTS check silently (classify() falls back to
        'legal_knowledge' if this returns anything unexpected)."""
        system_prompt = (
            "Classify the user's legal question into EXACTLY ONE of these "
            "categories: legal_knowledge, document_qa, explanation, drafting.\n"
            "- legal_knowledge: a general legal question not about a specific uploaded document.\n"
            "- document_qa: a targeted factual question about a specific uploaded document.\n"
            "- explanation: a request to explain/summarize/simplify an uploaded document.\n"
            "- drafting: a request to draft/generate a legal document (agreement, notice, "
            "affidavit, application, etc).\n"
            "Output ONLY the category name, nothing else — no punctuation, no explanation."
        )
        doc_note = "A document IS currently uploaded." if doc_present else "No document is uploaded."
        user_prompt = f"{doc_note}\n\nQuestion: {question}\n\nCategory:"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._generate(messages, max_new_tokens=10)

    def extract_slots(self, question: str, required_slots: list) -> dict:
        """Pulls out whatever slot values (names, dates, amounts, addresses)
        the user has ALREADY stated in their message, for draft_generator.py.
        Deliberately instructed to omit anything not explicitly stated rather
        than guess — draft_generator.py treats a missing key as 'ask the user',
        never as 'fill with a placeholder'. Returns {} on any parse failure so
        a malformed LLM response degrades to 'ask the user for everything'
        rather than silently fabricating a value."""
        system_prompt = (
            "Extract any of the following fields the user has explicitly "
            f"stated in their message: {', '.join(required_slots)}.\n"
            "Output ONLY a single JSON object with the fields you found — "
            "omit any field you did not find. Do not guess, infer, or invent "
            "a value for a field that wasn't stated. Output nothing except "
            "the JSON object."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        raw = self._generate(messages, max_new_tokens=200)
        try:
            # Strip markdown code fences if the model wraps its JSON in them
            # despite the instruction not to.
            cleaned = raw.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}

    def unload(self):
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
   
# ── Singleton, mirroring model_loader's pattern ──
critic_llm = CriticLLM()