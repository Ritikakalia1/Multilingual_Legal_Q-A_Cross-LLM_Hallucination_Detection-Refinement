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

import torch
import gc
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

    def unload(self):
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()


# ── Singleton, mirroring model_loader's pattern ──
critic_llm = CriticLLM()