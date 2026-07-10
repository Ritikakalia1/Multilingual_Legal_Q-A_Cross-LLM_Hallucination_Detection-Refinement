# entailment_checker.py
#
# Fixes the negation blind spot: cosine similarity between
# "State includes a Union territory" and "State does not include a
# Union territory" is high (~0.82) because negation words barely move
# sentence embeddings. Similarity answers "same topic?" — it cannot
# answer "same claim?". This module adds that missing signal.
#
# Uses XLM-R fine-tuned on XNLI, which covers Hindi as a training
# language and generalizes reasonably to Punjabi/Nepali via shared
# script/lexical overlap with Hindi (Devanagari) and Indo-Aryan roots.
# If Punjabi/Nepali accuracy is too low in practice, the fallback is to
# transliterate/translate both strings to English before running NLI —
# hook is included below (see `translate_fn`).

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NLI_MODEL = "joeddav/xlm-roberta-large-xnli"

# XNLI label order for this checkpoint: 0=contradiction, 1=neutral, 2=entailment
LABELS = ["contradiction", "neutral", "entailment"]


class EntailmentChecker:
    def __init__(self, translate_fn=None):
        """
        translate_fn: optional callable(text, lang) -> english_text.
        Pass this in if you already have a translation utility for
        pa/ne in your pipeline (e.g. reusing whatever normalizes your
        FAISS query embeddings). Leave None to run NLI on raw text.
        """
        self.tokenizer = None
        self.model = None
        self.translate_fn = translate_fn

    def _load(self):
        if self.model is not None:
            return
        logger.info(f"Loading NLI model: {NLI_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")
        logger.info("✅ NLI model loaded")

    def check(self, reference_answer: str, draft_answer: str, lang: str = "en") -> dict:
        """
        Premise = retrieved reference (ground truth).
        Hypothesis = draft/refined answer being evaluated.
        Returns which way it leans and how confidently.
        """
        self._load()

        premise = reference_answer
        hypothesis = draft_answer
        if self.translate_fn and lang != "en":
            premise = self.translate_fn(premise, lang)
            hypothesis = self.translate_fn(hypothesis, lang)

        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()

        scored = {LABELS[i]: round(probs[i], 4) for i in range(3)}
        top_label = max(scored, key=scored.get)

        return {
            "label": top_label,                 # "contradiction" | "neutral" | "entailment"
            "scores": scored,
            "is_contradiction": top_label == "contradiction" and scored["contradiction"] > 0.5,
        }


entailment_checker = EntailmentChecker()