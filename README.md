# Indian_Multilingual_Legal_Q-A_Cross-LLM_Hallucination_Detection-Refinement

**Cross-LLM refinement: fine-tuned draft → grounded critique → regeneration**

A legal question-answering system for Indian law (IPC, CrPC, Constitution) across **English, Hindi, Punjabi, and Nepali**, built on per-language fine-tuned adapters over a shared `Qwen2.5-3B-Instruct` base. Fine-tuned models hallucinate — inventing section numbers, contradicting the law they were trained on. This project adds a **three-signal hallucination detector** (citation grounding, semantic similarity, NLI entailment) and a **cross-LLM refinement loop** that catches and corrects hallucinated answers before returning them.

---

## Problem

Fine-tuned LLMs answering legal questions can confidently state incorrect section numbers, contradict the law via negation flips ("does not include" → "includes"), or drift off-topic — all while sounding fluent. Standard semantic similarity checks miss this: a negated sentence shares ~95% of its words with the correct one, so cosine similarity alone scores it as "grounded" even when it states the opposite of the truth.

---

## Architecture

```
User Question
     │
     ▼
LaBSE Embedding → FAISS Search (Top-3) → Retrieved Reference
     │                                    (falls back to a broader
     │                                     ChromaDB-backed legal
     │                                     corpus — PDFs + 10K QA
     │                                     pairs — if the curated
     │                                     dataset has no confident
     │                                     match for the topic)
     ▼
Qwen2.5-3B + LoRA Adapter (per-language) → Initial Answer
     │
     ▼
Hallucination Detector
  ├─ Citation Grounding Critic   (regex-extracted section/article numbers
  │                                checked against retrieved reference +
  │                                full dataset-wide citation index)
  ├─ Semantic Similarity Critic  (LaBSE cosine similarity + BLEU/ROUGE-L)
  └─ Entailment Critic           (XLM-R/XNLI: catches negation/contradiction
                                   that similarity alone misses)
     │
     ├── LIKELY GROUNDED ──────────────────► Return answer
     │
     └── HALLUCINATION DETECTED
              │
              ▼
     Critic LLM (LLM2) reviews draft strictly against retrieved
     reference — cannot introduce new facts, only flag discrepancies
              │
              ▼
     LLM1 regenerates using reference + critique as context
              │
              ▼
     Re-run Hallucination Detector → Final Answer
```

Separately, requests are routed by **intent** (`legal_knowledge` / `document_qa` / `explanation` / `drafting`) before any of the above runs:

- `document_qa` / `explanation` ground against a user-uploaded document (FIR, contract, judgment excerpt), chunked + embedded on the fly.
- `drafting` is template + slot-filling (no free generation, so nothing for the detector to check — output is a substitution, not invented text).

---

## Key design decisions

- **Retrieval is by question, not by generated answer.** Embedding a hallucinated answer for retrieval would pull an unrelated reference, compounding the error. Retrieval stays independent of whether the draft is trustworthy.
- **Citation checking is regex + set lookup, not an LLM judgment call** — it can't itself hallucinate. Every citation is checked both against the specific retrieved reference *and* the full dataset-wide citation index, to catch cases where a real citation is grounded in the wrong reference.
- **The critic LLM (LLM2) is explicitly forbidden from introducing new legal facts** — it can only compare the draft against the retrieved reference text it's shown, so a hallucinating critic can't inject fabricated law into the "corrected" answer.
- **Similarity alone cannot detect negation.** "Includes X" vs. "does not include X" score ~0.82 cosine similarity. A dedicated NLI entailment critic checks logical polarity independent of topical similarity, and forces the grounding score down whenever a contradiction is detected — regardless of how clean the other signals look.
- **Grounding score vs. hallucination score are kept mutually consistent** (`hallucination_score = 1 - grounding_score`) and both are always returned, so downstream consumers can't accidentally read the wrong polarity.
- **The curated FAISS dataset stays primary; a broader ChromaDB corpus is a fallback, not a replacement.** The curated dataset has a hand-verified citation index and narrow, reliable coverage; a second knowledge base built from raw `legal_corpus/` PDFs + a 10K QA set is only consulted when the curated dataset reports no confident reference — keeping the system's most-trusted source first in line.
- **Multi-turn conversation memory is scoped by grounding source**, not just by session — a follow-up to a document question only ever pulls context from prior document-grounded turns, never from an unrelated general-knowledge or static-dataset question asked in between the same session.

---

## Pipeline stages

| Stage | Component | File |
|---|---|---|
| Retrieval | FAISS + LaBSE | `build_faiss_index.py`, `hallucination_detector.py` |
| Broader fallback retrieval | ChromaDB + LaBSE (PDFs + 10K QA corpus) | `general_knowledge_base.py`, `build_chroma_index.py` |
| Generation (LLM1) | Qwen2.5-3B + per-language LoRA adapters | `model_loader.py` |
| Citation grounding | Regex extraction + dataset-wide citation index | `citation_utils.py` |
| Semantic similarity | LaBSE cosine similarity, BLEU, ROUGE-L | `hallucination_detector.py` |
| Entailment / contradiction | XLM-R fine-tuned on XNLI | `entailment_checker.py` |
| Critique (LLM2) | Qwen2.5-3B, grounded strictly in retrieved reference | `critic_llm.py` |
| Intent routing | Keyword heuristic + LLM fallback classifier | `intent_classifier.py` |
| Multi-turn memory | Source-scoped follow-up question rewriting | `conversation_memory.py` |
| Document grounding | Per-upload in-memory FAISS + reranker | `document_index.py` |
| Drafting | Template retrieval + slot extraction (no free generation) | `draft_generator.py` |
| Refinement orchestration | Full detect → critique → refine loop | `cross_llm_refinement.py` |
| Serving | Flask REST API + dashboard | `app.py`, `templates/index.html` |
| Calibration | Batch eval against gold-labeled test cases | `run_batch_eval.py` |

---

## Model training

Each language has its own LoRA adapter, trained independently on Kaggle's free-tier GPU via 4-bit QLoRA:

- **Base model:** `Qwen/Qwen2.5-3B-Instruct`
- **Quantization:** 4-bit NF4 (`bitsandbytes`), bf16 compute dtype, double quantization
- **LoRA config:** rank 16, alpha 32, dropout 0.05, targeting all attention and MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`) — ~30M trainable params (~1% of the 3.1B total)
- **Format:** each Q&A pair is rendered through the Qwen chat template with a system prompt fixing the target language, so one base model architecture serves all four adapters

Kaggle session limits meant training couldn't run start-to-finish in one sitting — each language's training was checkpointed and resumed across multiple sessions (in Punjabi's case, across a separate Kaggle account after exhausting another's weekly GPU quota). Checkpoints were re-uploaded as Kaggle datasets and reloaded via `PeftModel.from_pretrained(..., is_trainable=True)` to continue training exactly where it left off.

### Training results

| Language | Train samples | Steps | Final train loss | Final val loss |
|---|---|---|---|---|
| English | 12,537 | 783 | 7.04 | 0.92 |
| Hindi | 11,995 | 750 | 0.65 | 0.66 |
| Nepali | 12,132 | 759 | 0.58 | 0.57 |
| Punjabi | 12,006 | 751 | 0.41 | 0.40 |

**English's train/val loss gap looks like a bug, not a language difference.** Hindi, Nepali, and Punjabi all show training loss and validation loss tracking within ~0.02–0.05 of each other at every logged step (e.g. Punjabi step 100: train 0.589 / val 0.573; step 600: train 0.400 / val 0.408). English's training loss instead sits 6–7 points above its own validation loss throughout the run (train 7.04 vs. val 0.92 at the final step) — training loss should never be that far above validation loss for the same model on the same objective. Since three of four languages show train/val staying tightly coupled and only English diverges, this points to something specific to the English training cell (e.g. a difference in how loss is logged/reduced, gradient-accumulation scaling, or label masking) rather than English legal text being genuinely ~10x harder to model. **This should be root-caused — and the English run likely re-trained — before citing these numbers side-by-side as comparable across languages.**

Additionally, `hindi_part2.ipynb` includes a standalone adapter-vs-base-model evaluation: 50 held-out Hindi validation questions run through both the fine-tuned adapter and the un-tuned base model, scored on cosine similarity (via `all-MiniLM-L6-v2`), BLEU, and ROUGE-L against the gold answer. This is a distinct, smaller-scale sanity check from the embedder used in production retrieval/detection (`sentence-transformers/LaBSE`) — the two serve different purposes (one-off adapter quality check vs. live retrieval/grounding) and shouldn't be conflated.

---

## Calibration & evaluation

`run_batch_eval.py` runs the detector against a hand-labeled test set spanning 5 gold classes (`grounded`, `fabricated_citation`, `off_topic`, `contradiction`, `no_reference`) across all 4 languages, including out-of-distribution control questions (BNS — a law not present in the training corpus) to verify the detector correctly abstains rather than fabricating a confident answer. Output includes a full confusion matrix and per-class precision/recall/F1, written to `results_batch.csv`.

---

## Stack

`Qwen2.5-3B-Instruct` · 4-bit QLoRA (PEFT + bitsandbytes) · `sentence-transformers/LaBSE` · FAISS · ChromaDB · `xlm-roberta-large-xnli` · HuggingFace Transformers · PyTorch · Flask · Chart.js

---

## Run it

```bash
# builds FAISS index + citation index from data/
python build_faiss_index.py

# builds the broader ChromaDB corpus from legal_corpus/ PDFs + 10K QA json
python build_chroma_index.py

# calibration pass, writes results_batch.csv
python run_batch_eval.py

# serves the dashboard at localhost:5000
python app.py
```

---

## Limitations

- Calibration set size is limited; current scores should be read as directional rather than statistically robust until the eval set is scaled up.
- English's training run shows an unexplained train/val loss divergence not present in the other three languages (see note above) — likely a logging or setup bug in that specific notebook cell, not a genuine difficulty difference. Root cause is still open; treat the English adapter's training as provisional until it's resolved.
- The initial answer (LLM1's first pass) is generated without retrieval context — retrieval is only injected during the refinement step, after a hallucination is first detected. This is retrieval-augmented *correction*, not retrieval-augmented generation in the standard sense.
- The broader ChromaDB corpus is unverified relative to the curated FAISS dataset (chunked PDFs, unverified 10K QA pairs) — it's used strictly as a fallback, and its confidence threshold is a starting estimate pending its own calibration pass.
- Citation-grounding checks are numeral-only: if a question doesn't explicitly name a section/article number, a retrieval miss on an unrelated topic (e.g. a wrong Act entirely) isn't caught by the citation check and may fall through to a refinement pass that can't succeed against the wrong reference.
