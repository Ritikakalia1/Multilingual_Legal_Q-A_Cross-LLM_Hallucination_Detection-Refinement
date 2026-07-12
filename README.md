# Multilingual Legal Q&A with Cross-LLM Hallucination Detection & Refinement

A legal question-answering system for Indian law (IPC, CrPC, Constitution) across **English, Hindi, Punjabi, and Nepali**, built on a fine-tuned Qwen2.5-3B-Instruct model. Fine-tuned models hallucinate — inventing section numbers, contradicting the law they were trained on. This project adds a **three-signal hallucination detector** (citation grounding, semantic similarity, NLI entailment) and a **cross-LLM refinement loop** that catches and corrects hallucinated answers before returning them.

## Problem

Fine-tuned LLMs answering legal questions can confidently state incorrect section numbers, contradict the law via negation flips ("does not include" → "includes"), or drift off-topic — all while sounding fluent. Standard semantic similarity checks miss this: a negated sentence shares ~95% of its words with the correct one, so cosine similarity alone scores it as "grounded" even when it states the opposite of the truth.

## Architecture

```
User Question
      │
      ▼
LaBSE Embedding → FAISS Search (Top-3) → Retrieved Reference
      │
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

## Key design decisions

- **Retrieval is by question, not by generated answer.** Embedding a hallucinated answer for retrieval would pull an unrelated reference, compounding the error. Retrieval stays independent of whether the draft is trustworthy.
- **Citation checking is regex + set lookup, not an LLM judgment call** — it can't itself hallucinate. Every citation is checked both against the specific retrieved reference AND the full dataset-wide citation index, to catch cases where a real citation is grounded in the wrong reference.
- **The critic LLM (LLM2) is explicitly forbidden from introducing new legal facts** — it can only compare the draft against the retrieved reference text it's shown, so a hallucinating critic can't inject fabricated law into the "corrected" answer.
- **Similarity alone cannot detect negation.** "Includes X" vs "does not include X" score ~0.82 cosine similarity. A dedicated NLI entailment critic checks logical polarity independent of topical similarity, and forces the grounding score down whenever a contradiction is detected — regardless of how clean the other signals look.
- **Grounding score vs. hallucination score are kept mutually consistent** (`hallucination_score = 1 - grounding_score`) and both are always returned, so downstream consumers can't accidentally read the wrong polarity.

## Pipeline stages

| Stage | Component | File |
|---|---|---|
| Retrieval | FAISS + LaBSE | `build_faiss_index.py`, `hallucination_detector.py` |
| Generation (LLM1) | Qwen2.5-3B + per-language LoRA adapters | `model_loader.py` |
| Citation grounding | Regex extraction + dataset-wide citation index | `citation_utils.py` |
| Semantic similarity | LaBSE cosine similarity, BLEU, ROUGE-L | `hallucination_detector.py` |
| Entailment / contradiction | XLM-R fine-tuned on XNLI | `entailment_checker.py` |
| Critique (LLM2) | Qwen2.5-3B, grounded strictly in retrieved reference | `critic_llm.py` |
| Refinement orchestration | Full detect → critique → refine loop | `cross_llm_refinement.py` |
| Serving | Flask REST API + dashboard | `app.py`, `templates/index.html` |
| Calibration | Batch eval against gold-labeled test cases | `run_batch_eval.py` |

## Calibration & evaluation

`run_batch_eval.py` runs the detector against a hand-labeled test set spanning 5 gold classes (`grounded`, `fabricated_citation`, `off_topic`, `contradiction`, `no_reference`) across all 4 languages, including out-of-distribution control questions (BNS — a law not present in the training corpus) to verify the detector correctly abstains rather than fabricating a confident answer. Output includes a full confusion matrix and per-class precision/recall/F1, written to `results_batch.csv`.

## Stack

`Qwen2.5-3B-Instruct` (4-bit QLoRA via PEFT + bitsandbytes) · `sentence-transformers/LaBSE` · FAISS · `xlm-roberta-large-xnli` · HuggingFace Transformers · PyTorch · Flask · Chart.js

## Run it

```bash
python build_faiss_index.py       # builds FAISS index + citation index from data/
python run_batch_eval.py          # calibration pass, writes results_batch.csv
python app.py                     # serves the dashboard at localhost:5000
```

## Limitations

- Citation grounding critic can only verify citations that are numeric and pattern-matchable (`Section X`, `धारा X`, `ਧਾਰਾ X`); prose-only legal claims without a section number aren't checked by that critic.
- NLI entailment model is Hindi-trained with cross-lingual transfer to Punjabi/Nepali via XLM-R's shared representation — not independently validated per-language.
- Hindi/Punjabi/Nepali portions of the dataset are translated rather than natively authored; translation quality has not yet been independently spot-checked.
- Calibration set size is limited; current scores should be read as directional rather than statistically robust until the eval set is scaled up.
- The initial answer (LLM1's first pass) is generated without retrieval context — retrieval is only injected during the refinement step, after a hallucination is first detected. This is retrieval-augmented *correction*, not retrieval-augmented *generation* in the standard sense.
