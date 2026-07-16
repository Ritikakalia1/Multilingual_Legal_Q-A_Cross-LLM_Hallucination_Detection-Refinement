# Multilingual Legal Q&A — Cross-LLM Hallucination Detection & Refinement

**Cross-LLM refinement: fine-tuned draft → grounded critique → regeneration**

A legal question-answering system for Indian law (IPC, CrPC, Constitution) across **English, Hindi, Punjabi, and Nepali**, built on per-language fine-tuned LoRA adapters over a shared `Qwen2.5-3B-Instruct` base. Fine-tuned models hallucinate — inventing section numbers, contradicting the law they were trained on, or drifting off-topic — all while sounding fluent. This project adds a **three-signal hallucination detector** (citation grounding, semantic similarity, NLI entailment), a **retrieval verifier + cross-encoder reranker**, and a **cross-LLM refinement loop** that catches and corrects hallucinated answers before returning them — across four distinct request types (legal Q&A, document Q&A, document explanation, and document drafting), with source-scoped multi-turn memory tying it all together.

---

## Problem

Fine-tuned LLMs answering legal questions can confidently state incorrect section numbers, contradict the law via negation flips ("does not include" → "includes"), or drift off-topic — all while sounding fluent. Standard semantic similarity checks miss this: a negated sentence shares ~95% of its words with the correct one, so cosine similarity alone scores it as "grounded" even when it states the opposite of the truth. Retrieval itself can also silently fail — a question about Section 303 can retrieve a passage about Section 304B if nothing checks that the numbers actually match.

---

## Architecture

Requests are routed by **intent** before any generation happens, via `intent_classifier.py` (a cheap keyword heuristic, falling back to an LLM call only when ambiguous):

| Intent | Trigger | Handler | Grounding source |
|---|---|---|---|
| `legal_knowledge` | default — general legal question, no document uploaded | `cross_llm_refine()` | curated FAISS dataset → broader ChromaDB corpus (fallback) |
| `document_qa` | targeted factual question about an uploaded document | `cross_llm_refine_document()` | uploaded document chunks |
| `explanation` | "explain / summarize / simplify this document" | `explain_document()` | whole document (map-reduce), **not** scored by the detector |
| `drafting` | "draft a rental agreement / legal notice / RTI application" | `draft_generator.draft()` | fixed template + slot-filling, **no free generation** |

### `legal_knowledge` pipeline (the core hallucination-detection loop)

```
User Question
     │
     ▼
Language-routed embedding (BGE for English / LaBSE for hi·pa·ne)
     │
     ▼
FAISS bi-encoder search (wide pool) → Cross-encoder rerank (reranker.py)
     │
     ▼
Retrieval Verifier — if the question names a Section/Article number,
promote whichever candidate actually cites that number to the top
(retrieval_verifier.py)
     │
     ▼
Retrieved Reference
     │      (falls back to a broader ChromaDB-backed legal corpus —
     │       PDFs + a 10K QA dataset — if the curated FAISS dataset
     │       has no confident match for the topic)
     ▼
Qwen2.5-3B + LoRA Adapter (per-language) → Initial Answer
     │
     ▼
Hallucination Detector
  ├─ Citation Grounding Critic   (regex-extracted section/article numbers
  │                                checked against the retrieved reference
  │                                AND the full dataset-wide citation index)
  ├─ Semantic Similarity Critic  (language-routed cosine similarity +
  │                                BLEU/ROUGE-L)
  └─ Entailment Critic           (XLM-R/XNLI: catches negation/contradiction
                                   that similarity alone misses)
     │
     ├── LIKELY GROUNDED ──────────────────────────► Return answer
     ├── NO CONFIDENT REFERENCE / ABSTAIN ─────────► Return with a caveat
     │                                                (or fall back to the
     │                                                 general KB)
     └── HALLUCINATION DETECTED
              │
              ▼
     Critic LLM (LLM2) reviews the draft strictly against the retrieved
     reference — cannot introduce new facts, only flag discrepancies
              │
              ▼
     LLM1 regenerates using reference + critique as context, with an
     explicit instruction to attribute cited facts to the reference
              │
              ▼
     Re-run Hallucination Detector → Final Answer (+ before/after scores)
```

### Other three branches

- **`document_qa`** — a pasted document (FIR, contract, judgment excerpt) is chunked and embedded into an in-memory, per-upload FAISS index at `/upload_document` time. Retrieval happens **before** the first generation (not just at refinement time), since the fine-tuned adapter has zero built-in knowledge of an arbitrary uploaded document. Same three-step detect → critique → regenerate cycle as above, grounded in the document's chunks instead of the dataset.
- **`explanation`** — a map-reduce summarization over *all* of the document's chunks (or the whole document in one shot if it's short enough), not a targeted retrieval-then-answer. Explicitly **not** run through the hallucination detector or the critic — "does this answer match one retrieved passage" doesn't fit "does this summary faithfully compress the whole document." Flagged with `"verified": False` in the response so the frontend doesn't imply the same scrutiny as the other branches.
- **`drafting`** — template retrieval (keyword match over a small fixed library: rental agreement, legal notice, RTI application) + slot extraction via a constrained LLM call. Missing required fields are returned as `missing_fields` for the frontend to prompt for, rather than the model guessing/inventing a value. Only once every slot is filled does the template get substituted — a substitution, not a generation, so nothing in the final document is LLM-invented text.

---

## Key design decisions

- **Dual embedder, language-routed.** English text is embedded with `BAAI/bge-base-en-v1.5` (better retrieval quality on English-only text than a multilingual model); Hindi/Punjabi/Nepali text stays on `sentence-transformers/LaBSE` (the one model here that actually understands those languages). Because the two live in different, incompatible vector spaces, there are **two separate FAISS indices**, **two separate Chroma collections**, and every embed/compare call routes through whichever model matches the entry's `lang` field — mixing vectors from the two models would be meaningless. BGE also needs a query-side instruction prefix (`BGE_QUERY_INSTRUCTION`) that passages never get, per BGE's own training convention.
- **Retrieval is by question, not by generated answer.** Embedding a hallucinated answer for retrieval would pull an unrelated reference, compounding the error. Retrieval stays independent of whether the draft is trustworthy.
- **Two-stage retrieval: bi-encoder recall, cross-encoder precision.** FAISS (or Chroma) pulls a wide candidate pool cheaply; a multilingual cross-encoder (`reranker.py`) then re-scores just that shortlist by reading the query and each candidate together — a finer-grained signal than two independently-embedded vectors being compared by cosine similarity alone.
- **Retrieval verification catches wrong-section retrieval before it reaches the critics.** If a question explicitly names a section/article number, `retrieval_verifier.py` checks whether *any* retrieved candidate actually cites that number and promotes it to the top if so — a correct-but-lower-similarity candidate can beat a higher-similarity but wrong-section one. If nothing matches, the grounding score is capped and the pipeline reports a wrong-source problem rather than trying to refine against the wrong reference.
- **Citation checking is regex + set lookup, not an LLM judgment call** — it can't itself hallucinate. Every citation is checked both against the specific retrieved reference *and* the full dataset-wide citation index, to catch cases where a real citation is grounded in the wrong reference.
- **The critic LLM (LLM2) is explicitly forbidden from introducing new legal facts** — it can only compare the draft against the retrieved reference text it's shown, so a hallucinating critic can't inject fabricated law into the "corrected" answer. It's also reused (same loaded model, different prompts) for three other small, tightly-constrained jobs: rewriting follow-up questions, classifying intent, and extracting drafting slot values — all deliberately "no new facts, output-format-only" tasks so none of them becomes its own hallucination source.
- **Similarity alone cannot detect negation.** "Includes X" vs. "does not include X" score ~0.82 cosine similarity. A dedicated NLI entailment critic (XLM-R fine-tuned on XNLI) checks logical polarity independent of topical similarity, and forces the grounding score down whenever a contradiction is detected — regardless of how clean the other signals look.
- **Grounding score vs. hallucination score are kept mutually consistent** (`hallucination_score = 1 - grounding_score`) and both are always returned, so downstream consumers can't accidentally read the wrong polarity.
- **The curated FAISS dataset stays primary; a broader ChromaDB corpus is a fallback, not a replacement.** The curated dataset has a hand-verified citation index and narrow, reliable coverage; a second knowledge base built from raw `legal_corpus/` PDFs + a 10K QA set is only consulted when the curated dataset reports no confident reference — keeping the system's most-trusted source first in line. This fallback is fully opt-in: if the Chroma store hasn't been built, the general KB simply returns no candidates and the system behaves exactly as if the feature didn't exist.
- **Multi-turn conversation memory is scoped by grounding source, not just by session.** A follow-up ("what about the punishment for that?") is rewritten into a standalone question using recent turns as context — but only turns from the *same* source (`document`, `static_dataset`, or `general_kb`). A follow-up to a document question never pulls context from an unrelated general-knowledge question asked in between in the same session. A cheap regex gate (`looks_like_followup`) also skips the rewrite LLM call entirely for questions that clearly don't need it.
- **Drafting never free-generates.** Legal documents (agreements, notices, applications) are template substitutions with LLM-assisted slot extraction, not open-ended generation — the single request type in this system structurally immune to hallucination, by construction rather than by detection.

---

## Pipeline stages

| Stage | Component | File |
|---|---|---|
| Intent routing | Keyword heuristic + LLM fallback classifier | `intent_classifier.py` |
| Retrieval (dataset) | Dual FAISS indices (BGE for en, LaBSE for hi/pa/ne) | `build_faiss_index.py`, `hallucination_detector.py` |
| Retrieval (broader fallback) | Dual ChromaDB collections (PDFs + 10K QA corpus) | `general_knowledge_base.py`, `build_chroma_index.py` |
| Retrieval (uploaded document) | Per-upload in-memory FAISS, language-routed embedder | `document_index.py` |
| Reranking | Multilingual cross-encoder over the bi-encoder shortlist | `reranker.py` |
| Retrieval verification | Exact section/article number match promotes the right candidate | `retrieval_verifier.py` |
| Generation (LLM1) | Qwen2.5-3B + per-language LoRA adapters | `model_loader.py` |
| Citation grounding | Regex extraction + dataset-wide citation index | `citation_utils.py` |
| Semantic similarity | Language-routed cosine similarity, BLEU, ROUGE-L | `hallucination_detector.py` |
| Entailment / contradiction | XLM-R fine-tuned on XNLI | `entailment_checker.py` |
| Critique (LLM2) | Qwen2.5-3B, grounded strictly in retrieved reference | `critic_llm.py` |
| Multi-turn memory | Source-scoped follow-up question rewriting | `conversation_memory.py` |
| Drafting | Template retrieval + slot extraction (no free generation) | `draft_generator.py` |
| Refinement orchestration | Full detect → critique → refine loop, all four branches | `cross_llm_refinement.py` |
| Serving | Flask REST API + dashboard | `app.py`, `templates/index.html` |
| Calibration | Batch eval against gold-labeled test cases | `run_batch_eval.py`, `auto_sample_test_cases.py` |
| Diagnostics | Citation-coverage ceiling check, adapter training-distribution sanity check | `check_citation_coverage.py`, `test_model.py` |

---

## Model training

Each language has its own LoRA adapter, trained independently on Kaggle's free-tier GPU via 4-bit QLoRA:

- **Base model:** `Qwen/Qwen2.5-3B-Instruct`
- **Quantization:** 4-bit NF4 (`bitsandbytes`), bf16 compute dtype, double quantization
- **LoRA config:** rank 16, alpha 32, dropout 0.05, targeting all attention and MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`) — ~30M trainable params (~1% of the 3.1B total)
- **Format:** each Q&A pair is rendered through the Qwen chat template with a system prompt fixing the target language, so one base model architecture serves all four adapters
- **Hot-swapping:** `model_loader.py` loads the 4-bit base model once and attaches/swaps LoRA adapters per language at request time via PEFT's multi-adapter support, rather than reloading the full base model for every language switch

Kaggle session limits meant training couldn't run start-to-finish in one sitting — each language's training was checkpointed and resumed across multiple sessions (in Punjabi's case, across a separate Kaggle account after exhausting another's weekly GPU quota). Checkpoints were re-uploaded as Kaggle datasets and reloaded via `PeftModel.from_pretrained(..., is_trainable=True)` to continue training exactly where it left off.

### Training results

| Language | Train samples | Steps | Final train loss | Final val loss |
|---|---|---|---|---|
| English | 12,537 | 783 | 7.04 | 0.92 |
| Hindi | 11,995 | 750 | 0.65 | 0.66 |
| Nepali | 12,132 | 759 | 0.58 | 0.57 |
| Punjabi | 12,006 | 751 | 0.41 | 0.40 |

**English's train/val loss gap looks like a bug, not a language difference.** Hindi, Nepali, and Punjabi all show training loss and validation loss tracking within ~0.02–0.05 of each other at every logged step (e.g. Punjabi step 100: train 0.589 / val 0.573; step 600: train 0.400 / val 0.408). English's training loss instead sits 6–7 points above its own validation loss throughout the run (train 7.04 vs. val 0.92 at the final step) — training loss should never be that far above validation loss for the same model on the same objective. Since three of four languages show train/val staying tightly coupled and only English diverges, this points to something specific to the English training cell (e.g. a difference in how loss is logged/reduced, gradient-accumulation scaling, or label masking) rather than English legal text being genuinely ~10x harder to model. **This should be root-caused — and the English run likely re-trained — before citing these numbers side-by-side as comparable across languages.**

Additionally, `hindi_part2.ipynb` includes a standalone adapter-vs-base-model evaluation: 50 held-out Hindi validation questions run through both the fine-tuned adapter and the un-tuned base model, scored on cosine similarity (via `all-MiniLM-L6-v2`), BLEU, and ROUGE-L against the gold answer. This is a distinct, smaller-scale sanity check from the embedders used in production retrieval/detection (`BAAI/bge-base-en-v1.5` for English, `sentence-transformers/LaBSE` for hi/pa/ne) — the two serve different purposes (one-off adapter quality check vs. live retrieval/grounding) and shouldn't be conflated.

`test_model.py` runs a separate, complementary sanity check directly against the fine-tuned adapters: each language's adapter is asked 3 in-distribution IPC questions (pulled near-verbatim from the training data, with the gold answer included for eyeball comparison) plus 1 out-of-distribution question about the BNS (a newer law absent from the training corpus) — isolating "the adapter is broken" from "the adapter was simply never taught this."

---

## Calibration & evaluation

`run_batch_eval.py` runs the detector against a hand-labeled test set of real fine-tuned-model outputs (3 in-distribution + 1 out-of-distribution BNS control question per language, across all 4 languages), spanning 5 mutually-exclusive gold labels:

- `grounded` — answer is correct / adequately grounded
- `fabricated_citation` — cites a section/article that doesn't exist or doesn't apply
- `off_topic` — answer doesn't address the question / unrelated content
- `contradiction` — answer inverts or negates the correct fact
- `no_reference` — question is outside dataset coverage (e.g. BNS)

The detector's free-text verdict is mapped onto this same label space (`verdict_to_label()`), producing a full confusion matrix and per-class precision/recall/F1, written to `results_batch.csv` — which `app.py`'s `/batch_results` endpoint serves to the dashboard for a live before/after comparison view. `auto_sample_test_cases.py` complements this with a narrower sanity check: it feeds the dataset's own gold question/answer pairs back through the detector (which should always score `LIKELY GROUNDED`, since there's no fabrication to catch) — a check that the pipeline isn't over-flagging correct answers, not a hallucination test per se.

`check_citation_coverage.py` is a diagnostic, not a test: it measures what fraction of the dataset's own entries mention any Section/Article number at all, which sets the real ceiling on what the Citation Grounding Critic can ever verify — if most entries are general/procedural with no section numbers, that's a scope limitation worth documenting, not a bug.

---

## Stack

`Qwen2.5-3B-Instruct` · 4-bit QLoRA (PEFT + bitsandbytes) · `BAAI/bge-base-en-v1.5` (English embedding) · `sentence-transformers/LaBSE` (hi/pa/ne embedding) · `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (reranking) · FAISS · ChromaDB · `xlm-roberta-large-xnli` (entailment) · pdfplumber (PDF ingestion) · HuggingFace Transformers · PyTorch · Flask · Chart.js

---

## Run it

```bash
# builds the two FAISS indices (en/BGE + hi-pa-ne/LaBSE) + the combined
# citation index, from data/*.json
python build_faiss_index.py

# builds the two Chroma collections (en/BGE + hi-pa-ne/LaBSE) from
# legal_corpus/ PDFs + the 10K IndicLegalQA json
python build_chroma_index.py

# optional diagnostics
python check_citation_coverage.py     # citation-coverage ceiling check
python test_model.py                  # adapter training-distribution sanity check
python auto_sample_test_cases.py      # gold-pair "does it over-flag correct answers" check

# calibration pass, writes results_batch.csv
python run_batch_eval.py

# serves the dashboard at localhost:5000
python app.py
```

### API surface (`app.py`)

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the dashboard (`templates/index.html`) |
| `/upload_document` | POST | Chunks + embeds a pasted document; returns a `doc_id` |
| `/ask` | POST | Main entry point — classifies intent, runs the matching branch, returns before/after results |
| `/reset_session` | POST | Clears conversation memory for a `session_id` |
| `/batch_results` | GET | Serves `results_batch.csv` as JSON for the calibration dashboard table |

`/ask` accepts `question`, `lang` (`en`/`hi`/`pa`/`ne`), and optionally `doc_id` (grounds against an uploaded document), `session_id` (enables multi-turn follow-up resolution), and `known_slots` (accumulated drafting form fields from prior turns).

---

## Limitations

- Calibration set size is limited; current scores should be read as directional rather than statistically robust until the eval set is scaled up.
- English's training run shows an unexplained train/val loss divergence not present in the other three languages (see note above) — likely a logging or setup bug in that specific notebook cell, not a genuine difficulty difference. Root cause is still open; treat the English adapter's training as provisional until it's resolved.
- The initial answer (LLM1's first pass) is generated without retrieval context for the `legal_knowledge` and `general_kb` branches — retrieval is only injected during the refinement step, after a hallucination is first detected. This is retrieval-augmented *correction*, not retrieval-augmented generation in the standard sense. (The `document_qa` branch is the exception: retrieval happens before the first generation, since the adapter has no built-in knowledge of an arbitrary uploaded document.)
- The broader ChromaDB corpus is unverified relative to the curated FAISS dataset (chunked PDFs, unverified 10K QA pairs) — it's used strictly as a fallback, and its confidence threshold is a starting estimate pending its own calibration pass.
- Citation-grounding checks are numeral-only: if a question doesn't explicitly name a section/article number, a retrieval miss on an unrelated topic (e.g. a wrong Act entirely) isn't caught by the citation check and may fall through to a refinement pass that can't succeed against the wrong reference.
- `build_chroma_index.py` currently tags every ingested PDF chunk as `lang="en"` by default, regardless of the PDF's actual language — this needs a per-folder or per-file language override before non-English PDFs in `legal_corpus/` (if any are added) would route to the correct embedder/collection.
- The entailment checker (XLM-R/XNLI) is trained on Hindi as its closest in-family language and generalizes to Punjabi/Nepali only via shared script/lexical overlap — accuracy on those two languages hasn't been separately validated, and a translate-to-English fallback hook exists in the code (`translate_fn`) but isn't wired up by default.
- `explain_document()` deliberately isn't scored by the hallucination detector or critic — there's currently no dedicated summary-faithfulness check (e.g. entailment against a synthesized reference), so a document explanation carries less verification than the other three branches.
