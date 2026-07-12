# build_chroma_index.py
#
# Ingests TWO sources into GeneralKnowledgeBase's Chroma collections:
#   1. legal_corpus/**/*.pdf   — chunked the same word-sliding-window way
#      document_index.py chunks uploaded documents.
#   2. legal_corpus/IndicLegalQA_Dataset_10K_Revised.json — treated as
#      QA-pair entries, not chunks.
#
# ── DUAL-EMBEDDER EXTENSION ──
# Entries are split by language BEFORE ingestion — English entries go into
# GeneralKnowledgeBase's "en" collection (BGE), hi/pa/ne entries go into
# the "multi" collection (LaBSE). This mirrors build_faiss_index.py's
# en_entries/multi_entries split.
#
# Re-running this script is safe for GROWING the corpus (new PDFs added to
# legal_corpus/ later): add_entries() appends starting from each
# collection's current count rather than resetting it. It is NOT safe for
# re-ingesting an EDITED PDF (you'd get a duplicate copy of the old text)
# — for that, delete chroma_store/ and rebuild from scratch.
#
# Run:
#   python build_chroma_index.py

import os
import json
import glob

import pdfplumber
from sentence_transformers import SentenceTransformer

from general_knowledge_base import GeneralKnowledgeBase
from config import EMBED_MODEL_EN, EMBED_MODEL_MULTI

CORPUS_DIR = "legal_corpus"
QA_JSON_NAME = "IndicLegalQA_Dataset_10K_Revised.json"  # adjust to your exact filename

CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40


def chunk_text(text: str, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS) -> list:
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def extract_pdf_text(path: str) -> str:
    """Concatenates all pages' text. If a PDF is scanned/image-only,
    pdfplumber returns empty strings per page — this silently produces no
    chunks for that file rather than crashing; check the printed per-file
    chunk counts at the end of a run to catch that case."""
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
    return "\n".join(text_parts)


def act_name_from_path(path: str) -> str:
    """legal_corpus/women&law/some_act.pdf -> 'women&law' (the subfolder
    name), so retrieval results can show which part of the corpus an
    answer came from."""
    rel = os.path.relpath(path, CORPUS_DIR)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else "UNKNOWN"


def load_pdf_entries() -> list:
    """Returns entries tagged with 'lang' — adjust the default "en" below
    per-folder if some PDFs are actually hi/pa/ne (e.g. by checking
    act_name_from_path against a known list of non-English subfolders)."""
    pdf_paths = glob.glob(os.path.join(CORPUS_DIR, "**", "*.pdf"), recursive=True)
    entries = []
    for path in pdf_paths:
        raw_text = extract_pdf_text(path)
        chunks = chunk_text(raw_text)
        if not chunks:
            print(f"⚠️  No extractable text: {path} (scanned/image PDF? needs OCR first)")
            continue
        act = act_name_from_path(path)
        for idx, chunk in enumerate(chunks):
            entries.append({
                "text": chunk,
                "source_file": os.path.basename(path),
                "act": act,
                "lang": "en",       # adjust per-folder if some PDFs are hi/pa/ne
                "chunk_index": idx,
                "type": "corpus_chunk",
            })
        print(f"  {path}: {len(chunks)} chunks")
    return entries


def load_qa_json_entries() -> list:
    path = os.path.join(CORPUS_DIR, QA_JSON_NAME)
    if not os.path.exists(path):
        print(f"⚠️  {path} not found — skipping QA-json ingestion.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw_entries = json.load(f)

    entries = []
    for e in raw_entries:
        question = e.get("question", "")
        answer = e.get("answer", "")
        if not question or not answer:
            continue
        entries.append({
            "text": question,
            "question": question,
            "answer": answer,
            "question_en": e.get("question_en", question if e.get("lang", "en") == "en" else ""),
            "answer_en": e.get("answer_en", answer if e.get("lang", "en") == "en" else ""),
            "act": e.get("act", "GENERAL"),
            "lang": e.get("lang", "en"),
            "type": "qa_pair",
        })
    return entries


def split_by_lang_group(entries: list):
    """Splits a mixed-language entry list into (en_entries, multi_entries)
    — same split logic build_faiss_index.py uses."""
    en_entries = [e for e in entries if e.get("lang", "en") == "en"]
    multi_entries = [e for e in entries if e.get("lang", "en") != "en"]
    return en_entries, multi_entries


def build():
    print(f"Loading English embedding model: {EMBED_MODEL_EN} ...")
    embedder_en = SentenceTransformer(EMBED_MODEL_EN)

    print(f"Loading multilingual embedding model: {EMBED_MODEL_MULTI} ...")
    embedder_multi = SentenceTransformer(EMBED_MODEL_MULTI)

    kb = GeneralKnowledgeBase(embedder_en, embedder_multi)

    print("\n── Ingesting PDFs ──")
    pdf_entries = load_pdf_entries()
    print(f"Total PDF chunks: {len(pdf_entries)}")

    print("\n── Ingesting 10K QA JSON ──")
    qa_entries = load_qa_json_entries()
    print(f"Total QA entries: {len(qa_entries)}")

    all_entries = pdf_entries + qa_entries
    if not all_entries:
        raise RuntimeError("Nothing to ingest — check CORPUS_DIR and QA_JSON_NAME.")

    en_entries, multi_entries = split_by_lang_group(all_entries)
    print(f"\nPer-language split: en={len(en_entries)}, hi/pa/ne={len(multi_entries)}")

    if en_entries:
        print(f"\nEmbedding + adding {len(en_entries)} English entries (BGE) to Chroma...")
        kb.add_entries(en_entries, lang_group="en")
    else:
        print("\n⚠️  No English entries to ingest.")

    if multi_entries:
        print(f"\nEmbedding + adding {len(multi_entries)} hi/pa/ne entries (LaBSE) to Chroma...")
        kb.add_entries(multi_entries, lang_group="multi")
    else:
        print("\n⚠️  No hi/pa/ne entries to ingest.")

    print("\nRebuilding citation index over full corpus (both language groups combined)...")
    citations = kb.rebuild_citation_index(all_entries)
    print(f"✅ Found {len(citations)} unique citation numbers across the general corpus.")

    print(f"\n✅ English collection: {kb.collection_en.count()} vectors.")
    print(f"✅ Multilingual collection: {kb.collection_multi.count()} vectors.")


if __name__ == "__main__":
    build()