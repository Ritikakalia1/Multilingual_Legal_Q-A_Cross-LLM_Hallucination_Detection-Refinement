# build_faiss_index.py
#
# Builds TWO FAISS indices instead of one — vectors from different
# embedding models live in different, incompatible vector spaces, so they
# can never share an index or be compared by cosine similarity to each
# other.
#
#   legal_qa_en.index / legal_qa_en_metadata.pkl       — English entries,
#       embedded with BAAI/bge-base-en-v1.5.
#   legal_qa_multi.index / legal_qa_multi_metadata.pkl — hi/pa/ne entries,
#       embedded with LaBSE.
#
# citation_index.pkl stays a SINGLE combined set across all languages —
# citation extraction is regex-based, language-agnostic, and independent
# of which embedding model retrieved the entry.
#
# Run this once (and again any time the dataset changes):
#   python build_faiss_index.py

import json
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

from citation_utils import extract_citations
from config import FAISS_DIR, DATA_DIR, EMBED_MODEL_EN, EMBED_MODEL_MULTI

EN_INDEX_PATH = os.path.join(FAISS_DIR, "legal_qa_en.index")
EN_METADATA_PATH = os.path.join(FAISS_DIR, "legal_qa_en_metadata.pkl")
MULTI_INDEX_PATH = os.path.join(FAISS_DIR, "legal_qa_multi.index")
MULTI_METADATA_PATH = os.path.join(FAISS_DIR, "legal_qa_multi_metadata.pkl")
CITATION_INDEX_PATH = os.path.join(FAISS_DIR, "citation_index.pkl")

DATASET_FILES = [
    "ipc_qa.json", "ipc_qa_hi.json", "ipc_qa_pa.json", "ipc_qa_ne.json",
    "crpc_qa.json", "crpc_qa_hi.json", "crpc_qa_pa.json", "crpc_qa_ne.json",
    "constitution_qa.json", "constitution_qa_hi.json",
    "constitution_qa_pa.json", "constitution_qa_ne.json",
]


def infer_lang_from_filename(filename: str) -> str:
    """ipc_qa_hi.json -> 'hi', ipc_qa.json -> 'en' (base file = English)."""
    name = filename.replace(".json", "")
    for suffix, lang in [("_hi", "hi"), ("_pa", "pa"), ("_ne", "ne")]:
        if name.endswith(suffix):
            return lang
    return "en"


def load_all_qa_pairs():
    """Load every QA entry from all dataset files, tagging source act + language."""
    all_entries = []

    for filename in DATASET_FILES:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"⚠️  Skipping missing file: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        file_lang = infer_lang_from_filename(filename)
        act_name = filename.replace(".json", "").replace(f"_{file_lang}", "").replace("_qa", "").upper()
        if act_name == "":
            act_name = "UNKNOWN"

        for entry in entries:
            lang = entry.get("lang", file_lang)
            all_entries.append({
                "question": entry["question"],
                "answer": entry["answer"],
                "question_en": entry.get("question_en", entry["question"] if lang == "en" else ""),
                "answer_en": entry.get("answer_en", entry["answer"] if lang == "en" else ""),
                "act": entry.get("act", act_name),
                "lang": lang,
                "source_file": filename,
            })

    print(f"Loaded {len(all_entries)} QA pairs total.")
    from collections import Counter
    lang_counts = Counter(e["lang"] for e in all_entries)
    print(f"Per-language counts: {dict(lang_counts)}")
    return all_entries


def build_citation_index(entries):
    """Union of every citation number mentioned anywhere in the dataset —
    used by the detector's citation_exists_in_dataset() check. UNCHANGED
    by the embedder split — citation extraction is regex, not embedding."""
    citation_index = set()
    for e in entries:
        combined_text = " ".join([
            e.get("question", ""), e.get("answer", ""),
            e.get("question_en", ""), e.get("answer_en", ""),
        ])
        citation_index |= extract_citations(combined_text)

    print(f"Found {len(citation_index)} unique citation numbers across dataset.")
    with open(CITATION_INDEX_PATH, "wb") as f:
        pickle.dump(citation_index, f)
    print(f"✅ Saved citation index to {CITATION_INDEX_PATH}")
    return citation_index


def build_group_index(entries, embedder, index_path, metadata_path, group_label):
    """Builds one FAISS index for a single language group (English subset
    or hi/pa/ne subset), embedding QUESTION text — retrieval matches the
    input question to the closest reference question, then uses that
    entry's answer as ground truth (embedding answers instead would be a
    weaker apples-to-oranges match, and embedding the GENERATED answer
    would let a hallucination drag retrieval toward the wrong reference)."""
    if not entries:
        print(f"⚠️  No entries for group '{group_label}' — skipping index build.")
        return

    question_texts = [e["question"] for e in entries]

    print(f"Embedding {len(question_texts)} '{group_label}' reference questions...")
    embeddings = embedder.encode(
        question_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner-product == cosine similarity
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(entries, f)

    print(f"✅ '{group_label}' index built: {index.ntotal} vectors, dim={dim}")
    print(f"✅ Saved to {index_path} / {metadata_path}")


def build_index():
    entries = load_all_qa_pairs()
    if not entries:
        raise RuntimeError(
            f"No QA entries found. Check that {DATA_DIR}/ contains {DATASET_FILES}."
        )

    build_citation_index(entries)

    en_entries = [e for e in entries if e["lang"] == "en"]
    multi_entries = [e for e in entries if e["lang"] != "en"]

    print(f"\nLoading English embedding model: {EMBED_MODEL_EN} ...")
    en_embedder = SentenceTransformer(EMBED_MODEL_EN)
    build_group_index(en_entries, en_embedder, EN_INDEX_PATH, EN_METADATA_PATH, "english")

    print(f"\nLoading multilingual embedding model: {EMBED_MODEL_MULTI} ...")
    multi_embedder = SentenceTransformer(EMBED_MODEL_MULTI)
    build_group_index(multi_entries, multi_embedder, MULTI_INDEX_PATH, MULTI_METADATA_PATH, "hi/pa/ne")

    print("\n✅ Both indices built.")


if __name__ == "__main__":
    build_index()