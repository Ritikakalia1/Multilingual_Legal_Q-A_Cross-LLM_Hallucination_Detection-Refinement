# build_faiss_index.py
#
# Builds one FAISS index covering all languages (en/hi/pa/ne) over your
# ipc_qa.json / crpc_qa.json / constitution_qa.json files. This index is
# the retrieval backbone for the hallucination detector: given a generated
# answer, we retrieve the closest gold QA pair and check the answer against it.
#
# Also builds citation_index.pkl in the SAME run, so the FAISS index and
# the citation index can never go stale relative to each other — there's
# only one build step to remember when data/ changes.
#
# Run this once (and again any time the dataset changes):
#   python build_faiss_index.py

import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from citation_utils import extract_citations

# ── Paths ──
DATA_DIR = "data"  # folder containing ipc_qa.json, crpc_qa.json, constitution_qa.json
FAISS_DIR = "faiss_index"
os.makedirs(FAISS_DIR, exist_ok=True)

INDEX_PATH = os.path.join(FAISS_DIR, "legal_qa.index")
METADATA_PATH = os.path.join(FAISS_DIR, "legal_qa_metadata.pkl")
CITATION_INDEX_PATH = os.path.join(FAISS_DIR, "citation_index.pkl")

# ── Embedding model ──
# LaBSE handles 100+ languages including Hindi, Punjabi, Nepali, English in
# one shared embedding space — needed since we're comparing across languages.
# (multilingual-e5-base is a good alternative if LaBSE feels slow.)
EMBED_MODEL_NAME = "sentence-transformers/LaBSE"

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
        # act name = filename stem minus language suffix, e.g. "ipc_qa_hi" -> "ipc"
        act_name = filename.replace(".json", "").replace(f"_{file_lang}", "").replace("_qa", "").upper()
        if act_name == "":
            act_name = "UNKNOWN"

        for entry in entries:
            # Some entries (like your merged multilingual paste) may carry
            # their own "lang" field — trust that if present, otherwise fall
            # back to the filename-inferred language.
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

    # Quick per-language breakdown so you can sanity-check counts before embedding
    from collections import Counter
    lang_counts = Counter(e["lang"] for e in all_entries)
    print(f"Per-language counts: {dict(lang_counts)}")

    return all_entries


def build_citation_index(entries):
    """Union of every citation number mentioned anywhere in the dataset —
    used by the detector's citation_exists_in_dataset() check. Scans
    question + answer + question_en + answer_en for every entry, using the
    SAME extraction logic (citation_utils.extract_citations) the detector
    uses on generated answers, so the two stay consistent."""
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


def build_index():
    entries = load_all_qa_pairs()
    if not entries:
        raise RuntimeError(
            f"No QA entries found. Check that {DATA_DIR}/ contains "
            f"{DATASET_FILES}."
        )

    # Build the citation index alongside the FAISS index, from the same
    # `entries` list, in the same run — keeps them from drifting apart.
    build_citation_index(entries)

    print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # Embed the QUESTION text — retrieval works by matching the input
    # question to the closest reference question, then we use THAT entry's
    # answer as ground truth. (Embedding answers instead would mean matching
    # a question against answers, which is a weaker/apples-to-oranges match —
    # and if we ever embedded the *generated* answer instead of the question,
    # a hallucinated answer could drag retrieval toward the wrong reference
    # entirely, since it'd be searching based on fabricated content.)
    question_texts = [e["question"] for e in entries]

    print(f"Embedding {len(question_texts)} reference questions...")
    embeddings = embedder.encode(
        question_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner-product == cosine similarity
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine sim
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(entries, f)

    print(f"✅ Index built: {index.ntotal} vectors, dim={dim}")
    print(f"✅ Saved index to {INDEX_PATH}")
    print(f"✅ Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    build_index()