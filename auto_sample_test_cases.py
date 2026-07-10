# auto_sample_test_cases.py
#
# Auto-generates "expected: grounded" test cases by pulling ONE entry from
# each data/*.json file and feeding its own gold question+answer back through
# the detector. Since these are gold pairs, the detector SHOULD return
# "LIKELY GROUNDED" — this is a sanity check that the pipeline doesn't
# over-flag correct answers, not a hallucination test (gold data has no
# fabricated answers to test against — those need real model outputs,
# added separately in run_batch_eval.py's TEST_CASES).

import json
import os
import csv
from hallucination_detector import detector

DATA_DIR = "data"

DATASET_FILES = [
    "ipc_qa.json", "ipc_qa_hi.json", "ipc_qa_pa.json", "ipc_qa_ne.json",
    "crpc_qa.json", "crpc_qa_hi.json", "crpc_qa_pa.json", "crpc_qa_ne.json",
    "constitution_qa.json", "constitution_qa_hi.json",
    "constitution_qa_pa.json", "constitution_qa_ne.json",
]


def infer_lang_from_filename(filename: str) -> str:
    name = filename.replace(".json", "")
    for suffix, lang in [("_hi", "hi"), ("_pa", "pa"), ("_ne", "ne")]:
        if name.endswith(suffix):
            return lang
    return "en"


def sample_one_per_file(pick_index: int = 0):
    """Pull entry at `pick_index` from each file (default: first entry).
    Change pick_index if you want a different sample, e.g. entry 5."""
    samples = []
    for filename in DATASET_FILES:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"⚠️  Skipping missing file: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        if not entries:
            print(f"⚠️  {filename} is empty, skipping.")
            continue

        idx = min(pick_index, len(entries) - 1)
        entry = entries[idx]
        lang = entry.get("lang", infer_lang_from_filename(filename))

        samples.append({
            "source_file": filename,
            "lang": lang,
            "question": entry["question"],
            "gold_answer": entry["answer"],
        })

    return samples


def run_sanity_check(pick_index: int = 0):
    samples = sample_one_per_file(pick_index)
    if not samples:
        print("No samples found — check data/ folder.")
        return

    rows = []
    for s in samples:
        # Feed the GOLD answer back as if it were the "generated" answer —
        # this should score as grounded, since it literally IS the reference.
        result = detector.evaluate(s["question"], s["gold_answer"])

        top1_score = result["top_matches"][0]["score"] if result.get("top_matches") else None

        row = {
            "source_file": s["source_file"],
            "lang": s["lang"],
            "question": s["question"],
            "gold_answer": s["gold_answer"],
            "top1_retrieval_score": round(top1_score, 4) if top1_score is not None else None,
            "verdict": result["verdict"],
            "hallucination_score": result.get("hallucination_score"),
            "similarity_score": result["similarity_check"]["similarity_score"] if result.get("similarity_check") else None,
        }
        rows.append(row)

        print("=" * 100)
        print(f"[{s['source_file']}]  lang={s['lang']}")
        print(f"Q: {s['question']}")
        print(f"Gold A: {s['gold_answer']}")
        print(f"  top1 retrieval score : {row['top1_retrieval_score']}")
        print(f"  verdict              : {row['verdict']}")
        print(f"  similarity_score     : {row['similarity_score']}")
        print()

    fieldnames = list(rows[0].keys())
    with open("results_gold_sanity_check.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 100)
    print(f"✅ Wrote {len(rows)} results to results_gold_sanity_check.csv")

    from collections import Counter
    verdict_counts = Counter(r["verdict"] for r in rows)
    print("\nVerdict breakdown (expect all/mostly LIKELY GROUNDED):")
    for v, c in verdict_counts.items():
        print(f"  {v}: {c}")


if __name__ == "__main__":
    run_sanity_check(pick_index=0)