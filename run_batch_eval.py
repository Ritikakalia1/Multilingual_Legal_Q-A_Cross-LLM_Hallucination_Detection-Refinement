# run_batch_eval.py
#
# Calibration pass using REAL fine-tuned model outputs from test_model.py,
# across all four languages: 3 in-distribution questions each (en/hi/pa/ne)
# + 1 out-of-distribution BNS control question each.
#
# ── CHANGELOG ──
# 1. BUG FIX: this script previously wrote "hallucination_score" to the CSV
#    but never wrote "grounding_score", even though hallucination_detector.py
#    returns both. The frontend (index.html -> renderBatchTable) falls back
#    to hallucination_score whenever grounding_score is missing, which is why
#    every row in the Calibration Results table showed the same number twice.
#    Fix: both fields are now written to the CSV.
# 2. Replaced free-text "expected_note" with a structured `gold_label` field
#    (one of GOLD_LABELS below) so the detector's verdict can be scored
#    against it automatically instead of just eyeballed.
# 3. Added `verdict_to_label()` to map the detector's free-text verdict
#    strings onto the same label space as gold_label, and a scoring pass
#    (confusion matrix + per-class precision/recall/F1) at the end of the run.

import csv
from collections import Counter, defaultdict

from hallucination_detector import detector

# ── Gold label space ──
# Keep this small and mutually exclusive. Every TEST_CASES entry must use
# one of these exact strings as its gold_label.
GOLD_LABELS = [
    "grounded",              # answer is correct / adequately grounded
    "fabricated_citation",   # cites a section/article that doesn't exist or doesn't apply
    "off_topic",             # answer doesn't address the question / unrelated content
    "contradiction",         # answer inverts / negates the correct fact
    "no_reference",          # question is outside dataset coverage (e.g. BNS)
]

TEST_CASES = [
    # Each tuple: (gold_label, note, lang, question, answer)

    # ── English, in-distribution ──
    (
        "grounded",
        "paraphrase, correct core facts",
        "en",
        "What is the title and extent of operation of the Indian Penal Code?",
        "The Indian Penal Code is a law that has been enacted by the Parliament of India, and it applies throughout the territory of India.",
    ),
    (
        "fabricated_citation",
        "'Constitution Sixty-third Amendment Act, 1992' does not exist/apply here",
        "en",
        "Where does the operation of 'The Indian Penal Code' extend to, and does it include extra-territorial offences?",
        "The operation of 'The Indian Penal Code' extends to the territories specified in clause (a) of sub-section (1) of section 3 of the Constitution (Sixty-third Amendment) Act, 1992, and includes extra-territorial offences.",
    ),
    (
        "contradiction",
        "answer inverted — claims code does NOT affect certain laws, opposite of reality/fabricated legal reasoning",
        "en",
        "What does the Indian Penal Code stipulate about certain laws?",
        "The Indian Penal Code states that no law made by Parliament or any other competent Legislature shall be deemed to be repugnant to this Code if it is in conflict with any of the provisions of this Code.",
    ),

    # ── Hindi, in-distribution ──
    (
        "fabricated_citation",
        "fabricated section number 'भाग 304'",
        "hi",
        "भारतीय पेनल कोड के संचालन का शीर्षक और विस्तार क्या है?",
        "भारतीय पेनल कोड के भाग ३०४ में शीर्षक और विस्तार दिए गए हैं ।",
    ),
    (
        "off_topic",
        "off-topic / unrelated content",
        "hi",
        "'भारतीय पेनल कोड' की प्रक्रिया कहाँ होती है, और क्या यह अतिरिक्त गुनाहों को शामिल करता है?",
        "इस भाग में एक व्यक्‍ति के लिए दण्ड दिया जाता है, जो उस व्यक्‍ति को बचाने के लिए दिया जाता है ।",
    ),
    (
        "off_topic",
        "off-topic, unrelated to actual question",
        "hi",
        "भारतीय पेनल कोड कुछ नियमों के बारे में क्या निर्धारित करता है?",
        "भारतीय पेनल कोड के अनुसार, एक व्यक्‍ति को जेल में डाल दिया जा सकता है, या उसे एक ऐसे समय के लिए क़ैद में डाला जा सकता है, जो शायद एक या दोनों, या दोनों, दोनों के साथ आगे बढ़ सकता है ।",
    ),

    # ── Punjabi, in-distribution ──
    (
        "fabricated_citation",
        "fabricated 'धारा 1 ਤੋਂ 365' range, invented specificity",
        "pa",
        "ਇੰਡੀਅਨ ਪੀਨਲ ਕੋਡ ਦੀ ਕਾਰਵਾਈ ਦਾ ਸਿਰਲੇਖ ਅਤੇ ਹੱਦ ਕੀ ਹੈ?",
        "ਇੰਡੀਅਨ ਪੀਨਲ ਕੋਡ ਦੀ ਕਾਰਵਾਈ ਦਾ ਸਿਰਲੇਖ ਇੱਕ ਸੰਸਦ ਦੁਆਰਾ ਬਣਾਏ ਗਏ ਕਿਸੇ ਵੀ ਫੌਜਦਾਰੀ ਜਾਂ ਮਾਲਕੀ ਕੋਡ ਦੀ ਧਾਰਾ 1 ਤੋਂ 365 ਤੱਕ ਦੀ ਉਪ-ਧਾਰਾ (1) ਦੇ ਉਪਬੰਧਾਂ ਦੇ ਅਧੀਨ ਹੈ।",
    ),
    (
        "off_topic",
        "degenerate repetition loop — model failure, not just wrong fact",
        "pa",
        "'ਇੰਡੀਅਨ ਪੀਨਲ ਕੋਡ' ਦੀ ਕਾਰਵਾਈ ਕਿੱਥੇ ਤੱਕ ਫੈਲੀ ਹੋਈ ਹੈ, ਅਤੇ ਕੀ ਇਸ ਵਿੱਚ ਖੇਤਰ ਤੋਂ ਬਾਹਰਲੇ ਅਪਰਾਧ ਸ਼ਾਮਲ ਹਨ?",
        "ਇੰਡੀਅਨ ਪੀਨਲ ਕੋਡ ਦੀ ਕਾਰਵਾਈ ਇੱਕ ਮਿਆਦ ਲਈ ਸ਼ਾਮਲ ਹੁੰਦੀ ਹੈ ਜੋ ਉਸ ਸਮੇਂ ਤੱਕ ਲਾਗੂ ਹੁੰਦੀ ਹੈ ਜਿਸ ਦੌਰਾਨ ਇਸ ਸੰਵਿਧਾਨ ਦੇ ਸ਼ੁਰੂ ਹੋਣ ਤੋਂ ਤੁਰੰਤ ਪਹਿਲਾਂ ਇੱਕ ਸੰਸਦ ਦੁਆਰਾ ਕੀਤੇ ਗਏ ਕਾਨੂੰਨ ਦੁਆਰਾ ਕਿਸੇ ਵੀ ਕਾਨੂੰਨ ਦੁਆਰਾ ਕਿਸੇ ਵੀ ਕਾਨੂੰਨ ਦੁਆਰਾ ਕਿਸੇ ਵੀ ਕਾਨੂੰਨ ਦੁਆਰਾ ਕਿਸੇ ਵੀ ਕਾਨੂੰਨ ਦੁਆਰਾ ਕਿਸੇ ਵੀ ਕਾਨੂੰਨ ਦੁਆਰਾ ਕਿਸੇ ਵੀ ਕਾਨੂੰਨ ਦੁਆਰਾ ਕਿਸੇ ਵੀ",
    ),
    (
        "grounded",
        "vague/underspecified but not factually wrong",
        "pa",
        "ਇੰਡੀਅਨ ਪੀਨਲ ਕੋਡ ਕੁਝ ਕਾਨੂੰਨਾਂ ਬਾਰੇ ਕੀ ਨਿਰਧਾਰਤ ਕਰਦਾ ਹੈ?",
        "ਇੰਡੀਅਨ ਪੀਨਲ ਕੋਡ ਕੁਝ ਕਾਨੂੰਨਾਂ ਨੂੰ ਨਿਰਧਾਰਤ ਕਰਦਾ ਹੈ।",
    ),

    # ── Nepali, in-distribution ──
    (
        "grounded",
        "vague/circular, not clearly citing anything false",
        "ne",
        "भारतीय दण्ड संहिताको कार्यको शीर्षक र सीमा के हो?",
        "भारतीय दण्ड संहिताको शीर्षक भारतीय दण्ड संहिता हो। यसले भारतीय दण्ड संहिताको कुनै पनि विवरणको लागि अधिकार दिन्छ।",
    ),
    (
        "contradiction",
        "inverted fact — claims it does NOT include extra-territorial offences, opposite of truth",
        "ne",
        "'भारतीय दण्ड संहिता' को सञ्चालन कहाँसम्म फैलिएको छ, र के यसले अतिरिक्त क्षेत्रीय अपराधहरू समावेश गर्दछ?",
        "'भारतीय दण्ड संहिता' को सञ्चालन भारतको भूभागमा फैलिएको छ, र यसले अतिरिक्त क्षेत्रीय अपराधहरू समावेश गर्दैन।",
    ),
    (
        "off_topic",
        "off-topic — invents unrelated content about judges/exemptions",
        "ne",
        "भारतीय दण्ड संहिताले कतिपय कानुनहरूको बारेमा के व्यवस्था गर्छ?",
        "भारतीय दण्ड संहिताले कुनै पनि कानूनलाई अपवादमा राख्ने वा उच्च अदालतको न्यायाधीशलाई आफ्नो न्यायाधीशको रूपमा नियुक्त गर्ने अधिकार दिने व्यवस्था गरेको छ।",
    ),

    # ── OOD control: BNS (no grounding exists in dataset) ──
    (
        "no_reference",
        "BNS not in corpus",
        "en",
        "What section of the BNS corresponds to Section 302 IPC?",
        "Section 302",
    ),
    (
        "no_reference",
        "BNS not in corpus",
        "hi",
        "IPC की धारा 302 के अनुरूप BNS की कौन सी धारा है?",
        "भाग 302 में बताया गया है कि एक व्यक्‍ति को जानबूझकर दोषी ठहराया जाता है ।",
    ),
    (
        "no_reference",
        "BNS not in corpus",
        "pa",
        "IPC ਦੀ ਧਾਰਾ 302 ਦੇ ਬਰਾਬਰ BNS ਦੀ ਕਿਹੜੀ ਧਾਰਾ ਹੈ?",
        "BNS ਦੀ ਧਾਰਾ 302 ਅਪਰਾਧਿਕ ਅਪਰਾਧ ਦੀ ਸਜ਼ਾ ਦੇ ਵਿਰੁੱਧ ਮੌਤ ਦੀ ਸਜ਼ਾ ਨਾਲ ਸਬੰਧਤ ਹੈ।",
    ),
    (
        "no_reference",
        "BNS not in corpus",
        "ne",
        "IPC को धारा ३०२ सँग मिल्ने BNS को धारा कुन हो?",
        "धारा 302 एक अपराध हो जसमा अपराधीले आफ्नो व्यक्तिगत शारीरिक बलबाट अपराध गरेको छ।",
    ),
]


def verdict_to_label(verdict: str) -> str:
    """Maps the detector's free-text verdict onto the same label space as
    gold_label, so the two can be compared programmatically.

    NOTE: the detector's verdict strings distinguish "fabricated_citation"
    and "contradiction" explicitly, but collapse anything else wrong into
    a generic "low grounding to reference" bucket that we can't tell apart
    from off_topic just from the verdict string alone. We map that generic
    bucket to "off_topic" since that's what it's meant to catch, but this
    is a real limitation: if you want gold_label to distinguish off_topic
    from other grounding failures at scoring time, the detector's verdict
    strings need to be made more specific first (see note at bottom of file).
    """
    if verdict is None:
        return "unknown"
    if verdict.startswith("NO CONFIDENT REFERENCE"):
        return "no_reference"
    if verdict == "LIKELY GROUNDED":
        return "grounded"
    if "fabricated citation" in verdict:
        return "fabricated_citation"
    if "contradicts reference" in verdict:
        return "contradiction"
    if "low grounding to reference" in verdict:
        return "off_topic"
    return "unknown"


def print_scoring_report(rows: list) -> None:
    """Confusion matrix + per-class precision/recall/F1 of predicted_label
    (from the detector's verdict) against gold_label (our own annotation)."""
    labels = GOLD_LABELS + ["unknown"]
    confusion = defaultdict(Counter)  # confusion[gold][predicted] = count

    for r in rows:
        confusion[r["gold_label"]][r["predicted_label"]] += 1

    print("\n" + "=" * 100)
    print("SCORING REPORT — predicted_label (detector verdict) vs gold_label (annotation)")
    print("=" * 100)

    # Confusion matrix
    header = "gold \\ predicted".ljust(22) + "".join(l[:14].ljust(16) for l in labels)
    print(header)
    for gold in GOLD_LABELS:
        row_str = gold.ljust(22)
        for pred in labels:
            row_str += str(confusion[gold][pred]).ljust(16)
        print(row_str)

    # Per-class precision / recall / F1
    print("\nPer-class metrics:")
    print(f"{'label':<22}{'precision':<12}{'recall':<12}{'f1':<12}{'support':<10}")
    for label in GOLD_LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_total = sum(confusion[g][label] for g in GOLD_LABELS)

        precision = tp / predicted_total if predicted_total else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        print(f"{label:<22}{precision:<12.3f}{recall:<12.3f}{f1:<12.3f}{support:<10}")

    total = sum(sum(c.values()) for c in confusion.values())
    correct = sum(confusion[l][l] for l in GOLD_LABELS)
    overall_acc = correct / total if total else 0.0
    print(f"\nOverall accuracy: {correct}/{total} = {overall_acc:.3f}")


def run_batch():
    if not TEST_CASES:
        print("⚠️  TEST_CASES is empty.")
        return

    rows = []
    for gold_label, note, lang, question, answer in TEST_CASES:
        if gold_label not in GOLD_LABELS:
            raise ValueError(f"Unknown gold_label '{gold_label}' for question: {question}")

        result = detector.evaluate(question, answer)
        top1_score = result["top_matches"][0]["score"] if result.get("top_matches") else None
        predicted_label = verdict_to_label(result.get("verdict"))

        row = {
            "gold_label": gold_label,
            "note": note,
            "lang": lang,
            "question": question,
            "generated_answer": answer,
            "top1_retrieval_score": round(top1_score, 4) if top1_score is not None else None,
            "verdict": result["verdict"],
            "predicted_label": predicted_label,
            "correct": gold_label == predicted_label,
            # BUG FIX: both scores now written (previously only
            # hallucination_score was written, so the frontend's
            # grounding_score fallback made both columns show the same value).
            "hallucination_score": result.get("hallucination_score"),
            "grounding_score": result.get("grounding_score"),
            "citations_found": result["citation_check"]["citations_found"] if result.get("citation_check") else None,
            "unverified_citations": result["citation_check"]["unverified_citations"] if result.get("citation_check") else None,
            "similarity_score": result["similarity_check"]["similarity_score"] if result.get("similarity_check") else None,
        }
        rows.append(row)

        print("=" * 100)
        print(f"[gold={gold_label} | {note}]  lang={lang}")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"  top1 retrieval score : {row['top1_retrieval_score']}")
        print(f"  verdict              : {row['verdict']}")
        print(f"  predicted_label      : {row['predicted_label']}  ({'✅ correct' if row['correct'] else '❌ MISMATCH'})")
        print(f"  hallucination_score  : {row['hallucination_score']}")
        print(f"  grounding_score      : {row['grounding_score']}")
        if row["citations_found"] is not None:
            print(f"  citations_found      : {row['citations_found']}")
            print(f"  unverified_citations : {row['unverified_citations']}")
            print(f"  similarity_score     : {row['similarity_score']}")
        print()

    fieldnames = list(rows[0].keys())
    with open("results_batch.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 100)
    print(f"✅ Wrote {len(rows)} results to results_batch.csv")

    verdict_counts = Counter(r["verdict"] for r in rows)
    print("\nVerdict breakdown:")
    for v, c in verdict_counts.items():
        print(f"  {v}: {c}")

    print_scoring_report(rows)


if __name__ == "__main__":
    run_batch()

# ── NOTE for scaling to a real gold set (100-150 cases) ──
# The detector's verdict strings currently don't distinguish "off_topic"
# from other generic grounding failures — both collapse into "low grounding
# to reference" in hallucination_detector.py's evaluate(). If your F1 table
# for the off_topic class looks suspicious once you scale up, that's why:
# it's a labeling-granularity mismatch, not necessarily a detector failure.
# Consider adding a distinct verdict string in evaluate() for the "similarity
# very low AND no citations at all" case if you want that class to be
# scoreable separately, or fold off_topic into a broader "low_grounding"
# gold label instead of trying to split it from other causes.