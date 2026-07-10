# check_citation_coverage.py
#
# Quick diagnostic: how many entries in the dataset actually mention a
# specific Section/Article number? This tells us the real ceiling for what
# the Citation Grounding Critic can verify — if most entries are general/
# procedural (no section numbers at all), that's an important scope note
# for the paper, not a bug to fix.

import json
import glob
import re

SECTION_PATTERN = re.compile(
    r"(?:Section|Article|धारा|अनुच्छेद|ਧਾਰਾ)\s*\d+", re.IGNORECASE
)

files = glob.glob("data/*_qa.json") + glob.glob("data/*_qa_*.json")
files = sorted(set(files))

total = 0
with_section = 0
per_file_counts = {}

for f in files:
    with open(f, encoding="utf-8") as fh:
        entries = json.load(fh)

    file_total = len(entries)
    file_with_section = 0

    for e in entries:
        combined_text = " ".join([
            e.get("question", ""), e.get("answer", ""),
            e.get("question_en", ""), e.get("answer_en", ""),
        ])
        if SECTION_PATTERN.search(combined_text):
            file_with_section += 1

    total += file_total
    with_section += file_with_section
    per_file_counts[f] = (file_total, file_with_section)

print(f"{'File':40s} {'Total':>8s} {'With section#':>15s} {'%':>6s}")
for f, (t, w) in per_file_counts.items():
    pct = (w / t * 100) if t else 0
    print(f"{f:40s} {t:8d} {w:15d} {pct:5.1f}%")

print("-" * 75)
overall_pct = (with_section / total * 100) if total else 0
print(f"{'TOTAL':40s} {total:8d} {with_section:15d} {overall_pct:5.1f}%")