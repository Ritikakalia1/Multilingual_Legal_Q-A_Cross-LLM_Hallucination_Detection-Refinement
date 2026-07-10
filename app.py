# app.py
#
# Flask backend running the full cross-LLM refinement pipeline:
#   LLM1 (fine-tuned adapter, model_loader.py) drafts an answer
#   -> hallucination_detector retrieves reference + checks citations
#   -> LLM2 (critic_llm.py, a separate general model) critiques the draft
#      strictly against that retrieved reference
#   -> LLM1 regenerates using reference + critique as grounding context
#   -> hallucination_detector re-evaluates the refined answer
#
# Both the initial and refined results are returned so the frontend can
# show a direct before/after comparison, matching the calibration data in
# results_batch.csv.

from flask import Flask, request, jsonify, render_template
from cross_llm_refinement import cross_llm_refine
import csv
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

SUPPORTED_LANGS = ["en", "hi", "pa", "ne"]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    payload = request.json or {}
    question = payload.get('question', '').strip()
    lang = payload.get('lang', 'en').strip().lower()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400
    if lang not in SUPPORTED_LANGS:
        return jsonify({"error": f"Unsupported language '{lang}'. Choose from {SUPPORTED_LANGS}."}), 400

    result = cross_llm_refine(lang, question)
    result["question"] = question
    result["lang"] = lang

    return jsonify(result)


@app.route('/batch_results')
def batch_results():
    """Serves results_batch.csv (your calibration set) as JSON, so the
    frontend can render the summary table showing detector performance
    across the real test cases."""
    path = "results_batch.csv"
    if not os.path.exists(path):
        return jsonify({"error": "results_batch.csv not found — run run_batch_eval.py first."}), 404

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return jsonify(rows)


if __name__ == '__main__':
    app.run(debug=True, port=5000)