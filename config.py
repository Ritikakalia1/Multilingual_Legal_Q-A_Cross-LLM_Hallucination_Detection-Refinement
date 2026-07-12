# config.py
import os
import secrets

# ── Base paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(BASE_DIR, "adapters")
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
DATA_DIR    = os.path.join(BASE_DIR, "data")
FAISS_DIR   = os.path.join(BASE_DIR, "faiss_index")

# ── Create dirs if not exist ──
for d in [UPLOAD_DIR, DATA_DIR, FAISS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Model ──
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# ── Adapter paths ──
ADAPTERS = {
    "en": os.path.join(ADAPTER_DIR, "legal_llm_adapter_en"),
    "hi": os.path.join(ADAPTER_DIR, "legal_llm_adapter_hi"),
    "pa": os.path.join(ADAPTER_DIR, "legal_llm_adapter_pa"),
    "ne": os.path.join(ADAPTER_DIR, "legal_llm_adapter_ne"),
}

# ── Language names ──
LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "pa": "Punjabi",
    "ne": "Nepali",
}

# ── Model settings ──
MAX_NEW_TOKENS = 512
MAX_INPUT_LENGTH = 1024
DEVICE = "cuda"

# ── RAG settings ──
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_DOCS = 3

# ── Flask settings ──
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

# ── Critic thresholds ──
REFINEMENT_THRESHOLD = 0.6  # below this score → refine answer

# ── Embedding models (NEW — dual-embedder setup) ──
# English gets a retrieval-specialized model (better on English-only text
# than a multilingual one). hi/pa/ne stay on LaBSE, which is the one model
# here that actually understands those languages at all.
EMBED_MODEL_EN = "BAAI/bge-base-en-v1.5"
EMBED_MODEL_MULTI = "sentence-transformers/LaBSE"

# BGE models are trained asymmetrically: the QUERY side needs this
# instruction prefix for good retrieval quality; the PASSAGE/document side
# does not. Skipping this on queries measurably hurts retrieval with
# bge-base/large-v1.5. LaBSE has no equivalent requirement.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "