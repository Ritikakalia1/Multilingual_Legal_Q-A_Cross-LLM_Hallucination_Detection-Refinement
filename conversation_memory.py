# conversation_memory.py
#
# In-memory per-session conversation history + follow-up question
# resolution. A follow-up like "what about the punishment for that?" can't
# be embedded for retrieval as-is — "that" carries no meaning to LaBSE or
# FAISS. So before running the existing pipeline unchanged, we rewrite the
# raw follow-up into a standalone question using the last 1-2 turns as
# context, via a small LLM call (reusing critic_llm's already-loaded model
# — no new model, no retraining).
#
# Sessions are ephemeral, in-memory, keyed by a client-generated session_id
# (crypto.randomUUID() from the browser). No persistence across server
# restarts by design — this is conversational scratch memory, not a
# database of user history.
#
# --- SOURCE-SCOPED HISTORY (fix) ---------------------------------------
# History used to be one flat list shared across ALL grounding sources
# (document_qa, static legal_knowledge, general_kb fallback) within a
# session. That let an unrelated question asked in between two document
# questions get folded into the follow-up rewrite for the second document
# question (e.g. an FIR question inheriting "Para Legal Volunteers Scheme"
# context from an unrelated general-KB question).
#
# Fix: every turn is now tagged with a `source` string when it's added,
# and recent_context_text() only pulls turns matching the CURRENT source
# when building context for the rewriter. Turns from other sources are
# still stored (for potential future cross-topic features / debugging)
# but never bleed into an unrelated follow-up rewrite.
# -------------------------------------------------------------------------

import time
import re

SESSION_TTL_SECONDS = 3600      # evict idle sessions after 1 hour
MAX_HISTORY_TURNS = 4           # how many prior turns to keep/pass as context (per source)
MAX_REWRITE_CONTEXT_TURNS = 2   # how many prior turns to actually show the rewriter (keep prompt short)

VALID_SOURCES = {"document", "static_dataset", "general_kb"}

# Heuristic cues that a question is very likely a follow-up that needs
# rewriting — skips an unnecessary LLM call on the common case of a fresh,
# standalone question with no pronouns/ellipsis.
FOLLOWUP_CUES = re.compile(
    r"\b(that|this|it|those|these|the same|what about|and (?:the|what|who|when|where|how)|"
    r"he|she|they|him|her|them|his|her|their)\b",
    re.IGNORECASE,
)


def looks_like_followup(question: str) -> bool:
    """Cheap heuristic gate: only bother calling the rewriter LLM if the
    question actually contains a pronoun/reference cue. A fully independent
    question ('What is Section 302 IPC?') skips rewriting entirely."""
    return bool(FOLLOWUP_CUES.search(question))


class ConversationSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.turns = []  # list of {"question", "resolved_question", "answer", "source"}
        self.last_used = time.time()

    def add_turn(self, question: str, resolved_question: str, answer: str, source: str = "static_dataset"):
        """`source` should be one of VALID_SOURCES: "document", "static_dataset",
        or "general_kb". Unrecognized values are still stored as-is (so a
        caller passing a new source string doesn't crash), but you should
        add it to VALID_SOURCES if it becomes a real category."""
        self.turns.append({
            "question": question,
            "resolved_question": resolved_question,
            "answer": answer,
            "source": source,
        })
        # Trim per-source so one chatty source can't push another source's
        # history out of the window entirely.
        self._trim_per_source()
        self.last_used = time.time()

    def _trim_per_source(self):
        by_source = {}
        for t in self.turns:
            by_source.setdefault(t.get("source", "static_dataset"), []).append(t)
        trimmed = []
        for src, turns in by_source.items():
            trimmed.extend(turns[-MAX_HISTORY_TURNS:])
        # Keep overall chronological order stable-ish (not critical, since
        # recent_context_text always filters + re-slices anyway).
        self.turns = sorted(trimmed, key=lambda t: self.turns.index(t))

    def recent_context_text(self, n: int = MAX_REWRITE_CONTEXT_TURNS, source: str = None) -> str:
        """Renders the last n turns as plain Q/A text for the rewriter prompt.

        If `source` is given, only turns from that source are considered —
        this is what prevents an unrelated topic (e.g. a general-KB question
        asked in between) from leaking into a document follow-up's context.
        If `source` is None, falls back to the old flat behavior (all turns),
        useful for callers that don't care about scoping.
        """
        relevant = [t for t in self.turns if source is None or t.get("source") == source]
        recent = relevant[-n:]
        return "\n\n".join(f"Q: {t['resolved_question']}\nA: {t['answer']}" for t in recent)

    def has_history(self, source: str = None) -> bool:
        if source is None:
            return bool(self.turns)
        return any(t.get("source") == source for t in self.turns)


class ConversationManager:
    def __init__(self):
        self._sessions = {}

    def get_or_create(self, session_id: str) -> ConversationSession:
        self._evict_stale()
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(session_id)
        session = self._sessions[session_id]
        session.last_used = time.time()
        return session

    def reset(self, session_id: str):
        self._sessions.pop(session_id, None)

    def _evict_stale(self):
        now = time.time()
        for k in [k for k, v in self._sessions.items() if now - v.last_used > SESSION_TTL_SECONDS]:
            del self._sessions[k]


def resolve_followup_question(question: str, session: ConversationSession, rewriter_fn, source: str = None) -> str:
    """Returns a standalone version of `question`, rewriting only if:
      (a) there IS prior history in this session FOR THIS SOURCE, and
      (b) the question actually looks like it depends on that history.
    Otherwise returns the question unchanged (no wasted LLM call, and no
    risk of the rewriter "helpfully" mangling an already-clear question,
    and — critically — no risk of pulling in unrelated-topic context).

    source: the grounding source of the CURRENT question ("document",
    "static_dataset", or "general_kb"). Callers must pass this so the
    context lookup stays scoped to the right topic. If omitted, behaves
    like the old flat (unscoped) lookup.

    rewriter_fn: callable(context_text, question) -> str, supplied by the
    caller (app.py -> cross_llm_refinement.py wires this to
    critic_llm.rewrite_followup so we reuse the already-loaded critic model
    instead of loading a new one).
    """
    if not session.has_history(source=source):
        return question
    if not looks_like_followup(question):
        return question

    context_text = session.recent_context_text(source=source)
    if not context_text:
        return question
    try:
        rewritten = rewriter_fn(context_text, question)
        rewritten = rewritten.strip().strip('"')
        return rewritten if rewritten else question
    except Exception:
        # If rewriting fails for any reason, fall back to the raw question
        # rather than breaking the turn entirely.
        return question


conversation_manager = ConversationManager()