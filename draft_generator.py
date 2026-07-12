# draft_generator.py
#
# Fourth pipeline branch (see intent_classifier.py): template-retrieval +
# slot-filling document drafting, NOT free-generation. Deliberately avoids
# asking LLM1/LLM2 to "write a rental agreement" from scratch — that's
# exactly the kind of open-ended generation most likely to invent clauses,
# misstate statutory requirements, or omit legally-required language.
# Instead:
#
#   1. Match the request to a known TEMPLATE (keyword match over a small,
#      fixed library — few enough templates that this doesn't need FAISS).
#   2. Extract whatever slot values the user already stated in their
#      message (names, dates, amounts, addresses) via a constrained LLM
#      call, reusing critic_llm's loaded model — same pattern as
#      rewrite_followup/classify_intent.
#   3. Diff against the template's required_slots. If anything's missing,
#      return it as `missing_fields` instead of guessing/inventing a
#      placeholder value — app.py surfaces this as a follow-up prompt
#      ("What is the monthly rent amount?") rather than the frontend
#      showing a document with fabricated details.
#   4. Only once all required slots are filled does this fill the template
#      and return the completed document text — a substitution, not a
#      generation, so nothing in the final draft is text the LLM invented.
#
# Extend TEMPLATES (or move to data/templates/*.json + real retrieval) as
# the library grows past what keyword matching can disambiguate.

TEMPLATES = {
    "rental_agreement": {
        "match_keywords": ["rental agreement", "rent agreement", "lease agreement", "tenancy agreement"],
        "required_slots": ["landlord_name", "tenant_name", "property_address", "monthly_rent", "start_date", "duration_months"],
        "slot_prompts": {
            "landlord_name": "the landlord's full name",
            "tenant_name": "the tenant's full name",
            "property_address": "the full address of the rented property",
            "monthly_rent": "the monthly rent amount",
            "start_date": "the agreement's start date",
            "duration_months": "the duration of the lease, in months",
        },
        "template_text": (
            "RENTAL AGREEMENT\n\n"
            "This Rental Agreement is made on {start_date} between {landlord_name} "
            "(\"Landlord\") and {tenant_name} (\"Tenant\") for the property located at "
            "{property_address}.\n\n"
            "1. TERM: This agreement is valid for {duration_months} months from {start_date}.\n"
            "2. RENT: The Tenant shall pay a monthly rent of {monthly_rent} on or before "
            "the 5th day of each month.\n"
            "3. USE: The premises shall be used for residential purposes only.\n"
            "4. MAINTENANCE: The Tenant shall maintain the premises in good condition.\n\n"
            "Signed:\n_______________________          _______________________\n"
            "Landlord ({landlord_name})              Tenant ({tenant_name})"
        ),
    },
    "legal_notice": {
        "match_keywords": ["legal notice", "notice to"],
        "required_slots": ["sender_name", "recipient_name", "subject", "grievance_details", "demand"],
        "slot_prompts": {
            "sender_name": "who the notice is being sent from",
            "recipient_name": "who the notice is being sent to",
            "subject": "a short subject line for the notice",
            "grievance_details": "a brief description of the grievance/issue",
            "demand": "what specific action or remedy is being demanded",
        },
        "template_text": (
            "LEGAL NOTICE\n\n"
            "To: {recipient_name}\nFrom: {sender_name}\nSubject: {subject}\n\n"
            "This notice is being served regarding the following matter:\n"
            "{grievance_details}\n\n"
            "You are hereby called upon to {demand} within 15 days of receipt of this "
            "notice, failing which appropriate legal proceedings will be initiated "
            "against you, entirely at your risk as to costs and consequences.\n\n"
            "Sincerely,\n{sender_name}"
        ),
    },
    "rti_application": {
        "match_keywords": ["rti", "right to information"],
        "required_slots": ["applicant_name", "applicant_address", "public_authority", "information_sought"],
        "slot_prompts": {
            "applicant_name": "the applicant's full name",
            "applicant_address": "the applicant's address",
            "public_authority": "the name of the public authority/department the request is addressed to",
            "information_sought": "the specific information being requested",
        },
        "template_text": (
            "APPLICATION UNDER THE RIGHT TO INFORMATION ACT, 2005\n\n"
            "To,\nThe Public Information Officer,\n{public_authority}\n\n"
            "From,\n{applicant_name}\n{applicant_address}\n\n"
            "Subject: Request for information under the RTI Act, 2005\n\n"
            "I would like to request the following information:\n{information_sought}\n\n"
            "I have deposited the requisite fee as prescribed under the Act.\n\n"
            "Yours faithfully,\n{applicant_name}"
        ),
    },
}


def match_template(question: str):
    """Keyword match against the request text. Returns (template_key,
    template_dict) or (None, None) if nothing matches — caller treats that
    as 'not actually a drafting request I can handle', not a silent
    best-guess."""
    q_lower = question.lower()
    for key, tmpl in TEMPLATES.items():
        if any(kw in q_lower for kw in tmpl["match_keywords"]):
            return key, tmpl
    return None, None


def extract_slots(question: str, template: dict, extractor_fn) -> dict:
    """
    extractor_fn: callable(question, required_slots) -> dict, supplied by
    the caller. app.py wires this to critic_llm.extract_slots — same
    reused-model pattern as the rest of the pipeline. Returns only the
    slots the LLM was confident it found; anything it couldn't find is
    OMITTED from the returned dict, not filled with a guess.
    """
    try:
        found = extractor_fn(question, template["required_slots"])
        if not isinstance(found, dict):
            return {}
        # Defensive: only keep known slot keys, only keep non-empty values.
        return {
            k: v.strip() for k, v in found.items()
            if k in template["required_slots"] and isinstance(v, str) and v.strip()
        }
    except Exception:
        return {}


def draft(question: str, extractor_fn, known_slots: dict = None) -> dict:
    """
    Main entry point. `known_slots` lets a caller pass in slot values
    collected across multiple turns (e.g. the user answered a follow-up
    prompt for a missing field) — merged with whatever's freshly extracted
    from `question` itself, with fresh extraction taking precedence on
    conflicts (the latest turn is likeliest to be current).

    Returns one of:
      {"status": "no_template_match"}
      {"status": "missing_fields", "template": key, "missing_fields": [...],
       "missing_fields_prompts": {...}, "collected_slots": {...}}
      {"status": "drafted", "template": key, "document_text": "...",
       "collected_slots": {...}}
    """
    template_key, template = match_template(question)
    if template is None:
        return {"status": "no_template_match"}

    slots = dict(known_slots or {})
    slots.update(extract_slots(question, template, extractor_fn))

    missing = [s for s in template["required_slots"] if s not in slots]
    if missing:
        return {
            "status": "missing_fields",
            "template": template_key,
            "missing_fields": missing,
            "missing_fields_prompts": {m: template["slot_prompts"][m] for m in missing},
            "collected_slots": slots,
        }

    document_text = template["template_text"].format(**slots)
    return {
        "status": "drafted",
        "template": template_key,
        "document_text": document_text,
        "collected_slots": slots,
    }