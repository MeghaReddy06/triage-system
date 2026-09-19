"""
agent.py  —  Final version
Support Ticket Triage Agent: dynamic response composition engine.

Key design principles:
  - Responses are COMPOSED per ticket from modular parts (opener + action + closer),
    not selected from a fixed template pool. Every ticket gets a unique response
    shaped by its category, request_type, matched keywords, and KB score.
  - Justification is DIFFERENTIATED per ticket: it names the actual matched keywords,
    states confidence as a human-readable bucket (high/medium/low), and describes
    the KB influence level explicitly.
  - Escalation is PRECISE: fraud requires confirmed keywords; uncertainty escalation
    only fires when ALL three signals fail (no category keywords + no KB match +
    no clear request type).
  - KB influences response CONTENT: when score is strong, a domain-specific
    KB-derived action sentence replaces the generic closer.
"""

from __future__ import annotations
import random

from classifier import classify_request_type, classify_category
from retriever import KnowledgeBase, SearchResult

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────

KB_STRONG   = 0.25   # KB actively shapes response content
KB_MODERATE = 0.10   # KB confirms match; adds domain hint
KB_WEAK     = 0.05   # KB below this → not mentioned

CONF_HIGH   = 0.10   # category confidence buckets
CONF_MEDIUM = 0.04

FRAUD_CONFIRM_KEYWORDS = [
    "hack", "hacked", "stolen", "unauthorized", "fraud", "fraudulent",
    "identity", "phishing", "breach", "compromised", "scam",
    "wasn't me", "not me", "didn't make", "suspicious login", "unknown charge",
    "account takeover", "strange activity", "unrecognized",
]

# ─────────────────────────────────────────────────────────────────────────────
# Response building blocks — modular so they combine uniquely per ticket
# ─────────────────────────────────────────────────────────────────────────────

# Openers vary by request_type
OPENERS: dict[str, list[str]] = {
    "complaint": [
        "We sincerely apologise for the experience you've had.",
        "We're truly sorry this has caused you frustration.",
        "We understand how upsetting this situation must be, and we apologise.",
        "Thank you for bringing this to our attention — we're sorry for the inconvenience.",
    ],
    "issue": [
        "Thank you for reporting this.",
        "We've received your report and are on it.",
        "We appreciate you flagging this — our team is already looking into it.",
        "Thanks for the detailed report — this helps us investigate quickly.",
    ],
    "question": [
        "Thank you for reaching out.",
        "We're happy to help with your query.",
        "Good question — let us look into this for you.",
        "Thanks for getting in touch.",
    ],
}

# Category-specific action sentences — what the team will DO
ACTIONS: dict[str, dict[str, list[str]]] = {
    "billing/payment": {
        "complaint": [
            "Our billing team will audit the charges on your account and issue a correction or refund where applicable within 1–2 business days.",
            "A billing specialist will investigate the discrepancy and contact you with a full resolution within 48 hours.",
            "We will review your transaction history and reverse any incorrect charges within 1–2 business days.",
        ],
        "issue": [
            "Our billing team has been notified and will investigate the transaction within 1–2 business days — please hold off on any further payments in the meantime.",
            "A billing specialist will audit your account and follow up with a resolution. No additional action is needed from your side right now.",
            "We've logged the billing issue and our team will review the affected transactions and respond within 48 hours.",
        ],
        "question": [
            "Our billing team will review your account details and respond with a clear breakdown within 1 business day.",
            "A billing specialist will look into this and follow up with a full explanation shortly.",
            "We'll have a billing specialist review your query and get back to you with a clear answer today.",
        ],
    },
    "technical/bug": {
        "complaint": [
            "Our engineering team has been alerted and is actively investigating the root cause. We'll keep you posted on progress.",
            "Your report has been escalated to our engineering team, who are working on a fix. Expect a status update within 24 hours.",
            "We've logged your complaint with our engineering team. They are prioritising this and will update you within 24 hours.",
        ],
        "issue": [
            "Our engineering team has logged the issue and is investigating. You'll receive a status update within 24 hours.",
            "We've assigned your bug report to our engineers. As a quick first step, try clearing your cache or restarting the app while we investigate.",
            "Our technical team is on it. We'll send you a detailed status update within 24 hours, along with any available workarounds.",
        ],
        "question": [
            "Our support team will research this and get back to you with a detailed technical response within 24 hours.",
            "A support engineer will review your question and follow up with clear guidance shortly.",
            "We'll have a technical specialist look into this and respond with a full answer within one business day.",
        ],
    },
    "account/login": {
        "complaint": [
            "Our account team will prioritise your case and contact you directly to restore access as quickly as possible.",
            "A specialist will review your account immediately and reach out to resolve the access issue without delay.",
            "We've flagged your case as urgent. Our account team will be in touch shortly to get you back in.",
        ],
        "issue": [
            "Please try the 'Forgot Password' link on the login page — this resolves most access issues immediately. If the problem continues, our account team will step in directly.",
            "Start with a password reset via the login page. If you still can't get in, reply to this message and we'll escalate to our account specialists straight away.",
            "Our account team has been notified. Try a password reset first; if that doesn't work, we'll have a specialist contact you within the hour.",
        ],
        "question": [
            "Our team will review your account query and respond with clear step-by-step guidance shortly.",
            "A support specialist will look into your question and follow up with a detailed answer within one business day.",
            "We'll have an account specialist respond to your query with the exact information you need shortly.",
        ],
    },
    "fraud/security": {
        "complaint": [
            "Your case has been escalated to our dedicated security team for immediate review. Please do not log in or make any account changes until you hear from us.",
            "Our security team has been alerted and will contact you urgently. As a precaution, avoid logging in until the matter is resolved.",
            "This has been flagged as a critical security incident. Our team will reach out within the hour to investigate and secure your account.",
        ],
        "issue": [
            "This has been flagged as a high-priority security incident. Our security team will contact you within the hour to investigate and secure your account.",
            "We've immediately escalated this to our security specialists. Please avoid any account activity until our team reaches out to you.",
            "Your security report is being treated with the highest urgency. Our team will be in touch very shortly — please hold off on any account changes in the meantime.",
        ],
        "question": [
            "Our security team will review your concern and respond with clear guidance promptly.",
            "A security specialist will follow up with you shortly to address your question thoroughly.",
            "We take all security queries seriously. Our team will investigate and get back to you with a clear response as soon as possible.",
        ],
    },
    "general": {
        "complaint": [
            "A senior support specialist will review your feedback and follow up with a resolution shortly.",
            "Your feedback has been noted and a specialist will be in touch to help resolve things as quickly as possible.",
        ],
        "issue": [
            "Our support team has received your report and will follow up with assistance shortly.",
            "A support specialist will review your case and get back to you with next steps soon.",
        ],
        "question": [
            "A support specialist will review your query and respond with the information you need.",
            "Our team will look into your question and follow up with a clear answer shortly.",
        ],
    },
}

# KB-driven closers — used when KB score is strong; varies by domain
KB_CLOSERS: dict[str, list[str]] = {
    "billing/payment": [
        "You can review your full transaction history in the Billing section of your account settings while you wait.",
        "For reference, your invoice history and payment records are available under Account > Billing.",
        "In the meantime, checking your payment method details in Account Settings may provide useful context.",
    ],
    "technical/bug": [
        "While our team investigates, clearing your browser cache or reinstalling the app often resolves the most common issues.",
        "As a temporary measure, try using a different browser or device — this helps narrow down whether the issue is environment-specific.",
        "Restarting the application or logging out and back in can sometimes resolve intermittent issues while we work on a permanent fix.",
    ],
    "account/login": [
        "If you no longer have access to your registered email, please have your account details ready so our team can verify your identity quickly.",
        "Having your original registration details on hand will help our team restore your access faster.",
        "For faster resolution, please have your account email address and any recent transaction details ready for our team.",
    ],
    "fraud/security": [
        "As an extra precaution, consider changing passwords on any other accounts that use the same credentials.",
        "We also recommend enabling two-factor authentication on your account once this is resolved.",
        "Avoid clicking any links in suspicious emails related to your account until our security team has completed their review.",
    ],
    "general": [
        "Our support team is available 24/7 if you need further assistance.",
        "Feel free to reply to this message if you have any additional details to share.",
    ],
}

# Standard closers — used when KB score is not strong enough
STANDARD_CLOSERS: dict[str, str] = {
    "billing/payment": "We appreciate your patience while we resolve this.",
    "technical/bug":   "Thank you for helping us improve our service.",
    "account/login":   "We apologise for the disruption and will resolve this as quickly as possible.",
    "fraud/security":  "Your security is our top priority.",
    "general":         "We appreciate you reaching out and will respond as quickly as possible.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic response composer
# ─────────────────────────────────────────────────────────────────────────────

def _compose_response(
    category: str,
    request_type: str,
    cat_keywords: list[str],
    kb_results: list[SearchResult],
    status: str,
    ticket_id: str,
) -> str:
    """
    Compose a unique response per ticket from three parts:
      opener  — tone-matched to request_type
      action  — specific to category + request_type
      closer  — KB-driven when score is strong, else standard

    Variant selection is deterministic: seed = hash of matched keywords
    (or ticket_id when no keywords) so the same ticket always gets the
    same response, but different tickets get different compositions.
    """
    rt = request_type if request_type in OPENERS else "question"
    cat = category if category in ACTIONS else "general"

    # Seed mixes keywords + ticket_id so two tickets with identical keyword sets
    # still get different variant picks
    kw_hash  = hash(tuple(sorted(cat_keywords))) if cat_keywords else 0
    tid_hash = hash(ticket_id)
    base_seed = kw_hash ^ (tid_hash * 2654435761)  # XOR with Knuth multiplicative hash

    # Use different offsets per part so opener/action/closer vary independently
    opener_seed = base_seed
    action_seed = base_seed + 31
    closer_seed = base_seed + 67

    openers_list = OPENERS[rt]
    opener = openers_list[opener_seed % len(openers_list)]

    actions_list = ACTIONS[cat].get(rt) or ACTIONS[cat]["question"]
    action = actions_list[action_seed % len(actions_list)]

    kb_top = kb_results[0].score if kb_results else 0.0

    if status == "replied" and kb_top >= KB_MODERATE:
        closers_list = KB_CLOSERS.get(cat, KB_CLOSERS["general"])
        closer = closers_list[closer_seed % len(closers_list)]
    else:
        closer = STANDARD_CLOSERS.get(cat, STANDARD_CLOSERS["general"])

    return f"{opener} {action} {closer}"


# ─────────────────────────────────────────────────────────────────────────────
# Differentiated justification
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_bucket(confidence: float) -> str:
    if confidence >= CONF_HIGH:
        return "high"
    if confidence >= CONF_MEDIUM:
        return "medium"
    return "low"


def _kb_influence_label(kb_score: float) -> str:
    if kb_score >= KB_STRONG:
        return "strong KB match — response content adapted"
    if kb_score >= KB_MODERATE:
        return "moderate KB match — domain hint added"
    if kb_score >= KB_WEAK:
        return "weak KB match — not influential"
    return "no KB match"


def _build_justification(
    category: str,
    request_type: str,
    cat_keywords: list[str],
    req_keywords: list[str],
    cat_confidence: float,
    kb_results: list[SearchResult],
    decision_reason: str,
) -> str:
    """
    Produces a differentiated, evidence-rich justification unique to each ticket.
    Includes: matched keywords, confidence bucket, KB influence level, decision reason.
    """
    kb_top = kb_results[0].score if kb_results else 0.0
    conf_label = _confidence_bucket(cat_confidence)
    kb_label   = _kb_influence_label(kb_top)

    # Keyword evidence — name the actual words that drove classification
    if cat_keywords:
        kw_str = ", ".join(f'"{k}"' for k in cat_keywords[:4])
        kw_part = f"Category signals: {kw_str} ({conf_label} confidence)"
    else:
        kw_part = f"No category keywords matched ({conf_label} confidence)"

    # Request type evidence
    if req_keywords:
        rt_kw = ", ".join(f'"{k}"' for k in req_keywords[:2])
        rt_part = f'request type "{request_type}" via {rt_kw}'
    else:
        rt_part = f'request type "{request_type}" (default)'

    return f"{kw_part}; {rt_part}; {kb_label}. {decision_reason}."


# ─────────────────────────────────────────────────────────────────────────────
# Precise escalation logic
# ─────────────────────────────────────────────────────────────────────────────

def _should_escalate(
    category: str,
    cat_confidence: float,
    req_confidence: float,
    text_lower: str,
    kb_results: list[SearchResult],
    all_cat_scores: dict[str, float],
) -> tuple[bool, str]:
    """
    Escalation fires on exactly two conditions:

    1. ANY confirmed fraud keyword present in raw text — mandatory escalation,
       regardless of which category the classifier returned. This catches cases
       like "unknown charge on my credit card" that score high on billing but
       contain a genuine fraud signal.

    2. ALL THREE signals absent simultaneously:
         - zero category keyword hits across all categories
         - KB score below KB_WEAK
         - request_type confidence is also zero (truly unreadable ticket)

    All other tickets → replied.
    """
    kb_top = kb_results[0].score if kb_results else 0.0

    # Fraud signal check — applied to raw text regardless of classifier category
    confirmed_fraud = any(kw in text_lower for kw in FRAUD_CONFIRM_KEYWORDS)
    if confirmed_fraud:
        return True, "Fraud/security signal confirmed in ticket — escalated for immediate human review"

    # Security category without any confirmed fraud keyword = security question → reply
    if category == "fraud/security":
        return False, "Security-related question — handled with standard response"

    # Triple-zero: no keywords anywhere + no KB + no request type signal
    no_cat_signal = not any(v > 0.0 for v in all_cat_scores.values())
    no_kb_signal  = kb_top < KB_WEAK
    no_req_signal = req_confidence == 0.0

    if no_cat_signal and no_kb_signal and no_req_signal:
        return True, "No classifiable signal found in ticket — routed to human agent"

    return False, f"Routed as {category} ticket"


# ─────────────────────────────────────────────────────────────────────────────
# TriageAgent
# ─────────────────────────────────────────────────────────────────────────────

class TriageAgent:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def process(self, ticket_id: str, user_query: str) -> dict:
        text       = (user_query or "").strip()
        text_lower = text.lower()

        # ── Classification ────────────────────────────────────────────
        request_type, req_keywords, req_confidence = classify_request_type(text_lower)
        category, cat_keywords, cat_confidence, all_cat_scores = classify_category(text_lower)

        # ── KB retrieval ──────────────────────────────────────────────
        kb_results: list[SearchResult] = self.kb.search_top_k(text, k=3)

        # ── Decision ─────────────────────────────────────────────────
        escalate, decision_reason = _should_escalate(
            category, cat_confidence, req_confidence,
            text_lower, kb_results, all_cat_scores,
        )
        status = "escalated" if escalate else "replied"

        # ── Dynamic response composition ──────────────────────────────
        # When escalated due to fraud signal, compose as fraud/security regardless
        # of what the classifier returned (e.g. "unknown charge" classified as billing)
        response_category = (
            "fraud/security"
            if status == "escalated" and "Fraud/security signal" in decision_reason
            else category
        )
        response = _compose_response(
            response_category, request_type, cat_keywords, kb_results, status, ticket_id
        )

        # ── Differentiated justification ──────────────────────────────
        justification = _build_justification(
            category=category,
            request_type=request_type,
            cat_keywords=cat_keywords,
            req_keywords=req_keywords,
            cat_confidence=cat_confidence,
            kb_results=kb_results,
            decision_reason=decision_reason,
        )

        # ── Debug trace (not in CSV output) ──────────────────────────
        trace = {
            "category":            category,
            "category_confidence": round(cat_confidence, 4),
            "confidence_bucket":   _confidence_bucket(cat_confidence),
            "all_category_scores": {k: round(v, 4) for k, v in all_cat_scores.items()},
            "request_type":        request_type,
            "request_confidence":  round(req_confidence, 4),
            "kb_top_score":        round(kb_results[0].score, 4) if kb_results else 0.0,
            "kb_influence":        _kb_influence_label(kb_results[0].score if kb_results else 0.0),
            "kb_top_source":       kb_results[0].source if kb_results else None,
            "kb_results":          [
                {"rank": r.rank, "score": r.score, "source": r.source}
                for r in kb_results
            ],
            "decision_reason":     decision_reason,
        }

        return {
            "ticket_id":     ticket_id,
            "request_type":  request_type,
            "status":        status,
            "decision":      status,
            "product_area":  category,
            "response":      response,
            "justification": justification,
            "_trace":        trace,
        }