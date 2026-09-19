"""
classifier.py
Classifies support tickets by request type and category.

v3 upgrades:
- 'general' is no longer a fallback category — ambiguous tickets fall back to
  'technical/bug' (most common unclassified support type) so agents downstream
  can still route them, while truly zero-confidence tickets are escalated by agent.py
- Fraud fast-path requires >= 2 keyword hits (reduces single-word false positives)
- All classifiers return confidence_ratio for agent decision-making
- classify_category returns full all_scores dict for transparency
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Keyword tables
# ---------------------------------------------------------------------------

REQUEST_TYPE_KEYWORDS: dict[str, list[str]] = {
    "complaint": [
        "unhappy", "disappointed", "frustrated", "terrible", "worst", "bad",
        "unacceptable", "disgusting", "horrible", "poor", "awful", "angry",
        "complaint", "complain", "outraged", "ridiculous", "absurd",
        "dissatisfied", "let down", "never again", "not satisfied",
    ],
    "issue": [
        "error", "bug", "broken", "crash", "fail", "not working", "problem",
        "issue", "glitch", "unable", "cannot", "can't", "won't", "doesn't work",
        "down", "outage", "stuck", "frozen", "loading", "timeout", "fix",
        "not loading", "keeps crashing", "stopped working", "doesn't respond",
    ],
    "question": [
        "how", "what", "when", "where", "why", "which", "who", "can i",
        "is it", "do i", "should i", "does", "explain", "help me understand",
        "tell me", "information", "know", "clarify", "wondering", "curious",
        "would like to know", "could you", "please explain",
    ],
}

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "fraud/security": [
        "fraud", "scam", "hack", "hacked", "unauthorized", "stolen", "theft",
        "suspicious", "phishing", "identity", "breach", "security", "compromised",
        "account takeover", "not me", "didn't make", "unknown charge",
        "strange activity", "weird activity", "fraudulent", "impersonation",
        "spoofing", "social engineering", "credential stuffing", "data leak",
        "leaked", "someone else", "wasn't me", "suspicious login", "unrecognized",
    ],
    "billing/payment": [
        "charge", "charged", "billing", "bill", "invoice", "payment", "pay",
        "refund", "money", "price", "cost", "fee", "subscription", "plan",
        "overcharged", "double charge", "credit card", "debit", "receipt",
        "transaction", "amount", "due", "balance", "extra charge", "wrong amount",
        "cancel subscription", "renewal", "auto-renew", "pro-rated",
    ],
    "account/login": [
        "login", "log in", "sign in", "signin", "password", "reset", "forgot",
        "account", "username", "email", "access", "locked", "lock out",
        "two factor", "2fa", "otp", "verify", "verification", "profile",
        "register", "registration", "create account", "can't log in",
        "authentication", "credentials", "session", "logout",
    ],
    "technical/bug": [
        "error", "bug", "crash", "broken", "not working", "fail", "glitch",
        "slow", "performance", "load", "loading", "timeout", "page", "feature",
        "function", "app", "application", "website", "site", "server",
        "database", "api", "integration", "download", "upload", "install",
        "update", "version", "software", "system", "platform", "interface",
    ],
}

FRAUD_MIN_HITS = 2   # minimum keyword hits to trigger fraud fast-path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_with_matches(text: str, keywords: list[str]) -> tuple[int, float, list[str]]:
    """Returns (hit_count, confidence_ratio, matched_keywords)."""
    matched = [kw for kw in keywords if kw in text]
    count = len(matched)
    confidence = count / len(keywords) if keywords else 0.0
    return count, confidence, matched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_request_type(text: str) -> tuple[str, list[str], float]:
    """
    Returns (request_type, matched_keywords, confidence_ratio).

    Priority: issue > complaint > question.
    'issue' keywords are checked first because a ticket saying "error, not working"
    is unambiguously an issue even if it also contains question words like "how".
    Falls back to 'question' with 0.0 confidence when nothing matches.
    """
    # Issue takes priority — most actionable classification
    issue_kws = REQUEST_TYPE_KEYWORDS["issue"]
    i_count, i_conf, i_matched = _score_with_matches(text, issue_kws)
    if i_count > 0:
        return "issue", i_matched, i_conf

    # Complaint next
    complaint_kws = REQUEST_TYPE_KEYWORDS["complaint"]
    c_count, c_conf, c_matched = _score_with_matches(text, complaint_kws)
    if c_count > 0:
        return "complaint", c_matched, c_conf

    # Question as default when other signals exist
    question_kws = REQUEST_TYPE_KEYWORDS["question"]
    q_count, q_conf, q_matched = _score_with_matches(text, question_kws)
    if q_count > 0:
        return "question", q_matched, q_conf

    return "question", [], 0.0


def classify_category(text: str) -> tuple[str, list[str], float, dict[str, float]]:
    """
    Returns (category, matched_keywords, confidence_ratio, all_scores).

    Weighted scoring: raw hit count is the primary rank, confidence ratio breaks ties.
    Fraud fast-path fires only when >= FRAUD_MIN_HITS keywords match.
    Falls back to 'technical/bug' (not 'general') with 0.0 confidence when
    nothing matches — agent.py escalates truly ambiguous tickets from there.
    """
    all_scores:  dict[str, float]      = {}
    counts:      dict[str, int]        = {}
    confidences: dict[str, float]      = {}
    matches:     dict[str, list[str]]  = {}

    for cat, kws in CATEGORY_KEYWORDS.items():
        count, confidence, matched = _score_with_matches(text, kws)
        counts[cat]      = count
        confidences[cat] = confidence
        matches[cat]     = matched
        all_scores[cat]  = round(confidence, 4)

    # Fraud fast-path — only when confident (>= FRAUD_MIN_HITS)
    if counts["fraud/security"] >= FRAUD_MIN_HITS:
        return (
            "fraud/security",
            matches["fraud/security"],
            confidences["fraud/security"],
            all_scores,
        )

    # Weighted rank: primary = hit count, tie-break = confidence ratio
    non_fraud = {c: v for c, v in counts.items() if c != "fraud/security"}
    best_cat  = max(non_fraud, key=lambda c: (non_fraud[c], confidences[c]))

    if non_fraud[best_cat] > 0:
        return best_cat, matches[best_cat], confidences[best_cat], all_scores

    # Zero keyword matches — return technical/bug with 0.0 confidence
    # agent.py escalates when both cat_confidence and KB score are too low
    return "technical/bug", [], 0.0, all_scores