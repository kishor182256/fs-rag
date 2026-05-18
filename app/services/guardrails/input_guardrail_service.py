import re

from app.schemas.agentic import GuardrailResult


PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"disregard\s+all\s+above",
    r"system\s+prompt",
    r"developer\s+message",
    r"jailbreak",
]

PII_PATTERNS = {
    "email_detected": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone_detected": r"\b(?:\+?\d{1,3}[- ]?)?\d{10}\b",
    "aadhaar_like_detected": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "pan_like_detected": r"\b[A-Z]{5}\d{4}[A-Z]\b",
}


def check_input_guardrails(query: str) -> GuardrailResult:
    lowered = query.lower()
    risk_flags: list[str] = []

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            risk_flags.append("prompt_injection_or_jailbreak")
            return GuardrailResult(
                allowed=False,
                sanitized_query=query,
                risk_flags=risk_flags,
                action="block",
            )

    for flag, pattern in PII_PATTERNS.items():
        if re.search(pattern, query):
            risk_flags.append(flag)

    sanitized_query = query.strip()
    action = "allow"
    if risk_flags:
        # Soft guardrail: allow factual QA but keep risk tags for audit and output control.
        action = "sanitize"
        sanitized_query = re.sub(r"\s+", " ", sanitized_query)

    return GuardrailResult(
        allowed=True,
        sanitized_query=sanitized_query,
        risk_flags=sorted(set(risk_flags)),
        action=action,
    )
