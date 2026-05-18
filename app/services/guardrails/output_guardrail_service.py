import re


UNSAFE_PATTERNS = [
    r"\bkill\b",
    r"\battack\b",
    r"\bhate\b",
]


def enforce_output_guardrails(
    answer: str,
    *,
    require_citations: bool,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    lowered = answer.lower()
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, lowered):
            issues.append("unsafe_content_detected")
            break

    if require_citations:
        # At least one citation marker must be present when an answer is returned.
        if not re.search(r"\(chunk_[A-Za-z0-9_-]+\s+p\d+-\d+\)", answer):
            issues.append("missing_required_citations")

    return len(issues) == 0, issues
