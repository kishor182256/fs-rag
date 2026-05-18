from dataclasses import dataclass
import re


MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

TOPIC_KEYWORDS = {
    "banking": {"rbi", "bank", "repo rate", "npa", "ifsc", "basel"},
    "economy": {"gdp", "inflation", "fiscal", "budget", "deficit", "index"},
    "government_schemes": {"scheme", "yojana", "mission", "portal", "beneficiary"},
    "international": {"imf", "world bank", "un", "g20", "asean", "brics"},
    "awards": {"award", "prize", "medal", "honour", "recipient"},
    "sports": {"olympics", "cup", "trophy", "championship", "ipl"},
    "science_tech": {"isro", "ai", "quantum", "satellite", "launch", "nasa"},
}

ENTITY_PATTERN = re.compile(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b")


@dataclass
class EnrichedMetadata:
    months: list[str]
    topics: list[str]
    entities: list[str]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())


def extract_metadata(text: str) -> EnrichedMetadata:
    lowered = text.lower()
    tokens = set(_tokenize(text))

    months = sorted({month for month in MONTHS if month in tokens})

    topics: list[str] = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            topics.append(topic)

    entities = sorted({match.group(0) for match in ENTITY_PATTERN.finditer(text)})
    if len(entities) > 20:
        entities = entities[:20]

    return EnrichedMetadata(months=months, topics=topics, entities=entities)
