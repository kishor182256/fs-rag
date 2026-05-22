from app.schemas.agentic import AgentPlan


def _classify_intent(query: str) -> AgentPlan:
    q = query.lower()
    list_markers = ["list", "key highlights", "highlights", "winners", "countries joined", "participants"]
    explain_markers = ["explain", "brief about", "overview", "what is", "define", "describe"]

    if any(token in q for token in ["compare", "difference", "vs", "versus"]):
        intent = "comparison"
    elif any(token in q for token in list_markers):
        intent = "fact_lookup"
    elif any(token in q for token in explain_markers):
        intent = "multi_hop"
    elif any(token in q for token in ["why", "how", "impact", "analyse", "analyze"]):
        intent = "multi_hop"
    elif any(token in q for token in ["create", "send", "book", "schedule", "update"]):
        intent = "action_request"
    elif any(token in q for token in ["what", "when", "who", "where", "list"]):
        intent = "fact_lookup"
    else:
        intent = "unknown"

    sub_queries = [query.strip()]
    if intent == "comparison":
        sub_queries.append(f"key differences for: {query.strip()}")
    elif intent == "multi_hop":
        sub_queries.append(f"supporting evidence for: {query.strip()}")

    success_criteria = [
        "retrieve relevant chunks with citations",
        "answer only from retrieved evidence",
    ]
    if intent in {"comparison", "multi_hop"}:
        success_criteria.append("cover all requested facets without hallucination")

    return AgentPlan(intent=intent, sub_queries=sub_queries, success_criteria=success_criteria)


def build_plan(query: str) -> AgentPlan:
    return _classify_intent(query)
