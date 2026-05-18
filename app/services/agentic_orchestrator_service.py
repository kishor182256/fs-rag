from app.schemas.agentic import (
    AgentStep,
    AgenticQueryRequest,
    AgenticQueryResponse,
    EvidenceItem,
)
from app.services.agents.critic_agent import evaluate_answer
from app.services.agents.planner_agent import build_plan
from app.services.agents.retriever_agent import gather_evidence
from app.services.agents.synthesizer_agent import generate_answer
from app.services.guardrails.input_guardrail_service import check_input_guardrails
from app.services.guardrails.output_guardrail_service import enforce_output_guardrails
from app.services.guardrails.retrieval_guardrail_service import apply_retrieval_guardrails
from app.services.retrieval_types import RetrievalCandidate


def _mode_flags(mode: str) -> tuple[bool, bool, bool]:
    normalized = (mode or "compact").strip().lower()
    if normalized == "full":
        return True, True, True
    if normalized == "balanced":
        return True, False, True
    return False, False, False


def _as_evidence_items(candidates: list[RetrievalCandidate]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for candidate in candidates:
        items.append(
            EvidenceItem(
                chunk_id=candidate.chunk_id,
                source_file=candidate.source_file,
                page_start=candidate.page_start,
                page_end=candidate.page_end,
                score=round(candidate.rerank_score, 4),
                snippet=(candidate.snippet or candidate.text)[:1200],
                metadata={
                    "months": candidate.metadata.get("months", []),
                    "topics": candidate.metadata.get("topics", []),
                    "entities": candidate.metadata.get("entities", []),
                },
                modality="image" if candidate.modality == "image" else "text",
                image_path=candidate.image_path or None,
                image_name=candidate.image_name or None,
                citation=f"({candidate.chunk_id} p{candidate.page_start}-{candidate.page_end})",
            )
        )
    return items


def _append_steps(steps: list[AgentStep], step: str, status: str, detail: str) -> None:
    steps.append(AgentStep(step=step, status=status, detail=detail))


async def run_agentic_query(request: AgenticQueryRequest) -> AgenticQueryResponse:
    steps: list[AgentStep] = []
    include_debug, include_evidence, include_steps = _mode_flags(request.response_mode)

    input_guardrails = check_input_guardrails(request.query)
    if not input_guardrails.allowed:
        _append_steps(steps, "input_guardrails", "blocked", "Hard guardrail blocked the query.")
        return AgenticQueryResponse(
            query=request.query,
            status="blocked",
            final_answer="Query blocked by security guardrails.",
            input_guardrails=input_guardrails if include_debug else None,
            steps=steps if include_steps else None,
        )
    _append_steps(steps, "input_guardrails", "ok", f"Input guardrail action: {input_guardrails.action}.")

    plan = build_plan(input_guardrails.sanitized_query)
    _append_steps(steps, "planner_agent", "ok", f"Intent: {plan.intent}; sub-queries: {len(plan.sub_queries)}.")

    working_request = request.model_copy(update={"query": input_guardrails.sanitized_query})
    candidates, vector_status = await gather_evidence(working_request)
    guarded_candidates, retrieval_report = apply_retrieval_guardrails(candidates)
    if not retrieval_report.allowed:
        _append_steps(steps, "retrieval_guardrails", "blocked", "All evidence blocked by retrieval guardrails.")
        return AgenticQueryResponse(
            query=request.query,
            status="abstained",
            final_answer="Insufficient trusted evidence after retrieval guardrails.",
            planner=plan if include_debug else None,
            input_guardrails=input_guardrails if include_debug else None,
            retrieval_guardrails=retrieval_report if include_debug else None,
            steps=steps if include_steps else None,
            vector_status=vector_status,
        )
    if retrieval_report.conflicts:
        _append_steps(steps, "retrieval_guardrails", "warn", "; ".join(retrieval_report.conflicts))
    else:
        _append_steps(steps, "retrieval_guardrails", "ok", "Evidence passed retrieval guardrails.")

    evidence = _as_evidence_items(guarded_candidates)
    answer, answer_status, answer_model = await generate_answer(
        query=working_request.query,
        candidates=guarded_candidates,
        use_llm=working_request.use_llm,
    )
    _append_steps(steps, "synthesizer_agent", "ok", f"Initial synthesis status: {answer_status}.")

    critic = evaluate_answer(answer=answer, candidates=guarded_candidates, require_citations=request.require_citations)
    correction_loops = 0
    while not critic.passed and correction_loops < request.max_corrections:
        correction_loops += 1
        _append_steps(steps, "critic_agent", "warn", f"Critic requested retry: {', '.join(critic.issues)}")
        retry_request = working_request.model_copy(
            update={
                "top_k": min(20, working_request.top_k + 2),
                "vector_top_k": min(40, working_request.vector_top_k + 4),
            }
        )
        retry_candidates, vector_status = await gather_evidence(retry_request)
        guarded_candidates, retrieval_report = apply_retrieval_guardrails(retry_candidates)
        evidence = _as_evidence_items(guarded_candidates)
        answer, answer_status, answer_model = await generate_answer(
            query=working_request.query,
            candidates=guarded_candidates,
            use_llm=working_request.use_llm,
        )
        critic = evaluate_answer(
            answer=answer,
            candidates=guarded_candidates,
            require_citations=request.require_citations,
        )

    output_ok, output_issues = enforce_output_guardrails(answer, require_citations=request.require_citations)
    if not output_ok:
        _append_steps(steps, "output_guardrails", "blocked", f"Output blocked: {', '.join(output_issues)}")
        return AgenticQueryResponse(
            query=request.query,
            status="blocked",
            final_answer="Answer blocked by output guardrails.",
            planner=plan if include_debug else None,
            input_guardrails=input_guardrails if include_debug else None,
            retrieval_guardrails=retrieval_report if include_debug else None,
            critic=critic if include_debug else None,
            evidence=evidence if include_evidence else None,
            citations=[e.citation for e in evidence[: request.top_k]],
            steps=steps if include_steps else None,
            answer_model=answer_model,
            vector_status=vector_status,
        )

    _append_steps(steps, "output_guardrails", "ok", "Output passed safety and citation checks.")

    if not critic.passed:
        _append_steps(steps, "critic_agent", "warn", "Critic did not pass after correction budget.")
        return AgenticQueryResponse(
            query=request.query,
            status="abstained",
            final_answer="Insufficient confidence to provide a final answer.",
            planner=plan if include_debug else None,
            input_guardrails=input_guardrails if include_debug else None,
            retrieval_guardrails=retrieval_report if include_debug else None,
            critic=critic if include_debug else None,
            evidence=evidence if include_evidence else None,
            citations=[e.citation for e in evidence[: request.top_k]],
            steps=steps if include_steps else None,
            answer_model=answer_model,
            vector_status=vector_status,
        )

    _append_steps(steps, "critic_agent", "ok", "Critic accepted final answer.")
    return AgenticQueryResponse(
        query=request.query,
        status="completed",
        final_answer=answer,
        planner=plan if include_debug else None,
        input_guardrails=input_guardrails if include_debug else None,
        retrieval_guardrails=retrieval_report if include_debug else None,
        critic=critic if include_debug else None,
        evidence=evidence if include_evidence else None,
        citations=[e.citation for e in evidence[: request.top_k]],
        steps=steps if include_steps else None,
        answer_model=answer_model,
        vector_status=vector_status,
    )
