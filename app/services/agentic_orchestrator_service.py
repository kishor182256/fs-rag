import re
import logging

from app.schemas.agentic import (
    AgentStep,
    AgenticQueryRequest,
    AgenticQueryResponse,
    EvidenceItem,
    ImageReference,
    ModelOutput,
    ResponseMetadata,
    ResponseSource,
)
from app.services.agents.critic_agent import evaluate_answer
from app.services.agents.planner_agent import build_plan
from app.services.agents.retriever_agent import gather_evidence
from app.services.agents.synthesizer_agent import generate_answer
from app.services.guardrails.input_guardrail_service import check_input_guardrails
from app.services.guardrails.output_guardrail_service import enforce_output_guardrails
from app.services.guardrails.retrieval_guardrail_service import apply_retrieval_guardrails
from app.services.retrieval_types import RetrievalCandidate

logger = logging.getLogger(__name__)


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


def _citations_for_response(answer: str, evidence: list[EvidenceItem], limit: int) -> list[str]:
    cited_ids = re.findall(r"\(((?:chunk|image)_[A-Za-z0-9_-]+\s+p\d+(?:-\d+)?)\)", answer or "")
    if cited_ids:
        unique: list[str] = []
        seen: set[str] = set()
        for cited in cited_ids:
            marker = f"({cited})"
            if marker not in seen:
                unique.append(marker)
                seen.add(marker)

        # Collapse redundant citations for the same chunk/image id by keeping the widest page span.
        # Example: keep (chunk_00004 p5-8) and drop (chunk_00004 p5).
        parsed_pattern = re.compile(r"^\(((?:chunk|image)_[A-Za-z0-9_-]+)\s+p(\d+)(?:-(\d+))?\)$")
        selected: list[str] = []
        by_id: dict[str, tuple[int, int, int]] = {}
        # value: id -> (position_in_selected, start_page, end_page)
        for marker in unique:
            parsed = parsed_pattern.match(marker)
            if not parsed:
                selected.append(marker)
                continue

            cite_id = parsed.group(1)
            start = int(parsed.group(2))
            end = int(parsed.group(3) or parsed.group(2))
            current = by_id.get(cite_id)
            if current is None:
                by_id[cite_id] = (len(selected), start, end)
                selected.append(marker)
                continue

            pos, old_start, old_end = current
            old_span = old_end - old_start
            new_span = end - start
            if new_span > old_span or (new_span == old_span and start < old_start):
                selected[pos] = marker
                by_id[cite_id] = (pos, start, end)

        return selected[: max(1, limit)]

    return [e.citation for e in evidence[:limit]]


def _guardrail_fallback_answer(
    query: str,
    candidates: list[RetrievalCandidate],
    max_lines: int = 5,
    response_format: str = "auto",
) -> str:
    if not candidates:
        return "Insufficient local evidence."

    def _focused_snippet(text: str, query_text: str, limit: int = 220) -> str:
        clean = re.sub(r"\s+", " ", (text or "")).strip().replace("|", " ")
        clean = re.sub(r"\s{2,}", " ", clean).strip()
        if not clean:
            return ""

        query_terms = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", query_text)
            if len(token) > 2
        }
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        scored: list[tuple[int, str]] = []
        for sentence in sentences:
            sentence_terms = {
                token.lower()
                for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", sentence)
                if len(token) > 2
            }
            score = len(query_terms & sentence_terms)
            if score > 0:
                scored.append((score, sentence.strip()))

        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            best = scored[0][1]
        else:
            best = clean[:limit]

        if len(best) > limit:
            best = best[: limit - 3] + "..."
        return best

    format_mode = (response_format or "auto").strip().lower()
    if format_mode == "table":
        lines = ["| Item | Evidence | Citation |", "|---|---|---|"]
        for idx, candidate in enumerate(candidates[: max(1, max_lines)], start=1):
            snippet = _focused_snippet(candidate.snippet or candidate.text, query, limit=180)
            lines.append(f"| {idx} | {snippet} | ({candidate.chunk_id} p{candidate.page_start}-{candidate.page_end}) |")
        return "\n".join(lines)

    lowered = (query or "").lower()
    list_style = format_mode == "points" or any(marker in lowered for marker in ["list", "key highlights", "highlights", "points"])
    lines: list[str] = []

    if list_style:
        lines.append("Key Points")
    else:
        lines.extend(["Basic Info", ""])

    for candidate in candidates[: max(1, max_lines)]:
        snippet = _focused_snippet(candidate.snippet or candidate.text, query, limit=220)
        lines.append(f"- {snippet} ({candidate.chunk_id} p{candidate.page_start}-{candidate.page_end})")

    if not list_style:
        lines.extend(
            [
                "",
                "Conclusion",
                "The above points are extracted from retrieved document evidence and citations.",
            ]
        )
    return "\n".join(lines)


def _build_sources(evidence: list[EvidenceItem], limit: int) -> list[ResponseSource]:
    sources: list[ResponseSource] = []
    seen: set[str] = set()
    for item in evidence[: max(1, limit)]:
        pages = (
            str(item.page_start)
            if item.page_start == item.page_end
            else f"{item.page_start}-{item.page_end}"
        )
        key = f"{item.source_file}::{pages}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(ResponseSource(document=item.source_file, pages=pages))
    return sources


def _build_image_references(evidence: list[EvidenceItem], limit: int) -> list[ImageReference]:
    refs: list[ImageReference] = []
    seen: set[str] = set()
    for item in evidence:
        if item.modality != "image":
            continue
        key = f"{item.source_file}:{item.page_start}:{item.image_name or ''}"
        if key in seen:
            continue
        seen.add(key)
        refs.append(
            ImageReference(
                document=item.source_file,
                page=item.page_start,
                image_name=item.image_name,
                image_path=item.image_path,
                caption=item.snippet,
                citation=item.citation,
            )
        )
        if len(refs) >= max(1, limit):
            break
    return refs


def _retrieval_method(use_vector: bool, vector_status: str) -> str:
    if use_vector and vector_status == "used":
        return "hybrid_vector_bm25"
    if use_vector and vector_status in {"unavailable", "error"}:
        return "bm25_fallback"
    if use_vector:
        return "vector_requested"
    return "bm25"


def _response_metadata(
    *,
    model: str | None,
    use_vector: bool,
    vector_status: str,
    grounded: bool,
) -> ResponseMetadata:
    return ResponseMetadata(
        model=model,
        retrieval_method=_retrieval_method(use_vector=use_vector, vector_status=vector_status),
        grounded=grounded,
    )


async def run_agentic_query(request: AgenticQueryRequest) -> AgenticQueryResponse:
    steps: list[AgentStep] = []
    include_debug = False
    include_evidence = False
    include_steps = False
    try:
        include_debug, include_evidence, include_steps = _mode_flags(request.response_mode)

        input_guardrails = check_input_guardrails(request.query)
        if not input_guardrails.allowed:
            _append_steps(steps, "input_guardrails", "blocked", "Hard guardrail blocked the query.")
            return AgenticQueryResponse(
                query=request.query,
                status="blocked",
                final_answer="Query blocked by security guardrails.",
                image_references=[],
                metadata=_response_metadata(
                    model=None,
                    use_vector=request.use_vector,
                    vector_status="disabled",
                    grounded=False,
                ),
                input_guardrails=input_guardrails if include_debug else None,
                steps=steps if include_steps else None,
            )
        _append_steps(steps, "input_guardrails", "ok", f"Input guardrail action: {input_guardrails.action}.")

        plan = build_plan(input_guardrails.sanitized_query)
        _append_steps(steps, "planner_agent", "ok", f"Intent: {plan.intent}; sub-queries: {len(plan.sub_queries)}.")

        working_request = request.model_copy(update={"query": input_guardrails.sanitized_query})
        candidates, vector_status = await gather_evidence(working_request)
        guarded_candidates, retrieval_report = apply_retrieval_guardrails(candidates, query=working_request.query)
        if not retrieval_report.allowed:
            _append_steps(steps, "retrieval_guardrails", "blocked", "All evidence blocked by retrieval guardrails.")
            return AgenticQueryResponse(
                query=request.query,
                status="abstained",
                final_answer="Insufficient trusted evidence after retrieval guardrails.",
                planner=plan if include_debug else None,
                input_guardrails=input_guardrails if include_debug else None,
                retrieval_guardrails=retrieval_report if include_debug else None,
                sources=[],
                image_references=[],
                metadata=_response_metadata(
                    model=None,
                    use_vector=request.use_vector,
                    vector_status=vector_status,
                    grounded=False,
                ),
                steps=steps if include_steps else None,
                vector_status=vector_status,
            )
        if retrieval_report.conflicts:
            _append_steps(steps, "retrieval_guardrails", "warn", "; ".join(retrieval_report.conflicts))
        else:
            _append_steps(steps, "retrieval_guardrails", "ok", "Evidence passed retrieval guardrails.")

        evidence = _as_evidence_items(guarded_candidates)
        provider_override = None
        if request.primary_provider != "auto":
            provider_override = request.primary_provider
        elif request.model_provider != "auto":
            provider_override = request.model_provider
        primary_model_override = (request.primary_model or "").strip() or None
        answer, answer_status, answer_model = await generate_answer(
            query=working_request.query,
            candidates=guarded_candidates,
            use_llm=working_request.use_llm,
            response_format=working_request.response_format,
            provider_override=provider_override,
            model_override=primary_model_override,
        )
        primary_provider = provider_override or "auto"
        model_outputs: list[ModelOutput] = [
            ModelOutput(
                provider=primary_provider,
                model=answer_model or primary_model_override,
                status=answer_status,
                answer=answer or "",
            )
        ]
        secondary_model_override = (request.secondary_model or "").strip() or None
        if request.compare_models and secondary_model_override:
            secondary_provider_override = provider_override
            if request.secondary_provider != "auto":
                secondary_provider_override = request.secondary_provider
            alt_answer, alt_status, alt_model = await generate_answer(
                query=working_request.query,
                candidates=guarded_candidates,
                use_llm=working_request.use_llm,
                response_format=working_request.response_format,
                provider_override=secondary_provider_override,
                model_override=secondary_model_override,
            )
            model_outputs.append(
                ModelOutput(
                    provider=secondary_provider_override or "auto",
                    model=alt_model or secondary_model_override,
                    status=alt_status,
                    answer=alt_answer or "",
                )
            )
            if alt_status == "generated" and answer_status != "generated":
                answer = alt_answer
                answer_status = alt_status
                answer_model = alt_model or secondary_model_override
        _append_steps(steps, "synthesizer_agent", "ok", f"Initial synthesis status: {answer_status}.")

        critic = evaluate_answer(
            answer=answer,
            candidates=guarded_candidates,
            require_citations=request.require_citations,
            query=working_request.query,
        )
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
            guarded_candidates, retrieval_report = apply_retrieval_guardrails(retry_candidates, query=working_request.query)
            evidence = _as_evidence_items(guarded_candidates)
            answer, answer_status, answer_model = await generate_answer(
                query=working_request.query,
                candidates=guarded_candidates,
                use_llm=working_request.use_llm,
                response_format=working_request.response_format,
                provider_override=provider_override,
                model_override=primary_model_override,
            )
            critic = evaluate_answer(
                answer=answer,
                candidates=guarded_candidates,
                require_citations=request.require_citations,
                query=working_request.query,
            )

        output_ok, output_issues = enforce_output_guardrails(answer, require_citations=request.require_citations)
        if not output_ok:
            if "unsafe_content_detected" not in output_issues:
                fallback_answer = _guardrail_fallback_answer(
                    query=working_request.query,
                    candidates=guarded_candidates,
                    max_lines=min(request.top_k, 6),
                    response_format=working_request.response_format,
                )
                fallback_ok, fallback_issues = enforce_output_guardrails(
                    fallback_answer,
                    require_citations=request.require_citations,
                )
                if fallback_ok:
                    answer = fallback_answer
                    answer_model = answer_model or "deterministic_fallback"
                    critic = evaluate_answer(
                        answer=answer,
                        candidates=guarded_candidates,
                        require_citations=request.require_citations,
                        query=working_request.query,
                    )
                    _append_steps(
                        steps,
                        "output_guardrails",
                        "warn",
                        f"Applied citation-safe fallback due to: {', '.join(output_issues)}",
                    )
                else:
                    _append_steps(
                        steps,
                        "output_guardrails",
                        "blocked",
                        f"Output blocked after fallback: {', '.join(fallback_issues)}",
                    )
                    return AgenticQueryResponse(
                        query=request.query,
                        status="blocked",
                        final_answer="Answer blocked by output guardrails.",
                        planner=plan if include_debug else None,
                        input_guardrails=input_guardrails if include_debug else None,
                        retrieval_guardrails=retrieval_report if include_debug else None,
                        critic=critic if include_debug else None,
                        evidence=evidence if include_evidence else None,
                        sources=_build_sources(evidence, request.top_k),
                        image_references=_build_image_references(evidence, request.top_k),
                        metadata=_response_metadata(
                            model=answer_model,
                            use_vector=request.use_vector,
                            vector_status=vector_status,
                            grounded=False,
                        ),
                        citations=_citations_for_response(answer, evidence, request.top_k),
                        steps=steps if include_steps else None,
                        answer_model=answer_model,
                        model_outputs=model_outputs if request.compare_models else None,
                        vector_status=vector_status,
                    )
            else:
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
                    sources=_build_sources(evidence, request.top_k),
                    image_references=_build_image_references(evidence, request.top_k),
                    metadata=_response_metadata(
                        model=answer_model,
                        use_vector=request.use_vector,
                        vector_status=vector_status,
                        grounded=False,
                    ),
                    citations=_citations_for_response(answer, evidence, request.top_k),
                    steps=steps if include_steps else None,
                    answer_model=answer_model,
                    model_outputs=model_outputs if request.compare_models else None,
                    vector_status=vector_status,
                )

        if output_ok:
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
                sources=_build_sources(evidence, request.top_k),
                image_references=_build_image_references(evidence, request.top_k),
                metadata=_response_metadata(
                    model=answer_model,
                    use_vector=request.use_vector,
                    vector_status=vector_status,
                    grounded=False,
                ),
                citations=_citations_for_response(answer, evidence, request.top_k),
                steps=steps if include_steps else None,
                answer_model=answer_model,
                model_outputs=model_outputs if request.compare_models else None,
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
            sources=_build_sources(evidence, request.top_k),
            image_references=_build_image_references(evidence, request.top_k),
            metadata=_response_metadata(
                model=answer_model,
                use_vector=request.use_vector,
                vector_status=vector_status,
                grounded=bool(critic.passed),
            ),
            citations=_citations_for_response(answer, evidence, request.top_k),
            steps=steps if include_steps else None,
            answer_model=answer_model,
            model_outputs=model_outputs if request.compare_models else None,
            vector_status=vector_status,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("agentic_orchestrator_unhandled_error query=%s", (request.query or "")[:160])
        err_type = type(exc).__name__
        err_msg = str(exc).strip()
        if len(err_msg) > 220:
            err_msg = err_msg[:220] + "..."
        detail = f"Unhandled error: {err_type}"
        if err_msg:
            detail = f"{detail} - {err_msg}"
        _append_steps(steps, "orchestrator", "error", detail)
        return AgenticQueryResponse(
            query=request.query,
            status="failed",
            final_answer="Query processing failed due to an internal error. Please retry.",
            sources=[],
            image_references=[],
            metadata=_response_metadata(
                model=None,
                use_vector=request.use_vector,
                vector_status="error",
                grounded=False,
            ),
            citations=[],
            steps=steps if include_steps else None,
            vector_status="error",
        )
