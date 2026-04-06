from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from cycle_budget import CycleBudget, CycleBudgetExhausted

from jump_types import (
    JumpSearchResult,
    PreStage1DiagnosticsBundle,
    PreStage1ExecutionResult,
    PreStage1QueryPlan,
    PreStage1State,
    RenderedJumpPacket,
)


@dataclass(frozen=True)
class PreStage1Dependencies:
    build_jump_search_queries: Callable[[dict, str, str], list[str]]
    jump_source_shaped_terms: Callable[[str, dict], list[str]]
    tokenize_query_terms: Callable[[str], list[str]]
    jump_solution_marker_count: Callable[[str], int]
    has_intervention_query_label: Callable[[list[str] | tuple[str, ...]], bool]
    jump_result_anchor_context: Callable[
        [dict, str, str, list[str]],
        tuple[set[str], list[str], set[str]],
    ]
    sanitize: Callable[[object], str]
    normalize_jump_result_host: Callable[[str], str]
    jump_result_mentions_source_domain: Callable[[str, str, str, str, str], bool]
    host_matches_jump_include_domains: Callable[[str, tuple[str, ...]], bool]
    classify_weak_jump_result: Callable[
        [str, str, str, list[str], set[str]],
        tuple[bool, dict[str, object]],
    ]
    build_jump_search_content: Callable[
        [list[dict], set[str], set[str]],
        tuple[str, list[dict], list[str], list[dict], bool, dict[str, object]],
    ]
    should_attempt_alternate_jump_retrieval: Callable[[list[dict], list[dict]], bool]
    build_alternate_jump_search_query: Callable[[dict, str, str, str], str | None]
    tavily_search: Callable[..., dict[str, Any]]
    increment_tavily_calls: Callable[[int], None]
    academic_jump_include_domains: tuple[str, ...]


def build_pre_stage1_query_plan(
    pattern: dict,
    source_domain: str,
    source_category: str,
    deps: PreStage1Dependencies,
) -> PreStage1QueryPlan:
    queries = deps.build_jump_search_queries(
        pattern,
        source_domain,
        source_category,
    )
    query_labels = [
        str(label).strip()
        for label in (
            getattr(deps.build_jump_search_queries, "last_query_labels", []) or []
        )
        if str(label).strip()
    ]
    while len(query_labels) < len(queries):
        index = len(query_labels)
        if index == 0:
            query_labels.append("base")
        elif index == 1:
            query_labels.append("solution-biased")
        else:
            query_labels.append(f"variant-{index + 1}")

    built_jump_query = queries[0] if queries else ""
    transferable_query_profile = dict(
        getattr(
            deps.build_jump_search_queries,
            "last_transferable_query_profile",
            {},
        )
        or {}
    )
    source_shape_terms = deps.jump_source_shaped_terms(built_jump_query, pattern)
    return PreStage1QueryPlan(
        built_jump_query=built_jump_query,
        built_jump_queries=list(queries),
        built_jump_query_labels=query_labels[: len(queries)],
        legacy_built_jump_query=str(
            getattr(deps.build_jump_search_queries, "last_legacy_query", "") or ""
        ).strip(),
        transferable_query_profile=transferable_query_profile,
        query_collision_guard_applied=bool(
            getattr(
                deps.build_jump_search_queries,
                "last_collision_guard_applied",
                False,
            )
        ),
        transferable_fallback_gate_blocked=not bool(
            transferable_query_profile.get("usable")
        ),
        transferable_used_but_source_shaped=bool(
            transferable_query_profile.get("usable")
        )
        and len(source_shape_terms) >= 2,
        transferable_used_source_shape_terms=source_shape_terms[:4],
    )


def run_pre_stage1(
    pattern: dict,
    source_domain: str,
    source_category: str,
    *,
    cycle_budget: CycleBudget | None = None,
    deps: PreStage1Dependencies,
) -> PreStage1ExecutionResult:
    raw_search_query = str(pattern.get("search_query", "") or "").strip()
    query_plan = build_pre_stage1_query_plan(
        pattern,
        source_domain,
        source_category,
        deps,
    )
    diagnostics = PreStage1DiagnosticsBundle.from_query_plan(
        raw_search_query=raw_search_query,
        query_plan=query_plan,
    )
    if not query_plan.built_jump_query:
        return PreStage1ExecutionResult(
            diagnostics=diagnostics,
            stage1_outcome="no_results",
            stage1_failure_hint="empty_jump_query",
        )

    state = _build_pre_stage1_state(
        pattern,
        source_domain,
        source_category,
        query_plan,
        deps,
    )

    for index, current_query in enumerate(query_plan.built_jump_queries):
        query_label = (
            query_plan.built_jump_query_labels[index]
            if index < len(query_plan.built_jump_query_labels)
            else f"variant-{index + 1}"
        )
        raw_results = _execute_search(
            query=current_query,
            callsite="stage1_search",
            failure_message=(
                f"  [!] Tavily search failed for jump query '{current_query}': {{error}}"
            ),
            cycle_budget=cycle_budget,
            deps=deps,
        )
        if isinstance(raw_results, CycleBudgetExhausted):
            return _budget_exhausted_result(diagnostics, raw_results)
        if raw_results is None:
            state.query_error_count += 1
            continue
        state.general_result_count += len(raw_results)
        _merge_search_results(state, raw_results, query_label, deps)

    academic_results = _execute_search(
        query=query_plan.built_jump_query,
        callsite="stage1_academic_search",
        failure_message=(
            "  [!] Tavily academic jump search failed for query "
            f"'{query_plan.built_jump_query}': {{error}}"
        ),
        cycle_budget=cycle_budget,
        deps=deps,
        include_domains=deps.academic_jump_include_domains,
    )
    if isinstance(academic_results, CycleBudgetExhausted):
        return _budget_exhausted_result(diagnostics, academic_results)
    if academic_results is None:
        state.query_error_count += 1
        academic_results = []
    state.academic_result_count = len(academic_results)
    _merge_search_results(
        state,
        academic_results,
        "academic",
        deps,
        include_domains=deps.academic_jump_include_domains,
    )
    print(
        f"[Jump] general_results={state.general_result_count} "
        f"academic_results={state.academic_result_count}"
    )

    diagnostics.general_result_count = state.general_result_count
    diagnostics.academic_result_count = state.academic_result_count
    diagnostics.filtered_result_count = state.filtered_result_count
    diagnostics.filtered_result_reason_counts = dict(state.filtered_result_reason_counts)

    if state.query_error_count == len(query_plan.built_jump_queries) + 1:
        return PreStage1ExecutionResult(
            diagnostics=diagnostics,
            stage1_outcome="no_results",
            stage1_failure_hint="search_error",
        )

    _refresh_rendered_packet(state, diagnostics, deps)
    if (
        state.packet is not None
        and deps.should_attempt_alternate_jump_retrieval(
            state.merged_results,
            state.packet.clustered_results,
        )
    ):
        alternate_query = deps.build_alternate_jump_search_query(
            pattern,
            source_domain,
            source_category,
            query_plan.built_jump_query,
        )
        if alternate_query and alternate_query not in query_plan.built_jump_queries:
            diagnostics.alternate_retrieval_attempted = True
            diagnostics.alternate_jump_query = alternate_query
            alternate_results = _execute_search(
                query=alternate_query,
                callsite="stage1_alternate_search",
                failure_message=(
                    "  [!] Tavily alternate jump search failed for query "
                    f"'{alternate_query}': {{error}}"
                ),
                cycle_budget=cycle_budget,
                deps=deps,
            )
            if isinstance(alternate_results, CycleBudgetExhausted):
                return _budget_exhausted_result(diagnostics, alternate_results)
            if alternate_results is None:
                alternate_results = []
            state.alternate_result_count = len(alternate_results)
            diagnostics.alternate_result_count = state.alternate_result_count
            _merge_search_results(state, alternate_results, "alternate", deps)
            _refresh_rendered_packet(state, diagnostics, deps)

    if state.packet is None or not state.packet.search_content.strip():
        return PreStage1ExecutionResult(
            diagnostics=diagnostics,
            stage1_outcome="no_results",
            stage1_failure_hint="no_usable_results",
        )

    diagnostics.benchmark_snapshot = {
        "source_domain": source_domain,
        "source_category": source_category,
        "pattern_name": str(pattern.get("pattern_name", "") or "").strip() or "Unknown",
        "abstract_structure": str(pattern.get("abstract_structure", "") or "").strip(),
        "built_jump_query": query_plan.built_jump_query,
        "search_results": state.packet.search_content,
    }
    return PreStage1ExecutionResult(
        diagnostics=diagnostics,
        state=state,
    )


def augment_pre_stage1_with_query(
    state: PreStage1State,
    diagnostics: PreStage1DiagnosticsBundle,
    *,
    query: str,
    general_query_label: str,
    academic_query_label: str,
    general_callsite: str,
    academic_callsite: str,
    general_failure_message: str,
    academic_failure_message: str,
    cycle_budget: CycleBudget | None = None,
    deps: PreStage1Dependencies,
) -> PreStage1ExecutionResult:
    general_results = _execute_search(
        query=query,
        callsite=general_callsite,
        failure_message=general_failure_message,
        cycle_budget=cycle_budget,
        deps=deps,
    )
    if isinstance(general_results, CycleBudgetExhausted):
        return _budget_exhausted_result(diagnostics, general_results)
    if general_results is None:
        general_results = []
    state.general_result_count += len(general_results)
    _merge_search_results(state, general_results, general_query_label, deps)

    academic_results = _execute_search(
        query=query,
        callsite=academic_callsite,
        failure_message=academic_failure_message,
        cycle_budget=cycle_budget,
        deps=deps,
        include_domains=deps.academic_jump_include_domains,
    )
    if isinstance(academic_results, CycleBudgetExhausted):
        return _budget_exhausted_result(diagnostics, academic_results)
    if academic_results is None:
        academic_results = []
    state.academic_result_count += len(academic_results)
    _merge_search_results(
        state,
        academic_results,
        academic_query_label,
        deps,
        include_domains=deps.academic_jump_include_domains,
    )

    _refresh_rendered_packet(state, diagnostics, deps)
    return PreStage1ExecutionResult(
        diagnostics=diagnostics,
        state=state,
    )


def _build_pre_stage1_state(
    pattern: dict,
    source_domain: str,
    source_category: str,
    query_plan: PreStage1QueryPlan,
    deps: PreStage1Dependencies,
) -> PreStage1State:
    blocked_cluster_tokens = set(deps.tokenize_query_terms(source_domain))
    blocked_cluster_tokens.update(deps.tokenize_query_terms(source_category))
    _blocked_anchor_tokens, preferred_anchor_phrases, strong_anchor_tokens = (
        deps.jump_result_anchor_context(
            pattern,
            source_domain,
            source_category,
            query_plan.built_jump_queries,
        )
    )
    return PreStage1State(
        source_domain=source_domain,
        source_category=source_category,
        query_plan=query_plan,
        blocked_cluster_tokens=blocked_cluster_tokens,
        preferred_anchor_phrases=preferred_anchor_phrases,
        strong_anchor_tokens=strong_anchor_tokens,
    )


def _budget_exhausted_result(
    diagnostics: PreStage1DiagnosticsBundle,
    exhausted: CycleBudgetExhausted,
) -> PreStage1ExecutionResult:
    return PreStage1ExecutionResult(
        diagnostics=diagnostics,
        stage1_outcome="budget_exhausted_pre_stage1",
        stage1_failure_hint="cycle_budget_exhausted",
        budget_stop=exhausted.to_diagnostic(),
    )


def _execute_search(
    *,
    query: str,
    callsite: str,
    failure_message: str,
    cycle_budget: CycleBudget | None,
    deps: PreStage1Dependencies,
    include_domains: tuple[str, ...] | None = None,
) -> list[dict] | None | CycleBudgetExhausted:
    try:
        if cycle_budget is not None:
            cycle_budget.consume_tavily(
                outcome="budget_exhausted_pre_stage1",
                callsite=callsite,
            )
        search_kwargs: dict[str, object] = {
            "query": query,
            "max_results": 5,
            "include_answer": False,
            "search_depth": "basic",
        }
        if include_domains:
            search_kwargs["include_domains"] = list(include_domains)
        results = deps.tavily_search(**search_kwargs)
        deps.increment_tavily_calls(1)
    except CycleBudgetExhausted as exhausted:
        return exhausted
    except Exception as error:
        print(failure_message.format(error=error))
        return None

    raw_results = results.get("results", [])
    if not isinstance(raw_results, list):
        return []
    return raw_results


def _excerpt_rank(text: str, labels: list[str], deps: PreStage1Dependencies) -> tuple[int, int, int, int, int]:
    tokens = deps.tokenize_query_terms(text)
    return (
        int(deps.jump_solution_marker_count(text)),
        len(set(tokens)),
        len(tokens),
        len(text),
        1 if deps.has_intervention_query_label(labels) else 0,
    )


def _merge_search_results(
    state: PreStage1State,
    raw_results: list[dict],
    query_label: str,
    deps: PreStage1Dependencies,
    *,
    include_domains: tuple[str, ...] | None = None,
) -> None:
    category_lower = state.source_category.lower()
    for result in raw_results:
        title_text = str(result.get("title", "") or "").strip()
        title = title_text.lower()
        content = result.get("content", "")
        if category_lower and category_lower in title:
            continue
        clean = deps.sanitize(content)
        if not clean:
            continue
        url = str(result.get("url", "") or "").strip()
        normalized_host = deps.normalize_jump_result_host(url)
        if deps.jump_result_mentions_source_domain(
            title_text,
            clean,
            url,
            normalized_host,
            state.source_domain,
        ):
            continue
        if include_domains and not deps.host_matches_jump_include_domains(
            normalized_host,
            include_domains,
        ):
            continue
        should_drop, weak_result_context = deps.classify_weak_jump_result(
            title_text,
            url,
            clean,
            state.preferred_anchor_phrases,
            state.strong_anchor_tokens,
        )
        if should_drop:
            filtered_key = (url or title_text or clean).lower()
            existing_reason_codes = state.filtered_result_reason_keys.setdefault(
                filtered_key,
                set(),
            )
            if not existing_reason_codes:
                state.filtered_result_count += 1
            for reason_code in weak_result_context.get("reason_codes") or []:
                if reason_code in existing_reason_codes:
                    continue
                existing_reason_codes.add(reason_code)
                state.filtered_result_reason_counts[reason_code] = (
                    state.filtered_result_reason_counts.get(reason_code, 0) + 1
                )
            continue
        merged_result = _build_merged_search_result(
            title_text,
            clean,
            url,
            query_label,
            weak_result_context,
        )
        dedupe_key = (url or title_text or clean).lower()
        existing_index = state.merged_result_index.get(dedupe_key)
        if existing_index is None:
            state.merged_result_index[dedupe_key] = len(state.merged_results)
            state.merged_results.append(merged_result)
            continue
        existing_result = state.merged_results[existing_index]
        existing_labels = list(existing_result.get("query_labels") or [])
        incoming_labels = [query_label]
        incoming_rank = _excerpt_rank(clean, incoming_labels, deps)
        existing_rank = _excerpt_rank(
            str(existing_result.get("clean", "") or ""),
            existing_labels,
            deps,
        )
        if incoming_rank > existing_rank:
            existing_result["clean"] = clean
            if title_text:
                existing_result["title_text"] = title_text
            existing_result["anchor_overlap"] = int(
                merged_result.get("anchor_overlap") or 0
            )
            existing_result["preferred_phrase_match"] = bool(
                merged_result.get("preferred_phrase_match")
            )
            existing_result["solution_marker_count"] = int(
                merged_result.get("solution_marker_count") or 0
            )
            existing_result["adjacent_strength"] = int(
                merged_result.get("adjacent_strength") or 0
            )
            existing_result["intervention_marker_count"] = int(
                merged_result.get("intervention_marker_count") or 0
            )
            existing_result["intervention_evidence"] = bool(
                merged_result.get("intervention_evidence")
            )
            existing_result["intervention_signal"] = str(
                merged_result.get("intervention_signal") or ""
            ).strip()
            existing_result["triage_class"] = str(
                merged_result.get("triage_class") or "keep"
            ).strip() or "keep"
        else:
            existing_result["adjacent_strength"] = max(
                int(existing_result.get("adjacent_strength") or 0),
                int(merged_result.get("adjacent_strength") or 0),
            )
            existing_result["intervention_marker_count"] = max(
                int(existing_result.get("intervention_marker_count") or 0),
                int(merged_result.get("intervention_marker_count") or 0),
            )
            existing_result["intervention_evidence"] = bool(
                existing_result.get("intervention_evidence")
            ) or bool(merged_result.get("intervention_evidence"))
            if str(merged_result.get("intervention_signal") or "").strip() and not str(
                existing_result.get("intervention_signal") or ""
            ).strip():
                existing_result["intervention_signal"] = str(
                    merged_result.get("intervention_signal") or ""
                ).strip()
            existing_triage = str(existing_result.get("triage_class") or "keep").strip() or "keep"
            incoming_triage = str(merged_result.get("triage_class") or "keep").strip() or "keep"
            if existing_triage != "keep":
                existing_result["triage_class"] = (
                    "keep" if incoming_triage == "keep" else "adjacent"
                )
        existing_labels = list(existing_result.get("query_labels") or [])
        if query_label not in existing_labels:
            existing_labels.append(query_label)
            existing_result["query_labels"] = existing_labels


def _build_merged_search_result(
    title_text: str,
    clean: str,
    url: str,
    query_label: str,
    weak_result_context: dict[str, object],
) -> JumpSearchResult:
    return {
        "title_text": title_text,
        "clean": clean,
        "url": url,
        "query_labels": [query_label],
        "anchor_overlap": int(weak_result_context.get("anchor_overlap") or 0),
        "preferred_phrase_match": bool(
            weak_result_context.get("preferred_phrase_match")
        ),
        "solution_marker_count": int(
            weak_result_context.get("solution_marker_count") or 0
        ),
        "adjacent_strength": int(weak_result_context.get("adjacent_strength") or 0),
        "intervention_marker_count": int(
            weak_result_context.get("intervention_marker_count") or 0
        ),
        "intervention_evidence": bool(
            weak_result_context.get("intervention_evidence")
        ),
        "intervention_signal": str(
            weak_result_context.get("intervention_signal") or ""
        ).strip(),
        "triage_class": str(weak_result_context.get("triage_class") or "keep").strip()
        or "keep",
    }


def _refresh_rendered_packet(
    state: PreStage1State,
    diagnostics: PreStage1DiagnosticsBundle,
    deps: PreStage1Dependencies,
) -> None:
    (
        search_content,
        raw_target_candidates,
        top_titles,
        clustered_results,
        enriched_packet,
        packet_observability,
    ) = deps.build_jump_search_content(
        state.merged_results,
        state.blocked_cluster_tokens,
        state.strong_anchor_tokens,
    )
    state.packet = RenderedJumpPacket(
        search_content=search_content,
        raw_target_candidates=raw_target_candidates,
        top_titles=top_titles,
        clustered_results=clustered_results,
        enriched_packet=enriched_packet,
        packet_observability=packet_observability,
    )
    diagnostics.general_result_count = state.general_result_count
    diagnostics.academic_result_count = state.academic_result_count
    diagnostics.alternate_result_count = state.alternate_result_count
    diagnostics.filtered_result_count = state.filtered_result_count
    diagnostics.filtered_result_reason_counts = dict(state.filtered_result_reason_counts)
    diagnostics.highlighted_evidence_count = int(
        packet_observability.get("highlighted_evidence_count") or 0
    )
    diagnostics.adjacent_highlighted_count = int(
        packet_observability.get("adjacent_highlighted_count") or 0
    )
    diagnostics.adjacent_suppressed_count = int(
        packet_observability.get("adjacent_suppressed_count") or 0
    )
    diagnostics.packet_quality = (
        str(packet_observability.get("packet_quality") or "focused").strip()
        or "focused"
    )
    diagnostics.intervention_promoted_result_count = sum(
        1 for result in state.merged_results if result.get("intervention_evidence")
    )
    adjacent_result_count = sum(
        1 for result in state.merged_results if result.get("triage_class") == "adjacent"
    )
    diagnostics.adjacent_result_count = adjacent_result_count
    diagnostics.adjacent_retained_result_count = adjacent_result_count
    diagnostics.retained_adjacent_result_count = adjacent_result_count
    diagnostics.result_count = len(state.merged_results)
    diagnostics.cluster_count = len(clustered_results)
    diagnostics.top_cluster_hints = [
        str(cluster.get("cluster_hint", "") or "").strip()
        for cluster in clustered_results[:3]
        if str(cluster.get("cluster_hint", "") or "").strip()
    ]
    diagnostics.top_cluster_intervention_scores = [
        int(cluster.get("intervention_score") or 0)
        for cluster in clustered_results[:3]
    ]
    diagnostics.top_result_titles = list(top_titles)
    diagnostics.enriched_packet = enriched_packet
