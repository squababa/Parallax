from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from cycle_budget import CycleBudget, CycleBudgetExhausted
from hypothesis_validation import (
    CORE_TARGET_BROAD_PAGE_MARKERS,
    CORE_TARGET_WEAK_SOURCE_MARKERS,
)

from jump_types import (
    JumpQueryBuildResult,
    JumpSearchResult,
    PreStage1DiagnosticsBundle,
    PreStage1ExecutionResult,
    PreStage1QueryPlan,
    PreStage1State,
    RenderedJumpPacket,
)
from jump_support import (
    GENERIC_QUERY_TOKENS,
    JUMP_TITLE_SIGNATURE_NOISE_TOKENS,
    MECHANISM_QUERY_TOKENS,
    OVERLOADED_JUMP_QUERY_TOKENS,
    QUERY_PHRASE_STOPWORDS,
    WEAK_QUERY_TOKENS,
    _build_jump_title_signature,
    _classify_jump_intervention_evidence,
    _host_matches_jump_include_domains,
    _is_specific_jump_query_token,
    _jump_query_support_context,
    _jump_result_mentions_source_domain,
    _jump_solution_marker_count,
    _jump_source_shaped_terms,
    _normalize_jump_result_host,
    _preferred_jump_query_anchor_phrases,
    _select_best_jump_anchor_phrase,
    _tokenize_query_terms,
)
from sanitize import sanitize as _sanitize


@dataclass(frozen=True)
class PreStage1Dependencies:
    # Production contract: query builders return JumpQueryBuildResult and
    # search execution flows through explicit external boundaries.
    build_jump_search_queries: Callable[[dict, str, str], JumpQueryBuildResult]
    tavily_search: Callable[..., dict[str, Any]]
    increment_tavily_calls: Callable[[int], None]
    academic_jump_include_domains: tuple[str, ...]


def _default_query_label(index: int) -> str:
    if index == 0:
        return "base"
    if index == 1:
        return "solution-biased"
    return f"variant-{index + 1}"


def _normalize_jump_query_build_result(
    query_build_output: JumpQueryBuildResult,
) -> JumpQueryBuildResult:
    built_jump_queries = list(query_build_output.built_jump_queries)
    built_jump_query_labels = [
        str(label).strip()
        for label in query_build_output.built_jump_query_labels
        if str(label).strip()
    ]
    built_jump_query = query_build_output.built_jump_query
    while len(built_jump_query_labels) < len(built_jump_queries):
        built_jump_query_labels.append(_default_query_label(len(built_jump_query_labels)))

    if built_jump_query is None and built_jump_queries:
        built_jump_query = built_jump_queries[0]

    return JumpQueryBuildResult(
        built_jump_query=built_jump_query,
        built_jump_queries=built_jump_queries,
        built_jump_query_labels=built_jump_query_labels[: len(built_jump_queries)],
        legacy_built_jump_query=str(
            query_build_output.legacy_built_jump_query or ""
        ).strip(),
        transferable_query_profile=dict(
            query_build_output.transferable_query_profile or {}
        ),
        query_collision_guard_applied=bool(
            query_build_output.query_collision_guard_applied
        ),
    )


def _build_pre_stage1_query_build_result(
    pattern: dict,
    source_domain: str,
    source_category: str,
    deps: PreStage1Dependencies,
) -> JumpQueryBuildResult:
    query_build_output = deps.build_jump_search_queries(
        pattern,
        source_domain,
        source_category,
    )
    if not isinstance(query_build_output, JumpQueryBuildResult):
        raise TypeError(
            "PreStage1Dependencies.build_jump_search_queries must return "
            "JumpQueryBuildResult."
        )
    return _normalize_jump_query_build_result(query_build_output)


def _has_intervention_query_label(labels: list[str] | tuple[str, ...]) -> bool:
    return any(
        str(label).strip() in {"solution-biased", "intervention-family"}
        for label in labels
        if str(label).strip()
    )


def _build_alternate_jump_search_query(
    pattern: dict,
    source_domain: str,
    source_category: str,
    base_query: str,
) -> str:
    clean_base_query = re.sub(r"\s+", " ", str(base_query or "").strip())
    _blocked_tokens, preferred_anchor_phrases, support_tokens = (
        _jump_query_support_context(
            pattern,
            source_domain,
            source_category,
        )
    )
    selected: list[str] = []
    covered_tokens: set[str] = set()
    base_tokens = set(_tokenize_query_terms(clean_base_query))

    def _append_part(part: str) -> None:
        normalized = re.sub(r"\s+", " ", str(part or "").strip().lower())
        if not normalized or normalized in selected:
            return
        selected.append(normalized)
        covered_tokens.update(_tokenize_query_terms(normalized))

    anchor_phrase = _select_best_jump_anchor_phrase(preferred_anchor_phrases)
    if anchor_phrase:
        _append_part(anchor_phrase)

    for token in support_tokens:
        if (
            token in covered_tokens
            or token in base_tokens
            or token in GENERIC_QUERY_TOKENS
            or token in WEAK_QUERY_TOKENS
            or token in OVERLOADED_JUMP_QUERY_TOKENS
            or len(token) <= 2
        ):
            continue
        _append_part(token)
        if len(_tokenize_query_terms(" ".join(selected))) >= 4:
            break

    for term in ("control", "mechanism", "workaround"):
        if term not in covered_tokens:
            _append_part(term)

    alternate_query = " ".join(_tokenize_query_terms(" ".join(selected))[:8]).strip()
    if len(_tokenize_query_terms(alternate_query)) < 4:
        fallback_tokens = [
            token
            for token in _tokenize_query_terms(clean_base_query)
            if (
                token not in GENERIC_QUERY_TOKENS
                and token not in WEAK_QUERY_TOKENS
                and token not in OVERLOADED_JUMP_QUERY_TOKENS
            )
        ]
        alternate_query = " ".join(
            (fallback_tokens[:4] + ["control", "mechanism", "workaround"])[:8]
        ).strip()
    if alternate_query == clean_base_query:
        alternate_query = f"{alternate_query} control workaround".strip()
    return alternate_query


def _should_attempt_alternate_jump_retrieval(
    merged_results: list[dict],
    clustered_results: list[dict],
) -> bool:
    if not merged_results or not clustered_results:
        return False

    top_cluster_results = list(clustered_results[0].get("results") or [])
    if not top_cluster_results:
        return False

    top_cluster_keep_hits = sum(
        1
        for result in top_cluster_results
        if str(result.get("triage_class") or "keep").strip() == "keep"
    )
    top_cluster_anchor_max = max(
        (int(result.get("anchor_overlap") or 0) for result in top_cluster_results),
        default=0,
    )
    top_cluster_intervention_hits = sum(
        1 for result in top_cluster_results if result.get("intervention_evidence")
    )
    total_keep_count = sum(
        1
        for result in merged_results
        if str(result.get("triage_class") or "keep").strip() == "keep"
    )
    adjacent_count = sum(
        1
        for result in merged_results
        if str(result.get("triage_class") or "keep").strip() == "adjacent"
    )
    thin_underanchored_top_cluster = (
        top_cluster_anchor_max < 2
        and top_cluster_intervention_hits == 0
        and top_cluster_keep_hits <= 1
    )
    low_coherence_top_cluster = (
        len(clustered_results) >= 2 and len(top_cluster_results) <= 1
    )
    underanchored_adjacent_packet = (
        adjacent_count > 0
        and len(merged_results) >= 2
        and total_keep_count <= 1
        and thin_underanchored_top_cluster
    )
    broad_underanchored_packet = (
        low_coherence_top_cluster
        and len(merged_results) >= 2
        and adjacent_count > 0
        and total_keep_count <= 1
        and thin_underanchored_top_cluster
    )
    return underanchored_adjacent_packet or broad_underanchored_packet


def _jump_result_anchor_context(
    pattern: dict,
    source_domain: str,
    source_category: str,
    queries: list[str],
) -> tuple[set[str], list[str], set[str]]:
    blocked_tokens = set(_tokenize_query_terms(source_domain))
    blocked_tokens.update(_tokenize_query_terms(source_category))
    preferred_anchor_phrases = _preferred_jump_query_anchor_phrases(
        pattern,
        blocked_tokens,
    )
    strong_anchor_tokens: set[str] = set()
    for text in (
        str(pattern.get("control_lever", "") or ""),
        str(pattern.get("abstract_structure", "") or ""),
        str(pattern.get("measurable_signal", "") or ""),
        str(pattern.get("pattern_name", "") or ""),
        str(pattern.get("transfer_rationale", "") or ""),
        *(str(query or "") for query in queries),
    ):
        for token in _tokenize_query_terms(text):
            if (
                token in blocked_tokens
                or token in GENERIC_QUERY_TOKENS
                or token in WEAK_QUERY_TOKENS
                or token in OVERLOADED_JUMP_QUERY_TOKENS
                or len(token) <= 2
                or not _is_specific_jump_query_token(token)
            ):
                continue
            strong_anchor_tokens.add(token)
    return blocked_tokens, preferred_anchor_phrases, strong_anchor_tokens


def _score_jump_result_anchor_overlap(
    title_text: str,
    url: str,
    clean: str,
    preferred_anchor_phrases: list[str],
    strong_anchor_tokens: set[str],
) -> tuple[int, bool]:
    combined_text = " ".join(
        part for part in (str(title_text or ""), str(clean or "")) if part
    ).lower()
    preferred_phrase_match = any(
        phrase and phrase in combined_text for phrase in preferred_anchor_phrases
    )
    text_tokens = set(_tokenize_query_terms(f"{title_text} {clean} {url}"))
    anchor_overlap = len(text_tokens.intersection(strong_anchor_tokens))
    return anchor_overlap, preferred_phrase_match


def _classify_weak_jump_result(
    title_text: str,
    url: str,
    clean: str,
    preferred_anchor_phrases: list[str],
    strong_anchor_tokens: set[str],
) -> tuple[bool, dict[str, object]]:
    reference_text = " ".join(
        part for part in (str(title_text or ""), str(url or "")) if part
    ).lower()
    weak_source = any(
        marker in reference_text for marker in CORE_TARGET_WEAK_SOURCE_MARKERS
    )
    broad_page = any(
        marker in reference_text for marker in CORE_TARGET_BROAD_PAGE_MARKERS
    )
    anchor_overlap, preferred_phrase_match = _score_jump_result_anchor_overlap(
        title_text,
        url,
        clean,
        preferred_anchor_phrases,
        strong_anchor_tokens,
    )
    solution_marker_count = _jump_solution_marker_count(clean)
    specificity_score = len(
        {
            token
            for token in _tokenize_query_terms(f"{title_text} {clean}")
            if (
                len(token) >= 5
                and token not in GENERIC_QUERY_TOKENS
                and token not in WEAK_QUERY_TOKENS
                and token not in JUMP_TITLE_SIGNATURE_NOISE_TOKENS
                and token not in QUERY_PHRASE_STOPWORDS
                and token
                not in {
                    "background",
                    "broad",
                    "context",
                    "generic",
                    "general",
                    "introduction",
                    "only",
                    "overview",
                    "performance",
                    "system",
                    "systems",
                    "tutorial",
                }
            )
        }
    )
    reliable_solution_evidence = solution_marker_count > 0 and specificity_score >= 4
    intervention_context = _classify_jump_intervention_evidence(
        title_text,
        clean,
        anchor_overlap=anchor_overlap,
        preferred_phrase_match=preferred_phrase_match,
        strong_anchor_tokens=strong_anchor_tokens,
        solution_marker_count=solution_marker_count,
        specificity_score=specificity_score,
    )
    intervention_marker_count = int(
        intervention_context.get("intervention_marker_count") or 0
    )
    intervention_evidence = bool(intervention_context.get("intervention_evidence"))
    adjacent_strength = min(anchor_overlap, 2)
    if preferred_phrase_match:
        adjacent_strength += 2
    adjacent_strength += min(solution_marker_count, 2)
    if specificity_score >= 4:
        adjacent_strength += 1
    if specificity_score >= 6:
        adjacent_strength += 1
    if intervention_evidence:
        adjacent_strength += 2
    adjacent_strength += min(intervention_marker_count, 2)
    strong_grounding_signal = intervention_evidence or reliable_solution_evidence
    adjacent_retained = (
        anchor_overlap < 2
        and not preferred_phrase_match
        and (
            ((weak_source or broad_page) and strong_grounding_signal)
            or (not weak_source and not broad_page and specificity_score >= 6)
        )
    )
    should_drop = (
        (weak_source or broad_page)
        and anchor_overlap < 2
        and not preferred_phrase_match
        and not strong_grounding_signal
    )
    triage_class = "drop" if should_drop else ("adjacent" if adjacent_retained else "keep")
    reason_codes: list[str] = []
    if should_drop:
        if weak_source:
            reason_codes.append("weak_source")
        if broad_page:
            reason_codes.append("broad_page")
        if anchor_overlap < 2:
            reason_codes.append("low_anchor_overlap")
    return should_drop, {
        "weak_source": weak_source,
        "broad_page": broad_page,
        "anchor_overlap": anchor_overlap,
        "preferred_phrase_match": preferred_phrase_match,
        "solution_marker_count": solution_marker_count,
        "specificity_score": specificity_score,
        "adjacent_strength": adjacent_strength,
        "triage_class": triage_class,
        "intervention_marker_count": intervention_marker_count,
        "intervention_evidence": intervention_evidence,
        "intervention_signal": str(
            intervention_context.get("intervention_signal") or ""
        ).strip(),
        "reason_codes": reason_codes,
    }


def _build_jump_search_content(
    merged_results: list[dict],
    blocked_cluster_tokens: set[str],
    strong_anchor_tokens: set[str],
) -> tuple[str, list[dict], list[str], list[dict], bool, dict[str, object]]:
    def _excerpt_rank(text: str, labels: list[str]) -> tuple[int, int, int, int, int]:
        tokens = _tokenize_query_terms(text)
        return (
            _jump_solution_marker_count(text),
            len(set(tokens)),
            len(tokens),
            len(text),
            1 if _has_intervention_query_label(labels) else 0,
        )

    def _clustered_result_rank(
        result: dict,
    ) -> tuple[int, int, int, int, int, int, int, int, int, int, int]:
        clean = str(result.get("clean", "") or "").strip()
        query_labels = [
            str(label).strip()
            for label in (result.get("query_labels") or [])
            if str(label).strip()
        ]
        title_signature = tuple(result.get("title_signature") or ())
        excerpt_rank = _excerpt_rank(clean, query_labels)
        intervention_evidence = bool(result.get("intervention_evidence"))
        triage_class = str(result.get("triage_class") or "keep").strip() or "keep"
        return (
            1 if triage_class == "keep" else 0,
            int(result.get("anchor_overlap") or 0),
            1 if result.get("preferred_phrase_match") else 0,
            1 if intervention_evidence else 0,
            int(result.get("intervention_marker_count") or 0)
            if intervention_evidence
            else 0,
            1 if _has_intervention_query_label(query_labels) else 0,
            int(result.get("solution_marker_count") or excerpt_rank[0]),
            len(title_signature),
            excerpt_rank[1],
            excerpt_rank[2],
            excerpt_rank[3],
        )

    def _is_adjacent_result(result: dict) -> bool:
        return str(result.get("triage_class") or "keep").strip() == "adjacent"

    def _adjacent_strength(result: dict) -> int:
        return int(result.get("adjacent_strength") or 0)

    def _stage_one_mechanism_evidence_rank(
        result: dict,
    ) -> tuple[int, int, int, int, int, int, int]:
        clean = str(result.get("clean", "") or "").strip()
        title_text = str(result.get("title_text", "") or "").strip()
        query_labels = [
            str(label).strip()
            for label in (result.get("query_labels") or [])
            if str(label).strip()
        ]
        token_set = set(_tokenize_query_terms(f"{title_text} {clean}"))
        mechanism_token_count = len(
            token_set.intersection(MECHANISM_QUERY_TOKENS)
        )
        anchor_token_count = len(token_set.intersection(strong_anchor_tokens))
        excerpt_rank = _excerpt_rank(clean, query_labels)
        triage_class = str(result.get("triage_class") or "keep").strip() or "keep"
        return (
            1 if triage_class == "keep" else 0,
            1 if result.get("preferred_phrase_match") else 0,
            int(result.get("anchor_overlap") or 0),
            anchor_token_count + mechanism_token_count,
            excerpt_rank[0],
            excerpt_rank[1],
            excerpt_rank[2],
        )

    def _stage_one_intervention_evidence_rank(
        result: dict,
    ) -> tuple[int, int, int, int, int, int, int]:
        clean = str(result.get("clean", "") or "").strip()
        query_labels = [
            str(label).strip()
            for label in (result.get("query_labels") or [])
            if str(label).strip()
        ]
        excerpt_rank = _excerpt_rank(clean, query_labels)
        return (
            1 if result.get("intervention_evidence") else 0,
            int(result.get("intervention_marker_count") or 0),
            int(result.get("solution_marker_count") or 0),
            1 if _has_intervention_query_label(query_labels) else 0,
            int(result.get("anchor_overlap") or 0),
            1 if result.get("preferred_phrase_match") else 0,
            excerpt_rank[1],
        )

    def _mechanism_highlight_predicate(result: dict) -> bool:
        return (
            bool(result.get("preferred_phrase_match"))
            or int(result.get("anchor_overlap") or 0) >= 2
            or _stage_one_mechanism_evidence_rank(result)[3] >= 3
        )

    def _intervention_highlight_predicate(result: dict) -> bool:
        return bool(result.get("intervention_evidence")) or int(
            result.get("solution_marker_count") or 0
        ) > 0

    def _operator_response_highlight_predicate(result: dict) -> bool:
        return (
            bool(str(result.get("intervention_signal", "") or "").strip())
            or bool(result.get("intervention_evidence"))
            or int(result.get("solution_marker_count") or 0) > 0
        )

    def _stage_one_packet_result_key(result: dict) -> str:
        return (
            str(result.get("url", "") or "").strip().lower()
            or str(result.get("title_text", "") or "").strip().lower()
            or str(result.get("clean", "") or "").strip().lower()
        )

    def _compress_adjacent_display_results(
        cluster_results: list[dict],
        *,
        adjacent_heavy_packet: bool,
    ) -> list[dict]:
        if not adjacent_heavy_packet:
            return cluster_results
        adjacent_results = [
            result for result in cluster_results if _is_adjacent_result(result)
        ]
        keep_results = [
            result for result in cluster_results if not _is_adjacent_result(result)
        ]
        if len(adjacent_results) < 4 or len(adjacent_results) < len(keep_results) + 2:
            return cluster_results

        selected_adjacent_keys: set[str] = set()

        def _select_adjacent(predicate, ranker) -> None:
            candidates = [
                result
                for result in adjacent_results
                if predicate(result)
                and _stage_one_packet_result_key(result) not in selected_adjacent_keys
            ]
            if not candidates:
                return
            best_result = max(candidates, key=ranker)
            selected_adjacent_keys.add(_stage_one_packet_result_key(best_result))

        _select_adjacent(
            _mechanism_highlight_predicate,
            lambda result: (
                _adjacent_strength(result),
                *_stage_one_mechanism_evidence_rank(result),
            ),
        )
        _select_adjacent(
            _intervention_highlight_predicate,
            lambda result: (
                _adjacent_strength(result),
                *_stage_one_intervention_evidence_rank(result),
            ),
        )
        _select_adjacent(
            _operator_response_highlight_predicate,
            lambda result: (
                1 if str(result.get("intervention_signal", "") or "").strip() else 0,
                _adjacent_strength(result),
                *_stage_one_intervention_evidence_rank(result),
            ),
        )
        remaining_adjacent = [
            result
            for result in adjacent_results
            if _stage_one_packet_result_key(result) not in selected_adjacent_keys
        ]
        if remaining_adjacent:
            best_remaining_adjacent = max(
                remaining_adjacent,
                key=lambda result: (
                    _adjacent_strength(result),
                    *_stage_one_intervention_evidence_rank(result),
                    *_stage_one_mechanism_evidence_rank(result),
                ),
            )
            if _adjacent_strength(best_remaining_adjacent) >= 4 or not selected_adjacent_keys:
                selected_adjacent_keys.add(
                    _stage_one_packet_result_key(best_remaining_adjacent)
                )

        return [
            result
            for result in cluster_results
            if not _is_adjacent_result(result)
            or _stage_one_packet_result_key(result) in selected_adjacent_keys
        ]

    def _stage_one_adjacent_fallback_rank(
        result: dict,
    ) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int]:
        return (
            _adjacent_strength(result),
            *_stage_one_intervention_evidence_rank(result),
            *_stage_one_mechanism_evidence_rank(result),
        )

    def _stage_one_adjacent_highlight_extra_rank(
        highlight: dict[str, object],
    ) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]:
        result = dict(highlight.get("result") or {})
        return (
            _adjacent_strength(result),
            len(
                [
                    label
                    for label in (highlight.get("labels") or [])
                    if str(label).strip()
                ]
            ),
            *_stage_one_intervention_evidence_rank(result),
            *_stage_one_mechanism_evidence_rank(result),
        )

    def _select_stage_one_evidence_highlights(
        cluster_results: list[dict],
    ) -> list[dict[str, object]]:
        highlight_order: list[str] = []
        highlights_by_key: dict[str, dict[str, object]] = {}

        def _pick_highlight(label: str, predicate, ranker) -> None:
            candidates = [result for result in cluster_results if predicate(result)]
            if not candidates:
                return
            best_result = max(candidates, key=ranker)
            key = _stage_one_packet_result_key(best_result)
            existing = highlights_by_key.get(key)
            if existing is None:
                highlight_order.append(key)
                highlights_by_key[key] = {
                    "result": best_result,
                    "labels": [label],
                }
                return
            labels = existing.setdefault("labels", [])
            if label not in labels:
                labels.append(label)

        _pick_highlight(
            "Mechanism evidence",
            _mechanism_highlight_predicate,
            _stage_one_mechanism_evidence_rank,
        )
        _pick_highlight(
            "Intervention/workaround evidence",
            _intervention_highlight_predicate,
            _stage_one_intervention_evidence_rank,
        )
        _pick_highlight(
            "Operator response evidence",
            _operator_response_highlight_predicate,
            lambda result: (
                1 if str(result.get("intervention_signal", "") or "").strip() else 0,
                *_stage_one_intervention_evidence_rank(result),
            ),
        )
        return [highlights_by_key[key] for key in highlight_order]

    clustered_results: list[dict] = []
    for merged_result in merged_results:
        title_signature = _build_jump_title_signature(
            str(merged_result.get("title_text", "") or ""),
            blocked_cluster_tokens,
        )
        normalized_host = _normalize_jump_result_host(str(merged_result.get("url", "") or ""))
        result_entry = {
            **merged_result,
            "title_signature": title_signature,
            "normalized_host": normalized_host,
        }
        matching_cluster: dict | None = None
        if title_signature:
            signature_set = set(title_signature)
            for cluster in clustered_results:
                cluster_signature = tuple(cluster.get("title_signature") or ())
                if cluster_signature and cluster_signature == title_signature:
                    matching_cluster = cluster
                    break
            if matching_cluster is None and normalized_host:
                for cluster in clustered_results:
                    cluster_signature = tuple(cluster.get("title_signature") or ())
                    if (
                        not cluster_signature
                        or str(cluster.get("normalized_host", "") or "") != normalized_host
                        or len(signature_set.intersection(cluster_signature)) < 2
                    ):
                        continue
                    matching_cluster = cluster
                    break
        if matching_cluster is None:
            clustered_results.append(
                {
                    "title_signature": title_signature,
                    "normalized_host": normalized_host,
                    "results": [result_entry],
                }
            )
        else:
            matching_cluster["results"].append(result_entry)

    for cluster in clustered_results:
        cluster_results = sorted(
            cluster.get("results") or [],
            key=_clustered_result_rank,
            reverse=True,
        )
        cluster["results"] = cluster_results
        cluster_signature = tuple(cluster.get("title_signature") or ())
        best_result = cluster_results[0] if cluster_results else {}
        cluster_hint = " ".join(cluster_signature).strip()
        if not cluster_hint:
            cluster_hint = str(best_result.get("title_text", "") or "").strip()
        if not cluster_hint:
            cluster_hint = str(cluster.get("normalized_host", "") or "").strip()
        cluster["cluster_hint"] = cluster_hint or "singleton result"
        marker_density = (
            sum(
                int(
                    result.get("solution_marker_count")
                    or _jump_solution_marker_count(str(result.get("clean", "") or ""))
                )
                for result in cluster_results
            )
            / max(len(cluster_results), 1)
        )
        anchor_overlap_total = sum(
            int(result.get("anchor_overlap") or 0) for result in cluster_results
        )
        preferred_phrase_matches = sum(
            1 for result in cluster_results if result.get("preferred_phrase_match")
        )
        keep_hits = sum(
            1
            for result in cluster_results
            if str(result.get("triage_class") or "keep").strip() == "keep"
        )
        intervention_hits = sum(
            1 for result in cluster_results if result.get("intervention_evidence")
        )
        intervention_score = sum(
            int(result.get("intervention_marker_count") or 0)
            for result in cluster_results
            if result.get("intervention_evidence")
        ) + intervention_hits * 2
        cluster_intervention_signal = next(
            (
                str(result.get("intervention_signal") or "").strip()
                for result in cluster_results
                if result.get("intervention_evidence")
                and str(result.get("intervention_signal") or "").strip()
            ),
            "",
        )
        cluster["intervention_score"] = intervention_score
        cluster["intervention_signal"] = cluster_intervention_signal
        best_result_rank = (
            _clustered_result_rank(best_result)
            if cluster_results
            else (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
        cluster["rank"] = (
            len(cluster_results),
            keep_hits,
            intervention_hits,
            intervention_score,
            anchor_overlap_total,
            preferred_phrase_matches,
            1
            if any(
                _has_intervention_query_label(result.get("query_labels") or [])
                for result in cluster_results
            )
            else 0,
            marker_density,
            best_result_rank[4],
            best_result_rank[5],
            best_result_rank[6],
            len(cluster_signature),
        )

    clustered_results.sort(
        key=lambda cluster: cluster.get("rank")
        or (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        reverse=True,
    )
    adjacent_result_count = sum(
        1 for result in merged_results if _is_adjacent_result(result)
    )
    keep_result_count = sum(
        1 for result in merged_results if not _is_adjacent_result(result)
    )
    adjacent_heavy_packet = (
        adjacent_result_count >= 4
        and adjacent_result_count >= keep_result_count + 2
    )

    search_content: list[str] = []
    raw_target_candidates: list[dict] = []
    top_titles: list[str] = []
    enriched_packet = False
    highlighted_evidence_count = 0
    adjacent_highlighted_count = 0
    adjacent_suppressed_count = 0
    cluster_packet_sections: list[dict[str, object]] = []

    for cluster in clustered_results:
        cluster_hint = str(cluster.get("cluster_hint", "") or "").strip() or "Unknown"
        cluster_results = list(cluster.get("results") or [])
        display_cluster_results = _compress_adjacent_display_results(
            cluster_results,
            adjacent_heavy_packet=adjacent_heavy_packet,
        )
        adjacent_suppressed_count += max(
            sum(1 for result in cluster_results if _is_adjacent_result(result))
            - sum(
                1
                for result in display_cluster_results
                if _is_adjacent_result(result)
            ),
            0,
        )
        fallback_results: list[dict] = []
        for merged_result in cluster_results:
            title_text = str(merged_result.get("title_text", "") or "").strip()
            clean = str(merged_result.get("clean", "") or "").strip()
            url = str(merged_result.get("url", "") or "").strip()
            source_reference = url or title_text
            raw_target_candidates.append(
                {
                    "target_excerpt": clean[:500],
                    "target_url": source_reference or None,
                    "evaluation_source_reference": " ".join(
                        part for part in (title_text, url) if part
                    )
                    or source_reference
                    or None,
                }
            )
            if title_text and title_text not in top_titles and len(top_titles) < 3:
                top_titles.append(title_text)
        cluster_packet_sections.append(
            {
                "cluster": cluster,
                "cluster_hint": cluster_hint,
                "cluster_results": cluster_results,
                "display_cluster_results": display_cluster_results,
                "highlights": _select_stage_one_evidence_highlights(
                    display_cluster_results
                ),
                "fallback_results": fallback_results,
            }
        )

    adjacent_highlight_candidates = [
        highlight
        for section in cluster_packet_sections
        for highlight in list(section.get("highlights") or [])
        if _is_adjacent_result(dict(highlight.get("result") or {}))
    ]
    adjacent_highlight_candidate_count = len(adjacent_highlight_candidates)
    if adjacent_heavy_packet and len(adjacent_highlight_candidates) > 3:
        selected_adjacent_highlights: dict[str, dict[str, object]] = {}

        def _adjacent_highlight_key(highlight: dict[str, object]) -> str:
            return _stage_one_packet_result_key(dict(highlight.get("result") or {}))

        def _pick_adjacent_highlight(label_text: str, ranker) -> None:
            candidates = [
                highlight
                for highlight in adjacent_highlight_candidates
                if label_text in (highlight.get("labels") or [])
                and _adjacent_highlight_key(highlight)
                not in selected_adjacent_highlights
            ]
            if not candidates:
                return
            best_highlight = max(candidates, key=ranker)
            selected_adjacent_highlights[_adjacent_highlight_key(best_highlight)] = (
                best_highlight
            )

        _pick_adjacent_highlight(
            "Mechanism evidence",
            lambda highlight: (
                _adjacent_strength(dict(highlight.get("result") or {})),
                *_stage_one_mechanism_evidence_rank(
                    dict(highlight.get("result") or {})
                ),
            ),
        )
        _pick_adjacent_highlight(
            "Intervention/workaround evidence",
            lambda highlight: (
                _adjacent_strength(dict(highlight.get("result") or {})),
                *_stage_one_intervention_evidence_rank(
                    dict(highlight.get("result") or {})
                ),
            ),
        )
        _pick_adjacent_highlight(
            "Operator response evidence",
            lambda highlight: (
                1
                if str(
                    dict(highlight.get("result") or {}).get("intervention_signal") or ""
                ).strip()
                else 0,
                _adjacent_strength(dict(highlight.get("result") or {})),
                *_stage_one_intervention_evidence_rank(
                    dict(highlight.get("result") or {})
                ),
            ),
        )
        remaining_adjacent_highlights = [
            highlight
            for highlight in adjacent_highlight_candidates
            if _adjacent_highlight_key(highlight) not in selected_adjacent_highlights
        ]
        if remaining_adjacent_highlights:
            best_remaining_adjacent_highlight = max(
                remaining_adjacent_highlights,
                key=_stage_one_adjacent_highlight_extra_rank,
            )
            if (
                _adjacent_strength(
                    dict(best_remaining_adjacent_highlight.get("result") or {})
                )
                >= 6
                or not selected_adjacent_highlights
            ):
                selected_adjacent_highlights[
                    _adjacent_highlight_key(best_remaining_adjacent_highlight)
                ] = best_remaining_adjacent_highlight

        kept_adjacent_highlight_keys = set(selected_adjacent_highlights)
        for section in cluster_packet_sections:
            section["highlights"] = [
                highlight
                for highlight in list(section.get("highlights") or [])
                if not _is_adjacent_result(dict(highlight.get("result") or {}))
                or _stage_one_packet_result_key(dict(highlight.get("result") or {}))
                in kept_adjacent_highlight_keys
            ]

    highlighted_evidence_count = sum(
        1
        for section in cluster_packet_sections
        for _highlight in list(section.get("highlights") or [])
    )
    adjacent_highlighted_count = sum(
        1
        for section in cluster_packet_sections
        for highlight in list(section.get("highlights") or [])
        if _is_adjacent_result(dict(highlight.get("result") or {}))
    )
    enriched_packet = any(
        section.get("highlights") for section in cluster_packet_sections
    )

    for section in cluster_packet_sections:
        highlighted_result_keys = {
            _stage_one_packet_result_key(dict(highlight.get("result") or {}))
            for highlight in list(section.get("highlights") or [])
        }
        section["fallback_results"] = [
            merged_result
            for merged_result in list(section.get("display_cluster_results") or [])
            if _stage_one_packet_result_key(merged_result) not in highlighted_result_keys
        ]

    adjacent_fallback_budget = 1 if adjacent_highlighted_count > 0 else 2
    adjacent_fallback_candidates = [
        result
        for section in cluster_packet_sections
        for result in list(section.get("fallback_results") or [])
        if _is_adjacent_result(result)
    ]
    if (
        adjacent_heavy_packet
        and len(adjacent_fallback_candidates) > adjacent_fallback_budget
    ):
        kept_adjacent_fallback_keys = {
            _stage_one_packet_result_key(result)
            for result in sorted(
                adjacent_fallback_candidates,
                key=_stage_one_adjacent_fallback_rank,
                reverse=True,
            )[:adjacent_fallback_budget]
        }
        for section in cluster_packet_sections:
            filtered_fallback_results: list[dict] = []
            for result in list(section.get("fallback_results") or []):
                if not _is_adjacent_result(result):
                    filtered_fallback_results.append(result)
                    continue
                if _stage_one_packet_result_key(result) in kept_adjacent_fallback_keys:
                    filtered_fallback_results.append(result)
                    continue
                adjacent_suppressed_count += 1
            section["fallback_results"] = filtered_fallback_results

    visible_cluster_sections = [
        section
        for section in cluster_packet_sections
        if section.get("highlights") or section.get("fallback_results")
    ]

    for cluster_index, section in enumerate(visible_cluster_sections, start=1):
        cluster = dict(section.get("cluster") or {})
        cluster_hint = str(section.get("cluster_hint", "") or "").strip() or "Unknown"
        cluster_results = list(section.get("cluster_results") or [])
        highlights = list(section.get("highlights") or [])
        fallback_results = list(section.get("fallback_results") or [])
        search_content.append(f"Candidate cluster {cluster_index}:")
        search_content.append(f"Cluster hint: {cluster_hint}")
        search_content.append(f"Supporting results: {len(cluster_results)}")
        if str(cluster.get("intervention_signal", "") or "").strip():
            search_content.append(
                "Intervention signal: "
                f"{str(cluster.get('intervention_signal') or '').strip()}"
            )
        elif int(cluster.get("intervention_score") or 0) > 0:
            search_content.append("Intervention evidence: yes")
        for highlight in highlights:
            highlighted_result = dict(highlight.get("result") or {})
            title_text = str(highlighted_result.get("title_text", "") or "").strip()
            clean = str(highlighted_result.get("clean", "") or "").strip()
            url = str(highlighted_result.get("url", "") or "").strip()
            query_label_text = ", ".join(
                str(label).strip()
                for label in (highlighted_result.get("query_labels") or [])
                if str(label).strip()
            )
            label_texts = [
                str(label).strip()
                for label in (highlight.get("labels") or [])
                if str(label).strip()
            ]
            for label_text in label_texts:
                search_content.append(f"{label_text}:")
            if query_label_text:
                search_content.append(f"Retrieved via: {query_label_text}")
            search_content.append(f"Title: {title_text or 'Unknown'}")
            if url:
                search_content.append(f"URL: {url}")
            search_content.append(f"Snippet: {clean}")
        for result_index, merged_result in enumerate(fallback_results[:2], start=1):
            title_text = str(merged_result.get("title_text", "") or "").strip()
            clean = str(merged_result.get("clean", "") or "").strip()
            url = str(merged_result.get("url", "") or "").strip()
            if result_index <= 2:
                search_content.append(f"Search result {result_index}:")
                search_content.append(
                    f"Retrieved via: {', '.join(merged_result.get('query_labels', []))}"
                )
                search_content.append(f"Title: {title_text or 'Unknown'}")
                if url:
                    search_content.append(f"URL: {url}")
                search_content.append(f"Snippet: {clean}")
        search_content.append("")

    packet_quality = "focused"
    if adjacent_heavy_packet:
        packet_quality = (
            "adjacent_compressed"
            if (
                adjacent_suppressed_count > 0
                or adjacent_highlighted_count < adjacent_highlight_candidate_count
            )
            else "adjacent_heavy"
        )

    return (
        "\n".join(search_content),
        raw_target_candidates,
        top_titles,
        clustered_results,
        enriched_packet,
        {
            "packet_quality": packet_quality,
            "highlighted_evidence_count": highlighted_evidence_count,
            "adjacent_highlighted_count": adjacent_highlighted_count,
            "adjacent_suppressed_count": adjacent_suppressed_count,
        },
    )


def build_pre_stage1_query_plan(
    pattern: dict,
    source_domain: str,
    source_category: str,
    deps: PreStage1Dependencies,
) -> PreStage1QueryPlan:
    query_build_result = _build_pre_stage1_query_build_result(
        pattern,
        source_domain,
        source_category,
        deps,
    )
    source_shape_terms = _jump_source_shaped_terms(
        query_build_result.built_jump_query or "",
        pattern,
    )
    return PreStage1QueryPlan(
        built_jump_query=query_build_result.built_jump_query,
        built_jump_queries=list(query_build_result.built_jump_queries),
        built_jump_query_labels=list(query_build_result.built_jump_query_labels),
        legacy_built_jump_query=query_build_result.legacy_built_jump_query,
        transferable_query_profile=dict(
            query_build_result.transferable_query_profile
        ),
        query_collision_guard_applied=query_build_result.query_collision_guard_applied,
        transferable_fallback_gate_blocked=not bool(
            query_build_result.transferable_query_profile.get("usable")
        ),
        transferable_used_but_source_shaped=bool(
            query_build_result.transferable_query_profile.get("usable")
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
        and _should_attempt_alternate_jump_retrieval(
            state.merged_results,
            state.packet.clustered_results,
        )
    ):
        alternate_query = _build_alternate_jump_search_query(
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
    blocked_cluster_tokens = set(_tokenize_query_terms(source_domain))
    blocked_cluster_tokens.update(_tokenize_query_terms(source_category))
    _blocked_anchor_tokens, preferred_anchor_phrases, strong_anchor_tokens = (
        _jump_result_anchor_context(
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
    tokens = _tokenize_query_terms(text)
    return (
        int(_jump_solution_marker_count(text)),
        len(set(tokens)),
        len(tokens),
        len(text),
        1 if _has_intervention_query_label(labels) else 0,
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
        clean = _sanitize(content)
        if not clean:
            continue
        url = str(result.get("url", "") or "").strip()
        normalized_host = _normalize_jump_result_host(url)
        if _jump_result_mentions_source_domain(
            title_text,
            clean,
            url,
            normalized_host,
            state.source_domain,
        ):
            continue
        if include_domains and not _host_matches_jump_include_domains(
            normalized_host,
            include_domains,
        ):
            continue
        should_drop, weak_result_context = _classify_weak_jump_result(
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
    ) = _build_jump_search_content(
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
