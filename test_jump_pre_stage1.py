import copy
import re
from urllib.parse import urlparse

from jump_pre_stage1 import (
    PreStage1Dependencies,
    augment_pre_stage1_with_query,
    run_pre_stage1,
)


ACADEMIC_DOMAINS = ("academic.test",)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _has_intervention_query_label(labels: list[str] | tuple[str, ...]) -> bool:
    return any(
        str(label).strip() in {"solution-biased", "intervention-family"}
        for label in labels
        if str(label).strip()
    )


def _make_deps(
    search_responses: dict[tuple[str, tuple[str, ...]], list[dict]],
    search_calls: list[tuple[str, tuple[str, ...]]],
) -> PreStage1Dependencies:
    def build_jump_search_queries(*_args, **_kwargs) -> list[str]:
        return ["primary query", "backup query"]

    build_jump_search_queries.last_query_labels = ["mechanism-family"]
    build_jump_search_queries.last_collision_guard_applied = True
    build_jump_search_queries.last_legacy_query = "legacy primary query"
    build_jump_search_queries.last_transferable_query_profile = {"usable": True}

    def jump_source_shaped_terms(*_args, **_kwargs) -> list[str]:
        return ["relay", "gating", "mismatch"]

    def jump_result_anchor_context(*_args, **_kwargs):
        return set(), ["relay gating"], {"relay", "gating"}

    def sanitize(text: object) -> str:
        return " ".join(str(text or "").split()).strip()

    def normalize_jump_result_host(url: str) -> str:
        return urlparse(url).netloc.lower()

    def jump_result_mentions_source_domain(*_args, **_kwargs) -> bool:
        return False

    def host_matches_jump_include_domains(
        normalized_host: str,
        include_domains: tuple[str, ...],
    ) -> bool:
        return normalized_host in include_domains

    def classify_weak_jump_result(
        title_text: str,
        _url: str,
        clean: str,
        _preferred_anchor_phrases: list[str],
        _strong_anchor_tokens: set[str],
    ) -> tuple[bool, dict[str, object]]:
        title_lower = title_text.lower()
        clean_lower = clean.lower()
        if "broad" in title_lower:
            return True, {
                "reason_codes": ["broad_page"],
            }
        adjacent = "adjacent" in title_lower
        intervention_evidence = "workaround" in clean_lower or "operator" in clean_lower
        solution_marker_count = int(intervention_evidence)
        return False, {
            "anchor_overlap": 3 if "anchored" in title_lower else 1,
            "preferred_phrase_match": "anchored" in title_lower,
            "solution_marker_count": solution_marker_count,
            "adjacent_strength": 6 if adjacent else 2,
            "intervention_marker_count": solution_marker_count,
            "intervention_evidence": intervention_evidence,
            "intervention_signal": "operator response" if "operator" in clean_lower else "",
            "triage_class": "adjacent" if adjacent else "keep",
        }

    def build_jump_search_content(
        merged_results: list[dict],
        _blocked_cluster_tokens: set[str],
        _strong_anchor_tokens: set[str],
    ):
        titles = [
            str(result.get("title_text", "") or "").strip()
            for result in merged_results
            if str(result.get("title_text", "") or "").strip()
        ]
        adjacent_count = sum(
            1 for result in merged_results if result.get("triage_class") == "adjacent"
        )
        intervention_count = sum(
            1 for result in merged_results if result.get("intervention_evidence")
        )
        return (
            "\n".join(f"Title: {title}" for title in titles),
            [
                {"url": str(result.get("url", "") or "").strip()}
                for result in merged_results
            ],
            titles,
            [
                {
                    "cluster_hint": titles[0] if titles else "",
                    "intervention_score": intervention_count,
                }
            ]
            if titles
            else [],
            adjacent_count > 0,
            {
                "packet_quality": "adjacent_compressed" if adjacent_count else "focused",
                "highlighted_evidence_count": len(merged_results),
                "adjacent_highlighted_count": adjacent_count,
                "adjacent_suppressed_count": 0,
            },
        )

    def should_attempt_alternate_jump_retrieval(*_args, **_kwargs) -> bool:
        return False

    def build_alternate_jump_search_query(*_args, **_kwargs) -> str | None:
        return None

    def tavily_search(**kwargs):
        query = str(kwargs.get("query") or "").strip()
        include_domains = tuple(kwargs.get("include_domains") or ())
        search_calls.append((query, include_domains))
        return {
            "results": copy.deepcopy(search_responses.get((query, include_domains), []))
        }

    def increment_tavily_calls(_count: int = 1) -> None:
        return None

    def jump_solution_marker_count(text: str) -> int:
        text_lower = str(text or "").lower()
        return int("workaround" in text_lower or "operator" in text_lower)

    return PreStage1Dependencies(
        build_jump_search_queries=build_jump_search_queries,
        jump_source_shaped_terms=jump_source_shaped_terms,
        tokenize_query_terms=_tokenize,
        jump_solution_marker_count=jump_solution_marker_count,
        has_intervention_query_label=_has_intervention_query_label,
        jump_result_anchor_context=jump_result_anchor_context,
        sanitize=sanitize,
        normalize_jump_result_host=normalize_jump_result_host,
        jump_result_mentions_source_domain=jump_result_mentions_source_domain,
        host_matches_jump_include_domains=host_matches_jump_include_domains,
        classify_weak_jump_result=classify_weak_jump_result,
        build_jump_search_content=build_jump_search_content,
        should_attempt_alternate_jump_retrieval=should_attempt_alternate_jump_retrieval,
        build_alternate_jump_search_query=build_alternate_jump_search_query,
        tavily_search=tavily_search,
        increment_tavily_calls=increment_tavily_calls,
        academic_jump_include_domains=ACADEMIC_DOMAINS,
    )


def test_run_pre_stage1_builds_query_plan_merges_results_and_populates_diagnostics() -> None:
    search_calls: list[tuple[str, tuple[str, ...]]] = []
    deps = _make_deps(
        {
            ("primary query", ()): [
                {
                    "title": "Anchored relay result",
                    "content": "relay gating evidence",
                    "url": "https://target.test/relay",
                },
                {
                    "title": "Broad background result",
                    "content": "broad context only",
                    "url": "https://target.test/background",
                },
            ],
            ("backup query", ()): [
                {
                    "title": "Anchored relay result",
                    "content": "relay gating workaround evidence",
                    "url": "https://target.test/relay",
                }
            ],
            ("primary query", ACADEMIC_DOMAINS): [
                {
                    "title": "Adjacent academic operator note",
                    "content": "operator workaround from academic lane",
                    "url": "https://academic.test/operator",
                }
            ],
        },
        search_calls,
    )

    result = run_pre_stage1(
        {
            "pattern_name": "Relay-gated mismatch suppression",
            "abstract_structure": "relay gating suppresses mismatch faults",
            "search_query": "relay gating mismatch suppression",
        },
        "Network Protocols",
        "Technology",
        deps=deps,
    )

    assert result.stage1_outcome is None
    assert result.state is not None
    assert result.state.packet is not None
    assert result.diagnostics.built_jump_query == "primary query"
    assert result.diagnostics.built_jump_query_labels == [
        "mechanism-family",
        "solution-biased",
    ]
    assert result.diagnostics.query_collision_guard_applied is True
    assert result.diagnostics.transferable_used_but_source_shaped is True
    assert result.diagnostics.general_result_count == 3
    assert result.diagnostics.academic_result_count == 1
    assert result.diagnostics.filtered_result_count == 1
    assert result.diagnostics.filtered_result_reason_counts == {"broad_page": 1}
    assert result.diagnostics.result_count == 2
    assert result.diagnostics.adjacent_result_count == 1
    assert result.diagnostics.intervention_promoted_result_count == 2
    assert result.diagnostics.packet_quality == "adjacent_compressed"
    assert result.diagnostics.top_cluster_hints == ["Anchored relay result"]
    assert result.diagnostics.benchmark_snapshot == {
        "source_domain": "Network Protocols",
        "source_category": "Technology",
        "pattern_name": "Relay-gated mismatch suppression",
        "abstract_structure": "relay gating suppresses mismatch faults",
        "built_jump_query": "primary query",
        "search_results": (
            "Title: Anchored relay result\n"
            "Title: Adjacent academic operator note"
        ),
    }
    assert search_calls == [
        ("primary query", ()),
        ("backup query", ()),
        ("primary query", ACADEMIC_DOMAINS),
    ]
    assert result.state.merged_results[0]["query_labels"] == [
        "mechanism-family",
        "solution-biased",
    ]


def test_augment_pre_stage1_with_query_accumulates_counts_and_refreshes_packet() -> None:
    search_calls: list[tuple[str, tuple[str, ...]]] = []
    deps = _make_deps(
        {
            ("primary query", ()): [
                {
                    "title": "Anchored relay result",
                    "content": "relay gating evidence",
                    "url": "https://target.test/relay",
                }
            ],
            ("backup query", ()): [],
            ("primary query", ACADEMIC_DOMAINS): [],
            ("recovery query", ()): [
                {
                    "title": "Adjacent recovery result",
                    "content": "operator workaround after recovery",
                    "url": "https://target.test/recovery",
                }
            ],
            ("recovery query", ACADEMIC_DOMAINS): [
                {
                    "title": "Recovery academic note",
                    "content": "operator workaround from academic lane",
                    "url": "https://academic.test/recovery",
                }
            ],
        },
        search_calls,
    )

    initial = run_pre_stage1(
        {
            "pattern_name": "Relay-gated mismatch suppression",
            "abstract_structure": "relay gating suppresses mismatch faults",
            "search_query": "relay gating mismatch suppression",
        },
        "Network Protocols",
        "Technology",
        deps=deps,
    )
    assert initial.state is not None

    augmented = augment_pre_stage1_with_query(
        initial.state,
        initial.diagnostics,
        query="recovery query",
        general_query_label="soft-gate",
        academic_query_label="soft-gate-academic",
        general_callsite="stage1_soft_gate_search",
        academic_callsite="stage1_soft_gate_academic_search",
        general_failure_message="general failure: {error}",
        academic_failure_message="academic failure: {error}",
        deps=deps,
    )

    assert augmented.stage1_outcome is None
    assert augmented.state is not None
    assert augmented.state.packet is not None
    assert augmented.diagnostics.general_result_count == 2
    assert augmented.diagnostics.academic_result_count == 1
    assert augmented.diagnostics.result_count == 3
    assert augmented.diagnostics.adjacent_result_count == 1
    assert augmented.diagnostics.intervention_promoted_result_count == 2
    assert augmented.diagnostics.top_result_titles == [
        "Anchored relay result",
        "Adjacent recovery result",
        "Recovery academic note",
    ]
    assert augmented.state.packet.search_content == (
        "Title: Anchored relay result\n"
        "Title: Adjacent recovery result\n"
        "Title: Recovery academic note"
    )
    assert search_calls == [
        ("primary query", ()),
        ("backup query", ()),
        ("primary query", ACADEMIC_DOMAINS),
        ("recovery query", ()),
        ("recovery query", ACADEMIC_DOMAINS),
    ]
