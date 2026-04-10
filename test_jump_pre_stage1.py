import copy

from jump_pre_stage1 import (
    PreStage1Dependencies,
    augment_pre_stage1_with_query,
    run_pre_stage1,
)
from jump_types import JumpQueryBuildResult


ACADEMIC_DOMAINS = ("academic.test",)
PRIMARY_QUERY = "relay gating mismatch"
BACKUP_QUERY = "backup relay query"


def _make_jump_query_build_result(
    queries: list[str] | tuple[str, ...],
    *,
    labels: list[str] | tuple[str, ...] | None = None,
    legacy_built_jump_query: str | None = None,
    transferable_query_profile: dict[str, object] | None = None,
    query_collision_guard_applied: bool = False,
) -> JumpQueryBuildResult:
    clean_queries = [str(query).strip() for query in queries if str(query).strip()]
    return JumpQueryBuildResult(
        built_jump_query=clean_queries[0] if clean_queries else "",
        built_jump_queries=clean_queries,
        built_jump_query_labels=[
            str(label).strip()
            for label in (labels or [])
            if str(label).strip()
        ],
        legacy_built_jump_query=(
            str(legacy_built_jump_query).strip()
            if legacy_built_jump_query is not None
            else (clean_queries[0] if clean_queries else "")
        ),
        transferable_query_profile=dict(transferable_query_profile or {}),
        query_collision_guard_applied=query_collision_guard_applied,
    )


def _make_deps(
    search_responses: dict[tuple[str, tuple[str, ...]], list[dict]],
    search_calls: list[tuple[str, tuple[str, ...]]],
) -> PreStage1Dependencies:
    def build_jump_search_queries(*_args, **_kwargs) -> JumpQueryBuildResult:
        return _make_jump_query_build_result(
            [PRIMARY_QUERY, BACKUP_QUERY],
            labels=["mechanism-family"],
            legacy_built_jump_query="legacy primary query",
            transferable_query_profile={"usable": True},
            query_collision_guard_applied=True,
        )

    def tavily_search(**kwargs):
        query = str(kwargs.get("query") or "").strip()
        include_domains = tuple(kwargs.get("include_domains") or ())
        search_calls.append((query, include_domains))
        return {
            "results": copy.deepcopy(search_responses.get((query, include_domains), []))
        }

    def increment_tavily_calls(_count: int = 1) -> None:
        return None

    return PreStage1Dependencies(
        build_jump_search_queries=build_jump_search_queries,
        tavily_search=tavily_search,
        increment_tavily_calls=increment_tavily_calls,
        academic_jump_include_domains=ACADEMIC_DOMAINS,
    )


def test_run_pre_stage1_builds_query_plan_merges_results_and_populates_diagnostics() -> None:
    search_calls: list[tuple[str, tuple[str, ...]]] = []
    deps = _make_deps(
        {
            (PRIMARY_QUERY, ()): [
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
            (BACKUP_QUERY, ()): [
                {
                    "title": "Anchored relay result",
                    "content": "relay gating workaround evidence",
                    "url": "https://target.test/relay",
                }
            ],
            (PRIMARY_QUERY, ACADEMIC_DOMAINS): [
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
            "grounded": {
                "source_control": "relay gating",
                "source_metric": "mismatch faults",
            },
        },
        "Network Protocols",
        "Technology",
        deps=deps,
    )

    assert result.stage1_outcome is None
    assert result.state is not None
    assert result.state.packet is not None
    assert result.diagnostics.built_jump_query == PRIMARY_QUERY
    assert result.diagnostics.built_jump_query_labels == [
        "mechanism-family",
        "solution-biased",
    ]
    assert result.diagnostics.query_collision_guard_applied is True
    assert result.diagnostics.transferable_used_but_source_shaped is True
    assert result.diagnostics.general_result_count == 3
    assert result.diagnostics.academic_result_count == 1
    assert result.diagnostics.filtered_result_count == 0
    assert result.diagnostics.filtered_result_reason_counts == {}
    assert result.diagnostics.result_count == 3
    assert result.diagnostics.adjacent_result_count == 0
    assert result.diagnostics.intervention_promoted_result_count == 0
    assert result.diagnostics.packet_quality == "focused"
    assert result.diagnostics.top_cluster_hints == [
        "Anchored relay result",
        "adjacent academic operator",
        "Broad background result",
    ]
    assert result.diagnostics.benchmark_snapshot is not None
    assert result.diagnostics.benchmark_snapshot["source_domain"] == "Network Protocols"
    assert result.diagnostics.benchmark_snapshot["source_category"] == "Technology"
    assert result.diagnostics.benchmark_snapshot["pattern_name"] == (
        "Relay-gated mismatch suppression"
    )
    assert result.diagnostics.benchmark_snapshot["abstract_structure"] == (
        "relay gating suppresses mismatch faults"
    )
    assert result.diagnostics.benchmark_snapshot["built_jump_query"] == PRIMARY_QUERY
    assert "Anchored relay result" in result.diagnostics.benchmark_snapshot["search_results"]
    assert (
        "Adjacent academic operator note"
        in result.diagnostics.benchmark_snapshot["search_results"]
    )
    assert (
        "Broad background result"
        in result.diagnostics.benchmark_snapshot["search_results"]
    )
    assert (
        "Retrieved via: mechanism-family, solution-biased"
        in result.diagnostics.benchmark_snapshot["search_results"]
    )
    assert "Retrieved via: academic" in result.diagnostics.benchmark_snapshot["search_results"]
    assert search_calls == [
        (PRIMARY_QUERY, ()),
        (BACKUP_QUERY, ()),
        (PRIMARY_QUERY, ACADEMIC_DOMAINS),
    ]
    assert result.state.merged_results[0]["query_labels"] == [
        "mechanism-family",
        "solution-biased",
    ]


def test_run_pre_stage1_consumes_typed_query_build_result_directly() -> None:
    search_calls: list[tuple[str, tuple[str, ...]]] = []
    deps = _make_deps(
        {
            (PRIMARY_QUERY, ()): [
                {
                    "title": "Anchored relay result",
                    "content": "relay gating evidence",
                    "url": "https://target.test/relay",
                }
            ],
            (BACKUP_QUERY, ()): [],
            (PRIMARY_QUERY, ACADEMIC_DOMAINS): [],
        },
        search_calls,
    )

    result = run_pre_stage1(
        {
            "pattern_name": "Relay-gated mismatch suppression",
            "abstract_structure": "relay gating suppresses mismatch faults",
            "search_query": "relay gating mismatch suppression",
            "grounded": {
                "source_control": "relay gating",
                "source_metric": "mismatch faults",
            },
        },
        "Network Protocols",
        "Technology",
        deps=deps,
    )

    assert result.stage1_outcome is None
    assert result.diagnostics.built_jump_query == PRIMARY_QUERY
    assert result.diagnostics.built_jump_queries == [PRIMARY_QUERY, BACKUP_QUERY]
    assert result.diagnostics.built_jump_query_labels == [
        "mechanism-family",
        "solution-biased",
    ]
    assert result.diagnostics.legacy_built_jump_query == "legacy primary query"
    assert result.diagnostics.query_collision_guard_applied is True


def test_run_pre_stage1_rejects_legacy_query_builder_format() -> None:
    search_calls: list[tuple[str, tuple[str, ...]]] = []
    deps = _make_deps({}, search_calls)

    def legacy_build_jump_search_queries(*_args, **_kwargs) -> list[str]:
        return ["primary query", "backup query"]

    object.__setattr__(
        deps,
        "build_jump_search_queries",
        legacy_build_jump_search_queries,
    )

    try:
        run_pre_stage1(
            {
                "pattern_name": "Relay-gated mismatch suppression",
                "abstract_structure": "relay gating suppresses mismatch faults",
                "search_query": "relay gating mismatch suppression",
                "grounded": {
                    "source_control": "relay gating",
                    "source_metric": "mismatch faults",
                },
            },
            "Network Protocols",
            "Technology",
            deps=deps,
        )
    except TypeError as exc:
        assert "JumpQueryBuildResult" in str(exc)
    else:
        raise AssertionError("Expected TypeError for legacy query builder format.")


def test_augment_pre_stage1_with_query_accumulates_counts_and_refreshes_packet() -> None:
    search_calls: list[tuple[str, tuple[str, ...]]] = []
    deps = _make_deps(
        {
            (PRIMARY_QUERY, ()): [
                {
                    "title": "Anchored relay result",
                    "content": "relay gating evidence",
                    "url": "https://target.test/relay",
                }
            ],
            (BACKUP_QUERY, ()): [],
            (PRIMARY_QUERY, ACADEMIC_DOMAINS): [],
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
            "grounded": {
                "source_control": "relay gating",
                "source_metric": "mismatch faults",
            },
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
    assert augmented.diagnostics.adjacent_result_count == 0
    assert augmented.diagnostics.intervention_promoted_result_count == 0
    assert set(augmented.diagnostics.top_result_titles) == {
        "Anchored relay result",
        "Adjacent recovery result",
        "Recovery academic note",
    }
    assert augmented.diagnostics.packet_quality == "focused"
    assert "Recovery academic note" in augmented.state.packet.search_content
    assert "Anchored relay result" in augmented.state.packet.search_content
    assert "Adjacent recovery result" in augmented.state.packet.search_content
    assert "Retrieved via: soft-gate-academic" in augmented.state.packet.search_content
    assert "Retrieved via: soft-gate" in augmented.state.packet.search_content
    assert search_calls == [
        (PRIMARY_QUERY, ()),
        (BACKUP_QUERY, ()),
        (PRIMARY_QUERY, ACADEMIC_DOMAINS),
        ("recovery query", ()),
        ("recovery query", ACADEMIC_DOMAINS),
    ]
