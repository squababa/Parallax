from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict


class JumpSearchResult(TypedDict, total=False):
    title_text: str
    clean: str
    url: str
    query_labels: list[str]
    anchor_overlap: int
    preferred_phrase_match: bool
    solution_marker_count: int
    adjacent_strength: int
    intervention_marker_count: int
    intervention_evidence: bool
    intervention_signal: str
    triage_class: str


class JumpClusterSummary(TypedDict, total=False):
    cluster_hint: str
    intervention_score: int


class JumpPacketObservability(TypedDict, total=False):
    highlighted_evidence_count: int
    adjacent_highlighted_count: int
    adjacent_suppressed_count: int
    packet_quality: str


@dataclass
class RenderedJumpPacket:
    search_content: str
    raw_target_candidates: list[dict[str, Any]]
    top_titles: list[str]
    clustered_results: list[JumpClusterSummary]
    enriched_packet: bool
    packet_observability: JumpPacketObservability


@dataclass
class PreStage1QueryPlan:
    built_jump_query: str | None
    built_jump_queries: list[str] = field(default_factory=list)
    built_jump_query_labels: list[str] = field(default_factory=list)
    legacy_built_jump_query: str = ""
    transferable_query_profile: dict[str, Any] = field(default_factory=dict)
    query_collision_guard_applied: bool = False
    transferable_fallback_gate_blocked: bool = False
    transferable_used_but_source_shaped: bool = False
    transferable_used_source_shape_terms: list[str] = field(default_factory=list)


@dataclass
class PreStage1DiagnosticsBundle:
    raw_search_query: str = ""
    legacy_built_jump_query: str = ""
    built_jump_query: str | None = None
    built_jump_queries: list[str] = field(default_factory=list)
    built_jump_query_labels: list[str] = field(default_factory=list)
    transferable_query_profile: dict[str, Any] = field(default_factory=dict)
    transferable_fallback_gate_blocked: bool = False
    transferable_used_but_source_shaped: bool = False
    transferable_used_source_shape_terms: list[str] = field(default_factory=list)
    query_collision_guard_applied: bool = False
    result_count: int = 0
    general_result_count: int = 0
    academic_result_count: int = 0
    filtered_result_count: int = 0
    filtered_result_reason_counts: dict[str, int] = field(default_factory=dict)
    cluster_count: int = 0
    top_cluster_hints: list[str] = field(default_factory=list)
    top_cluster_intervention_scores: list[int] = field(default_factory=list)
    intervention_promoted_result_count: int = 0
    adjacent_result_count: int = 0
    adjacent_retained_result_count: int = 0
    retained_adjacent_result_count: int = 0
    highlighted_evidence_count: int = 0
    adjacent_highlighted_count: int = 0
    adjacent_suppressed_count: int = 0
    packet_quality: str = "focused"
    alternate_retrieval_attempted: bool = False
    alternate_jump_query: str | None = None
    alternate_result_count: int = 0
    enriched_packet: bool = False
    top_result_titles: list[str] = field(default_factory=list)
    benchmark_snapshot: dict[str, Any] | None = None

    @classmethod
    def from_query_plan(
        cls,
        *,
        raw_search_query: str,
        query_plan: PreStage1QueryPlan,
    ) -> "PreStage1DiagnosticsBundle":
        return cls(
            raw_search_query=raw_search_query,
            legacy_built_jump_query=query_plan.legacy_built_jump_query,
            built_jump_query=query_plan.built_jump_query,
            built_jump_queries=list(query_plan.built_jump_queries),
            built_jump_query_labels=list(query_plan.built_jump_query_labels),
            transferable_query_profile=dict(query_plan.transferable_query_profile),
            transferable_fallback_gate_blocked=query_plan.transferable_fallback_gate_blocked,
            transferable_used_but_source_shaped=query_plan.transferable_used_but_source_shaped,
            transferable_used_source_shape_terms=list(
                query_plan.transferable_used_source_shape_terms
            ),
            query_collision_guard_applied=query_plan.query_collision_guard_applied,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreStage1State:
    source_domain: str
    source_category: str
    query_plan: PreStage1QueryPlan
    blocked_cluster_tokens: set[str]
    preferred_anchor_phrases: list[str]
    strong_anchor_tokens: set[str]
    merged_results: list[JumpSearchResult] = field(default_factory=list)
    merged_result_index: dict[str, int] = field(default_factory=dict)
    filtered_result_reason_keys: dict[str, set[str]] = field(default_factory=dict)
    general_result_count: int = 0
    academic_result_count: int = 0
    alternate_result_count: int = 0
    filtered_result_count: int = 0
    filtered_result_reason_counts: dict[str, int] = field(default_factory=dict)
    query_error_count: int = 0
    packet: RenderedJumpPacket | None = None


@dataclass
class PreStage1ExecutionResult:
    diagnostics: PreStage1DiagnosticsBundle
    state: PreStage1State | None = None
    stage1_outcome: str | None = None
    stage1_failure_hint: str | None = None
    budget_stop: dict[str, Any] | None = None
