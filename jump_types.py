from __future__ import annotations

import copy
from collections.abc import Iterator, MutableMapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypedDict


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


def _clean_optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _int_value(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bool_value(value: object) -> bool:
    return bool(value)


@dataclass
class JumpReplaySnapshot:
    source_domain: str = ""
    source_category: str = ""
    pattern_name: str = "Unknown"
    abstract_structure: str = ""
    built_jump_query: str | None = None
    search_results: str = ""
    stage_one_success: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source_domain": self.source_domain,
            "source_category": self.source_category,
            "pattern_name": self.pattern_name,
            "abstract_structure": self.abstract_structure,
            "built_jump_query": self.built_jump_query,
            "search_results": self.search_results,
        }
        if self.stage_one_success is not None:
            payload["stage_one_success"] = copy.deepcopy(self.stage_one_success)
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> "JumpReplaySnapshot | None":
        if isinstance(payload, cls):
            return copy.deepcopy(payload)
        if not isinstance(payload, dict):
            return None
        stage_one_success = (
            copy.deepcopy(payload.get("stage_one_success"))
            if isinstance(payload.get("stage_one_success"), dict)
            else None
        )
        return cls(
            source_domain=str(payload.get("source_domain", "") or "").strip(),
            source_category=str(payload.get("source_category", "") or "").strip(),
            pattern_name=(
                str(payload.get("pattern_name", "") or "").strip() or "Unknown"
            ),
            abstract_structure=str(payload.get("abstract_structure", "") or "").strip(),
            built_jump_query=_clean_optional_text(payload.get("built_jump_query")),
            search_results=str(payload.get("search_results", "") or "").strip(),
            stage_one_success=stage_one_success,
        )


@dataclass
class JumpAttemptDiagnostic(MutableMapping[str, Any]):
    pattern_name: str = "Unknown"
    abstract_structure: str = ""
    raw_search_query: str = ""
    built_jump_query: str | None = None
    result_count: int = 0
    general_result_count: int = 0
    academic_result_count: int = 0
    filtered_result_count: int = 0
    adjacent_result_count: int = 0
    packet_quality: str = "focused"
    alternate_retrieval_attempted: bool = False
    enriched_packet: bool = False
    top_result_titles: list[str] = field(default_factory=list)
    outcome: str | None = None
    target_domain: str | None = None
    failure_stage: str | None = None
    failure_hint: str | None = None
    budget_exhausted: bool = False
    stage1_soft_gate_attempted: bool = False
    stage1_soft_gate_recovered: bool = False
    stage1_failure_subtype: str | None = None
    stage2_failed_at: str | None = None
    stage2_incomplete_fields: list[str] = field(default_factory=list)
    budget_stop: dict[str, Any] | None = None
    replay_snapshot: JumpReplaySnapshot | None = None
    legacy_built_jump_query: str = ""
    built_jump_queries: list[str] = field(default_factory=list)
    built_jump_query_labels: list[str] = field(default_factory=list)
    transferable_query_profile: dict[str, Any] = field(default_factory=dict)
    transferable_fallback_gate_blocked: bool = False
    transferable_used_but_source_shaped: bool = False
    transferable_used_source_shape_terms: list[str] = field(default_factory=list)
    query_collision_guard_applied: bool = False
    filtered_result_reason_counts: dict[str, int] = field(default_factory=dict)
    cluster_count: int = 0
    top_cluster_hints: list[str] = field(default_factory=list)
    top_cluster_intervention_scores: list[int] = field(default_factory=list)
    intervention_promoted_result_count: int = 0
    highlighted_evidence_count: int = 0
    adjacent_highlighted_count: int = 0
    adjacent_suppressed_count: int = 0
    alternate_jump_query: str | None = None
    alternate_result_count: int = 0
    _stage1_outcome: str | None = field(default=None, repr=False)
    _stage1_target_domain: str | None = field(default=None, repr=False)
    _stage1_failure_hint: str | None = field(default=None, repr=False)
    _stage2_outcome: str | None = field(default=None, repr=False)
    _stage2_target_domain: str | None = field(default=None, repr=False)
    _stage2_failure_hint: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.pattern_name = self.pattern_name or "Unknown"
        self.top_result_titles = list(self.top_result_titles)
        self.built_jump_queries = list(self.built_jump_queries)
        self.built_jump_query_labels = list(self.built_jump_query_labels)
        self.transferable_query_profile = dict(self.transferable_query_profile)
        self.transferable_used_source_shape_terms = list(
            self.transferable_used_source_shape_terms
        )
        self.filtered_result_reason_counts = dict(self.filtered_result_reason_counts)
        self.top_cluster_hints = list(self.top_cluster_hints)
        self.top_cluster_intervention_scores = list(
            self.top_cluster_intervention_scores
        )
        self.stage2_incomplete_fields = list(self.stage2_incomplete_fields)
        if isinstance(self.replay_snapshot, dict):
            self.replay_snapshot = JumpReplaySnapshot.from_dict(self.replay_snapshot)
        if self._stage1_outcome is not None or self._stage2_outcome is not None:
            self._sync_public_view()
        elif self.outcome is not None:
            self._sync_legacy_view_from_public()
            self._sync_public_view()

    def _sync_public_view(self) -> None:
        self.target_domain = (
            _clean_optional_text(self._stage2_target_domain)
            or _clean_optional_text(self._stage1_target_domain)
        )
        stage1_outcome = _clean_optional_text(self._stage1_outcome)
        stage1_hint = _clean_optional_text(self._stage1_failure_hint)
        stage2_outcome = _clean_optional_text(self._stage2_outcome)
        stage2_hint = _clean_optional_text(self._stage2_failure_hint)

        self.outcome = None
        self.failure_stage = None
        self.failure_hint = None

        if stage2_outcome == "connection_found":
            self.outcome = "connection_found"
        elif stage1_outcome == "budget_exhausted_pre_stage1":
            self.outcome = "budget_exhausted"
            self.failure_stage = "pre_stage1"
            self.failure_hint = stage1_hint
        elif stage1_outcome == "budget_exhausted_stage1":
            self.outcome = "budget_exhausted"
            self.failure_stage = "stage1"
            self.failure_hint = stage1_hint
        elif stage2_outcome == "stage2_no_connection":
            self.outcome = "no_connection"
            self.failure_stage = "stage2"
            self.failure_hint = stage2_hint
        elif stage1_outcome == "weak_signal":
            self.outcome = "weak_signal"
            self.failure_stage = "stage1"
            self.failure_hint = stage1_hint
        elif stage1_outcome == "detect_no_signal":
            self.outcome = "no_connection"
            self.failure_stage = "stage1"
            self.failure_hint = stage1_hint
        elif stage1_outcome == "no_results":
            self.outcome = "no_results"
            self.failure_stage = (
                "pre_stage1"
                if stage1_hint in {"empty_jump_query", "search_error", "no_usable_results"}
                else "stage1"
            )
            self.failure_hint = stage1_hint

        self.budget_exhausted = self.outcome == "budget_exhausted"

    def _sync_legacy_view_from_public(self) -> None:
        outcome = _clean_optional_text(self.outcome)
        failure_stage = _clean_optional_text(self.failure_stage)
        failure_hint = _clean_optional_text(self.failure_hint)
        target_domain = _clean_optional_text(self.target_domain)

        self._stage1_outcome = None
        self._stage1_target_domain = None
        self._stage1_failure_hint = None
        self._stage2_outcome = None
        self._stage2_target_domain = None
        self._stage2_failure_hint = None

        if outcome == "connection_found":
            self._stage1_outcome = "detect_signal"
            self._stage1_target_domain = target_domain
            self._stage2_outcome = "connection_found"
            self._stage2_target_domain = target_domain
            return
        if outcome == "no_connection" and failure_stage == "stage2":
            self._stage1_outcome = "detect_signal"
            self._stage1_target_domain = target_domain
            self._stage2_outcome = "stage2_no_connection"
            self._stage2_failure_hint = failure_hint
            return
        if outcome == "no_connection":
            self._stage1_outcome = "detect_no_signal"
            self._stage1_target_domain = target_domain
            self._stage1_failure_hint = failure_hint
            return
        if outcome == "weak_signal":
            self._stage1_outcome = "weak_signal"
            self._stage1_target_domain = target_domain
            self._stage1_failure_hint = failure_hint
            return
        if outcome == "budget_exhausted":
            if failure_stage == "pre_stage1":
                self._stage1_outcome = "budget_exhausted_pre_stage1"
            else:
                self._stage1_outcome = "budget_exhausted_stage1"
            self._stage1_failure_hint = failure_hint or "cycle_budget_exhausted"
            return
        if outcome == "no_results":
            self._stage1_outcome = "no_results"
            self._stage1_failure_hint = failure_hint

    def to_dict(self) -> dict[str, Any]:
        snapshot_dict = (
            self.replay_snapshot.to_dict() if self.replay_snapshot is not None else None
        )
        payload: dict[str, Any] = {
            "pattern_name": self.pattern_name,
            "abstract_structure": self.abstract_structure,
            "raw_search_query": self.raw_search_query,
            "legacy_built_jump_query": self.legacy_built_jump_query,
            "built_jump_query": self.built_jump_query,
            "built_jump_queries": list(self.built_jump_queries),
            "built_jump_query_labels": list(self.built_jump_query_labels),
            "transferable_query_profile": copy.deepcopy(self.transferable_query_profile),
            "transferable_fallback_gate_blocked": self.transferable_fallback_gate_blocked,
            "transferable_used_but_source_shaped": self.transferable_used_but_source_shaped,
            "transferable_used_source_shape_terms": list(
                self.transferable_used_source_shape_terms
            ),
            "query_collision_guard_applied": self.query_collision_guard_applied,
            "result_count": self.result_count,
            "general_result_count": self.general_result_count,
            "academic_result_count": self.academic_result_count,
            "filtered_result_count": self.filtered_result_count,
            "filtered_result_reason_counts": copy.deepcopy(
                self.filtered_result_reason_counts
            ),
            "cluster_count": self.cluster_count,
            "top_cluster_hints": list(self.top_cluster_hints),
            "top_cluster_intervention_scores": list(
                self.top_cluster_intervention_scores
            ),
            "intervention_promoted_result_count": self.intervention_promoted_result_count,
            "highlighted_evidence_count": self.highlighted_evidence_count,
            "adjacent_result_count": self.adjacent_result_count,
            "adjacent_retained_result_count": self.adjacent_result_count,
            "retained_adjacent_result_count": self.adjacent_result_count,
            "adjacent_highlighted_count": self.adjacent_highlighted_count,
            "adjacent_suppressed_count": self.adjacent_suppressed_count,
            "packet_quality": self.packet_quality,
            "alternate_retrieval_attempted": self.alternate_retrieval_attempted,
            "alternate_jump_query": self.alternate_jump_query,
            "alternate_result_count": self.alternate_result_count,
            "enriched_packet": self.enriched_packet,
            "top_result_titles": list(self.top_result_titles),
            "stage1_outcome": self._stage1_outcome,
            "stage1_target_domain": self._stage1_target_domain,
            "stage1_failure_hint": self._stage1_failure_hint,
            "stage1_failure_subtype": self.stage1_failure_subtype,
            "stage1_soft_gate_attempted": self.stage1_soft_gate_attempted,
            "stage1_soft_gate_recovered": self.stage1_soft_gate_recovered,
            "stage2_outcome": self._stage2_outcome,
            "stage2_target_domain": self._stage2_target_domain,
            "stage2_failure_hint": self._stage2_failure_hint,
            "stage2_failed_at": self.stage2_failed_at,
            "benchmark_snapshot": snapshot_dict,
        }
        if self.stage2_incomplete_fields:
            payload["stage2_incomplete_fields"] = list(self.stage2_incomplete_fields)
        if self.budget_stop is not None:
            payload["budget_stop"] = copy.deepcopy(self.budget_stop)
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> "JumpAttemptDiagnostic | None":
        if isinstance(payload, cls):
            return copy.deepcopy(payload)
        if not isinstance(payload, dict):
            return None

        diagnostic = cls(
            pattern_name=str(payload.get("pattern_name", "") or "").strip() or "Unknown",
            abstract_structure=str(payload.get("abstract_structure", "") or "").strip(),
            raw_search_query=str(payload.get("raw_search_query", "") or "").strip(),
            built_jump_query=_clean_optional_text(payload.get("built_jump_query")),
            result_count=_int_value(payload.get("result_count")),
            general_result_count=_int_value(payload.get("general_result_count")),
            academic_result_count=_int_value(payload.get("academic_result_count")),
            filtered_result_count=_int_value(payload.get("filtered_result_count")),
            adjacent_result_count=_int_value(
                payload.get("adjacent_result_count")
                or payload.get("retained_adjacent_result_count")
                or payload.get("adjacent_retained_result_count")
            ),
            packet_quality=str(payload.get("packet_quality", "focused") or "focused").strip()
            or "focused",
            alternate_retrieval_attempted=_bool_value(
                payload.get("alternate_retrieval_attempted")
            ),
            enriched_packet=_bool_value(payload.get("enriched_packet")),
            top_result_titles=_string_list(payload.get("top_result_titles")),
            stage1_soft_gate_attempted=_bool_value(
                payload.get("stage1_soft_gate_attempted")
            ),
            stage1_soft_gate_recovered=_bool_value(
                payload.get("stage1_soft_gate_recovered")
            ),
            stage1_failure_subtype=_clean_optional_text(
                payload.get("stage1_failure_subtype")
            ),
            stage2_failed_at=_clean_optional_text(payload.get("stage2_failed_at")),
            stage2_incomplete_fields=_string_list(payload.get("stage2_incomplete_fields")),
            budget_stop=(
                copy.deepcopy(payload.get("budget_stop"))
                if isinstance(payload.get("budget_stop"), dict)
                else None
            ),
            replay_snapshot=JumpReplaySnapshot.from_dict(
                payload.get("replay_snapshot") or payload.get("benchmark_snapshot")
            ),
            legacy_built_jump_query=str(
                payload.get("legacy_built_jump_query", "") or ""
            ).strip(),
            built_jump_queries=_string_list(payload.get("built_jump_queries")),
            built_jump_query_labels=_string_list(payload.get("built_jump_query_labels")),
            transferable_query_profile=(
                copy.deepcopy(payload.get("transferable_query_profile"))
                if isinstance(payload.get("transferable_query_profile"), dict)
                else {}
            ),
            transferable_fallback_gate_blocked=_bool_value(
                payload.get("transferable_fallback_gate_blocked")
            ),
            transferable_used_but_source_shaped=_bool_value(
                payload.get("transferable_used_but_source_shaped")
            ),
            transferable_used_source_shape_terms=_string_list(
                payload.get("transferable_used_source_shape_terms")
            ),
            query_collision_guard_applied=_bool_value(
                payload.get("query_collision_guard_applied")
            ),
            filtered_result_reason_counts=(
                copy.deepcopy(payload.get("filtered_result_reason_counts"))
                if isinstance(payload.get("filtered_result_reason_counts"), dict)
                else {}
            ),
            cluster_count=_int_value(payload.get("cluster_count")),
            top_cluster_hints=_string_list(payload.get("top_cluster_hints")),
            top_cluster_intervention_scores=[
                _int_value(value)
                for value in (payload.get("top_cluster_intervention_scores") or [])
                if isinstance(payload.get("top_cluster_intervention_scores"), list)
            ],
            intervention_promoted_result_count=_int_value(
                payload.get("intervention_promoted_result_count")
            ),
            highlighted_evidence_count=_int_value(
                payload.get("highlighted_evidence_count")
            ),
            adjacent_highlighted_count=_int_value(
                payload.get("adjacent_highlighted_count")
            ),
            adjacent_suppressed_count=_int_value(
                payload.get("adjacent_suppressed_count")
            ),
            alternate_jump_query=_clean_optional_text(payload.get("alternate_jump_query")),
            alternate_result_count=_int_value(payload.get("alternate_result_count")),
            outcome=_clean_optional_text(payload.get("outcome")),
            target_domain=_clean_optional_text(payload.get("target_domain")),
            failure_stage=_clean_optional_text(payload.get("failure_stage")),
            failure_hint=_clean_optional_text(payload.get("failure_hint")),
            budget_exhausted=_bool_value(payload.get("budget_exhausted")),
        )

        diagnostic._stage1_outcome = _clean_optional_text(payload.get("stage1_outcome"))
        diagnostic._stage1_target_domain = _clean_optional_text(
            payload.get("stage1_target_domain")
        )
        diagnostic._stage1_failure_hint = _clean_optional_text(
            payload.get("stage1_failure_hint")
        )
        diagnostic._stage2_outcome = _clean_optional_text(payload.get("stage2_outcome"))
        diagnostic._stage2_target_domain = _clean_optional_text(
            payload.get("stage2_target_domain")
        )
        diagnostic._stage2_failure_hint = _clean_optional_text(
            payload.get("stage2_failure_hint")
        )
        if diagnostic._stage1_outcome or diagnostic._stage2_outcome:
            diagnostic._sync_public_view()
        elif diagnostic.outcome is not None:
            diagnostic._sync_legacy_view_from_public()
            diagnostic._sync_public_view()
        return diagnostic

    def __getitem__(self, key: str) -> Any:
        if key == "outcome":
            return self.outcome
        if key == "target_domain":
            return self.target_domain
        if key == "failure_stage":
            return self.failure_stage
        if key == "failure_hint":
            return self.failure_hint
        if key == "budget_exhausted":
            return self.budget_exhausted
        if key == "replay_snapshot":
            return (
                self.replay_snapshot.to_dict()
                if self.replay_snapshot is not None
                else None
            )
        if key == "benchmark_snapshot":
            return (
                self.replay_snapshot.to_dict()
                if self.replay_snapshot is not None
                else None
            )
        if key == "adjacent_retained_result_count":
            return self.adjacent_result_count
        if key == "retained_adjacent_result_count":
            return self.adjacent_result_count
        if key == "stage1_outcome":
            return self._stage1_outcome
        if key == "stage1_target_domain":
            return self._stage1_target_domain
        if key == "stage1_failure_hint":
            return self._stage1_failure_hint
        if key == "stage2_outcome":
            return self._stage2_outcome
        if key == "stage2_target_domain":
            return self._stage2_target_domain
        if key == "stage2_failure_hint":
            return self._stage2_failure_hint
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "benchmark_snapshot" or key == "replay_snapshot":
            self.replay_snapshot = JumpReplaySnapshot.from_dict(value)
            return
        if key in {"adjacent_result_count", "adjacent_retained_result_count", "retained_adjacent_result_count"}:
            self.adjacent_result_count = _int_value(value)
            return
        if key == "stage1_outcome":
            self._stage1_outcome = _clean_optional_text(value)
            self._sync_public_view()
            return
        if key == "stage1_target_domain":
            self._stage1_target_domain = _clean_optional_text(value)
            self._sync_public_view()
            return
        if key == "stage1_failure_hint":
            self._stage1_failure_hint = _clean_optional_text(value)
            self._sync_public_view()
            return
        if key == "stage2_outcome":
            self._stage2_outcome = _clean_optional_text(value)
            self._sync_public_view()
            return
        if key == "stage2_target_domain":
            self._stage2_target_domain = _clean_optional_text(value)
            self._sync_public_view()
            return
        if key == "stage2_failure_hint":
            self._stage2_failure_hint = _clean_optional_text(value)
            self._sync_public_view()
            return
        if key == "outcome":
            self.outcome = _clean_optional_text(value)
            self._sync_legacy_view_from_public()
            self._sync_public_view()
            return
        if key == "target_domain":
            self.target_domain = _clean_optional_text(value)
            self._sync_legacy_view_from_public()
            self._sync_public_view()
            return
        if key == "failure_stage":
            self.failure_stage = _clean_optional_text(value)
            self._sync_legacy_view_from_public()
            self._sync_public_view()
            return
        if key == "failure_hint":
            self.failure_hint = _clean_optional_text(value)
            self._sync_legacy_view_from_public()
            self._sync_public_view()
            return
        if key == "budget_exhausted":
            self.budget_exhausted = _bool_value(value)
            if self.budget_exhausted and self.outcome is None:
                self.outcome = "budget_exhausted"
            self._sync_legacy_view_from_public()
            self._sync_public_view()
            return
        if key == "built_jump_queries":
            self.built_jump_queries = _string_list(value)
            return
        if key == "built_jump_query_labels":
            self.built_jump_query_labels = _string_list(value)
            return
        if key == "top_result_titles":
            self.top_result_titles = _string_list(value)
            return
        if key == "stage2_incomplete_fields":
            self.stage2_incomplete_fields = _string_list(value)
            return
        if key == "top_cluster_hints":
            self.top_cluster_hints = _string_list(value)
            return
        if key == "top_cluster_intervention_scores":
            self.top_cluster_intervention_scores = [
                _int_value(item) for item in value
            ] if isinstance(value, list) else []
            return
        if key == "transferable_used_source_shape_terms":
            self.transferable_used_source_shape_terms = _string_list(value)
            return
        if key == "transferable_query_profile":
            self.transferable_query_profile = dict(value) if isinstance(value, dict) else {}
            return
        if key == "filtered_result_reason_counts":
            self.filtered_result_reason_counts = dict(value) if isinstance(value, dict) else {}
            return
        if key == "budget_stop":
            self.budget_stop = copy.deepcopy(value) if isinstance(value, dict) else None
            return
        if key in self.__dataclass_fields__:
            setattr(self, key, value)
            return
        raise KeyError(key)

    def __delitem__(self, key: str) -> None:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


JumpFailureAttribution = Literal[
    "pre_stage1_failure",
    "stage1_failure",
    "stage2_failure",
    "successful_connection",
    "ambiguous_failure",
]

_PRE_STAGE1_FAILURE_HINTS = {
    "empty_jump_query",
    "search_error",
    "no_usable_results",
}

_CLEAR_STAGE1_NO_RESULTS_HINTS = {
    "generation_failed",
    "invalid_json",
    "invalid_payload",
    "invalid_payload_non_object",
}


def classify_jump_attempt_attribution(payload: object) -> JumpFailureAttribution:
    """Classify where a jump attempt appears to have failed."""
    diagnostic = JumpAttemptDiagnostic.from_dict(payload)
    if diagnostic is None:
        return "ambiguous_failure"

    outcome = _clean_optional_text(diagnostic.outcome)
    failure_stage = _clean_optional_text(diagnostic.failure_stage)
    stage1_outcome = _clean_optional_text(diagnostic.get("stage1_outcome"))
    stage1_failure_hint = _clean_optional_text(diagnostic.get("stage1_failure_hint"))
    stage1_failure_subtype = _clean_optional_text(
        diagnostic.get("stage1_failure_subtype")
    )
    stage2_outcome = _clean_optional_text(diagnostic.get("stage2_outcome"))
    stage2_failure_hint = _clean_optional_text(diagnostic.get("stage2_failure_hint"))
    target_domain = _clean_optional_text(diagnostic.target_domain)
    stage2_failed_at = _clean_optional_text(diagnostic.stage2_failed_at)
    stage2_incomplete_fields = _string_list(diagnostic.stage2_incomplete_fields)

    if outcome == "connection_found" or stage2_outcome == "connection_found":
        return "successful_connection"
    if failure_stage == "stage2" or stage2_outcome == "stage2_no_connection":
        return "stage2_failure"
    if (
        stage1_outcome == "detect_signal"
        and target_domain
        and (
            stage2_failed_at
            or stage2_failure_hint
            or stage2_incomplete_fields
        )
    ):
        return "stage2_failure"
    if (
        failure_stage == "pre_stage1"
        or stage1_outcome == "budget_exhausted_pre_stage1"
        or (
            stage1_outcome == "no_results"
            and (stage1_failure_hint or stage2_failure_hint) in _PRE_STAGE1_FAILURE_HINTS
        )
    ):
        return "pre_stage1_failure"
    if stage1_outcome in {
        "detect_no_signal",
        "weak_signal",
        "budget_exhausted_stage1",
    }:
        return "stage1_failure"
    if failure_stage == "stage1" and stage1_outcome != "no_results":
        return "stage1_failure"
    if stage1_outcome == "no_results":
        if stage1_failure_subtype:
            return "stage1_failure"
        if stage1_failure_hint in _CLEAR_STAGE1_NO_RESULTS_HINTS:
            return "stage1_failure"
        return "ambiguous_failure"
    return "ambiguous_failure"


@dataclass
class RenderedJumpPacket:
    search_content: str
    raw_target_candidates: list[dict[str, Any]]
    top_titles: list[str]
    clustered_results: list[JumpClusterSummary]
    enriched_packet: bool
    packet_observability: JumpPacketObservability


@dataclass
class JumpQueryBuildResult:
    built_jump_query: str | None
    built_jump_queries: list[str] = field(default_factory=list)
    built_jump_query_labels: list[str] = field(default_factory=list)
    legacy_built_jump_query: str = ""
    transferable_query_profile: dict[str, Any] = field(default_factory=dict)
    query_collision_guard_applied: bool = False


@dataclass
class JumpSearchQueryMetadataResult:
    built_jump_query: str = ""
    legacy_built_jump_query: str = ""
    transferable_query_profile: dict[str, Any] = field(default_factory=dict)
    query_collision_guard_applied: bool = False


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
