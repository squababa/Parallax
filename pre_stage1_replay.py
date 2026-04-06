"""Deterministic Pre-stage1 replay harness for fixture-driven retrieval checks.

Fixture schema:
- `id`, `description`, `source_domain`, `source_category`
- `pattern`: one explore-style pattern payload
- `llm_jump_query`: optional string or null to stub the query-builder model output
- `built_queries_override`: optional list of exact built queries to bypass query building
- `built_query_labels`: optional labels used with `built_queries_override`
- `legacy_built_query`, `transferable_query_profile`, `query_collision_guard_applied`:
  optional metadata used with `built_queries_override`
- `search_responses`: list of `{query, lane, include_domains, results}` entries
- `stage1_stub`: `{data, failure_hint}` replay response
- `stage2_stub`: optional `{data, failure_hint, incomplete_fields}` replay response
- `expect`: exact scorecard values to compare against
"""

from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import sys
import types
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "pre_stage1"


def _ensure_env_defaults() -> None:
    os.environ.setdefault("LOCAL_LLM_ONLY", "1")
    os.environ.setdefault("TAVILY_API_KEY", "test-key")
    os.environ.setdefault("BLACKCLAW_DB_PATH", "/tmp/blackclaw-pre-stage1-replay.db")
    os.environ.setdefault("BLACKCLAW_MODEL", "qwen3:8b")


def _install_optional_dependency_stubs() -> None:
    try:
        import tavily  # noqa: F401
    except ImportError:
        module = types.ModuleType("tavily")

        class TavilyClient:  # pragma: no cover - exercised via harness import
            def __init__(self, api_key: str | None = None):
                self.api_key = api_key

            def search(self, **_kwargs: Any) -> dict[str, Any]:
                raise AssertionError("Tavily search should be monkeypatched in replay mode")

        module.TavilyClient = TavilyClient
        sys.modules["tavily"] = module

    try:
        import requests  # noqa: F401
    except ImportError:
        module = types.ModuleType("requests")

        class RequestException(Exception):
            pass

        class _StubResponse:
            status_code = 200
            text = '{"response": ""}'

            def json(self) -> dict[str, str]:
                return {"response": ""}

        def post(*_args: Any, **_kwargs: Any) -> _StubResponse:
            return _StubResponse()

        module.RequestException = RequestException
        module.post = post
        sys.modules["requests"] = module

    try:
        import bs4  # noqa: F401
    except ImportError:
        module = types.ModuleType("bs4")

        class BeautifulSoup:  # pragma: no cover - exercised via harness import
            def __init__(self, text: str, _parser: str | None = None):
                self._text = text

            def __call__(self, _names: Any) -> list[Any]:
                return []

            def get_text(self, separator: str = " ") -> str:
                return separator.join(str(self._text).split())

        module.BeautifulSoup = BeautifulSoup
        sys.modules["bs4"] = module


def _load_jump_module():
    _ensure_env_defaults()
    _install_optional_dependency_stubs()
    return importlib.import_module("jump")


def load_fixture_paths(fixtures_dir: Path = FIXTURE_DIR) -> list[Path]:
    return sorted(path for path in fixtures_dir.glob("*.json") if path.is_file())


def load_fixture(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    with fixture_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class _PatchSession:
    def __init__(self) -> None:
        self._originals: list[tuple[Any, str, Any]] = []

    def setattr(self, obj: Any, name: str, value: Any) -> None:
        self._originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def restore(self) -> None:
        for obj, name, value in reversed(self._originals):
            setattr(obj, name, value)


def _normalize_query(query: object) -> str:
    return " ".join(str(query or "").split()).strip()


def _search_key(
    jump_module,
    *,
    query: object,
    lane: str | None = None,
    include_domains: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, tuple[str, ...] | None]:
    clean_query = _normalize_query(query)
    if include_domains is not None:
        domains = tuple(str(domain).strip() for domain in include_domains if str(domain).strip())
    elif str(lane or "general").strip() == "academic":
        domains = tuple(jump_module.ACADEMIC_JUMP_INCLUDE_DOMAINS)
    else:
        domains = ()
    return clean_query, (domains or None)


def _build_search_response_index(jump_module, fixture: dict[str, Any]) -> dict[tuple[str, tuple[str, ...] | None], dict[str, Any]]:
    indexed: dict[tuple[str, tuple[str, ...] | None], dict[str, Any]] = {}
    for entry in fixture.get("search_responses") or []:
        if not isinstance(entry, dict):
            continue
        key = _search_key(
            jump_module,
            query=entry.get("query"),
            lane=str(entry.get("lane") or "general").strip() or "general",
            include_domains=entry.get("include_domains"),
        )
        indexed[key] = {"results": copy.deepcopy(list(entry.get("results") or []))}
    return indexed


def _default_stage2_stub(stage1_stub: dict[str, Any]) -> dict[str, Any]:
    data = stage1_stub.get("data") if isinstance(stage1_stub, dict) else None
    target_domain = ""
    if isinstance(data, dict):
        target_domain = str(data.get("target_domain") or "").strip()
    return {
        "data": {
            "no_connection": False,
            "target_domain": target_domain or "Replay Target",
        },
        "failure_hint": None,
        "incomplete_fields": None,
    }


def _make_stage_one_stub(fixture: dict[str, Any], captured: dict[str, Any]):
    stage1_stub = fixture.get("stage1_stub") if isinstance(fixture.get("stage1_stub"), dict) else {}

    def fake_stage_one(**kwargs: Any):
        captured["stage1_kwargs"] = copy.deepcopy(kwargs)
        payload = stage1_stub.get("data")
        failure_hint = stage1_stub.get("failure_hint")
        clean_failure_hint = (
            str(failure_hint).strip() if failure_hint is not None else None
        )
        return copy.deepcopy(payload) if isinstance(payload, dict) else None, (
            clean_failure_hint or None
        )

    return fake_stage_one


def _make_stage_two_stub(fixture: dict[str, Any], captured: dict[str, Any]):
    raw_stub = fixture.get("stage2_stub")
    stage1_stub = fixture.get("stage1_stub") if isinstance(fixture.get("stage1_stub"), dict) else {}
    stage2_stub = raw_stub if isinstance(raw_stub, dict) else _default_stage2_stub(stage1_stub)

    def fake_stage_two(**kwargs: Any):
        captured["stage2_kwargs"] = copy.deepcopy(kwargs)
        payload = stage2_stub.get("data")
        failure_hint = stage2_stub.get("failure_hint")
        incomplete_fields = stage2_stub.get("incomplete_fields")
        clean_failure_hint = (
            str(failure_hint).strip() if failure_hint is not None else None
        )
        return (
            copy.deepcopy(payload) if isinstance(payload, dict) else None,
            clean_failure_hint or None,
            copy.deepcopy(incomplete_fields),
        )

    return fake_stage_two


def _make_query_builder_stub(fixture: dict[str, Any]):
    llm_jump_query = fixture.get("llm_jump_query")

    def fake_generate_llm_jump_search_query(*_args: Any, **_kwargs: Any) -> str | None:
        clean = _normalize_query(llm_jump_query)
        return clean or None

    return fake_generate_llm_jump_search_query


def _make_built_queries_override(fixture: dict[str, Any]):
    built_queries = [
        _normalize_query(query)
        for query in list(fixture.get("built_queries_override") or [])
        if _normalize_query(query)
    ]
    built_query_labels = [
        str(label).strip()
        for label in list(fixture.get("built_query_labels") or [])
        if str(label).strip()
    ]
    while len(built_query_labels) < len(built_queries):
        built_query_labels.append(f"variant-{len(built_query_labels) + 1}")

    def fake_build_jump_search_queries(*_args: Any, **_kwargs: Any) -> list[str]:
        return list(built_queries)

    fake_build_jump_search_queries.last_collision_guard_applied = bool(
        fixture.get("query_collision_guard_applied")
    )
    fake_build_jump_search_queries.last_legacy_query = _normalize_query(
        fixture.get("legacy_built_query")
    )
    fake_build_jump_search_queries.last_transferable_query_profile = copy.deepcopy(
        fixture.get("transferable_query_profile") or {}
    )
    fake_build_jump_search_queries.last_query_labels = built_query_labels[: len(built_queries)]
    return fake_build_jump_search_queries


def _obvious_source_leakage(diagnostic: dict[str, Any]) -> bool:
    profile = diagnostic.get("transferable_query_profile")
    profile = profile if isinstance(profile, dict) else {}
    concerns = {
        str(concern).strip()
        for concern in list(profile.get("concerns") or [])
        if str(concern).strip()
    }
    strong_source_leakage_terms = list(profile.get("strong_source_leakage_terms") or [])
    strong_source_shape_terms = list(profile.get("strong_source_shape_terms") or [])
    return bool(
        diagnostic.get("transferable_fallback_gate_blocked")
        or diagnostic.get("transferable_used_but_source_shaped")
        or strong_source_leakage_terms
        or strong_source_shape_terms
        or (
            "transferable_source_leakage" in concerns
            and not bool(profile.get("usable", False))
        )
        or (
            "transferable_source_shaped" in concerns
            and not bool(profile.get("usable", False))
        )
    )


def _build_scorecard(
    fixture: dict[str, Any],
    diagnostic: dict[str, Any],
    search_calls: list[dict[str, Any]],
    unexpected_search_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    scorecard = {
        "fixture_id": str(fixture.get("id") or "").strip(),
        "built_queries": list(diagnostic.get("built_jump_queries") or []),
        "raw_result_count": int(diagnostic.get("general_result_count") or 0)
        + int(diagnostic.get("academic_result_count") or 0)
        + int(diagnostic.get("alternate_result_count") or 0),
        "filtered_result_count": int(diagnostic.get("filtered_result_count") or 0),
        "retained_result_count": int(diagnostic.get("result_count") or 0),
        "cluster_count": int(diagnostic.get("cluster_count") or 0),
        "highlighted_evidence_count": int(diagnostic.get("highlighted_evidence_count") or 0),
        "intervention_bearing_result_count": int(
            diagnostic.get("intervention_promoted_result_count") or 0
        ),
        "packet_quality": str(diagnostic.get("packet_quality") or "").strip() or "focused",
        "top_cluster_hints": list(diagnostic.get("top_cluster_hints") or []),
        "obvious_source_leakage": _obvious_source_leakage(diagnostic),
        "stage1_outcome": str(diagnostic.get("stage1_outcome") or "").strip() or None,
        "search_call_count": len(search_calls),
        "unexpected_search_calls": len(unexpected_search_calls),
    }
    return scorecard


def _compare_expected(scorecard: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = scorecard.get(key)
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: expected {expected_value!r}, got {actual_value!r}"
            )
    return mismatches


def run_fixture(path: str | Path | dict[str, Any]) -> dict[str, Any]:
    fixture = copy.deepcopy(path) if isinstance(path, dict) else load_fixture(path)
    jump_module = _load_jump_module()
    patch_session = _PatchSession()
    captured: dict[str, Any] = {}
    search_calls: list[dict[str, Any]] = []
    unexpected_search_calls: list[dict[str, Any]] = []
    search_response_index = _build_search_response_index(jump_module, fixture)

    def fake_search(**kwargs: Any) -> dict[str, Any]:
        key = _search_key(
            jump_module,
            query=kwargs.get("query"),
            include_domains=kwargs.get("include_domains"),
        )
        call = {
            "query": key[0],
            "include_domains": list(key[1] or []),
            "lane": "academic" if key[1] else "general",
        }
        search_calls.append(call)
        response = search_response_index.get(key)
        if response is None:
            unexpected_search_calls.append(call)
            return {"results": []}
        return copy.deepcopy(response)

    try:
        patch_session.setattr(jump_module._tavily, "search", fake_search)
        patch_session.setattr(jump_module, "increment_tavily_calls", lambda count=1: None)
        patch_session.setattr(jump_module, "increment_llm_calls", lambda count=1: None)
        patch_session.setattr(
            jump_module,
            "_stage_one_detect_with_diagnostics",
            _make_stage_one_stub(fixture, captured),
        )
        patch_session.setattr(
            jump_module,
            "_stage_two_hypothesize_with_diagnostics",
            _make_stage_two_stub(fixture, captured),
        )

        if fixture.get("built_queries_override"):
            patch_session.setattr(
                jump_module,
                "_build_jump_search_queries",
                _make_built_queries_override(fixture),
            )
        else:
            patch_session.setattr(
                jump_module,
                "_generate_llm_jump_search_query",
                _make_query_builder_stub(fixture),
            )

        connection, diagnostic = jump_module.lateral_jump_with_diagnostics(
            copy.deepcopy(fixture.get("pattern") or {}),
            str(fixture.get("source_domain") or "").strip(),
            str(fixture.get("source_category") or "").strip(),
        )
    finally:
        patch_session.restore()

    scorecard = _build_scorecard(
        fixture,
        diagnostic,
        search_calls,
        unexpected_search_calls,
    )
    expected = fixture.get("expect") if isinstance(fixture.get("expect"), dict) else {}
    mismatches = _compare_expected(scorecard, expected)
    return {
        "fixture": fixture,
        "connection": connection,
        "diagnostic": diagnostic,
        "scorecard": scorecard,
        "search_calls": search_calls,
        "unexpected_search_calls": unexpected_search_calls,
        "captured": captured,
        "mismatches": mismatches,
    }


def run_all_fixtures(fixtures_dir: Path = FIXTURE_DIR) -> list[dict[str, Any]]:
    return [run_fixture(path) for path in load_fixture_paths(fixtures_dir)]


def scorecards_by_fixture(fixtures_dir: Path = FIXTURE_DIR) -> dict[str, dict[str, Any]]:
    return {
        result["scorecard"]["fixture_id"]: result["scorecard"]
        for result in run_all_fixtures(fixtures_dir)
    }
