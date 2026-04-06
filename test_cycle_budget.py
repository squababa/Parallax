import json
import os
import sys
import types

import pytest

os.environ.setdefault("LOCAL_LLM_ONLY", "1")
os.environ.setdefault("BLACKCLAW_MODEL", "qwen3:8b")
os.environ.setdefault("TAVILY_API_KEY", "test")

if "tavily" not in sys.modules:
    fake_tavily = types.ModuleType("tavily")

    class _FakeTavilyClient:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, *args, **kwargs):
            return {"results": []}

    fake_tavily.TavilyClient = _FakeTavilyClient
    sys.modules["tavily"] = fake_tavily

if "llm_client" not in sys.modules:
    fake_llm_client = types.ModuleType("llm_client")

    class _DummyClient:
        def generate_content(self, *args, **kwargs):
            raise AssertionError("Tests should monkeypatch the LLM client.")

    fake_llm_client.get_llm_client = lambda: _DummyClient()
    fake_llm_client.get_provider_status = lambda: {"provider": "test", "status": "ok"}
    sys.modules["llm_client"] = fake_llm_client

if "sanitize" not in sys.modules:
    fake_sanitize = types.ModuleType("sanitize")
    fake_sanitize.sanitize = lambda value: value
    fake_sanitize.check_llm_output = lambda value: value
    sys.modules["sanitize"] = fake_sanitize

import explore
import jump
import main
import store


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate_content(self, *_args, **_kwargs):
        if self.calls >= len(self._responses):
            raise AssertionError("Unexpected LLM call in test.")
        response = FakeResponse(self._responses[self.calls])
        self.calls += 1
        return response


class FakeTavilyClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, *, query, **_kwargs):
        self.calls.append(str(query))
        return {
            "results": [
                {
                    "title": "Operational Threshold Control",
                    "url": "https://example.com/operator-threshold-control",
                    "content": (
                        "Queue threshold control reduces inflow when load rises and "
                        "stabilizes downstream delay."
                    ),
                }
            ]
        }


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "cycle_budget_test.db"
    monkeypatch.setattr(store, "DB_PATH", str(db_path))
    store.init_db()
    return db_path


def _manual_seed(*queries: str) -> dict:
    return {
        "name": "Synthetic Control Systems",
        "category": "Testing",
        "seed_queries": list(queries),
        "quality_profile": {
            "band": "high",
            "score": 0.84,
            "strengths": ["mechanism-rich operator framing"],
            "concerns": [],
        },
    }


def _pattern_payload() -> str:
    return json.dumps(
        {
            "patterns": [
                {
                    "pattern_name": "Queue-threshold congestion gating",
                    "description": (
                        "Queue threshold gating suppresses inflow and stabilizes delay."
                    ),
                    "abstract_structure": (
                        "Increasing load crosses a queue threshold and throttles inflow."
                    ),
                    "search_query": "queue threshold throttling latency",
                    "measurable_signal": "queue length and mean delay",
                    "control_lever": "adjust the congestion threshold",
                    "transfer_rationale": (
                        "Transfers to buffered systems that gate inflow under overload."
                    ),
                }
            ]
        }
    )


def _install_run_cycle_capture(monkeypatch) -> list[dict]:
    saved: list[dict] = []
    monkeypatch.setattr(main, "update_domain_visited", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "print_cycle_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "get_next_transmission_number", lambda: 1)
    monkeypatch.setattr(
        main,
        "save_exploration",
        lambda **kwargs: saved.append(dict(kwargs)) or len(saved),
    )
    return saved


def _stabilize_pattern_extraction(monkeypatch) -> None:
    monkeypatch.setattr(explore, "_is_low_signal_pattern", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(explore, "_rewrite_transferable_pattern_fields", lambda pattern, _seed: pattern)
    monkeypatch.setattr(
        explore,
        "_profile_pattern_quality",
        lambda *_args, **_kwargs: {"band": "high", "jump_ready": True, "concerns": []},
    )


def _pattern(name: str) -> dict:
    return {
        "pattern_name": name,
        "abstract_structure": f"{name} structure",
        "search_query": f"{name} query",
    }


def test_run_cycle_stops_at_seed_search_budget(monkeypatch, temp_db) -> None:
    saved = _install_run_cycle_capture(monkeypatch)
    tavily_client = FakeTavilyClient()

    monkeypatch.setattr(explore, "_tavily", tavily_client)
    monkeypatch.setattr(main, "MAX_TAVILY_CALLS_PER_CYCLE", 1)
    monkeypatch.setattr(main, "MAX_LLM_CALLS_PER_CYCLE", 5)

    transmitted = main.run_cycle(
        cycle_num=1,
        threshold=0.75,
        max_patterns=2,
        manual_seed=_manual_seed("first query", "second query"),
    )

    assert transmitted is False
    assert len(tavily_client.calls) == 1
    diagnostics = saved[-1]["pattern_diagnostics"]
    assert diagnostics["outcome"] == "budget_exhausted_seed_search"
    assert diagnostics["jump_outcome"] == "budget_exhausted_pre_stage1"
    assert diagnostics["budget_stop"]["callsite"] == "seed_search"


def test_run_cycle_stops_at_pattern_extraction_budget(monkeypatch, temp_db) -> None:
    saved = _install_run_cycle_capture(monkeypatch)
    tavily_client = FakeTavilyClient()
    llm_client = FakeLLMClient(["not valid json"])

    monkeypatch.setattr(explore, "_tavily", tavily_client)
    monkeypatch.setattr(explore, "get_llm_client", lambda: llm_client)
    monkeypatch.setattr(main, "MAX_TAVILY_CALLS_PER_CYCLE", 5)
    monkeypatch.setattr(main, "MAX_LLM_CALLS_PER_CYCLE", 1)

    transmitted = main.run_cycle(
        cycle_num=1,
        threshold=0.75,
        max_patterns=2,
        manual_seed=_manual_seed("single query"),
    )

    assert transmitted is False
    assert len(tavily_client.calls) == 1
    assert llm_client.calls == 1
    diagnostics = saved[-1]["pattern_diagnostics"]
    assert diagnostics["outcome"] == "budget_exhausted_pattern_extraction"
    assert diagnostics["jump_outcome"] == "budget_exhausted_pre_stage1"
    assert diagnostics["budget_stop"]["callsite"] == "pattern_extraction"


def test_run_cycle_stops_at_jump_retrieval_budget_before_stage1_detection(
    monkeypatch,
    temp_db,
) -> None:
    saved = _install_run_cycle_capture(monkeypatch)
    _stabilize_pattern_extraction(monkeypatch)
    tavily_client = FakeTavilyClient()
    llm_client = FakeLLMClient([_pattern_payload()])

    monkeypatch.setattr(explore, "_tavily", tavily_client)
    monkeypatch.setattr(jump, "_tavily", tavily_client)
    monkeypatch.setattr(explore, "get_llm_client", lambda: llm_client)
    monkeypatch.setattr(
        jump,
        "_build_jump_search_queries",
        lambda *_args, **_kwargs: ["stage1 transfer query"],
    )
    monkeypatch.setattr(main, "MAX_TAVILY_CALLS_PER_CYCLE", 2)
    monkeypatch.setattr(main, "MAX_LLM_CALLS_PER_CYCLE", 5)

    transmitted = main.run_cycle(
        cycle_num=1,
        threshold=0.75,
        max_patterns=2,
        manual_seed=_manual_seed("single query"),
    )

    assert transmitted is False
    assert len(tavily_client.calls) == 2
    assert llm_client.calls == 1
    diagnostics = saved[-1]["pattern_diagnostics"]
    assert diagnostics["jump_outcome"] == "budget_exhausted_pre_stage1"
    assert diagnostics["jump_attempts"][0]["stage1_outcome"] == "budget_exhausted_pre_stage1"
    assert diagnostics["jump_attempts"][0]["budget_stop"]["callsite"] == "stage1_academic_search"


def test_run_cycle_stops_at_stage1_llm_budget(monkeypatch, temp_db) -> None:
    saved = _install_run_cycle_capture(monkeypatch)
    _stabilize_pattern_extraction(monkeypatch)
    tavily_client = FakeTavilyClient()
    llm_client = FakeLLMClient([_pattern_payload()])

    monkeypatch.setattr(explore, "_tavily", tavily_client)
    monkeypatch.setattr(jump, "_tavily", tavily_client)
    monkeypatch.setattr(explore, "get_llm_client", lambda: llm_client)
    monkeypatch.setattr(jump, "_llm_client", llm_client)
    monkeypatch.setattr(
        jump,
        "_build_jump_search_queries",
        lambda *_args, **_kwargs: ["stage1 transfer query"],
    )
    monkeypatch.setattr(main, "MAX_TAVILY_CALLS_PER_CYCLE", 5)
    monkeypatch.setattr(main, "MAX_LLM_CALLS_PER_CYCLE", 1)

    transmitted = main.run_cycle(
        cycle_num=1,
        threshold=0.75,
        max_patterns=2,
        manual_seed=_manual_seed("single query"),
    )

    assert transmitted is False
    assert len(tavily_client.calls) == 3
    assert llm_client.calls == 1
    diagnostics = saved[-1]["pattern_diagnostics"]
    assert diagnostics["jump_outcome"] == "budget_exhausted_stage1"
    assert diagnostics["jump_attempts"][0]["stage1_outcome"] == "budget_exhausted_stage1"
    assert diagnostics["jump_attempts"][0]["budget_stop"]["callsite"] == "stage1_detect"


def test_run_cycle_treats_pre_stage1_jump_budget_stop_as_terminal_not_miss(
    monkeypatch,
    temp_db,
    capsys,
) -> None:
    saved = _install_run_cycle_capture(monkeypatch)
    jump_attempts = []

    monkeypatch.setattr(main, "describe_seed_quality", lambda seed: seed["quality_profile"])
    monkeypatch.setattr(main, "finalize_pattern_diagnostics", explore.finalize_pattern_diagnostics)
    monkeypatch.setattr(main, "append_jump_attempt_diagnostic", explore.append_jump_attempt_diagnostic)
    def fake_dive(seed: dict, *_args, **_kwargs) -> list[dict]:
        seed["pattern_diagnostics"] = {
            "seed_name": seed["name"],
            "raw_pattern_count": 2,
            "retained_pattern_count": 2,
            "high_quality_count": 2,
            "medium_quality_count": 0,
            "weak_quality_count": 0,
            "jump_ready_count": 2,
            "drop_counts": {
                "missing_required_fields": 0,
                "low_signal": 0,
                "weak_quality": 0,
            },
            "top_rejection_reasons": [],
            "outcome": "patterns_ready",
            "summary": "patterns_ready: kept 2/2 patterns (high=2, medium=0, weak_rejected=0)",
        }
        return [_pattern("first pattern"), _pattern("second pattern")]

    monkeypatch.setattr(main, "dive", fake_dive)

    def fake_lateral_jump(*_args, **_kwargs):
        jump_attempts.append("called")
        return (
            None,
            {
                "pattern_name": "first pattern",
                "stage1_outcome": "budget_exhausted_pre_stage1",
                "stage1_failure_hint": "cycle_budget_exhausted",
                "budget_stop": {
                    "summary": "tavily cycle budget exhausted at stage1_search (2/2)",
                    "callsite": "stage1_search",
                },
            },
        )

    monkeypatch.setattr(main, "lateral_jump_with_diagnostics", fake_lateral_jump)

    transmitted = main.run_cycle(
        cycle_num=1,
        threshold=0.75,
        max_patterns=2,
        manual_seed=_manual_seed("single query"),
    )

    output = capsys.readouterr().out

    assert transmitted is False
    assert len(jump_attempts) == 1
    assert "[Budget] tavily cycle budget exhausted at stage1_search (2/2)" in output
    assert "  [Jump] No connection found" not in output
    diagnostics = saved[-1]["pattern_diagnostics"]
    assert diagnostics["jump_outcome"] == "budget_exhausted_pre_stage1"
    assert len(diagnostics["jump_attempts"]) == 1
    assert diagnostics["jump_attempts"][0]["stage1_outcome"] == "budget_exhausted_pre_stage1"
