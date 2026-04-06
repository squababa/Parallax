import json
from pathlib import Path
import subprocess
import sys

from pre_stage1_replay import FIXTURE_DIR, load_fixture_paths, run_fixture


REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "replay_pre_stage1.py"


def _fixture_label(path: Path) -> str:
    return path.name


def test_pre_stage1_replay_matches_expected_scorecards() -> None:
    failures: list[str] = []
    for fixture_path in load_fixture_paths(FIXTURE_DIR):
        result = run_fixture(fixture_path)
        if not result["mismatches"]:
            continue
        joined = "\n".join(result["mismatches"])
        failures.append(f"{_fixture_label(fixture_path)}\n{joined}")
    assert not failures, "\n\n".join(failures)


def test_pre_stage1_replay_is_deterministic() -> None:
    failures: list[str] = []
    for fixture_path in load_fixture_paths(FIXTURE_DIR):
        first = run_fixture(fixture_path)
        second = run_fixture(fixture_path)
        if first["scorecard"] != second["scorecard"]:
            failures.append(f"{_fixture_label(fixture_path)} scorecard changed across runs")
        if first["search_calls"] != second["search_calls"]:
            failures.append(f"{_fixture_label(fixture_path)} search calls changed across runs")
        first_packet = (
            first["diagnostic"].get("benchmark_snapshot", {}) or {}
        ).get("search_results")
        second_packet = (
            second["diagnostic"].get("benchmark_snapshot", {}) or {}
        ).get("search_results")
        if first_packet != second_packet:
            failures.append(f"{_fixture_label(fixture_path)} packet text changed across runs")
    assert not failures, "\n".join(failures)


def test_replay_script_json_mode_emits_parseable_json_only() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""

    payload = json.loads(completed.stdout)
    assert isinstance(payload, list)
    assert len(payload) == len(load_fixture_paths(FIXTURE_DIR))
    assert "[Jump]" not in completed.stdout
    assert {
        str(item.get("fixture") or "").strip()
        for item in payload
        if isinstance(item, dict)
    } == {
        path.stem
        for path in load_fixture_paths(FIXTURE_DIR)
    }
