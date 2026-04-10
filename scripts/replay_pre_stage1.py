#!/usr/bin/env python3
"""Run deterministic Pre-stage1 replay fixtures and print a scorecard."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pre_stage1_replay import FIXTURE_DIR, load_fixture_paths, run_fixture


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay Pre-stage1 retrieval fixtures without live network or model calls.",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        default=[],
        help="Fixture file name or path. Repeat to run a subset.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a human summary.",
    )
    parser.add_argument(
        "--show-packet",
        action="store_true",
        help="Print the replayed Pre-stage1 packet for each fixture.",
    )
    return parser


def _resolve_fixture_paths(raw_fixtures: list[str]) -> list[Path]:
    if not raw_fixtures:
        return load_fixture_paths(FIXTURE_DIR)

    resolved: list[Path] = []
    for raw_fixture in raw_fixtures:
        candidate = Path(raw_fixture)
        if not candidate.is_absolute():
            fixture_dir_candidate = FIXTURE_DIR / raw_fixture
            if fixture_dir_candidate.exists():
                candidate = fixture_dir_candidate
            else:
                candidate = (REPO_ROOT / raw_fixture).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Fixture not found: {raw_fixture}")
        resolved.append(candidate)
    return resolved


def _human_summary(result: dict, *, show_packet: bool) -> str:
    fixture = result["fixture"]
    scorecard = result["scorecard"]
    diagnostic = result["diagnostic"]
    lines = [
        f"{scorecard['fixture_id']}: {fixture.get('description', '').strip()}",
        (
            "  built_queries="
            + json.dumps(scorecard["built_queries"], ensure_ascii=False)
        ),
        (
            "  raw="
            f"{scorecard['raw_result_count']} filtered={scorecard['filtered_result_count']} "
            f"retained={scorecard['retained_result_count']} clusters={scorecard['cluster_count']}"
        ),
        (
            "  highlights="
            f"{scorecard['highlighted_evidence_count']} intervention_results="
            f"{scorecard['intervention_bearing_result_count']} packet_quality={scorecard['packet_quality']}"
        ),
        (
            "  leakage="
            f"{'yes' if scorecard['obvious_source_leakage'] else 'no'} "
            f"stage1_outcome={scorecard['stage1_outcome'] or 'none'}"
        ),
        "  top_cluster_hints=" + json.dumps(scorecard["top_cluster_hints"], ensure_ascii=False),
        (
            "  search_calls="
            f"{scorecard['search_call_count']} unexpected={scorecard['unexpected_search_calls']}"
        ),
    ]
    if result["mismatches"]:
        lines.append("  status=FAIL")
        for mismatch in result["mismatches"]:
            lines.append(f"    - {mismatch}")
    else:
        lines.append("  status=PASS")
    if show_packet:
        packet = (diagnostic.get("benchmark_snapshot", {}) or {}).get("search_results") or ""
        lines.append("  packet:")
        if packet:
            for raw_line in str(packet).splitlines():
                lines.append(f"    {raw_line}")
        else:
            lines.append("    <empty>")
    return "\n".join(lines)


def _run_fixtures(
    fixture_paths: list[Path],
    *,
    suppress_replay_output: bool,
) -> list[dict]:
    if not suppress_replay_output:
        return [run_fixture(path) for path in fixture_paths]

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
        return [run_fixture(path) for path in fixture_paths]


def main() -> int:
    args = _build_parser().parse_args()
    fixture_paths = _resolve_fixture_paths(args.fixture)
    results = _run_fixtures(
        fixture_paths,
        suppress_replay_output=args.json,
    )

    if args.json:
        payload = [
            {
                "fixture": result["fixture"]["id"],
                "scorecard": result["scorecard"],
                "mismatches": result["mismatches"],
                "search_calls": result["search_calls"],
            }
            for result in results
        ]
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        for index, result in enumerate(results):
            if index:
                print()
            print(_human_summary(result, show_packet=args.show_packet))

    return 1 if any(result["mismatches"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
