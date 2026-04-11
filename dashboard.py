import hashlib
import json
import math
import os
import re
import sqlite3
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import quote

from flask import Flask, Response, jsonify, render_template, request


DEFAULT_EXPLORATION_LIMIT = 50
DEFAULT_STATS_WINDOW = 200
TOP_KILLED_LIMIT = 10
FRONTIER_LAUNCHPAD_LIMIT = 10
FRONTIER_STRONG_BRIDGE_LIMIT = 12
FRONTIER_SALVAGE_LIMIT = 8
FRONTIER_REGION_LIMIT = 3
FRONTIER_SEED_LIMIT = 5
FRONTIER_NEAR_MISS_SCORE = 0.75
FRONTIER_FRESHNESS_FULL_DAYS = 7
FRONTIER_FRESHNESS_DECAY_DAYS = 30
VALID_GRADES = ("A", "B+", "B", "B-", "C+", "C", "D", "F")
CLAUDE_SONNET_INPUT_RATE_PER_MTOK = 3.0
CLAUDE_SONNET_OUTPUT_RATE_PER_MTOK = 15.0
BLENDED_RATE_PER_MTOK = 9.0
API_USAGE_INPUT_COLUMNS = ("input_tokens", "prompt_tokens")
API_USAGE_OUTPUT_COLUMNS = ("output_tokens", "completion_tokens")
API_USAGE_MODEL_COLUMNS = ("model", "model_name")
API_USAGE_TIME_COLUMNS = ("timestamp", "created_at", "recorded_at", "date")
VALID_STRONG_REJECTION_STATUSES = ("open", "salvaged", "dismissed")
DB_PATH = str(
    Path(
        os.getenv("DB_PATH")
        or os.getenv("BLACKCLAW_DB_PATH")
        or "blackclaw.db"
    ).expanduser().resolve()
)
os.environ["BLACKCLAW_DB_PATH"] = DB_PATH

from store import (
    get_strong_rejection,
    get_strong_rejection_stats,
    get_prediction_evidence_hit,
    get_prediction_evidence_review_stats,
    get_prediction_outcome_review,
    get_prediction_outcome_suggestion_stats,
    init_db,
    list_strong_rejections,
    list_prediction_evidence_review_queue,
    list_prediction_outcome_review_queue,
    update_strong_rejection_status,
    update_prediction_evidence_review_status,
)

app = Flask(__name__)
init_db()


def _db_uri(mode: str = "ro") -> str:
    return f"{Path(DB_PATH).as_uri()}?mode={mode}"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_uri("ro"), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _connect_write() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_uri("rw"), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return dict(row)


def _parse_positive_int(raw_value: str | None, default: int) -> int:
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _clean_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_optional_json_object() -> dict:
    payload = request.get_json(silent=True)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object payload")
    return payload


def _parse_optional_note_payload(payload: dict) -> str | None:
    raw_note = payload.get("note")
    if raw_note is None:
        raw_note = payload.get("notes")
    return _clean_optional_text(raw_note)


def _update_evidence_review_status_response(
    evidence_id: int,
    review_status: str,
):
    try:
        payload = _parse_optional_json_object()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    updated = update_prediction_evidence_review_status(
        evidence_id,
        review_status,
        notes=_parse_optional_note_payload(payload),
    )
    if not updated:
        return jsonify({"error": "evidence hit not found", "id": evidence_id}), 404
    return jsonify(
        {
            "ok": True,
            "evidence": get_prediction_evidence_hit(evidence_id),
        }
    )


def _update_strong_rejection_status_response(
    rejection_id: int,
    status: str,
):
    try:
        payload = _parse_optional_json_object()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    updated = update_strong_rejection_status(
        rejection_id,
        status,
        notes=_parse_optional_note_payload(payload),
    )
    if not updated:
        return jsonify({"error": "strong rejection not found", "id": rejection_id}), 404
    return jsonify(
        {
            "ok": True,
            "strong_rejection": get_strong_rejection(rejection_id),
        }
    )


def _operator_home_outcome_sort_key(row: dict) -> tuple:
    priority = {
        "review_for_support": 0,
        "review_for_contradiction": 1,
        "conflicting_evidence": 2,
        "waiting_on_review": 3,
        "insufficient_evidence": 4,
    }
    support_hits = _coerce_int(row.get("accepted_support_hits"))
    contradiction_hits = _coerce_int(row.get("accepted_contradiction_hits"))
    unreviewed_hits = _coerce_int(row.get("unreviewed_reviewable_hits"))
    return (
        priority.get(str(row.get("recommendation") or ""), 99),
        -(support_hits + contradiction_hits),
        -max(support_hits, contradiction_hits),
        -unreviewed_hits,
        -_coerce_int(row.get("id")),
    )


def _operator_home_strong_rejection_sort_key(row: dict) -> tuple:
    score = row.get("total_score")
    safe_score = float(score) if isinstance(score, (int, float)) else 0.0
    return (
        score is None,
        -safe_score,
        -_coerce_int(row.get("id")),
    )


def _operator_home_prediction_summary(row: dict) -> str:
    prediction_json = row.get("prediction_json")
    if isinstance(prediction_json, dict):
        statement = _clean_optional_text(prediction_json.get("statement"))
        if statement is not None:
            return statement
    return _clean_optional_text(row.get("prediction")) or "—"


def _build_operator_home_snapshot() -> dict:
    evidence_stats = get_prediction_evidence_review_stats()
    outcome_stats = get_prediction_outcome_suggestion_stats()
    strong_rejection_stats = get_strong_rejection_stats()

    evidence_backlog_rows = list_prediction_evidence_review_queue(limit=5)

    outcome_backlog_rows = [
        row
        for row in list_prediction_outcome_review_queue(limit=None)
        if str(row.get("outcome_status") or "open").strip().lower() == "open"
    ]
    outcome_backlog_rows.sort(key=_operator_home_outcome_sort_key)

    strong_rejection_backlog_rows = list_strong_rejections(limit=100, status="open")
    strong_rejection_backlog_rows.sort(key=_operator_home_strong_rejection_sort_key)

    review_backlog = outcome_stats.get("review_backlog") or {}
    overall = outcome_stats.get("overall") or {}

    counts = {
        "unreviewed_evidence_hits": _coerce_int(
            (evidence_stats.get("by_review_status") or {}).get("unreviewed")
        ),
        "predictions_needing_review": _coerce_int(
            evidence_stats.get("predictions_needing_review")
        ),
        "open_strong_rejections": _coerce_int(strong_rejection_stats.get("open")),
        "open_predictions": _coerce_int(overall.get("open")),
        "review_for_support_candidates": _coerce_int(
            review_backlog.get("open_predictions_with_accepted_support_only")
        ),
        "review_for_contradiction_candidates": _coerce_int(
            review_backlog.get("open_predictions_with_accepted_contradiction_only")
        ),
        "conflicting_evidence_predictions": _coerce_int(
            review_backlog.get("open_predictions_with_accepted_conflicting_evidence")
        ),
    }

    return {
        "counts": counts,
        "flags": {
            "has_unreviewed_evidence": counts["unreviewed_evidence_hits"] > 0,
            "has_outcome_candidates": bool(outcome_backlog_rows),
            "has_open_strong_rejections": counts["open_strong_rejections"] > 0,
        },
        "evidence_backlog": [
            {
                "id": _coerce_int(row.get("id")),
                "prediction_id": _coerce_int(row.get("prediction_id")),
                "classification": row.get("classification") or "unclear",
                "score": row.get("score"),
                "title": row.get("title") or "Untitled result",
                "scan_timestamp": row.get("scan_timestamp"),
            }
            for row in evidence_backlog_rows[:5]
        ],
        "outcome_backlog": [
            {
                "id": _coerce_int(row.get("id")),
                "transmission_number": row.get("transmission_number"),
                "recommendation": row.get("recommendation")
                or "insufficient_evidence",
                "mechanism_type": row.get("mechanism_type") or "unknown",
                "utility_class": row.get("utility_class") or "unknown",
                "prediction_summary": _operator_home_prediction_summary(row),
            }
            for row in outcome_backlog_rows[:5]
        ],
        "strong_rejection_backlog": [
            {
                "id": _coerce_int(row.get("id")),
                "total_score": row.get("total_score"),
                "mechanism_type": row.get("mechanism_type") or "unknown",
                "seed_domain": row.get("seed_domain") or "—",
                "target_domain": row.get("target_domain") or "—",
                "salvage_reason": row.get("salvage_reason") or "—",
                "timestamp": row.get("timestamp"),
            }
            for row in strong_rejection_backlog_rows[:5]
        ],
    }


def _pick_existing_column(
    columns: set[str], candidates: tuple[str, ...]
) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _read_api_usage_columns(conn) -> set[str]:
    try:
        return {
            row["name"]
            for row in conn.execute("PRAGMA table_info(api_usage)").fetchall()
        }
    except sqlite3.Error:
        return set()


def _coerce_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _coerce_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp_float(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(maximum, max(minimum, float(value)))


def _safe_ratio(numerator, denominator) -> float:
    safe_denominator = _coerce_float(denominator)
    if safe_denominator <= 0:
        return 0.0
    return _coerce_float(numerator) / safe_denominator


def _parse_timestamp(value) -> datetime | None:
    text = _clean_optional_text(value)
    if text is None:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp_is_newer(candidate: str | None, current: str | None) -> bool:
    if candidate is None:
        return False
    if current is None:
        return True
    return candidate > current


def _normalized_log(value, max_value) -> float:
    safe_max_value = _coerce_float(max_value)
    if safe_max_value <= 0:
        return 0.0
    safe_value = max(0.0, _coerce_float(value))
    return _clamp_float(math.log1p(safe_value) / math.log1p(safe_max_value))


def _freshness_norm(timestamp_value: str | None, *, now: datetime | None = None) -> float:
    timestamp = _parse_timestamp(timestamp_value)
    if timestamp is None:
        return 0.0
    current_time = now or datetime.now(timezone.utc)
    age_days = max(0.0, (current_time - timestamp).total_seconds() / 86400.0)
    if age_days <= FRONTIER_FRESHNESS_FULL_DAYS:
        return 1.0
    if age_days >= FRONTIER_FRESHNESS_DECAY_DAYS:
        return 0.0
    decay_window = FRONTIER_FRESHNESS_DECAY_DAYS - FRONTIER_FRESHNESS_FULL_DAYS
    return _clamp_float((FRONTIER_FRESHNESS_DECAY_DAYS - age_days) / decay_window)


def _timestamp_sort_value(timestamp_value: str | None) -> float:
    timestamp = _parse_timestamp(timestamp_value)
    if timestamp is None:
        return 0.0
    return timestamp.timestamp()


def _pluralize(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return singular
    return plural or f"{singular}s"


def _domain_anchor_score(row: dict) -> int:
    return (
        _coerce_int(row.get("connection_count")) * 3
        + _coerce_int(row.get("transmitted_count")) * 4
        + _coerce_int(row.get("appearance_count"))
    )


def _cluster_id_for_domains(domain_ids: list[str]) -> str:
    digest = hashlib.sha1(
        "|".join(sorted(domain_ids, key=lambda value: str(value).lower())).encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    return f"cluster-{digest}"


def _frontier_region_why(entry: dict) -> list[str]:
    reasons: list[str] = []
    transmitted_link_count = _coerce_int(entry.get("transmitted_link_count"))
    near_miss_count = _coerce_int(entry.get("near_miss_count"))
    open_salvage_count = _coerce_int(entry.get("open_salvage_count"))
    avg_score = entry.get("avg_score")
    if transmitted_link_count > 0:
        reasons.append(
            f"{transmitted_link_count} "
            f"{_pluralize(transmitted_link_count, 'transmitting bridge')}"
        )
    if near_miss_count > 0:
        reasons.append(
            f"{near_miss_count} { _pluralize(near_miss_count, 'near miss', 'near misses')}"
        )
    if open_salvage_count > 0:
        reasons.append(
            f"{open_salvage_count} {_pluralize(open_salvage_count, 'salvage lead')}"
        )
    if isinstance(avg_score, (int, float)) and avg_score >= FRONTIER_NEAR_MISS_SCORE:
        reasons.append(f"avg score {avg_score:.3f}")
    if _freshness_norm(entry.get("latest_timestamp")) >= 0.95:
        reasons.append("fresh activity")
    if not reasons:
        link_count = _coerce_int(entry.get("link_count"))
        reasons.append(f"{link_count} {_pluralize(link_count, 'mapped bridge')}")
    return reasons[:3]


def _frontier_seed_why(entry: dict) -> list[str]:
    reasons: list[str] = []
    transmitted_count = _coerce_int(entry.get("transmitted_count"))
    near_miss_count = _coerce_int(entry.get("near_miss_count"))
    open_salvage_count = _coerce_int(entry.get("open_salvage_count"))
    transmitted_neighbor_count = _coerce_int(entry.get("transmitted_neighbor_count"))
    avg_score = entry.get("avg_score")
    if transmitted_count > 0:
        reasons.append(
            f"{transmitted_count} {_pluralize(transmitted_count, 'transmission')}"
        )
    if near_miss_count > 0:
        reasons.append(
            f"{near_miss_count} { _pluralize(near_miss_count, 'near miss', 'near misses')}"
        )
    if open_salvage_count > 0:
        reasons.append(
            f"{open_salvage_count} {_pluralize(open_salvage_count, 'salvage lead')}"
        )
    if transmitted_neighbor_count > 0:
        reasons.append(
            f"{transmitted_neighbor_count} "
            f"{_pluralize(transmitted_neighbor_count, 'transmitting neighbor')}"
        )
    if isinstance(avg_score, (int, float)) and avg_score >= FRONTIER_NEAR_MISS_SCORE:
        reasons.append(f"avg score {avg_score:.3f}")
    if _freshness_norm(entry.get("latest_timestamp")) >= 0.95:
        reasons.append("fresh activity")
    if not reasons:
        exploration_count = _coerce_int(entry.get("exploration_count"))
        reasons.append(
            f"{exploration_count} {_pluralize(exploration_count, 'exploration')}"
        )
    return reasons[:3]


def _uses_sonnet_pricing(model_name) -> bool:
    return "sonnet" in str(model_name or "").strip().lower()


def _estimate_usage_cost(rows) -> tuple[int, int, float]:
    total_input_tokens = 0
    total_output_tokens = 0
    estimated_cost = 0.0

    for row in rows:
        input_tokens = _coerce_int(row["input_tokens"])
        output_tokens = _coerce_int(row["output_tokens"])
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        if _uses_sonnet_pricing(row["model"]):
            estimated_cost += (
                (input_tokens / 1_000_000) * CLAUDE_SONNET_INPUT_RATE_PER_MTOK
            )
            estimated_cost += (
                (output_tokens / 1_000_000)
                * CLAUDE_SONNET_OUTPUT_RATE_PER_MTOK
            )
        else:
            estimated_cost += (
                (input_tokens + output_tokens) / 1_000_000
            ) * BLENDED_RATE_PER_MTOK

    return total_input_tokens, total_output_tokens, estimated_cost


def _get_transmissions() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """SELECT
            t.id,
            t.transmission_number,
            t.timestamp,
            t.exploration_id,
            t.formatted_output,
            t.mechanism_signature,
            t.signature_cluster_id,
            t.exportable,
            t.user_rating,
            t.user_notes,
            t.dive_result,
            t.dive_timestamp,
            e.seed_domain,
            e.jump_target_domain,
            e.connection_description,
            e.novelty_score,
            e.distance_score,
            e.depth_score,
            e.total_score
        FROM transmissions t
        LEFT JOIN explorations e ON e.id = t.exploration_id
        ORDER BY t.transmission_number DESC, t.id DESC"""
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _get_transmission_timeline() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """SELECT
            e.id AS exploration_id,
            COALESCE(t.timestamp, e.timestamp) AS timestamp,
            e.total_score,
            e.transmitted,
            t.transmission_number
        FROM explorations e
        LEFT JOIN transmissions t ON t.exploration_id = e.id
        WHERE e.total_score IS NOT NULL
          AND COALESCE(t.timestamp, e.timestamp) IS NOT NULL
        ORDER BY COALESCE(t.timestamp, e.timestamp) ASC, e.id ASC"""
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _get_transmission_export_row(transmission_id: int) -> dict | None:
    conn = _connect()
    row = conn.execute(
        """SELECT
            t.id,
            t.transmission_number,
            t.formatted_output,
            e.seed_domain,
            e.jump_target_domain,
            e.connection_description,
            e.novelty_score,
            e.depth_score,
            e.distance_score,
            e.total_score
        FROM transmissions t
        LEFT JOIN explorations e ON e.id = t.exploration_id
        WHERE t.id = ?""",
        (transmission_id,),
    ).fetchone()
    conn.close()
    return _row_to_dict(row)


def _format_score(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _get_recent_explorations(
    limit: int,
    seed_domain: str | None = None,
) -> list[dict]:
    clean_seed_domain = (seed_domain or "").strip() or None
    conn = _connect()
    if clean_seed_domain is None:
        rows = conn.execute(
            """SELECT *
            FROM explorations
            ORDER BY timestamp DESC, id DESC
            LIMIT ?""",
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT *
            FROM explorations
            WHERE seed_domain = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?""",
            (clean_seed_domain, limit),
        ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _get_exploration(exploration_id: int) -> dict | None:
    conn = _connect()
    row = conn.execute(
        """SELECT *
        FROM explorations
        WHERE id = ?""",
        (exploration_id,),
    ).fetchone()
    conn.close()
    return _row_to_dict(row)


def _get_adversarial_rubric(exploration_id: int):
    conn = _connect()
    row = conn.execute(
        """SELECT adversarial_rubric_json
        FROM explorations
        WHERE id = ?""",
        (exploration_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    raw_value = row["adversarial_rubric_json"]
    if raw_value is None:
        return {"adversarial": None}
    return {"adversarial": json.loads(raw_value)}


def _get_kill_stats(window: int) -> dict:
    conn = _connect()
    row = conn.execute(
        """WITH recent AS (
            SELECT
                timestamp,
                patterns_found,
                total_score,
                validation_json,
                adversarial_rubric_json,
                seed_url,
                target_url,
                distance_score,
                transmitted
            FROM explorations
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
        )
        SELECT
            COUNT(*) AS total_explorations,
            COALESCE(SUM(transmitted), 0) AS total_transmitted,
            COALESCE(
                SUM(
                    CASE
                        WHEN patterns_found IS NULL
                        OR TRIM(patterns_found) IN ('', '[]', '{}')
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS no_patterns_found,
            COALESCE(
                SUM(CASE WHEN total_score < 0.6 THEN 1 ELSE 0 END),
                0
            ) AS below_score_threshold,
            COALESCE(
                SUM(
                    CASE
                        WHEN validation_json IS NOT NULL AND transmitted = 0
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS validation_rejected,
            COALESCE(
                SUM(
                    CASE
                        WHEN adversarial_rubric_json IS NOT NULL AND transmitted = 0
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS adversarial_killed,
            COALESCE(
                SUM(
                    CASE
                        WHEN seed_url IS NULL OR target_url IS NULL
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS provenance_missing,
            COALESCE(
                SUM(CASE WHEN distance_score < 0.5 THEN 1 ELSE 0 END),
                0
            ) AS distance_too_low,
            MIN(timestamp) AS oldest_timestamp,
            MAX(timestamp) AS newest_timestamp,
            AVG(total_score) AS avg_total_score_all,
            AVG(CASE WHEN transmitted = 1 THEN total_score END)
                AS avg_total_score_transmitted
        FROM recent""",
        (window,),
    ).fetchone()
    conn.close()

    report = dict(row)
    total = int(report.get("total_explorations", 0) or 0)
    transmitted = int(report.get("total_transmitted", 0) or 0)
    transmission_rate = (transmitted / total * 100.0) if total else 0.0
    report["window_requested"] = window
    report["transmission_rate"] = round(transmission_rate, 1)
    return report


def _get_cost_stats(window: int = DEFAULT_STATS_WINDOW) -> dict:
    report = _get_kill_stats(window)
    total_explorations = _coerce_int(report.get("total_explorations"))
    total_transmissions = _coerce_int(report.get("total_transmitted"))
    window_start = report.get("oldest_timestamp")
    window_end = report.get("newest_timestamp")
    if total_explorations <= 0 or not window_start or not window_end:
        return {"available": False, "message": "No cost data available"}

    conn = _connect()
    try:
        columns = _read_api_usage_columns(conn)
        input_column = _pick_existing_column(columns, API_USAGE_INPUT_COLUMNS)
        output_column = _pick_existing_column(columns, API_USAGE_OUTPUT_COLUMNS)
        model_column = _pick_existing_column(columns, API_USAGE_MODEL_COLUMNS)
        time_column = _pick_existing_column(columns, API_USAGE_TIME_COLUMNS)
        if input_column is None or output_column is None or time_column is None:
            return {"available": False, "message": "No cost data available"}

        start_value = window_start[:10] if time_column == "date" else window_start
        end_value = window_end[:10] if time_column == "date" else window_end
        model_sql = (
            f"{model_column} AS model" if model_column is not None else "NULL AS model"
        )
        usage_rows = conn.execute(
            f"""SELECT
                {input_column} AS input_tokens,
                {output_column} AS output_tokens,
                {model_sql}
            FROM api_usage
            WHERE {time_column} BETWEEN ? AND ?
            ORDER BY {time_column} ASC""",
            (start_value, end_value),
        ).fetchall()
    except sqlite3.Error:
        return {"available": False, "message": "No cost data available"}
    finally:
        conn.close()

    if not usage_rows:
        return {"available": False, "message": "No cost data available"}

    total_input_tokens, total_output_tokens, estimated_cost = _estimate_usage_cost(
        usage_rows
    )
    total_tokens = total_input_tokens + total_output_tokens

    return {
        "available": True,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_total_cost": estimated_cost,
        "cost_per_transmission": (
            estimated_cost / total_transmissions
            if total_transmissions > 0
            else None
        ),
        "cost_per_exploration": (
            estimated_cost / total_explorations if total_explorations > 0 else None
        ),
        "tokens_per_exploration": (
            total_tokens / total_explorations if total_explorations > 0 else None
        ),
        "window_requested": window,
        "transmission_count": total_transmissions,
        "exploration_count": total_explorations,
    }


def _get_top_killed(limit: int = TOP_KILLED_LIMIT) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """SELECT *
        FROM explorations
        WHERE transmitted = 0
        ORDER BY total_score IS NULL ASC, total_score DESC, timestamp DESC, id DESC
        LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _get_domain_stats() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """SELECT
            e.seed_domain,
            COUNT(*) AS exploration_count,
            COALESCE(SUM(e.transmitted), 0) AS transmitted_count,
            AVG(e.total_score) AS avg_total_score,
            (
                SELECT e2.id
                FROM explorations e2
                WHERE e2.seed_domain = e.seed_domain
                ORDER BY e2.total_score IS NULL ASC,
                    e2.total_score DESC,
                    e2.timestamp DESC,
                    e2.id DESC
                LIMIT 1
            ) AS best_exploration_id,
            (
                SELECT e2.jump_target_domain
                FROM explorations e2
                WHERE e2.seed_domain = e.seed_domain
                ORDER BY e2.total_score IS NULL ASC,
                    e2.total_score DESC,
                    e2.timestamp DESC,
                    e2.id DESC
                LIMIT 1
            ) AS best_jump_target_domain,
            (
                SELECT e2.connection_description
                FROM explorations e2
                WHERE e2.seed_domain = e.seed_domain
                ORDER BY e2.total_score IS NULL ASC,
                    e2.total_score DESC,
                    e2.timestamp DESC,
                    e2.id DESC
                LIMIT 1
            ) AS best_connection_description,
            (
                SELECT e2.total_score
                FROM explorations e2
                WHERE e2.seed_domain = e.seed_domain
                ORDER BY e2.total_score IS NULL ASC,
                    e2.total_score DESC,
                    e2.timestamp DESC,
                    e2.id DESC
                LIMIT 1
            ) AS best_total_score
        FROM explorations e
        GROUP BY e.seed_domain
        ORDER BY exploration_count DESC, e.seed_domain ASC"""
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _get_graph_exploration_rows() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """SELECT
            seed_domain,
            jump_target_domain,
            total_score,
            transmitted,
            timestamp
        FROM explorations
        WHERE seed_domain IS NOT NULL
          AND TRIM(seed_domain) <> ''
          AND jump_target_domain IS NOT NULL
          AND TRIM(jump_target_domain) <> ''
        ORDER BY timestamp ASC, id ASC"""
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _get_open_salvage_rows() -> list[dict]:
    conn = None
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT
                seed_domain AS source,
                target_domain AS target,
                total_score,
                salvage_reason AS reason,
                status,
                timestamp
            FROM strong_rejections
            WHERE status = 'open'
            ORDER BY total_score IS NULL ASC,
                total_score DESC,
                timestamp DESC,
                id DESC"""
        ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error:
        return []
    finally:
        if conn is not None:
            conn.close()


def _get_domain_graph() -> dict:
    rows = _get_graph_exploration_rows()

    node_map: dict[str, dict] = {}
    link_map: dict[tuple[str, str], dict] = {}
    unique_transmitted_domains: set[str] = set()
    total_score = 0.0
    scored_rows = 0
    total_explorations = 0
    total_transmissions = 0

    for row in rows:
        seed_domain = _clean_optional_text(row.get("seed_domain"))
        target_domain = _clean_optional_text(row.get("jump_target_domain"))
        if seed_domain is None or target_domain is None:
            continue

        total_explorations += 1
        transmitted = 1 if _coerce_int(row.get("transmitted")) > 0 else 0
        total_transmissions += transmitted
        timestamp = _clean_optional_text(row.get("timestamp"))

        raw_score = row.get("total_score")
        if raw_score is None:
            score = None
        else:
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = None
        if score is not None:
            total_score += score
            scored_rows += 1

        for domain_name, is_seed in ((seed_domain, True), (target_domain, False)):
            node = node_map.setdefault(
                domain_name,
                {
                    "id": domain_name,
                    "appearance_count": 0,
                    "seed_count": 0,
                    "target_count": 0,
                    "transmitted_count": 0,
                    "incoming_edges": 0,
                    "outgoing_edges": 0,
                    "connected_domains": set(),
                    "score_total": 0.0,
                    "score_count": 0,
                    "max_score": None,
                    "latest_timestamp": None,
                },
            )
            node["appearance_count"] += 1
            if is_seed:
                node["seed_count"] += 1
            else:
                node["target_count"] += 1
            node["transmitted_count"] += transmitted
            if transmitted:
                unique_transmitted_domains.add(domain_name)
            if score is not None:
                node["score_total"] += score
                node["score_count"] += 1
                if node["max_score"] is None or score > node["max_score"]:
                    node["max_score"] = score
            if _timestamp_is_newer(timestamp, node["latest_timestamp"]):
                node["latest_timestamp"] = timestamp

        link = link_map.setdefault(
            (seed_domain, target_domain),
            {
                "source": seed_domain,
                "target": target_domain,
                "count": 0,
                "transmitted_count": 0,
                "score_total": 0.0,
                "score_count": 0,
                "max_score": None,
                "latest_timestamp": None,
            },
        )
        link["count"] += 1
        link["transmitted_count"] += transmitted
        if score is not None:
            link["score_total"] += score
            link["score_count"] += 1
            if link["max_score"] is None or score > link["max_score"]:
                link["max_score"] = score
        if _timestamp_is_newer(timestamp, link["latest_timestamp"]):
            link["latest_timestamp"] = timestamp

    for source, target in link_map:
        node_map[source]["outgoing_edges"] += 1
        node_map[target]["incoming_edges"] += 1
        node_map[source]["connected_domains"].add(target)
        node_map[target]["connected_domains"].add(source)

    nodes = []
    node_lookup: dict[str, dict] = {}
    for node in node_map.values():
        if node["seed_count"] > 0 and node["target_count"] > 0:
            role = "bridge"
        elif node["seed_count"] > 0:
            role = "seed"
        else:
            role = "target"
        score_count = node["score_count"]
        node_payload = {
            "id": node["id"],
            "label": node["id"],
            "role": role,
            "appearance_count": node["appearance_count"],
            "seed_count": node["seed_count"],
            "target_count": node["target_count"],
            "transmitted_count": node["transmitted_count"],
            "incoming_edges": node["incoming_edges"],
            "outgoing_edges": node["outgoing_edges"],
            "connection_count": len(node["connected_domains"]),
            "avg_score": node["score_total"] / score_count if score_count > 0 else None,
            "max_score": node["max_score"],
            "latest_timestamp": node["latest_timestamp"],
        }
        nodes.append(node_payload)
        node_lookup[node_payload["id"]] = node_payload

    links = []
    for link in link_map.values():
        score_count = link["score_count"]
        links.append(
            {
                "source": link["source"],
                "target": link["target"],
                "count": link["count"],
                "transmitted_count": link["transmitted_count"],
                "avg_score": link["score_total"] / score_count if score_count > 0 else None,
                "max_score": link["max_score"],
                "latest_timestamp": link["latest_timestamp"],
            }
        )

    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_lookup}
    for link in links:
        adjacency.setdefault(link["source"], set()).add(link["target"])
        adjacency.setdefault(link["target"], set()).add(link["source"])

    clusters = []
    visited: set[str] = set()
    cluster_metric_map: dict[str, dict] = {}
    for node_id in sorted(node_lookup, key=lambda value: str(value).lower()):
        if node_id in visited:
            continue
        stack = [node_id]
        component_ids: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component_ids.append(current)
            neighbors = sorted(
                adjacency.get(current, set()),
                key=lambda value: str(value).lower(),
                reverse=True,
            )
            stack.extend(neighbors)

        component_set = set(component_ids)
        component_nodes = [node_lookup[component_id] for component_id in component_ids]
        component_links = [
            link
            for link in links
            if link["source"] in component_set and link["target"] in component_set
        ]
        cluster_id = _cluster_id_for_domains(component_ids)
        ranked_domains = sorted(
            component_nodes,
            key=lambda row: (
                -_domain_anchor_score(row),
                str(row.get("id") or "").lower(),
            ),
        )
        top_domains = [row["id"] for row in ranked_domains[:5]]
        label = " / ".join(top_domains[:2]) if len(top_domains) > 1 else top_domains[0]
        if not label:
            label = component_ids[0]
        for component_id in component_ids:
            node_lookup[component_id]["cluster_id"] = cluster_id
        clusters.append(
            {
                "id": cluster_id,
                "label": label,
                "node_count": len(component_nodes),
                "link_count": len(component_links),
                "transmitted_link_count": sum(
                    1
                    for link in component_links
                    if _coerce_int(link.get("transmitted_count")) > 0
                ),
                "avg_score": None,
                "latest_timestamp": None,
                "top_domains": top_domains,
                "seed_count": sum(
                    1 for row in component_nodes if row.get("role") == "seed"
                ),
                "target_count": sum(
                    1 for row in component_nodes if row.get("role") == "target"
                ),
                "bridge_count": sum(
                    1 for row in component_nodes if row.get("role") == "bridge"
                ),
                "exploration_count": 0,
                "transmitted_exploration_count": 0,
                "open_salvage_count": 0,
            }
        )
        cluster_metric_map[cluster_id] = {
            "score_total": 0.0,
            "score_count": 0,
            "latest_timestamp": None,
            "exploration_count": 0,
            "transmitted_exploration_count": 0,
        }

    cluster_lookup = {cluster["id"]: cluster for cluster in clusters}
    for row in rows:
        seed_domain = _clean_optional_text(row.get("seed_domain"))
        target_domain = _clean_optional_text(row.get("jump_target_domain"))
        cluster_id = None
        if seed_domain is not None and seed_domain in node_lookup:
            cluster_id = node_lookup[seed_domain].get("cluster_id")
        if cluster_id is None and target_domain is not None and target_domain in node_lookup:
            cluster_id = node_lookup[target_domain].get("cluster_id")
        if cluster_id is None:
            continue
        metrics = cluster_metric_map[cluster_id]
        metrics["exploration_count"] += 1
        metrics["transmitted_exploration_count"] += (
            1 if _coerce_int(row.get("transmitted")) > 0 else 0
        )
        raw_score = row.get("total_score")
        if raw_score is not None:
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = None
            if score is not None:
                metrics["score_total"] += score
                metrics["score_count"] += 1
        timestamp = _clean_optional_text(row.get("timestamp"))
        if _timestamp_is_newer(timestamp, metrics["latest_timestamp"]):
            metrics["latest_timestamp"] = timestamp

    for cluster_id, metrics in cluster_metric_map.items():
        cluster = cluster_lookup[cluster_id]
        cluster["avg_score"] = (
            metrics["score_total"] / metrics["score_count"]
            if metrics["score_count"] > 0
            else None
        )
        cluster["latest_timestamp"] = metrics["latest_timestamp"]
        cluster["exploration_count"] = metrics["exploration_count"]
        cluster["transmitted_exploration_count"] = metrics[
            "transmitted_exploration_count"
        ]

    for row in _get_open_salvage_rows():
        affected_cluster_ids = {
            cluster_id
            for cluster_id in (
                node_lookup.get(_clean_optional_text(row.get("source")) or "", {}).get(
                    "cluster_id"
                ),
                node_lookup.get(_clean_optional_text(row.get("target")) or "", {}).get(
                    "cluster_id"
                ),
            )
            if cluster_id is not None
        }
        for cluster_id in affected_cluster_ids:
            cluster_lookup[cluster_id]["open_salvage_count"] += 1

    nodes.sort(
        key=lambda row: (
            -_coerce_int(row.get("connection_count")),
            -_coerce_int(row.get("transmitted_count")),
            -_coerce_int(row.get("appearance_count")),
            str(row.get("id") or "").lower(),
        )
    )
    links.sort(
        key=lambda row: (
            -_coerce_int(row.get("transmitted_count")),
            -(float(row["max_score"]) if row.get("max_score") is not None else -1.0),
            -_coerce_int(row.get("count")),
            str(row.get("source") or "").lower(),
            str(row.get("target") or "").lower(),
        )
    )
    clusters.sort(
        key=lambda row: (
            -_coerce_int(row.get("node_count")),
            -_coerce_int(row.get("transmitted_link_count")),
            -_coerce_float(row.get("avg_score")),
            str(row.get("label") or "").lower(),
        )
    )

    return {
        "stats": {
            "unique_domains": len(nodes),
            "unique_connections": len(links),
            "transmitted_connections": sum(
                1 for link in links if _coerce_int(link.get("transmitted_count")) > 0
            ),
            "successful_domains": len(unique_transmitted_domains),
            "total_explorations": total_explorations,
            "total_transmissions": total_transmissions,
            "avg_score": total_score / scored_rows if scored_rows > 0 else None,
            "cluster_count": len(clusters),
            "largest_cluster_size": max(
                (_coerce_int(cluster.get("node_count")) for cluster in clusters),
                default=0,
            ),
        },
        "nodes": nodes,
        "links": links,
        "clusters": clusters,
    }


def _get_frontier_snapshot() -> dict:
    graph = _get_domain_graph()
    exploration_rows = _get_graph_exploration_rows()
    salvage_candidates = _get_open_salvage_rows()

    node_cluster_map = {
        row["id"]: row.get("cluster_id")
        for row in graph.get("nodes", [])
        if row.get("id")
    }
    cluster_lookup = {
        row["id"]: row for row in graph.get("clusters", []) if row.get("id")
    }

    cluster_stats_map: dict[str, dict] = {}
    for cluster in graph.get("clusters", []):
        cluster_id = cluster.get("id")
        if not cluster_id:
            continue
        cluster_stats_map[cluster_id] = {
            "cluster_id": cluster_id,
            "label": cluster.get("label") or "Unknown cluster",
            "node_count": _coerce_int(cluster.get("node_count")),
            "link_count": _coerce_int(cluster.get("link_count")),
            "transmitted_link_count": _coerce_int(cluster.get("transmitted_link_count")),
            "avg_score": cluster.get("avg_score"),
            "near_miss_count": 0,
            "open_salvage_count": 0,
            "latest_timestamp": cluster.get("latest_timestamp"),
            "top_domains": list(cluster.get("top_domains") or []),
            "exploration_count": 0,
            "transmitted_exploration_count": 0,
        }

    seed_stats_map: dict[str, dict] = {}
    for row in exploration_rows:
        seed_domain = _clean_optional_text(row.get("seed_domain"))
        target_domain = _clean_optional_text(row.get("jump_target_domain"))
        if seed_domain is None or target_domain is None:
            continue

        cluster_id = node_cluster_map.get(seed_domain) or node_cluster_map.get(target_domain)
        if cluster_id is None:
            continue

        raw_score = row.get("total_score")
        if raw_score is None:
            score = None
        else:
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = None
        transmitted = 1 if _coerce_int(row.get("transmitted")) > 0 else 0
        near_miss = 1 if score is not None and score >= FRONTIER_NEAR_MISS_SCORE and transmitted == 0 else 0
        timestamp = _clean_optional_text(row.get("timestamp"))

        cluster_entry = cluster_stats_map.setdefault(
            cluster_id,
            {
                "cluster_id": cluster_id,
                "label": cluster_lookup.get(cluster_id, {}).get("label") or "Unknown cluster",
                "node_count": 0,
                "link_count": 0,
                "transmitted_link_count": 0,
                "avg_score": None,
                "near_miss_count": 0,
                "open_salvage_count": 0,
                "latest_timestamp": None,
                "top_domains": [],
                "exploration_count": 0,
                "transmitted_exploration_count": 0,
            },
        )
        cluster_entry["exploration_count"] += 1
        cluster_entry["transmitted_exploration_count"] += transmitted
        cluster_entry["near_miss_count"] += near_miss
        if _timestamp_is_newer(timestamp, cluster_entry["latest_timestamp"]):
            cluster_entry["latest_timestamp"] = timestamp

        seed_entry = seed_stats_map.setdefault(
            seed_domain,
            {
                "domain": seed_domain,
                "cluster_id": cluster_id,
                "exploration_count": 0,
                "transmitted_count": 0,
                "avg_score": None,
                "near_miss_count": 0,
                "open_salvage_count": 0,
                "latest_timestamp": None,
                "score_total": 0.0,
                "score_count": 0,
                "neighbor_ids": set(),
                "transmitted_neighbor_ids": set(),
                "frontier_score": 0.0,
            },
        )
        seed_entry["exploration_count"] += 1
        seed_entry["transmitted_count"] += transmitted
        seed_entry["near_miss_count"] += near_miss
        seed_entry["neighbor_ids"].add(target_domain)
        if transmitted:
            seed_entry["transmitted_neighbor_ids"].add(target_domain)
        if _timestamp_is_newer(timestamp, seed_entry["latest_timestamp"]):
            seed_entry["latest_timestamp"] = timestamp
        if score is not None:
            seed_entry["score_total"] += score
            seed_entry["score_count"] += 1

    for row in salvage_candidates:
        source_domain = _clean_optional_text(row.get("source"))
        target_domain = _clean_optional_text(row.get("target"))
        affected_cluster_ids = {
            cluster_id
            for cluster_id in (
                node_cluster_map.get(source_domain) if source_domain else None,
                node_cluster_map.get(target_domain) if target_domain else None,
            )
            if cluster_id is not None
        }
        for cluster_id in affected_cluster_ids:
            cluster_stats_map.setdefault(
                cluster_id,
                {
                    "cluster_id": cluster_id,
                    "label": cluster_lookup.get(cluster_id, {}).get("label")
                    or "Unknown cluster",
                    "node_count": 0,
                    "link_count": 0,
                    "transmitted_link_count": 0,
                    "avg_score": None,
                    "near_miss_count": 0,
                    "open_salvage_count": 0,
                    "latest_timestamp": None,
                    "top_domains": [],
                    "exploration_count": 0,
                    "transmitted_exploration_count": 0,
                },
            )["open_salvage_count"] += 1
        if source_domain is not None and source_domain in seed_stats_map:
            seed_stats_map[source_domain]["open_salvage_count"] += 1

    for seed_entry in seed_stats_map.values():
        seed_entry["avg_score"] = (
            seed_entry["score_total"] / seed_entry["score_count"]
            if seed_entry["score_count"] > 0
            else None
        )
        seed_entry["neighbor_count"] = len(seed_entry["neighbor_ids"])
        seed_entry["transmitted_neighbor_count"] = len(
            seed_entry["transmitted_neighbor_ids"]
        )

    max_region_activity = max(
        (_coerce_int(row.get("exploration_count")) for row in cluster_stats_map.values()),
        default=0,
    )
    max_seed_activity = max(
        (_coerce_int(row.get("exploration_count")) for row in seed_stats_map.values()),
        default=0,
    )
    max_seed_salvage = max(
        (_coerce_int(row.get("open_salvage_count")) for row in seed_stats_map.values()),
        default=0,
    )

    next_regions = []
    for cluster_entry in cluster_stats_map.values():
        avg_score = _clamp_float(_coerce_float(cluster_entry.get("avg_score")))
        transmission_rate = _safe_ratio(
            cluster_entry.get("transmitted_exploration_count"),
            cluster_entry.get("exploration_count"),
        )
        near_miss_rate = _safe_ratio(
            cluster_entry.get("near_miss_count"),
            cluster_entry.get("exploration_count"),
        )
        salvage_density = _clamp_float(
            _safe_ratio(
                cluster_entry.get("open_salvage_count"),
                cluster_entry.get("node_count"),
            )
        )
        activity_norm = _normalized_log(
            cluster_entry.get("exploration_count"),
            max_region_activity,
        )
        freshness_norm = _freshness_norm(cluster_entry.get("latest_timestamp"))
        frontier_score = 100.0 * (
            0.30 * avg_score
            + 0.20 * transmission_rate
            + 0.20 * near_miss_rate
            + 0.15 * salvage_density
            + 0.10 * activity_norm
            + 0.05 * freshness_norm
        )
        next_regions.append(
            {
                "cluster_id": cluster_entry.get("cluster_id"),
                "label": cluster_entry.get("label") or "Unknown cluster",
                "frontier_score": round(frontier_score, 2),
                "node_count": _coerce_int(cluster_entry.get("node_count")),
                "link_count": _coerce_int(cluster_entry.get("link_count")),
                "transmitted_link_count": _coerce_int(
                    cluster_entry.get("transmitted_link_count")
                ),
                "avg_score": cluster_entry.get("avg_score"),
                "near_miss_count": _coerce_int(cluster_entry.get("near_miss_count")),
                "open_salvage_count": _coerce_int(
                    cluster_entry.get("open_salvage_count")
                ),
                "latest_timestamp": cluster_entry.get("latest_timestamp"),
                "top_domains": list(cluster_entry.get("top_domains") or [])[:5],
                "why": _frontier_region_why(cluster_entry),
                "_transmitted_sort": _coerce_int(
                    cluster_entry.get("transmitted_exploration_count")
                ),
            }
        )

    next_regions.sort(
        key=lambda row: (
            -_coerce_float(row.get("frontier_score")),
            -_coerce_int(row.get("_transmitted_sort")),
            -_coerce_float(row.get("avg_score")),
            -_timestamp_sort_value(row.get("latest_timestamp")),
            str(row.get("label") or "").lower(),
        ),
    )

    next_seeds = []
    for seed_entry in seed_stats_map.values():
        avg_score = _clamp_float(_coerce_float(seed_entry.get("avg_score")))
        transmission_rate = _safe_ratio(
            seed_entry.get("transmitted_count"),
            seed_entry.get("exploration_count"),
        )
        near_miss_rate = _safe_ratio(
            seed_entry.get("near_miss_count"),
            seed_entry.get("exploration_count"),
        )
        salvage_norm = _normalized_log(
            seed_entry.get("open_salvage_count"),
            max_seed_salvage,
        )
        transmitted_neighbor_rate = _safe_ratio(
            seed_entry.get("transmitted_neighbor_count"),
            seed_entry.get("neighbor_count"),
        )
        activity_norm = _normalized_log(
            seed_entry.get("exploration_count"),
            max_seed_activity,
        )
        freshness_norm = _freshness_norm(seed_entry.get("latest_timestamp"))
        frontier_score = 100.0 * (
            0.30 * avg_score
            + 0.20 * transmission_rate
            + 0.20 * near_miss_rate
            + 0.15 * salvage_norm
            + 0.10 * transmitted_neighbor_rate
            + 0.05 * freshness_norm
        )
        seed_entry["frontier_score"] = round(frontier_score, 2)
        next_seeds.append(
            {
                "domain": seed_entry.get("domain"),
                "cluster_id": seed_entry.get("cluster_id"),
                "frontier_score": seed_entry["frontier_score"],
                "exploration_count": _coerce_int(seed_entry.get("exploration_count")),
                "transmitted_count": _coerce_int(seed_entry.get("transmitted_count")),
                "avg_score": seed_entry.get("avg_score"),
                "near_miss_count": _coerce_int(seed_entry.get("near_miss_count")),
                "open_salvage_count": _coerce_int(
                    seed_entry.get("open_salvage_count")
                ),
                "latest_timestamp": seed_entry.get("latest_timestamp"),
                "why": _frontier_seed_why(seed_entry),
                "_transmitted_neighbor_count": _coerce_int(
                    seed_entry.get("transmitted_neighbor_count")
                ),
            }
        )

    next_seeds.sort(
        key=lambda row: (
            -_coerce_float(row.get("frontier_score")),
            -_coerce_int(row.get("transmitted_count")),
            -_coerce_float(row.get("avg_score")),
            -_timestamp_sort_value(row.get("latest_timestamp")),
            str(row.get("domain") or "").lower(),
        ),
    )

    launchpads = []
    for seed_entry in seed_stats_map.values():
        launchpads.append(
            {
                "domain": seed_entry.get("domain"),
                "cluster_id": seed_entry.get("cluster_id"),
                "exploration_count": _coerce_int(seed_entry.get("exploration_count")),
                "transmitted_count": _coerce_int(seed_entry.get("transmitted_count")),
                "avg_score": seed_entry.get("avg_score"),
                "latest_timestamp": seed_entry.get("latest_timestamp"),
                "frontier_score": seed_entry.get("frontier_score", 0.0),
            }
        )

    launchpads.sort(
        key=lambda row: (
            -_coerce_int(row.get("transmitted_count")),
            -_coerce_float(row.get("avg_score")),
            -_coerce_int(row.get("exploration_count")),
            str(row.get("domain") or "").lower(),
        )
    )

    strong_bridges = []
    for row in graph.get("links", []):
        strong_bridges.append(
            {
                "source": row.get("source"),
                "target": row.get("target"),
                "count": _coerce_int(row.get("count")),
                "transmitted_count": _coerce_int(row.get("transmitted_count")),
                "avg_score": row.get("avg_score"),
                "max_score": row.get("max_score"),
                "latest_timestamp": row.get("latest_timestamp"),
            }
        )

    strong_bridges.sort(
        key=lambda row: (
            -_coerce_int(row.get("transmitted_count")),
            -_coerce_float(row.get("max_score")),
            -_coerce_int(row.get("count")),
            str(row.get("source") or "").lower(),
            str(row.get("target") or "").lower(),
        )
    )

    for row in next_regions:
        row.pop("_transmitted_sort", None)
    for row in next_seeds:
        row.pop("_transmitted_neighbor_count", None)

    return {
        "stats": {
            "high_signal_seeds": sum(
                1
                for row in launchpads
                if _coerce_int(row.get("transmitted_count")) > 0
                or _coerce_float(row.get("avg_score")) >= FRONTIER_NEAR_MISS_SCORE
            ),
            "high_signal_bridges": sum(
                1
                for row in strong_bridges
                if _coerce_int(row.get("transmitted_count")) > 0
                or _coerce_float(row.get("max_score")) >= 0.85
            ),
            "open_salvage_candidates": len(salvage_candidates),
        },
        "launchpads": launchpads[:FRONTIER_LAUNCHPAD_LIMIT],
        "strong_bridges": strong_bridges[:FRONTIER_STRONG_BRIDGE_LIMIT],
        "salvage_candidates": salvage_candidates[:FRONTIER_SALVAGE_LIMIT],
        "next_regions": next_regions[:FRONTIER_REGION_LIMIT],
        "next_seeds": next_seeds[:FRONTIER_SEED_LIMIT],
    }


def _extract_transmission_sections(formatted_output: str) -> dict[str, str]:
    section_names = {
        "1) PRIMARY CLAIM": "primary_claim",
        "2) PREDICTION": "prediction",
        "3) OPERATOR TAKEAWAY": "operator_takeaway",
        "4) TEST": "test",
        "5) MECHANISM": "mechanism",
        "7) VARIABLE MAPPING": "variable_mapping",
        "11) OPTIONAL SUMMARY": "optional_summary",
        "3) VARIABLE MAPPING": "variable_mapping",
        "4) MECHANISM": "mechanism",
        "5) PREDICTION": "prediction",
        "6) TEST": "test",
        "8) OPTIONAL SUMMARY": "optional_summary",
    }
    sections: dict[str, str] = {}
    current_section = None
    buffer: list[str] = []

    def _flush():
        nonlocal buffer, current_section
        if current_section is None:
            return
        cleaned_lines = []
        for line in buffer:
            if line.startswith("    "):
                cleaned_lines.append(line[4:])
            elif line.startswith("  "):
                cleaned_lines.append(line[2:])
            else:
                cleaned_lines.append(line)
        sections[current_section] = "\n".join(cleaned_lines).strip()

    for raw_line in str(formatted_output or "").splitlines():
        stripped = raw_line.strip()
        if stripped in section_names:
            _flush()
            current_section = section_names[stripped]
            buffer = []
            continue
        if current_section is not None:
            buffer.append(raw_line.rstrip())
    _flush()
    return sections


def _collapse_whitespace(value: str | None) -> str:
    if value is None:
        return "—"
    collapsed = " ".join(str(value).split())
    return collapsed or "—"


def _first_sentence(value: str | None) -> str:
    text = _collapse_whitespace(value)
    if text == "—":
        return text
    match = re.search(r"^.*?[.!?](?=\s|$)", text)
    return match.group(0).strip() if match else text


def _format_mapping_lines(mapping_text: str | None) -> str:
    if not mapping_text:
        return "- —"
    try:
        parsed = json.loads(mapping_text)
    except (TypeError, ValueError):
        parsed = None

    if isinstance(parsed, dict):
        lines = []
        for source, target in parsed.items():
            source_text = _collapse_whitespace(source)
            if isinstance(target, (dict, list)):
                target_text = json.dumps(target, ensure_ascii=False)
            else:
                target_text = _collapse_whitespace(target)
            lines.append(f"- {source_text} → {target_text}")
        return "\n".join(lines) if lines else "- —"

    if isinstance(parsed, list):
        lines = []
        for item in parsed:
            if isinstance(item, dict):
                source = item.get("source") or item.get("from") or item.get("left") or "—"
                target = item.get("target") or item.get("to") or item.get("right") or "—"
                lines.append(f"- {_collapse_whitespace(source)} → {_collapse_whitespace(target)}")
            else:
                lines.append(f"- {_collapse_whitespace(item)}")
        return "\n".join(lines) if lines else "- —"

    fallback_lines = [
        f"- {_collapse_whitespace(line)}"
        for line in str(mapping_text).splitlines()
        if _collapse_whitespace(line) != "—"
    ]
    return "\n".join(fallback_lines) if fallback_lines else "- —"


def _format_test_line(test_text: str | None) -> str:
    if not test_text:
        return "—"
    parts: dict[str, str] = {}
    for line in str(test_text).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parts[key.strip().lower()] = _collapse_whitespace(value)

    segments = []
    if parts.get("metric"):
        segments.append(f"Metric: {parts['metric']}")
    if parts.get("confirm"):
        segments.append(f"Confirm: {parts['confirm']}")
    if parts.get("falsify"):
        segments.append(f"Falsify: {parts['falsify']}")

    if segments:
        return " ".join(segments)
    return _collapse_whitespace(test_text)


def _build_transmission_markdown(row: dict) -> str:
    sections = _extract_transmission_sections(row.get("formatted_output") or "")
    mechanism_text = _collapse_whitespace(sections.get("mechanism"))
    claim_text = _first_sentence(
        sections.get("primary_claim")
        or sections.get("optional_summary")
        or row.get("connection_description")
        or sections.get("mechanism")
    )
    hook_text = _collapse_whitespace(
        sections.get("optional_summary")
        or row.get("connection_description")
    )
    prediction_text = _collapse_whitespace(sections.get("prediction"))
    mapping_lines = _format_mapping_lines(sections.get("variable_mapping"))
    test_line = _format_test_line(sections.get("test"))
    takeaway_text = _collapse_whitespace(sections.get("operator_takeaway"))
    if takeaway_text == "—":
        takeaway_text = test_line if test_line != "—" else prediction_text

    lines = [
        f"## {row.get('seed_domain') or 'Unknown Seed'} ↔ {row.get('jump_target_domain') or 'Unknown Target'}",
        "",
        f"**Primary claim:** {claim_text}",
        "",
        f"**Prediction:** {prediction_text}",
        "",
        f"**How to test it:** {test_line}",
        "",
        f"**Operator takeaway:** {takeaway_text}",
        "",
        f"**The mechanism:** {mechanism_text}",
        "",
        "**Variable mapping:**",
        mapping_lines,
    ]
    if hook_text != "—":
        lines.extend(
            [
                "",
                f"**Optional context:** {hook_text}",
            ]
        )
    lines.extend(
        [
            "",
            (
                "**Scores:** "
                f"Novelty {_format_score(row.get('novelty_score'))} | "
                f"Depth {_format_score(row.get('depth_score'))} | "
                f"Distance {_format_score(row.get('distance_score'))} | "
                f"Total {_format_score(row.get('total_score'))}"
            ),
            "",
            "*Found by BlackClaw — autonomous curiosity engine*",
        ]
    )
    return "\n".join(lines)


@app.errorhandler(sqlite3.OperationalError)
def handle_db_error(exc):
    return jsonify({"error": str(exc), "db_path": DB_PATH}), 500


@app.get("/api/transmissions")
def api_transmissions():
    return jsonify(_get_transmissions())


@app.get("/api/transmission-timeline")
def api_transmission_timeline():
    return jsonify(_get_transmission_timeline())


@app.get("/api/domain-graph")
def api_domain_graph():
    return jsonify(_get_domain_graph())


@app.get("/api/frontier")
def api_frontier():
    return jsonify(_get_frontier_snapshot())


@app.post("/api/transmissions/<int:transmission_id>/grade")
def api_grade_transmission(transmission_id: int):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "expected JSON object payload"}), 400
    grade = str(payload.get("grade") or "").strip()
    if grade not in VALID_GRADES:
        return jsonify({"error": "grade must be one of: " + ", ".join(VALID_GRADES)}), 400

    raw_notes = payload.get("notes")
    if raw_notes is None:
        clean_notes = None
    else:
        clean_notes = str(raw_notes).strip() or None

    conn = _connect_write()
    cursor = conn.execute(
        """UPDATE transmissions
        SET user_rating = ?, user_notes = ?
        WHERE id = ?""",
        (grade, clean_notes, transmission_id),
    )
    conn.commit()
    conn.close()

    if cursor.rowcount < 1:
        return jsonify({"error": "transmission not found", "id": transmission_id}), 404
    return jsonify({"ok": True})


@app.get("/api/transmissions/<int:transmission_id>/markdown")
def api_transmission_markdown(transmission_id: int):
    row = _get_transmission_export_row(transmission_id)
    if row is None:
        return jsonify({"error": "transmission not found", "id": transmission_id}), 404
    return Response(
        _build_transmission_markdown(row),
        mimetype="text/plain",
    )


@app.get("/api/explorations")
def api_explorations():
    limit = _parse_positive_int(request.args.get("limit"), DEFAULT_EXPLORATION_LIMIT)
    seed_domain = request.args.get("seed_domain")
    return jsonify(_get_recent_explorations(limit, seed_domain=seed_domain))


@app.get("/api/explorations/<int:exploration_id>")
def api_exploration(exploration_id: int):
    row = _get_exploration(exploration_id)
    if row is None:
        return jsonify({"error": "exploration not found", "id": exploration_id}), 404
    return jsonify(row)


@app.get("/api/explorations/<int:exploration_id>/adversarial")
def api_exploration_adversarial(exploration_id: int):
    payload = _get_adversarial_rubric(exploration_id)
    if payload is None:
        return jsonify({"error": "exploration not found", "id": exploration_id}), 404
    return jsonify(payload)


@app.get("/api/stats")
def api_stats():
    window = _parse_positive_int(request.args.get("window"), DEFAULT_STATS_WINDOW)
    return jsonify(_get_kill_stats(window))


@app.get("/api/costs")
def api_costs():
    window = _parse_positive_int(request.args.get("window"), DEFAULT_STATS_WINDOW)
    return jsonify(_get_cost_stats(window))


@app.get("/api/top-killed")
def api_top_killed():
    return jsonify(_get_top_killed())


@app.get("/api/evidence-review-queue")
def api_evidence_review_queue():
    limit = _parse_positive_int(request.args.get("limit"), 20)
    return jsonify(list_prediction_evidence_review_queue(limit=limit))


@app.get("/api/evidence-review-stats")
def api_evidence_review_stats():
    return jsonify(get_prediction_evidence_review_stats())


@app.get("/api/evidence/<int:evidence_id>")
def api_evidence_detail(evidence_id: int):
    row = get_prediction_evidence_hit(evidence_id)
    if row is None:
        return jsonify({"error": "evidence hit not found", "id": evidence_id}), 404
    return jsonify(row)


@app.post("/api/evidence/<int:evidence_id>/accept")
def api_accept_evidence(evidence_id: int):
    return _update_evidence_review_status_response(evidence_id, "accepted")


@app.post("/api/evidence/<int:evidence_id>/dismiss")
def api_dismiss_evidence(evidence_id: int):
    return _update_evidence_review_status_response(evidence_id, "dismissed")


@app.get("/api/outcome-review-queue")
def api_outcome_review_queue():
    limit = _parse_positive_int(request.args.get("limit"), 20)
    return jsonify(list_prediction_outcome_review_queue(limit=limit))


@app.get("/api/outcome-review/<int:prediction_id>")
def api_outcome_review_detail(prediction_id: int):
    row = get_prediction_outcome_review(prediction_id)
    if row is None:
        return jsonify({"error": "prediction not found", "id": prediction_id}), 404
    return jsonify(row)


@app.get("/api/outcome-suggestion-stats")
def api_outcome_suggestion_stats():
    return jsonify(get_prediction_outcome_suggestion_stats())


@app.get("/api/strong-rejections")
def api_strong_rejections():
    limit = _parse_positive_int(request.args.get("limit"), 20)
    raw_status = _clean_optional_text(request.args.get("status"))
    if raw_status is None:
        status = None
    else:
        status = raw_status.lower()
        if status not in VALID_STRONG_REJECTION_STATUSES:
            return (
                jsonify(
                    {
                        "error": "status must be one of: "
                        + ", ".join(VALID_STRONG_REJECTION_STATUSES)
                    }
                ),
                400,
            )
    return jsonify(
        list_strong_rejections(
            limit=limit,
            status=status,
            open_first=status is None,
        )
    )


@app.get("/api/strong-rejection/<int:rejection_id>")
def api_strong_rejection_detail(rejection_id: int):
    row = get_strong_rejection(rejection_id)
    if row is None:
        return jsonify({"error": "strong rejection not found", "id": rejection_id}), 404
    return jsonify(row)


@app.get("/api/strong-rejection-stats")
def api_strong_rejection_stats():
    return jsonify(get_strong_rejection_stats())


@app.post("/api/strong-rejection/<int:rejection_id>/salvage")
def api_salvage_strong_rejection(rejection_id: int):
    return _update_strong_rejection_status_response(rejection_id, "salvaged")


@app.post("/api/strong-rejection/<int:rejection_id>/dismiss")
def api_dismiss_strong_rejection(rejection_id: int):
    return _update_strong_rejection_status_response(rejection_id, "dismissed")


@app.get("/api/operator-home")
def api_operator_home():
    return jsonify(_build_operator_home_snapshot())


def _render_observatory_page(template_name: str, page_key: str, page_title: str):
    return render_template(
        template_name,
        page_key=page_key,
        page_title=page_title,
    )


@app.get("/")
def index():
    return _render_observatory_page("home.html", "home", "BlackClaw Observatory")


@app.get("/map")
def map_page():
    return _render_observatory_page("map.html", "map", "BlackClaw Map")


@app.get("/frontier")
def frontier_page():
    return _render_observatory_page("frontier.html", "frontier", "BlackClaw Frontier")


@app.get("/review")
def review_page():
    return _render_observatory_page("review.html", "review", "BlackClaw Review")


@app.get("/archive")
def archive_page():
    return _render_observatory_page("archive.html", "archive", "BlackClaw Archive")


def _legacy_dashboard_markup():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BlackClaw Dashboard</title>
  <style>
    :root {
      font-family: Menlo, Monaco, Consolas, monospace;
      color-scheme: dark;
      --bg: #1a1a2e;
      --panel-bg: #16213e;
      --text: #e0e0e0;
      --header-text: #ffffff;
      --link-color: #4fc3f7;
      --border-color: #333;
      --muted: #a9b4c2;
      --panel-alt: #1d2b4d;
      --input-bg: #0f172a;
      --accent: #00e676;
      --kill-high: #ff6b6b;
      --kill-low: #00e676;
      --row-hover: #1d2b4d;
    }
    [data-theme="light"] {
      color-scheme: light;
      --bg: #f5f5f5;
      --panel-bg: #ffffff;
      --text: #111;
      --header-text: #000;
      --link-color: #0366d6;
      --border-color: #d7d7d7;
      --muted: #666;
      --panel-alt: #fafafa;
      --input-bg: #ffffff;
      --accent: #008f4c;
      --kill-high: #c0392b;
      --kill-low: #008f4c;
      --row-hover: #f3f7fb;
    }
    body {
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
    }
    body,
    section,
    .stat,
    details,
    .transmission-item,
    .detail-panel,
    .detail-card,
    .detail-item,
    .theme-toggle,
    .grade-controls select,
    .grade-controls input,
    .grade-controls button,
    .review-actions input,
    .review-actions button,
    .adversarial-detail,
    .adversarial-detail pre,
    table,
    th,
    td {
      transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
    h1, h2, h3, th, summary {
      color: var(--header-text);
    }
    a {
      color: var(--link-color);
    }
    section {
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      padding: 16px;
      margin-bottom: 16px;
    }
    h1, h2 {
      margin: 0 0 12px;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
    }
    .theme-toggle {
      position: fixed;
      top: 20px;
      right: 24px;
      z-index: 10;
      font: inherit;
      padding: 8px 12px;
      background: var(--panel-bg);
      color: var(--header-text);
      border: 1px solid var(--border-color);
      border-radius: 999px;
      cursor: pointer;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .stat {
      border: 1px solid var(--border-color);
      padding: 12px;
      background: var(--panel-alt);
    }
    .stat-value {
      font-weight: 600;
    }
    .score-accent,
    .grade-summary {
      color: var(--accent);
    }
    .kill-high {
      color: var(--kill-high);
    }
    .kill-low {
      color: var(--kill-low);
    }
    details {
      border: 1px solid var(--border-color);
      padding: 10px 12px;
      margin-bottom: 10px;
      background: var(--panel-alt);
    }
    summary {
      cursor: pointer;
      font-weight: 600;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 12px 0 0;
      font-size: 13px;
      line-height: 1.4;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      text-align: left;
      padding: 8px;
      border-bottom: 1px solid var(--border-color);
      vertical-align: top;
    }
    .error {
      color: var(--kill-high);
      white-space: pre-wrap;
    }
    .transmission-item {
      border: 1px solid var(--border-color);
      padding: 12px;
      margin-bottom: 10px;
      background: var(--panel-alt);
    }
    .grade-summary {
      margin-bottom: 12px;
    }
    .grade-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 10px;
    }
    .grade-controls select,
    .grade-controls input,
    .grade-controls button {
      font: inherit;
      padding: 6px 8px;
      background: var(--input-bg);
      color: var(--text);
      border: 1px solid var(--border-color);
    }
    .grade-controls input {
      min-width: 220px;
    }
    .grade-status {
      color: var(--muted);
      font-size: 13px;
      min-width: 48px;
    }
    .copy-status {
      color: var(--accent);
      font-size: 13px;
      min-width: 56px;
    }
    .timeline-shell {
      border: 1px solid var(--border-color);
      padding: 12px;
      background: var(--panel-alt);
    }
    .timeline-chart {
      display: block;
      width: 100%;
      height: auto;
    }
    .timeline-meta,
    .timeline-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }
    .timeline-swatch {
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 6px;
      border-radius: 999px;
      vertical-align: middle;
    }
    .timeline-swatch-transmitted {
      background: var(--accent);
    }
    .timeline-swatch-untransmitted {
      background: var(--kill-high);
    }
    .top-killed-row {
      cursor: pointer;
    }
    .top-killed-row:hover {
      background: var(--row-hover);
    }
    .adversarial-cell {
      padding: 0;
      border-bottom: 1px solid var(--border-color);
    }
    .adversarial-detail {
      margin: 8px 0 8px 16px;
      padding: 12px 0 12px 12px;
      border-left: 3px solid var(--border-color);
    }
    .adversarial-detail h3 {
      margin: 0 0 8px;
      font-size: 14px;
    }
    .adversarial-detail p,
    .adversarial-detail ol {
      margin: 0 0 8px;
    }
    .adversarial-detail pre {
      margin-top: 8px;
      background: var(--panel-alt);
      padding: 8px;
      border: 1px solid var(--border-color);
    }
    .table-shell {
      overflow-x: auto;
    }
    .triage-panels {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .triage-panel {
      border: 1px solid var(--border-color);
      padding: 12px;
      background: var(--panel-alt);
    }
    .triage-panel h3 {
      margin: 0 0 10px;
    }
    .triage-panel p {
      margin: 0 0 10px;
    }
    .review-table-row {
      cursor: pointer;
    }
    .review-table-row:hover,
    .review-table-row.is-selected {
      background: var(--row-hover);
    }
    .detail-panel {
      margin-top: 12px;
      border: 1px solid var(--border-color);
      padding: 12px;
      background: var(--panel-alt);
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .detail-item {
      border: 1px solid var(--border-color);
      padding: 10px;
      background: var(--panel-bg);
    }
    .detail-label {
      display: block;
      margin-bottom: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    .detail-panel p {
      margin: 0 0 10px;
    }
    .detail-panel ul {
      margin: 0 0 10px 18px;
      padding: 0;
    }
    .detail-stack {
      display: grid;
      gap: 10px;
    }
    .detail-card {
      border: 1px solid var(--border-color);
      padding: 10px;
      background: var(--panel-bg);
    }
    .detail-card p {
      margin: 0 0 8px;
    }
    .status-pill {
      display: inline-block;
      border: 1px solid var(--border-color);
      border-radius: 999px;
      padding: 2px 8px;
      background: var(--panel-bg);
      font-size: 12px;
    }
    .review-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 12px;
    }
    .review-actions input,
    .review-actions button {
      font: inherit;
      padding: 6px 8px;
      background: var(--input-bg);
      color: var(--text);
      border: 1px solid var(--border-color);
    }
    .review-actions input {
      min-width: 240px;
    }
    .review-status {
      color: var(--muted);
      font-size: 13px;
      min-width: 56px;
    }
    :root {
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      --font-sans: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      --font-display: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      --font-mono: "SFMono-Regular", Menlo, Monaco, Consolas, monospace;
      --bg: #071317;
      --panel-bg: rgba(9, 20, 26, 0.84);
      --panel-alt: rgba(14, 31, 40, 0.9);
      --text: #e5eef3;
      --header-text: #f4fbff;
      --link-color: #8fe7ff;
      --border-color: rgba(151, 201, 220, 0.18);
      --muted: #9ab2be;
      --input-bg: rgba(6, 17, 22, 0.92);
      --accent: #59d39a;
      --kill-high: #ff7a59;
      --kill-low: #b5f38a;
      --row-hover: rgba(96, 167, 196, 0.12);
      --shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
    }
    [data-theme="light"] {
      --bg: #eef3ef;
      --panel-bg: rgba(255, 255, 255, 0.88);
      --panel-alt: rgba(246, 250, 248, 0.95);
      --text: #15222b;
      --header-text: #102029;
      --link-color: #0e7490;
      --border-color: rgba(30, 65, 80, 0.14);
      --muted: #59707b;
      --input-bg: rgba(255, 255, 255, 0.96);
      --accent: #1c8f67;
      --kill-high: #cf5f42;
      --kill-low: #4f8f3b;
      --row-hover: rgba(52, 117, 145, 0.09);
      --shadow: 0 22px 48px rgba(43, 70, 87, 0.08);
    }
    html {
      scroll-behavior: smooth;
    }
    body {
      max-width: 1500px;
      margin: 0 auto;
      padding: 32px 24px 96px;
      background:
        radial-gradient(circle at top left, rgba(96, 167, 196, 0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(89, 211, 154, 0.14), transparent 24%),
        linear-gradient(180deg, rgba(7, 19, 23, 0.96), rgba(7, 19, 23, 1));
      color: var(--text);
      font-family: var(--font-sans);
    }
    [data-theme="light"] body {
      background:
        radial-gradient(circle at top left, rgba(110, 194, 220, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(94, 180, 128, 0.14), transparent 22%),
        linear-gradient(180deg, rgba(244, 248, 245, 0.98), rgba(238, 243, 239, 1));
    }
    h1,
    h2,
    h3 {
      font-family: var(--font-display);
      letter-spacing: -0.03em;
    }
    h1 {
      font-size: clamp(2.4rem, 5vw, 4.2rem);
      line-height: 0.94;
      max-width: 12ch;
    }
    h2 {
      font-size: clamp(1.5rem, 2vw, 2rem);
    }
    th,
    .stat-value,
    .status-pill,
    .theme-toggle,
    .hero-link,
    .section-nav a,
    .control-field span,
    .graph-detail-panel button,
    pre {
      font-family: var(--font-mono);
    }
    a {
      text-underline-offset: 0.16em;
    }
    section {
      margin-bottom: 18px;
      padding: 20px;
      border-radius: 22px;
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }
    .hero {
      padding: 28px;
      overflow: hidden;
      background:
        linear-gradient(135deg, rgba(11, 31, 38, 0.98), rgba(8, 22, 27, 0.9)),
        var(--panel-bg);
    }
    [data-theme="light"] .hero {
      background:
        linear-gradient(135deg, rgba(251, 255, 253, 0.98), rgba(241, 248, 245, 0.95)),
        var(--panel-bg);
    }
    .hero-header {
      display: flex;
      gap: 24px;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 20px;
    }
    .eyebrow {
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-transform: uppercase;
    }
    .hero-copy {
      max-width: 62ch;
      font-size: 1rem;
      line-height: 1.6;
      color: var(--muted);
    }
    .hero-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: flex-end;
      min-width: 240px;
    }
    .hero-link,
    .graph-detail-panel button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid var(--border-color);
      background: var(--panel-alt);
      color: var(--header-text);
      text-decoration: none;
      cursor: pointer;
    }
    .section-nav {
      position: sticky;
      top: 14px;
      z-index: 8;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 18px;
      padding: 10px 12px;
      border-radius: 999px;
      border: 1px solid var(--border-color);
      background: rgba(7, 19, 23, 0.75);
      backdrop-filter: blur(12px);
    }
    [data-theme="light"] .section-nav {
      background: rgba(255, 255, 255, 0.78);
    }
    .section-nav a {
      padding: 8px 12px;
      border-radius: 999px;
      color: var(--muted);
      text-decoration: none;
    }
    .section-nav a:hover {
      background: var(--panel-alt);
      color: var(--header-text);
    }
    .theme-toggle {
      top: 22px;
      right: 24px;
      background: rgba(7, 19, 23, 0.82);
      backdrop-filter: blur(12px);
      box-shadow: var(--shadow);
    }
    [data-theme="light"] .theme-toggle {
      background: rgba(255, 255, 255, 0.9);
    }
    .stat,
    .transmission-item,
    .triage-panel,
    .detail-panel,
    .detail-card,
    .detail-item,
    details,
    .timeline-shell {
      border-radius: 16px;
    }
    .stat {
      padding: 14px;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.01), transparent), var(--panel-alt);
    }
    .stat-value {
      margin-top: 6px;
      font-size: 1.1rem;
      letter-spacing: -0.02em;
    }
    .muted {
      line-height: 1.5;
    }
    .section-heading {
      display: flex;
      gap: 16px;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 14px;
    }
    .graph-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: flex-end;
      align-items: flex-end;
      max-width: 560px;
    }
    .control-field {
      display: grid;
      gap: 6px;
      min-width: 160px;
    }
    .control-field span {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .control-field input,
    .control-field select,
    #graph-reset {
      font: inherit;
      padding: 10px 12px;
      color: var(--text);
      background: var(--input-bg);
      border: 1px solid var(--border-color);
      border-radius: 14px;
    }
    .control-search {
      min-width: min(320px, 100%);
      flex: 1 1 240px;
    }
    .control-range {
      min-width: 210px;
    }
    .graph-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) minmax(280px, 0.8fr);
      gap: 14px;
      align-items: stretch;
    }
    .graph-stage,
    .graph-detail-panel {
      border: 1px solid var(--border-color);
      background: var(--panel-alt);
      border-radius: 18px;
    }
    .graph-stage {
      position: relative;
      min-height: 580px;
      overflow: hidden;
    }
    .domain-web-canvas {
      display: block;
      width: 100%;
      height: 100%;
      cursor: default;
    }
    .graph-empty {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      text-align: center;
      color: var(--muted);
      background: linear-gradient(180deg, rgba(7, 19, 23, 0.18), rgba(7, 19, 23, 0.58));
    }
    .graph-caption,
    .graph-legend {
      position: absolute;
      left: 18px;
      right: 18px;
      z-index: 1;
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      align-items: center;
      pointer-events: none;
    }
    .graph-caption {
      top: 14px;
    }
    .graph-legend {
      bottom: 14px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(7, 19, 23, 0.62);
      border: 1px solid var(--border-color);
      font-size: 12px;
      color: var(--muted);
    }
    [data-theme="light"] .legend-item {
      background: rgba(255, 255, 255, 0.78);
    }
    .legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }
    .legend-dot-seed {
      background: #ffb74d;
    }
    .legend-dot-target {
      background: #67d5ff;
    }
    .legend-dot-bridge {
      background: #59d39a;
    }
    .legend-line {
      width: 18px;
      height: 2px;
      display: inline-block;
      background: linear-gradient(90deg, rgba(154, 178, 190, 0.3), rgba(89, 211, 154, 0.9));
    }
    .graph-detail-panel {
      padding: 16px;
      display: grid;
      align-content: start;
      gap: 12px;
    }
    .graph-detail-panel h3 {
      margin: 0;
    }
    .graph-detail-panel p,
    .graph-detail-panel ul {
      margin: 0;
    }
    .graph-neighbors {
      display: grid;
      gap: 8px;
    }
    .graph-neighbor {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 10px;
      align-items: center;
      justify-content: space-between;
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid var(--border-color);
      background: var(--panel-bg);
    }
    .graph-neighbor strong {
      font-size: 14px;
    }
    #graph-reset {
      cursor: pointer;
    }
    @media (max-width: 1024px) {
      .graph-layout {
        grid-template-columns: 1fr;
      }
      .graph-stage {
        min-height: 500px;
      }
      .hero-header,
      .section-heading {
        flex-direction: column;
      }
      .hero-actions,
      .graph-controls {
        justify-content: flex-start;
        width: 100%;
      }
    }
    @media (max-width: 720px) {
      body {
        padding: 18px 14px 80px;
      }
      section,
      .hero {
        padding: 16px;
      }
      .section-nav {
        top: 8px;
        border-radius: 20px;
      }
      .graph-stage {
        min-height: 420px;
      }
      .theme-toggle {
        position: static;
        margin-bottom: 12px;
      }
    }
  </style>
</head>
<body>
  <button id="theme-toggle" class="theme-toggle" type="button">Light Mode</button>
  <section class="hero">
    <div class="hero-header">
      <div>
        <p class="eyebrow">BlackClaw Observatory</p>
        <h1>Map the search space, not just the backlog.</h1>
        <p class="hero-copy">
          BlackClaw is already exploring a real web of domains. This dashboard now leads with that map so we can see what it has touched, where it keeps jumping, and which bridges are actually surviving.
        </p>
      </div>
      <div class="hero-actions">
        <a class="hero-link" href="#domain-web-section">Open Exploration Web</a>
        <a class="hero-link" href="/domains">Browse Domain Table</a>
      </div>
    </div>
    <div id="hero-summary" class="grid"><p class="muted">Loading live snapshot…</p></div>
  </section>

  <nav class="section-nav" aria-label="Dashboard sections">
    <a href="#domain-web-section">Exploration Web</a>
    <a href="#operator-home-section">Operator Home</a>
    <a href="#evidence-section">Evidence</a>
    <a href="#outcome-review-section">Outcomes</a>
    <a href="#strong-rejection-section">Rejections</a>
    <a href="#transmissions-section">Transmissions</a>
  </nav>

  <section id="domain-web-section">
    <div class="section-heading">
      <div>
        <p class="eyebrow">Exploration Web</p>
        <h2>Where BlackClaw Has Been</h2>
        <p class="muted">Each node is a domain BlackClaw has touched. Links represent real seed-to-target jumps. Brighter edges mean stronger or transmitted connections. Search a domain to isolate its local neighborhood.</p>
      </div>
      <div class="graph-controls">
        <label class="control-field">
          <span>View</span>
          <select id="graph-mode">
            <option value="all">All explored</option>
            <option value="successful">Transmitted only</option>
            <option value="strong">High signal</option>
          </select>
        </label>
        <label class="control-field control-search">
          <span>Focus</span>
          <input id="graph-search" type="text" placeholder="Try Locksmithing or Neuroscience">
        </label>
        <label class="control-field control-range">
          <span>Min score <strong id="graph-score-value">0.00</strong></span>
          <input id="graph-score" type="range" min="0" max="1" step="0.05" value="0">
        </label>
        <button id="graph-reset" type="button">Reset view</button>
      </div>
    </div>
    <div id="domain-web-summary" class="grid"><p class="muted">Loading graph…</p></div>
    <div class="graph-layout">
      <div class="graph-stage">
        <div class="graph-caption muted">Click a node to inspect a domain. Click a line to inspect a specific jump.</div>
        <canvas id="domain-web-canvas" class="domain-web-canvas"></canvas>
        <div id="domain-web-empty" class="graph-empty" hidden>No domains match the current filters.</div>
        <div class="graph-legend" aria-hidden="true">
          <span class="legend-item"><span class="legend-dot legend-dot-seed"></span>Seed-heavy</span>
          <span class="legend-item"><span class="legend-dot legend-dot-target"></span>Target-heavy</span>
          <span class="legend-item"><span class="legend-dot legend-dot-bridge"></span>Acts as both</span>
          <span class="legend-item"><span class="legend-line"></span>Stronger or transmitted jump</span>
        </div>
      </div>
      <div id="domain-web-detail" class="graph-detail-panel">
        <h3>Select Something In The Web</h3>
        <p class="muted">The old dashboard told you counts. This panel tells you where those counts live in the network. Pick a domain or edge to inspect it.</p>
      </div>
    </div>
  </section>

  <section id="operator-home-section">
    <h2>Operator Home</h2>
    <p class="muted">Local triage snapshot of what needs attention now. Click any row to load the existing detail panel in the relevant review section below.</p>
    <div id="operator-home-summary" class="grid"><p class="muted">Loading…</p></div>
    <div class="triage-panels">
      <div class="triage-panel">
        <h3>Evidence Backlog</h3>
        <p class="muted">Unreviewed evidence hits that are ready for manual review.</p>
        <div id="operator-home-evidence"></div>
      </div>
      <div class="triage-panel">
        <h3>Outcome Backlog</h3>
        <p class="muted">Open predictions that are closest to outcome adjudication.</p>
        <div id="operator-home-outcomes"></div>
      </div>
      <div class="triage-panel">
        <h3>Strong Rejection Backlog</h3>
        <p class="muted">Open salvage candidates sorted by local score strength.</p>
        <div id="operator-home-strong-rejections"></div>
      </div>
    </div>
  </section>

  <section id="stats-section">
    <h2>Kill Stats</h2>
    <div id="stats" class="grid"></div>
  </section>

  <section id="cost-section">
    <h2>Cost</h2>
    <div id="costs" class="grid"></div>
  </section>

  <section id="timeline-section">
    <h2>Transmission Timeline</h2>
    <div id="transmission-timeline"><p class="muted">Loading…</p></div>
  </section>

  <section id="top-killed-section">
    <h2>Top Killed Connections</h2>
    <div id="top-killed"></div>
  </section>

  <section id="evidence-section">
    <h2>Evidence Review</h2>
    <p class="muted">SQLite-only review queue for evidence hits. Click a row to inspect one hit and optionally mark it accepted or dismissed.</p>
    <div id="evidence-review-stats" class="grid"><p class="muted">Loading…</p></div>
    <div id="evidence-review-breakdown"></div>
    <div id="evidence-review-queue"></div>
    <div id="evidence-detail" class="detail-panel"><p class="muted">Select an evidence hit to inspect details.</p></div>
  </section>

  <section id="outcome-stats-section">
    <h2>Outcome Suggestion Stats</h2>
    <p class="muted">Open-prediction suggestion buckets computed from accepted and unreviewed local evidence only.</p>
    <div id="outcome-suggestion-buckets" class="grid"><p class="muted">Loading…</p></div>
    <div id="outcome-review-backlog" class="grid"></div>
  </section>

  <section id="outcome-review-section">
    <h2>Outcome Review</h2>
    <p class="muted">Manual outcome-review queue driven by local evidence counts. Click a row to inspect the current recommendation and example hits.</p>
    <div id="outcome-review-queue"></div>
    <div id="outcome-review-detail" class="detail-panel"><p class="muted">Select a prediction to inspect outcome review detail.</p></div>
  </section>

  <section id="strong-rejection-section">
    <h2>Strong Rejections</h2>
    <p class="muted">Salvage queue for locally stored high-scoring rejects. Open items are shown first by default; click a row to inspect and optionally mark it salvaged or dismissed.</p>
    <div id="strong-rejection-stats" class="grid"><p class="muted">Loading…</p></div>
    <div id="strong-rejection-queue"></div>
    <div id="strong-rejection-detail" class="detail-panel"><p class="muted">Select a strong rejection to inspect details.</p></div>
  </section>

  <section id="transmissions-section">
    <h2>Transmissions</h2>
    <div id="grade-summary" class="grade-summary muted" hidden></div>
    <p id="transmission-count" class="muted">Loading…</p>
    <div id="transmissions"></div>
  </section>

  <script>
    const GRADE_OPTIONS = ["A", "B+", "B", "B-", "C+", "C", "D", "F"];
    const OUTCOME_SUGGESTION_BUCKETS = [
      "review_for_support",
      "review_for_contradiction",
      "conflicting_evidence",
      "waiting_on_review",
      "insufficient_evidence",
    ];
    const OUTCOME_SUGGESTION_LABELS = {
      review_for_support: "Review for support",
      review_for_contradiction: "Review for contradiction",
      conflicting_evidence: "Conflicting evidence",
      waiting_on_review: "Waiting on review",
      insufficient_evidence: "Insufficient evidence",
    };
    let isDarkMode = true;
    let transmissionRows = [];
    let evidenceQueueRows = [];
    let outcomeQueueRows = [];
    let strongRejectionRows = [];
    let selectedEvidenceId = null;
    let selectedOutcomePredictionId = null;
    let selectedStrongRejectionId = null;
    let dashboardStatsPayload = null;
    let dashboardCostsPayload = null;
    let operatorHomeSnapshot = null;
    let domainGraphPayload = null;
    let currentDomainGraph = null;
    let currentDomainGraphLayout = null;
    let selectedGraphNodeId = null;
    let selectedGraphLinkKey = null;
    let hoveredGraphNodeId = null;
    let hoveredGraphLinkKey = null;
    let domainGraphControlsInitialized = false;
    let domainGraphSearchTimer = null;
    let domainGraphResizeTimer = null;
    const domainGraphLayoutCache = new Map();

    function applyTheme() {
      document.documentElement.dataset.theme = isDarkMode ? "dark" : "light";
      document.getElementById("theme-toggle").textContent = isDarkMode ? "Light Mode" : "Dark Mode";
      if (domainGraphPayload) {
        updateDomainGraph();
      }
    }

    async function fetchJson(url) {
      const response = await fetch(url);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Request failed");
      }
      return payload;
    }

    async function fetchText(url) {
      const response = await fetch(url);
      const payload = await response.text();
      if (!response.ok) {
        throw new Error(payload || "Request failed");
      }
      return payload;
    }

    async function postJson(url, payload = {}) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Request failed");
      }
      return data;
    }

    function formatScore(value) {
      return typeof value === "number" ? value.toFixed(3) : "n/a";
    }

    function formatInteger(value) {
      return typeof value === "number" ? value.toLocaleString() : "n/a";
    }

    function formatCurrency(value) {
      return typeof value === "number" ? `$${value.toFixed(4)}` : "n/a";
    }

    function formatAverage(value) {
      return typeof value === "number" ? value.toFixed(2) : "n/a";
    }

    function formatPercent(value) {
      return typeof value === "number" ? `${value.toFixed(1)}%` : "n/a";
    }

    function formatTimelineDate(date, includeYear = false) {
      return date.toLocaleString(undefined, {
        ...(includeYear ? { year: "numeric" } : {}),
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function inputValue(value) {
      return escapeHtml(String(value ?? "").replaceAll("\\n", " "));
    }

    function formatTimestamp(value) {
      if (!value) {
        return "n/a";
      }
      const time = Date.parse(value);
      if (!Number.isFinite(time)) {
        return String(value);
      }
      return formatTimelineDate(new Date(time), true);
    }

    function truncateText(value, maxLength = 120) {
      const text = String(value ?? "").trim();
      if (!text) {
        return "—";
      }
      return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
    }

    function clamp(value, min, max) {
      return Math.min(max, Math.max(min, value));
    }

    function hashString(value) {
      let hash = 0;
      const text = String(value ?? "");
      for (let index = 0; index < text.length; index += 1) {
        hash = ((hash << 5) - hash + text.charCodeAt(index)) | 0;
      }
      return Math.abs(hash);
    }

    function predictionSummary(row) {
      const predictionJson = row && typeof row.prediction_json === "object" ? row.prediction_json : null;
      const statement = predictionJson && typeof predictionJson.statement === "string"
        ? predictionJson.statement
        : null;
      return statement || row.prediction_summary || row.prediction || "—";
    }

    function safeHttpUrl(value) {
      const text = String(value ?? "").trim();
      const lower = text.toLowerCase();
      return lower.startsWith("http://") || lower.startsWith("https://") ? text : "";
    }

    function renderExternalLink(value) {
      const text = String(value ?? "").trim();
      if (!text) {
        return "—";
      }
      const safeUrl = safeHttpUrl(text);
      if (!safeUrl) {
        return escapeHtml(text);
      }
      return `<a href="${escapeHtml(safeUrl)}" target="_blank" rel="noreferrer">${escapeHtml(text)}</a>`;
    }

    function renderStatusPill(value) {
      return `<span class="status-pill">${escapeHtml(value || "unknown")}</span>`;
    }

    function renderDetailGrid(items) {
      return `
        <div class="detail-grid">
          ${items.map((item) => `
            <div class="detail-item">
              <span class="detail-label">${escapeHtml(item.label)}</span>
              <div>${item.valueHtml || escapeHtml(item.value ?? "—")}</div>
            </div>
          `).join("")}
        </div>
      `;
    }

    function formatJsonForDisplay(value) {
      if (value == null) {
        return null;
      }
      if (typeof value === "string") {
        const text = value.trim();
        return text || null;
      }
      try {
        return JSON.stringify(value, null, 2);
      } catch (error) {
        return String(value);
      }
    }

    function renderJsonDetailSection(title, value) {
      const text = formatJsonForDisplay(value);
      if (!text) {
        return "";
      }
      return `
        <details>
          <summary>${escapeHtml(title)}</summary>
          <pre>${escapeHtml(text)}</pre>
        </details>
      `;
    }

    function renderReasonList(items) {
      if (!Array.isArray(items) || !items.length) {
        return "—";
      }
      return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
    }

    function formatPathValue(path) {
      if (!Array.isArray(path) || !path.length) {
        return "—";
      }
      return path.join(" -> ");
    }

    function scrollToPanel(id) {
      const element = document.getElementById(id);
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }

    function killRateClass(count, total) {
      const safeTotal = Number(total || 0);
      const rate = safeTotal ? Number(count || 0) / safeTotal : 0;
      return rate >= 0.5 ? "kill-high" : "kill-low";
    }

    function renderHeroSummary() {
      const container = document.getElementById("hero-summary");
      if (!container) {
        return;
      }
      if (!dashboardStatsPayload && !dashboardCostsPayload && !operatorHomeSnapshot && !domainGraphPayload) {
        container.innerHTML = "<p class=\\"muted\\">Loading live snapshot…</p>";
        return;
      }

      const graphStats = domainGraphPayload ? domainGraphPayload.stats || {} : {};
      const operatorCounts = operatorHomeSnapshot ? operatorHomeSnapshot.counts || {} : {};
      const items = [
        {
          label: "Mapped domains",
          value: graphStats.unique_domains != null ? formatInteger(graphStats.unique_domains) : "…",
          valueClass: "score-accent",
        },
        {
          label: "Mapped jumps",
          value: graphStats.unique_connections != null ? formatInteger(graphStats.unique_connections) : "…",
        },
        {
          label: "Recent transmission rate",
          value: dashboardStatsPayload ? formatPercent(dashboardStatsPayload.transmission_rate) : "…",
          valueClass: "score-accent",
        },
        {
          label: "Unreviewed evidence",
          value: operatorHomeSnapshot ? formatInteger(operatorCounts.unreviewed_evidence_hits || 0) : "…",
        },
        {
          label: "Open strong rejections",
          value: operatorHomeSnapshot ? formatInteger(operatorCounts.open_strong_rejections || 0) : "…",
        },
        {
          label: "Cost per transmission",
          value: dashboardCostsPayload && dashboardCostsPayload.available
            ? formatCurrency(dashboardCostsPayload.cost_per_transmission)
            : "n/a",
          valueClass: "score-accent",
        },
      ];

      container.innerHTML = items.map((item) => `
        <div class="stat">
          <div class="muted">${escapeHtml(item.label)}</div>
          <div class="stat-value ${item.valueClass || ""}">${escapeHtml(item.value)}</div>
        </div>
      `).join("");
    }

    function renderStats(stats) {
      dashboardStatsPayload = stats;
      renderHeroSummary();
      const items = [
        { label: "Window", value: stats.window_requested },
        { label: "Total explorations", value: stats.total_explorations },
        { label: "Total transmitted", value: stats.total_transmitted, valueClass: "score-accent" },
        { label: "Transmission rate", value: `${stats.transmission_rate}%`, valueClass: "score-accent" },
        { label: "No patterns found", value: stats.no_patterns_found, valueClass: killRateClass(stats.no_patterns_found, stats.total_explorations) },
        { label: "Below score threshold", value: stats.below_score_threshold, valueClass: killRateClass(stats.below_score_threshold, stats.total_explorations) },
        { label: "Validation rejected", value: stats.validation_rejected, valueClass: killRateClass(stats.validation_rejected, stats.total_explorations) },
        { label: "Adversarial killed", value: stats.adversarial_killed, valueClass: killRateClass(stats.adversarial_killed, stats.total_explorations) },
        { label: "Provenance missing", value: stats.provenance_missing, valueClass: killRateClass(stats.provenance_missing, stats.total_explorations) },
        { label: "Distance too low", value: stats.distance_too_low, valueClass: killRateClass(stats.distance_too_low, stats.total_explorations) },
        { label: "Avg total_score (all)", value: formatScore(stats.avg_total_score_all), valueClass: "score-accent" },
        { label: "Avg total_score (transmitted)", value: formatScore(stats.avg_total_score_transmitted), valueClass: "score-accent" },
      ];
      document.getElementById("stats").innerHTML = items.map((item) => `
        <div class="stat">
          <div class="muted">${escapeHtml(item.label)}</div>
          <div class="stat-value ${item.valueClass || ""}">${escapeHtml(item.value)}</div>
        </div>
      `).join("");
    }

    function renderCosts(costs) {
      dashboardCostsPayload = costs;
      renderHeroSummary();
      const container = document.getElementById("costs");
      if (!costs.available) {
        container.innerHTML = "<p class=\\"muted\\">No cost data available</p>";
        return;
      }

      const items = [
        { label: "Total input tokens", value: formatInteger(costs.total_input_tokens), valueClass: "score-accent" },
        { label: "Total output tokens", value: formatInteger(costs.total_output_tokens), valueClass: "score-accent" },
        { label: "Estimated total cost", value: formatCurrency(costs.estimated_total_cost), valueClass: "score-accent" },
        { label: "Cost per transmission", value: formatCurrency(costs.cost_per_transmission), valueClass: "score-accent" },
        { label: "Cost per exploration", value: formatCurrency(costs.cost_per_exploration), valueClass: "score-accent" },
        { label: "Tokens per exploration (avg)", value: formatAverage(costs.tokens_per_exploration), valueClass: "score-accent" },
      ];

      container.innerHTML = items.map((item) => `
        <div class="stat">
          <div class="muted">${escapeHtml(item.label)}</div>
          <div class="stat-value ${item.valueClass || ""}">${escapeHtml(item.value)}</div>
        </div>
      `).join("");
    }

    function domainRoleLabel(role) {
      if (role === "seed") {
        return "seed-heavy";
      }
      if (role === "target") {
        return "target-heavy";
      }
      return "bridge";
    }

    function graphLinkKey(link) {
      return `${link.source}→${link.target}`;
    }

    function graphNodeColor(role, alpha = 1) {
      const palette = document.documentElement.dataset.theme === "light"
        ? {
            seed: `rgba(221, 134, 41, ${alpha})`,
            target: `rgba(34, 151, 201, ${alpha})`,
            bridge: `rgba(28, 143, 103, ${alpha})`,
          }
        : {
            seed: `rgba(255, 183, 77, ${alpha})`,
            target: `rgba(103, 213, 255, ${alpha})`,
            bridge: `rgba(89, 211, 154, ${alpha})`,
          };
      return palette[role] || palette.bridge;
    }

    function renderDomainGraph(payload) {
      domainGraphPayload = payload;
      renderHeroSummary();
      initializeDomainGraphControls();
      updateDomainGraph();
    }

    function initializeDomainGraphControls() {
      if (domainGraphControlsInitialized) {
        return;
      }
      const modeInput = document.getElementById("graph-mode");
      const scoreInput = document.getElementById("graph-score");
      const scoreValue = document.getElementById("graph-score-value");
      const searchInput = document.getElementById("graph-search");
      const resetButton = document.getElementById("graph-reset");
      const canvas = document.getElementById("domain-web-canvas");
      if (!modeInput || !scoreInput || !scoreValue || !searchInput || !resetButton || !canvas) {
        return;
      }

      scoreValue.textContent = Number(scoreInput.value || 0).toFixed(2);

      modeInput.addEventListener("change", () => {
        selectedGraphNodeId = null;
        selectedGraphLinkKey = null;
        updateDomainGraph();
      });
      scoreInput.addEventListener("input", () => {
        scoreValue.textContent = Number(scoreInput.value || 0).toFixed(2);
        selectedGraphLinkKey = null;
        updateDomainGraph();
      });
      searchInput.addEventListener("input", () => {
        window.clearTimeout(domainGraphSearchTimer);
        domainGraphSearchTimer = window.setTimeout(() => {
          selectedGraphNodeId = null;
          selectedGraphLinkKey = null;
          updateDomainGraph();
        }, 120);
      });
      resetButton.addEventListener("click", () => {
        modeInput.value = "all";
        scoreInput.value = "0";
        scoreValue.textContent = "0.00";
        searchInput.value = "";
        selectedGraphNodeId = null;
        selectedGraphLinkKey = null;
        hoveredGraphNodeId = null;
        hoveredGraphLinkKey = null;
        updateDomainGraph();
      });
      canvas.addEventListener("mousemove", handleDomainGraphCanvasMove);
      canvas.addEventListener("mouseleave", () => {
        hoveredGraphNodeId = null;
        hoveredGraphLinkKey = null;
        renderDomainGraphCanvas();
      });
      canvas.addEventListener("click", handleDomainGraphCanvasClick);
      window.addEventListener("resize", () => {
        window.clearTimeout(domainGraphResizeTimer);
        domainGraphResizeTimer = window.setTimeout(() => {
          updateDomainGraph();
        }, 120);
      });
      domainGraphControlsInitialized = true;
    }

    function buildFilteredDomainGraph(payload) {
      const mode = document.getElementById("graph-mode")?.value || "all";
      const minScore = Number(document.getElementById("graph-score")?.value || 0);
      const query = (document.getElementById("graph-search")?.value || "").trim().toLowerCase();
      const baseNodes = Array.isArray(payload?.nodes) ? payload.nodes : [];
      const baseLinks = Array.isArray(payload?.links) ? payload.links : [];

      let filteredLinks = baseLinks.filter((link) => {
        const maxScore = Number(link.max_score ?? link.avg_score ?? 0);
        if (maxScore < minScore) {
          return false;
        }
        if (mode === "successful") {
          return Number(link.transmitted_count || 0) > 0;
        }
        if (mode === "strong") {
          return Number(link.transmitted_count || 0) > 0 || maxScore >= Math.max(minScore, 0.8);
        }
        return true;
      });

      const matchedIds = new Set();
      if (query) {
        baseNodes.forEach((node) => {
          if (String(node.id || "").toLowerCase().includes(query)) {
            matchedIds.add(node.id);
          }
        });
        if (!matchedIds.size) {
          return {
            nodes: [],
            links: [],
            matchedIds,
            query,
            stats: {
              visible_domains: 0,
              visible_connections: 0,
              visible_transmitted_connections: 0,
              visible_explorations: 0,
              avg_score: null,
            },
          };
        }
        const focusIds = new Set(matchedIds);
        filteredLinks.forEach((link) => {
          if (matchedIds.has(link.source) || matchedIds.has(link.target)) {
            focusIds.add(link.source);
            focusIds.add(link.target);
          }
        });
        filteredLinks = filteredLinks.filter(
          (link) => focusIds.has(link.source) && focusIds.has(link.target)
        );
        if (!filteredLinks.length) {
          matchedIds.forEach((id) => focusIds.add(id));
        }
        const standaloneNodes = baseNodes
          .filter((node) => focusIds.has(node.id))
          .map((node) => ({
            ...node,
            degree: 0,
            radius: 8,
            isMatch: matchedIds.has(node.id),
          }));
        if (!filteredLinks.length) {
          return {
            nodes: standaloneNodes,
            links: [],
            matchedIds,
            query,
            stats: {
              visible_domains: standaloneNodes.length,
              visible_connections: 0,
              visible_transmitted_connections: 0,
              visible_explorations: 0,
              avg_score: null,
            },
          };
        }
      }

      const includedIds = new Set();
      filteredLinks.forEach((link) => {
        includedIds.add(link.source);
        includedIds.add(link.target);
      });
      const nodes = baseNodes
        .filter((node) => includedIds.has(node.id))
        .map((node) => ({
          ...node,
          degree: 0,
          radius: 7,
          isMatch: matchedIds.has(node.id),
        }));
      const nodeById = new Map(nodes.map((node) => [node.id, node]));
      const links = filteredLinks.filter(
        (link) => nodeById.has(link.source) && nodeById.has(link.target)
      );

      let totalScoredConnections = 0;
      let scoreSum = 0;
      let visibleExplorations = 0;
      links.forEach((link) => {
        visibleExplorations += Number(link.count || 0);
        const sourceNode = nodeById.get(link.source);
        const targetNode = nodeById.get(link.target);
        if (sourceNode) {
          sourceNode.degree += Number(link.count || 0);
        }
        if (targetNode) {
          targetNode.degree += Number(link.count || 0);
        }
        const score = Number(link.max_score ?? link.avg_score);
        if (Number.isFinite(score)) {
          totalScoredConnections += 1;
          scoreSum += score;
        }
      });
      nodes.forEach((node) => {
        node.radius = clamp(
          4 + Math.sqrt(Math.max(1, node.degree || node.connection_count || 1)) * 1.6
            + (Number(node.transmitted_count || 0) > 0 ? 1.5 : 0),
          5,
          18
        );
      });

      return {
        nodes,
        links,
        matchedIds,
        query,
        stats: {
          visible_domains: nodes.length,
          visible_connections: links.length,
          visible_transmitted_connections: links.filter(
            (link) => Number(link.transmitted_count || 0) > 0
          ).length,
          visible_explorations: visibleExplorations,
          avg_score: totalScoredConnections > 0 ? scoreSum / totalScoredConnections : null,
        },
      };
    }

    function renderDomainGraphSummary(graph) {
      const container = document.getElementById("domain-web-summary");
      if (!container) {
        return;
      }
      const overall = domainGraphPayload ? domainGraphPayload.stats || {} : {};
      const items = [
        {
          label: graph.query ? "Visible domains" : "Mapped domains",
          value: formatInteger(graph.query ? graph.stats.visible_domains : (overall.unique_domains || 0)),
          valueClass: "score-accent",
        },
        {
          label: graph.query ? "Visible connections" : "Mapped jumps",
          value: formatInteger(graph.query ? graph.stats.visible_connections : (overall.unique_connections || 0)),
        },
        {
          label: "Transmitted connections",
          value: formatInteger(
            graph.query
              ? graph.stats.visible_transmitted_connections
              : (overall.transmitted_connections || 0)
          ),
          valueClass: "score-accent",
        },
        {
          label: "Visible explorations",
          value: formatInteger(graph.query ? graph.stats.visible_explorations : (overall.total_explorations || 0)),
        },
        {
          label: "Average visible score",
          value: formatScore(graph.stats.avg_score ?? overall.avg_score),
          valueClass: "score-accent",
        },
        {
          label: "Search focus",
          value: graph.query ? truncateText(graph.query, 28) : "entire map",
        },
      ];

      container.innerHTML = items.map((item) => `
        <div class="stat">
          <div class="muted">${escapeHtml(item.label)}</div>
          <div class="stat-value ${item.valueClass || ""}">${escapeHtml(item.value)}</div>
        </div>
      `).join("");
    }

    function prepareDomainGraphCanvas() {
      const canvas = document.getElementById("domain-web-canvas");
      if (!canvas) {
        return null;
      }
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(320, Math.round(rect.width || 0));
      const height = Math.max(360, Math.round(rect.height || 0));
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
      }
      const context = canvas.getContext("2d");
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { canvas, context, width, height };
    }

    function computeDomainGraphLayout(graph, width, height) {
      const centerX = width / 2;
      const centerY = height / 2;
      const spread = Math.min(width, height) * 0.36;
      const nodes = graph.nodes.map((node) => {
        const cached = domainGraphLayoutCache.get(node.id);
        if (cached && Number.isFinite(cached.x) && Number.isFinite(cached.y)) {
          return {
            ...node,
            x: clamp(cached.x, 26, width - 26),
            y: clamp(cached.y, 26, height - 26),
            vx: 0,
            vy: 0,
          };
        }
        const seed = hashString(node.id);
        const angle = ((seed % 3600) / 3600) * Math.PI * 2;
        const ring = 0.18 + (((seed >> 3) % 1000) / 1000) * 0.82;
        const degreeBias = 1 - Math.min(0.62, Math.log1p(node.degree || 1) / 9);
        return {
          ...node,
          x: centerX + Math.cos(angle) * spread * ring * degreeBias,
          y: centerY + Math.sin(angle) * spread * ring * degreeBias,
          vx: 0,
          vy: 0,
        };
      });
      const nodeById = new Map(nodes.map((node) => [node.id, node]));
      const links = graph.links
        .map((link) => ({
          ...link,
          sourceNode: nodeById.get(link.source),
          targetNode: nodeById.get(link.target),
        }))
        .filter((link) => link.sourceNode && link.targetNode);

      if (nodes.length > 1) {
        const repulsionSamples = Math.min(
          24,
          Math.max(8, Math.round(Math.sqrt(nodes.length) * 1.4))
        );
        const ticks = Math.min(220, Math.max(90, 90 + Math.round(nodes.length / 5)));

        for (let tick = 0; tick < ticks; tick += 1) {
          const alpha = 1 - tick / ticks;

          links.forEach((link) => {
            const source = link.sourceNode;
            const target = link.targetNode;
            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const distance = Math.sqrt(dx * dx + dy * dy) || 0.001;
            const desiredDistance = clamp(
              78 - Math.min(22, Number(link.transmitted_count || 0) * 8) - Math.min(16, Number(link.count || 0) * 3),
              28,
              92
            );
            const springForce = (distance - desiredDistance) * 0.0022;
            const forceX = (dx / distance) * springForce;
            const forceY = (dy / distance) * springForce;
            source.vx += forceX;
            source.vy += forceY;
            target.vx -= forceX;
            target.vy -= forceY;
          });

          nodes.forEach((node, index) => {
            node.vx += (centerX - node.x) * 0.0007 * alpha;
            node.vy += (centerY - node.y) * 0.0007 * alpha;

            for (let sample = 1; sample <= repulsionSamples; sample += 1) {
              const other = nodes[(index + sample * 17 + tick * 13) % nodes.length];
              if (!other || other === node) {
                continue;
              }
              let dx = node.x - other.x;
              let dy = node.y - other.y;
              let distanceSquared = dx * dx + dy * dy;
              if (distanceSquared < 0.5) {
                dx = 0.1 + ((index + sample) % 3) * 0.07;
                dy = 0.1 + ((index + tick) % 3) * 0.07;
                distanceSquared = dx * dx + dy * dy;
              }

              const minDistance = node.radius + other.radius + 10;
              if (distanceSquared < minDistance * minDistance) {
                const distance = Math.sqrt(distanceSquared);
                const overlap = (minDistance - distance) / Math.max(distance, 0.001);
                const pushX = dx * overlap * 0.02;
                const pushY = dy * overlap * 0.02;
                node.vx += pushX;
                node.vy += pushY;
                other.vx -= pushX;
                other.vy -= pushY;
              }

              const repulsion = (1800 + (node.degree + other.degree) * 10) / (distanceSquared + 120);
              node.vx += dx * repulsion * 0.00045;
              node.vy += dy * repulsion * 0.00045;
            }
          });

          nodes.forEach((node) => {
            node.vx *= 0.84;
            node.vy *= 0.84;
            node.x = clamp(node.x + node.vx, 24, width - 24);
            node.y = clamp(node.y + node.vy, 24, height - 24);
          });
        }
      }

      nodes.forEach((node) => {
        domainGraphLayoutCache.set(node.id, { x: node.x, y: node.y });
      });

      return { nodes, links, width, height };
    }

    function distanceToSegment(pointX, pointY, x1, y1, x2, y2) {
      const dx = x2 - x1;
      const dy = y2 - y1;
      if (dx === 0 && dy === 0) {
        return Math.hypot(pointX - x1, pointY - y1);
      }
      const t = clamp(
        ((pointX - x1) * dx + (pointY - y1) * dy) / (dx * dx + dy * dy),
        0,
        1
      );
      const closestX = x1 + dx * t;
      const closestY = y1 + dy * t;
      return Math.hypot(pointX - closestX, pointY - closestY);
    }

    function findDomainGraphHit(pointX, pointY) {
      if (!currentDomainGraphLayout) {
        return null;
      }

      const nodeHit = [...currentDomainGraphLayout.nodes]
        .sort((left, right) => right.radius - left.radius)
        .find((node) => Math.hypot(pointX - node.x, pointY - node.y) <= node.radius + 3);
      if (nodeHit) {
        return { type: "node", item: nodeHit };
      }

      const linkHit = currentDomainGraphLayout.links.find((link) => {
        const threshold = 4 + Math.min(4, Number(link.count || 0));
        return distanceToSegment(
          pointX,
          pointY,
          link.sourceNode.x,
          link.sourceNode.y,
          link.targetNode.x,
          link.targetNode.y
        ) <= threshold;
      });
      if (linkHit) {
        return { type: "link", item: linkHit };
      }
      return null;
    }

    function renderDomainGraphCanvas() {
      const prepared = prepareDomainGraphCanvas();
      if (!prepared) {
        return;
      }
      const { context, width, height } = prepared;
      const emptyState = document.getElementById("domain-web-empty");

      if (!currentDomainGraph || !currentDomainGraph.nodes.length) {
        context.clearRect(0, 0, width, height);
        if (emptyState) {
          emptyState.hidden = false;
        }
        return;
      }

      if (emptyState) {
        emptyState.hidden = true;
      }

      const layoutNeedsRefresh =
        !currentDomainGraphLayout
        || currentDomainGraphLayout.width !== width
        || currentDomainGraphLayout.height !== height
        || currentDomainGraphLayout.nodes.length !== currentDomainGraph.nodes.length
        || currentDomainGraphLayout.links.length !== currentDomainGraph.links.length;

      if (layoutNeedsRefresh) {
        currentDomainGraphLayout = computeDomainGraphLayout(currentDomainGraph, width, height);
      }

      const isLight = document.documentElement.dataset.theme === "light";
      const background = context.createLinearGradient(0, 0, width, height);
      background.addColorStop(0, isLight ? "rgba(252, 255, 253, 0.96)" : "rgba(6, 19, 24, 0.96)");
      background.addColorStop(1, isLight ? "rgba(242, 247, 244, 0.96)" : "rgba(7, 16, 21, 0.99)");
      context.clearRect(0, 0, width, height);
      context.fillStyle = background;
      context.fillRect(0, 0, width, height);

      const activeNodeIds = new Set();
      if (selectedGraphNodeId) {
        activeNodeIds.add(selectedGraphNodeId);
        currentDomainGraph.links.forEach((link) => {
          if (link.source === selectedGraphNodeId || link.target === selectedGraphNodeId) {
            activeNodeIds.add(link.source);
            activeNodeIds.add(link.target);
          }
        });
      }
      if (selectedGraphLinkKey) {
        const selectedLink = currentDomainGraph.links.find(
          (link) => graphLinkKey(link) === selectedGraphLinkKey
        );
        if (selectedLink) {
          activeNodeIds.add(selectedLink.source);
          activeNodeIds.add(selectedLink.target);
        }
      }

      currentDomainGraphLayout.links.forEach((link) => {
        const key = graphLinkKey(link);
        const isSelected = key === selectedGraphLinkKey;
        const isHovered = key === hoveredGraphLinkKey;
        const touchesSelectedNode = selectedGraphNodeId
          && (link.source === selectedGraphNodeId || link.target === selectedGraphNodeId);
        const isDimmed = (
          (selectedGraphNodeId && !touchesSelectedNode)
          || (selectedGraphLinkKey && !isSelected)
        );
        const score = Number(link.max_score ?? link.avg_score ?? 0);
        const baseAlpha = clamp(
          0.16 + score * 0.36 + Number(link.transmitted_count || 0) * 0.18,
          0.12,
          0.88
        );
        const alpha = isSelected || isHovered ? 0.95 : isDimmed ? 0.08 : baseAlpha;
        context.beginPath();
        context.moveTo(link.sourceNode.x, link.sourceNode.y);
        context.lineTo(link.targetNode.x, link.targetNode.y);
        context.lineWidth = isSelected || isHovered
          ? 3.4
          : clamp(0.8 + Number(link.count || 0) * 0.55 + Number(link.transmitted_count || 0) * 0.7, 0.8, 4.2);
        if (Number(link.transmitted_count || 0) > 0) {
          context.strokeStyle = isLight
            ? `rgba(28, 143, 103, ${alpha})`
            : `rgba(89, 211, 154, ${alpha})`;
        } else if (score >= 0.85) {
          context.strokeStyle = isLight
            ? `rgba(41, 138, 176, ${alpha})`
            : `rgba(103, 213, 255, ${alpha})`;
        } else {
          context.strokeStyle = isLight
            ? `rgba(89, 112, 123, ${alpha})`
            : `rgba(154, 178, 190, ${alpha})`;
        }
        context.stroke();
      });

      const emphasizedNodes = new Set(activeNodeIds);
      if (hoveredGraphNodeId) {
        emphasizedNodes.add(hoveredGraphNodeId);
      }
      currentDomainGraphLayout.nodes
        .slice()
        .sort((left, right) => left.radius - right.radius)
        .forEach((node) => {
          const isSelected = node.id === selectedGraphNodeId;
          const isHovered = node.id === hoveredGraphNodeId;
          const isActive = !selectedGraphNodeId && !selectedGraphLinkKey
            ? true
            : emphasizedNodes.has(node.id);
          const alpha = isSelected || isHovered ? 1 : isActive ? 0.92 : 0.24;

          context.beginPath();
          context.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
          context.fillStyle = graphNodeColor(node.role, alpha);
          context.fill();
          context.lineWidth = isSelected || isHovered ? 2.4 : 1.2;
          context.strokeStyle = Number(node.transmitted_count || 0) > 0
            ? (isLight ? "rgba(16, 39, 49, 0.92)" : "rgba(244, 251, 255, 0.82)")
            : (isLight ? "rgba(16, 39, 49, 0.42)" : "rgba(244, 251, 255, 0.26)");
          context.stroke();
        });

      const labelCandidates = currentDomainGraphLayout.nodes
        .filter((node) => (
          node.isMatch
          || node.id === selectedGraphNodeId
          || node.id === hoveredGraphNodeId
          || Number(node.transmitted_count || 0) > 0
          || Number(node.connection_count || 0) >= 4
        ))
        .sort((left, right) => (
          Number(right.isMatch) - Number(left.isMatch)
          || Number(right.transmitted_count || 0) - Number(left.transmitted_count || 0)
          || Number(right.connection_count || 0) - Number(left.connection_count || 0)
        ))
        .slice(0, currentDomainGraph.query ? 28 : 18);

      context.font = '12px "SFMono-Regular", Menlo, Monaco, Consolas, monospace';
      context.textBaseline = "middle";
      labelCandidates.forEach((node) => {
        const label = truncateText(node.id, 34);
        const textX = node.x + node.radius + 8;
        const textY = node.y;
        const labelWidth = context.measureText(label).width + 10;
        context.fillStyle = isLight ? "rgba(255, 255, 255, 0.82)" : "rgba(7, 19, 23, 0.72)";
        context.fillRect(textX - 4, textY - 10, labelWidth, 20);
        context.fillStyle = isLight ? "rgba(16, 32, 41, 0.92)" : "rgba(244, 251, 255, 0.92)";
        context.fillText(label, textX, textY);
      });
    }

    function renderGraphNeighbors(neighbors) {
      if (!neighbors.length) {
        return "<p class=\\"muted\\">No adjacent domains under the current filters.</p>";
      }
      return `
        <div class="graph-neighbors">
          ${neighbors.map((item) => `
            <div class="graph-neighbor">
              <div>
                <strong>${escapeHtml(item.domain)}</strong>
                <div class="muted">
                  ${escapeHtml(formatInteger(item.count))} explorations
                  • ${escapeHtml(formatInteger(item.transmitted_count))} transmitted
                  • best ${escapeHtml(formatScore(item.max_score))}
                </div>
              </div>
              <button type="button" class="graph-focus-button" data-domain-id="${escapeHtml(item.domain)}">Focus</button>
            </div>
          `).join("")}
        </div>
      `;
    }

    function attachGraphDetailHandlers() {
      document.querySelectorAll(".graph-focus-button").forEach((button) => {
        button.addEventListener("click", () => {
          const domainId = button.dataset.domainId;
          const searchInput = document.getElementById("graph-search");
          if (!searchInput || !domainId) {
            return;
          }
          searchInput.value = domainId;
          selectedGraphNodeId = domainId;
          selectedGraphLinkKey = null;
          updateDomainGraph();
        });
      });
    }

    function renderDomainGraphDetail() {
      const panel = document.getElementById("domain-web-detail");
      if (!panel) {
        return;
      }
      if (!currentDomainGraph || !currentDomainGraph.nodes.length) {
        panel.innerHTML = `
          <h3>No Visible Domains</h3>
          <p class="muted">Reset the filters or lower the score threshold to bring the graph back.</p>
        `;
        return;
      }

      if (selectedGraphLinkKey) {
        const link = currentDomainGraph.links.find(
          (item) => graphLinkKey(item) === selectedGraphLinkKey
        );
        if (link) {
          panel.innerHTML = `
            <h3>${escapeHtml(link.source)} → ${escapeHtml(link.target)}</h3>
            <p class="muted">A directed BlackClaw jump from seed domain to target domain.</p>
            ${renderDetailGrid([
              { label: "Explorations", value: formatInteger(link.count) },
              { label: "Transmitted", value: formatInteger(link.transmitted_count) },
              { label: "Average score", value: formatScore(link.avg_score) },
              { label: "Best score", value: formatScore(link.max_score) },
              { label: "Last explored", value: formatTimestamp(link.latest_timestamp) },
            ])}
            <p>This edge survived filtering because BlackClaw actually made this jump. Stronger line weight means repeated passes; greener lines mean transmitted output made it through.</p>
            <div class="review-actions">
              <button type="button" class="graph-focus-button" data-domain-id="${escapeHtml(link.source)}">Focus source</button>
              <button type="button" class="graph-focus-button" data-domain-id="${escapeHtml(link.target)}">Focus target</button>
            </div>
          `;
          attachGraphDetailHandlers();
          return;
        }
      }

      if (selectedGraphNodeId) {
        const node = currentDomainGraph.nodes.find((item) => item.id === selectedGraphNodeId);
        if (node) {
          const neighbors = currentDomainGraph.links
            .filter((link) => link.source === node.id || link.target === node.id)
            .map((link) => ({
              domain: link.source === node.id ? link.target : link.source,
              count: Number(link.count || 0),
              transmitted_count: Number(link.transmitted_count || 0),
              max_score: link.max_score,
            }))
            .sort((left, right) => (
              right.transmitted_count - left.transmitted_count
              || (Number(right.max_score || 0) - Number(left.max_score || 0))
              || right.count - left.count
            ))
            .slice(0, 8);

          panel.innerHTML = `
            <h3>${escapeHtml(node.id)}</h3>
            <p class="muted">This domain appears as ${escapeHtml(domainRoleLabel(node.role))} in the current map.</p>
            ${renderDetailGrid([
              { label: "Connections", value: formatInteger(node.connection_count) },
              { label: "Outgoing jumps", value: formatInteger(node.outgoing_edges) },
              { label: "Incoming jumps", value: formatInteger(node.incoming_edges) },
              { label: "Exploration touches", value: formatInteger(node.appearance_count) },
              { label: "Transmitted touches", value: formatInteger(node.transmitted_count) },
              { label: "Average score", value: formatScore(node.avg_score) },
              { label: "Best score", value: formatScore(node.max_score) },
              { label: "Last seen", value: formatTimestamp(node.latest_timestamp) },
            ])}
            <p>Use this as a frontier read: domains with lots of outgoing edges are acting like launch pads, domains with lots of incoming edges are where BlackClaw keeps landing, and bridge nodes do both.</p>
            <h3>Strongest Adjacent Domains</h3>
            ${renderGraphNeighbors(neighbors)}
          `;
          attachGraphDetailHandlers();
          return;
        }
      }

      const topNodes = currentDomainGraph.nodes
        .slice()
        .sort((left, right) => (
          Number(right.transmitted_count || 0) - Number(left.transmitted_count || 0)
          || Number(right.connection_count || 0) - Number(left.connection_count || 0)
        ))
        .slice(0, 6)
        .map((node) => ({
          domain: node.id,
          count: Number(node.appearance_count || 0),
          transmitted_count: Number(node.transmitted_count || 0),
          max_score: node.max_score,
        }));

      panel.innerHTML = `
        <h3>Graph Snapshot</h3>
        <p class="muted">Pick a domain or a jump to inspect it directly. Until then, here are the most connected visible domains under the current filters.</p>
        ${renderGraphNeighbors(topNodes)}
      `;
      attachGraphDetailHandlers();
    }

    function updateDomainGraph() {
      if (!domainGraphPayload) {
        return;
      }
      currentDomainGraph = buildFilteredDomainGraph(domainGraphPayload);
      renderDomainGraphSummary(currentDomainGraph);

      if (
        selectedGraphNodeId
        && !currentDomainGraph.nodes.some((node) => node.id === selectedGraphNodeId)
      ) {
        selectedGraphNodeId = null;
      }
      if (
        selectedGraphLinkKey
        && !currentDomainGraph.links.some((link) => graphLinkKey(link) === selectedGraphLinkKey)
      ) {
        selectedGraphLinkKey = null;
      }

      currentDomainGraphLayout = null;
      renderDomainGraphCanvas();
      renderDomainGraphDetail();
    }

    function handleDomainGraphCanvasMove(event) {
      const prepared = prepareDomainGraphCanvas();
      if (!prepared || !currentDomainGraphLayout) {
        return;
      }
      const rect = prepared.canvas.getBoundingClientRect();
      const pointX = event.clientX - rect.left;
      const pointY = event.clientY - rect.top;
      const hit = findDomainGraphHit(pointX, pointY);
      prepared.canvas.style.cursor = hit ? "pointer" : "default";
      const nextHoveredNodeId = hit && hit.type === "node" ? hit.item.id : null;
      const nextHoveredLinkKey = hit && hit.type === "link" ? graphLinkKey(hit.item) : null;
      if (hoveredGraphNodeId === nextHoveredNodeId && hoveredGraphLinkKey === nextHoveredLinkKey) {
        return;
      }
      hoveredGraphNodeId = nextHoveredNodeId;
      hoveredGraphLinkKey = nextHoveredLinkKey;
      renderDomainGraphCanvas();
    }

    function handleDomainGraphCanvasClick(event) {
      const prepared = prepareDomainGraphCanvas();
      if (!prepared) {
        return;
      }
      const rect = prepared.canvas.getBoundingClientRect();
      const pointX = event.clientX - rect.left;
      const pointY = event.clientY - rect.top;
      const hit = findDomainGraphHit(pointX, pointY);
      if (!hit) {
        selectedGraphNodeId = null;
        selectedGraphLinkKey = null;
        renderDomainGraphCanvas();
        renderDomainGraphDetail();
        return;
      }
      if (hit.type === "node") {
        selectedGraphNodeId = selectedGraphNodeId === hit.item.id ? null : hit.item.id;
        selectedGraphLinkKey = null;
      } else {
        const key = graphLinkKey(hit.item);
        selectedGraphLinkKey = selectedGraphLinkKey === key ? null : key;
        selectedGraphNodeId = null;
      }
      renderDomainGraphCanvas();
      renderDomainGraphDetail();
    }

    function renderTransmissionTimeline(rows) {
      const container = document.getElementById("transmission-timeline");
      const points = rows
        .map((row) => {
          const time = Date.parse(row.timestamp);
          const score = Number(row.total_score);
          if (!Number.isFinite(time) || !Number.isFinite(score)) {
            return null;
          }
          return {
            time,
            date: new Date(time),
            score,
            transmitted: Number(row.transmitted || 0) > 0,
            transmissionNumber: row.transmission_number,
          };
        })
        .filter(Boolean)
        .sort((a, b) => a.time - b.time);

      if (!points.length) {
        container.innerHTML = "<p class=\\"muted\\">No scored transmission data yet. Once explorations are recorded, the timeline will appear here.</p>";
        return;
      }

      const width = 900;
      const height = 320;
      const padding = { top: 16, right: 20, bottom: 52, left: 58 };
      const minTime = points[0].time;
      const maxTime = points[points.length - 1].time;
      const timeRange = Math.max(1, maxTime - minTime);
      const rawMinScore = Math.min(...points.map((point) => point.score));
      const rawMaxScore = Math.max(...points.map((point) => point.score));
      const minScore = Math.min(0, rawMinScore);
      const maxScore = Math.max(1, rawMaxScore);
      const scoreRange = Math.max(0.001, maxScore - minScore);

      function xFor(time) {
        return padding.left + ((time - minTime) / timeRange) * (width - padding.left - padding.right);
      }

      function yFor(score) {
        return height - padding.bottom - ((score - minScore) / scoreRange) * (height - padding.top - padding.bottom);
      }

      const yTicks = [minScore, minScore + scoreRange / 2, maxScore];
      const xTicks = timeRange <= 1
        ? [minTime]
        : [minTime, minTime + timeRange / 2, maxTime];
      const linePoints = points
        .map((point) => `${xFor(point.time).toFixed(2)},${yFor(point.score).toFixed(2)}`)
        .join(" ");

      const gridLines = yTicks.map((tick) => {
        const y = yFor(tick).toFixed(2);
        return `
          <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="var(--border-color)" stroke-width="1" />
          <text x="${padding.left - 8}" y="${y}" fill="var(--muted)" font-size="12" text-anchor="end" dominant-baseline="middle">${escapeHtml(tick.toFixed(2))}</text>
        `;
      }).join("");

      const xLabels = xTicks.map((tick, index) => {
        const x = xFor(tick).toFixed(2);
        const anchor = index === 0 ? "start" : index === xTicks.length - 1 ? "end" : "middle";
        return `
          <text x="${x}" y="${height - 22}" fill="var(--muted)" font-size="12" text-anchor="${anchor}">${escapeHtml(formatTimelineDate(new Date(tick)))}</text>
        `;
      }).join("");

      const circles = points.map((point) => {
        const color = point.transmitted ? "var(--accent)" : "var(--kill-high)";
        const label = point.transmitted && point.transmissionNumber != null
          ? `Tx #${point.transmissionNumber}`
          : "Not transmitted";
        const title = `${formatTimelineDate(point.date, true)} | total_score ${point.score.toFixed(3)} | ${label}`;
        return `
          <circle
            cx="${xFor(point.time).toFixed(2)}"
            cy="${yFor(point.score).toFixed(2)}"
            r="${point.transmitted ? 4 : 3.5}"
            fill="${color}"
            opacity="${point.transmitted ? 0.95 : 0.75}"
          >
            <title>${escapeHtml(title)}</title>
          </circle>
        `;
      }).join("");

      container.innerHTML = `
        <div class="timeline-shell">
          <svg class="timeline-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Transmission timeline of total scores over time">
            <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}" stroke="var(--text)" stroke-width="1.5" />
            <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="var(--text)" stroke-width="1.5" />
            ${gridLines}
            ${points.length > 1 ? `<polyline fill="none" stroke="var(--muted)" stroke-width="1.5" opacity="0.7" points="${linePoints}" />` : ""}
            ${circles}
            ${xLabels}
            <text x="${(padding.left + width - padding.right) / 2}" y="${height - 4}" fill="var(--muted)" font-size="12" text-anchor="middle">Timestamp</text>
            <text x="18" y="${height / 2}" fill="var(--muted)" font-size="12" text-anchor="middle" transform="rotate(-90 18 ${height / 2})">total_score</text>
          </svg>
          <div class="timeline-meta">
            <div class="timeline-legend">
              <span><span class="timeline-swatch timeline-swatch-transmitted"></span>Transmitted</span>
              <span><span class="timeline-swatch timeline-swatch-untransmitted"></span>Not transmitted</span>
            </div>
            <span>${points.length} points</span>
          </div>
        </div>
      `;
    }

    function renderTopKilled(rows) {
      if (!rows.length) {
        document.getElementById("top-killed").innerHTML = "<p class=\\"muted\\">No non-transmitted explorations found.</p>";
        return;
      }
      const body = rows.map((row) => `
        <tr class="top-killed-row" data-exploration-id="${escapeHtml(row.id)}" aria-expanded="false">
          <td>${escapeHtml(row.id)}</td>
          <td class="score-accent">${escapeHtml(formatScore(row.total_score))}</td>
          <td>${escapeHtml(row.seed_domain)}</td>
          <td>${escapeHtml(row.jump_target_domain)}</td>
          <td>${escapeHtml(row.connection_description)}</td>
        </tr>
        <tr class="top-killed-detail-row" hidden>
          <td colspan="5" class="adversarial-cell">
            <div class="adversarial-detail">Click row to load adversarial detail.</div>
          </td>
        </tr>
      `).join("");
      document.getElementById("top-killed").innerHTML = `
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Total Score</th>
                <th>Seed</th>
                <th>Target</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      `;
      attachTopKilledHandlers();
    }

    function renderAdversarialDetail(payload) {
      if (!payload || payload.adversarial == null) {
        return "<div>Killed before adversarial stage</div>";
      }

      const adversarial = payload.adversarial;
      const killReasons = Array.isArray(adversarial.kill_reasons) ? adversarial.kill_reasons : [];
      const killReasonsHtml = killReasons.length
        ? `<ol>${killReasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}</ol>`
        : "<p>None</p>";

      const detailRows = [
        ["mapping_integrity", adversarial.mapping_integrity],
        ["invariant_validity", adversarial.invariant_validity],
        ["assumption_fragility", adversarial.assumption_fragility],
        ["test_discriminativeness", adversarial.test_discriminativeness],
        ["survival_score", adversarial.survival_score],
      ];

      return `
        <h3>Adversarial Detail</h3>
        <p><strong>Kill reasons</strong></p>
        ${killReasonsHtml}
        ${detailRows.map(([label, value]) => `
          <p><strong>${escapeHtml(label)}</strong>: ${escapeHtml(
            value === undefined || value === null ? "n/a" : value
          )}</p>
        `).join("")}
        <pre>${escapeHtml(JSON.stringify(adversarial, null, 2))}</pre>
      `;
    }

    function attachTopKilledHandlers() {
      document.querySelectorAll(".top-killed-row").forEach((row) => {
        row.addEventListener("click", async () => {
          const detailRow = row.nextElementSibling;
          const detail = detailRow.querySelector(".adversarial-detail");
          const explorationId = row.dataset.explorationId;

          if (!detailRow.hidden) {
            detailRow.hidden = true;
            row.setAttribute("aria-expanded", "false");
            return;
          }

          detailRow.hidden = false;
          row.setAttribute("aria-expanded", "true");

          if (detail.dataset.loaded === "true") {
            return;
          }

          detail.textContent = "Loading adversarial detail...";
          try {
            const payload = await fetchJson(`/api/explorations/${explorationId}/adversarial`);
            detail.innerHTML = renderAdversarialDetail(payload);
            detail.dataset.loaded = "true";
          } catch (error) {
            detail.innerHTML = `<div class="error">${escapeHtml(
              error instanceof Error ? error.message : "Failed to load adversarial detail"
            )}</div>`;
          }
        });
      });
    }

    function renderEvidenceReviewStats(stats) {
      const byReviewStatus = stats.by_review_status || {};
      const summaryItems = [
        { label: "Total hits", value: stats.total_hits || 0, valueClass: "score-accent" },
        { label: "Unreviewed", value: byReviewStatus.unreviewed || 0 },
        { label: "Accepted", value: byReviewStatus.accepted || 0, valueClass: "score-accent" },
        { label: "Dismissed", value: byReviewStatus.dismissed || 0 },
        { label: "Predictions needing review", value: stats.predictions_needing_review || 0 },
      ];
      document.getElementById("evidence-review-stats").innerHTML = summaryItems.map((item) => `
        <div class="stat">
          <div class="muted">${escapeHtml(item.label)}</div>
          <div class="stat-value ${item.valueClass || ""}">${escapeHtml(formatInteger(item.value))}</div>
        </div>
      `).join("");

      const byClassification = stats.by_classification || {};
      const rows = ["possible_support", "possible_contradiction", "unclear"].map((label) => {
        const row = byClassification[label] || {};
        return `
          <tr>
            <td>${escapeHtml(label)}</td>
            <td>${escapeHtml(formatInteger(row.unreviewed || 0))}</td>
            <td>${escapeHtml(formatInteger(row.accepted || 0))}</td>
            <td>${escapeHtml(formatInteger(row.dismissed || 0))}</td>
            <td>${escapeHtml(formatInteger(row.total || 0))}</td>
          </tr>
        `;
      }).join("");
      document.getElementById("evidence-review-breakdown").innerHTML = `
        <p class="muted">Classification breakdown across all stored evidence hits.</p>
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>Classification</th>
                <th>Unreviewed</th>
                <th>Accepted</th>
                <th>Dismissed</th>
                <th>Total</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    }

    function renderOperatorHome(snapshot) {
      operatorHomeSnapshot = snapshot;
      renderHeroSummary();
      const counts = snapshot.counts || {};
      const summaryItems = [
        { label: "Unreviewed evidence", value: counts.unreviewed_evidence_hits || 0, valueClass: "score-accent" },
        { label: "Predictions needing review", value: counts.predictions_needing_review || 0 },
        { label: "Open strong rejections", value: counts.open_strong_rejections || 0 },
        { label: "Open predictions", value: counts.open_predictions || 0 },
        { label: "Review-for-support candidates", value: counts.review_for_support_candidates || 0, valueClass: "score-accent" },
        { label: "Review-for-contradiction candidates", value: counts.review_for_contradiction_candidates || 0, valueClass: "score-accent" },
        { label: "Conflicting-evidence predictions", value: counts.conflicting_evidence_predictions || 0, valueClass: "score-accent" },
      ];
      document.getElementById("operator-home-summary").innerHTML = summaryItems.map((item) => `
        <div class="stat">
          <div class="muted">${escapeHtml(item.label)}</div>
          <div class="stat-value ${item.valueClass || ""}">${escapeHtml(formatInteger(item.value))}</div>
        </div>
      `).join("");

      renderOperatorHomeEvidence(snapshot.evidence_backlog || []);
      renderOperatorHomeOutcomes(snapshot.outcome_backlog || []);
      renderOperatorHomeStrongRejections(snapshot.strong_rejection_backlog || []);
    }

    function renderOperatorHomeEvidence(rows) {
      const container = document.getElementById("operator-home-evidence");
      if (!rows.length) {
        container.innerHTML = "<p class=\\"muted\\">No unreviewed evidence hits.</p>";
        return;
      }
      const body = rows.map((row) => `
        <tr
          class="review-table-row ${Number(row.id) === selectedEvidenceId ? "is-selected" : ""}"
          data-home-evidence-id="${escapeHtml(row.id)}"
        >
          <td>${escapeHtml(row.id)}</td>
          <td>${escapeHtml(row.prediction_id)}</td>
          <td>${renderStatusPill(row.classification)}</td>
          <td class="score-accent">${escapeHtml(formatScore(row.score))}</td>
          <td>${escapeHtml(truncateText(row.title, 72))}</td>
        </tr>
      `).join("");
      container.innerHTML = `
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>Evidence ID</th>
                <th>Prediction ID</th>
                <th>Classification</th>
                <th>Score</th>
                <th>Short title</th>
              </tr>
            </thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      `;
      document.querySelectorAll("#operator-home-evidence [data-home-evidence-id]").forEach((row) => {
        row.addEventListener("click", async () => {
          await loadEvidenceDetail(Number(row.dataset.homeEvidenceId));
          scrollToPanel("evidence-detail");
        });
      });
    }

    function renderOperatorHomeOutcomes(rows) {
      const container = document.getElementById("operator-home-outcomes");
      if (!rows.length) {
        container.innerHTML = "<p class=\\"muted\\">No open outcome candidates right now.</p>";
        return;
      }
      const body = rows.map((row) => `
        <tr
          class="review-table-row ${Number(row.id) === selectedOutcomePredictionId ? "is-selected" : ""}"
          data-home-prediction-id="${escapeHtml(row.id)}"
        >
          <td>${escapeHtml(row.id)}</td>
          <td>${escapeHtml(row.transmission_number)}</td>
          <td>${renderStatusPill(row.recommendation || "insufficient_evidence")}</td>
          <td>${escapeHtml(row.mechanism_type || "unknown")}</td>
          <td>${escapeHtml(row.utility_class || "unknown")}</td>
          <td>${escapeHtml(truncateText(row.prediction_summary || "—", 78))}</td>
        </tr>
      `).join("");
      container.innerHTML = `
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>Prediction ID</th>
                <th>Tx #</th>
                <th>Recommendation</th>
                <th>Mechanism type</th>
                <th>Utility</th>
                <th>Short prediction</th>
              </tr>
            </thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      `;
      document.querySelectorAll("#operator-home-outcomes [data-home-prediction-id]").forEach((row) => {
        row.addEventListener("click", async () => {
          await loadOutcomeReviewDetail(Number(row.dataset.homePredictionId));
          scrollToPanel("outcome-review-detail");
        });
      });
    }

    function renderOperatorHomeStrongRejections(rows) {
      const container = document.getElementById("operator-home-strong-rejections");
      if (!rows.length) {
        container.innerHTML = "<p class=\\"muted\\">No open strong rejections.</p>";
        return;
      }
      const body = rows.map((row) => `
        <tr
          class="review-table-row ${Number(row.id) === selectedStrongRejectionId ? "is-selected" : ""}"
          data-home-strong-rejection-id="${escapeHtml(row.id)}"
        >
          <td>${escapeHtml(row.id)}</td>
          <td class="score-accent">${escapeHtml(formatScore(row.total_score))}</td>
          <td>${escapeHtml(row.mechanism_type || "unknown")}</td>
          <td>${escapeHtml(`${row.seed_domain || "—"} -> ${row.target_domain || "—"}`)}</td>
          <td>${escapeHtml(truncateText(row.salvage_reason || "—", 72))}</td>
        </tr>
      `).join("");
      container.innerHTML = `
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>Rejection ID</th>
                <th>Total score</th>
                <th>Mechanism type</th>
                <th>Seed -> target</th>
                <th>Salvage reason</th>
              </tr>
            </thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      `;
      document.querySelectorAll("#operator-home-strong-rejections [data-home-strong-rejection-id]").forEach((row) => {
        row.addEventListener("click", async () => {
          await loadStrongRejectionDetail(Number(row.dataset.homeStrongRejectionId));
          scrollToPanel("strong-rejection-detail");
        });
      });
    }

    function renderEvidenceReviewQueue(rows) {
      evidenceQueueRows = rows;
      const container = document.getElementById("evidence-review-queue");
      if (!rows.length) {
        container.innerHTML = "<p class=\\"muted\\">No unreviewed evidence hits found.</p>";
        return;
      }
      const body = rows.map((row) => `
        <tr
          class="review-table-row ${Number(row.id) === selectedEvidenceId ? "is-selected" : ""}"
          data-evidence-id="${escapeHtml(row.id)}"
        >
          <td>${escapeHtml(row.id)}</td>
          <td>${escapeHtml(row.prediction_id)}</td>
          <td>${renderStatusPill(row.classification)}</td>
          <td>${renderStatusPill(row.review_status)}</td>
          <td class="score-accent">${escapeHtml(formatScore(row.score))}</td>
          <td>${escapeHtml(truncateText(row.title, 96))}</td>
          <td>${escapeHtml(formatTimestamp(row.scan_timestamp))}</td>
        </tr>
      `).join("");
      container.innerHTML = `
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>Evidence ID</th>
                <th>Prediction ID</th>
                <th>Classification</th>
                <th>Review status</th>
                <th>Score</th>
                <th>Title</th>
                <th>Scan timestamp</th>
              </tr>
            </thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      `;
      attachEvidenceQueueHandlers();
    }

    function attachEvidenceQueueHandlers() {
      document.querySelectorAll("#evidence-review-queue [data-evidence-id]").forEach((row) => {
        row.addEventListener("click", () => {
          loadEvidenceDetail(Number(row.dataset.evidenceId));
        });
      });
    }

    function renderEvidenceDetail(payload) {
      return `
        ${renderDetailGrid([
          { label: "Evidence ID", value: payload.id },
          { label: "Prediction ID", value: payload.prediction_id },
          { label: "Classification", valueHtml: renderStatusPill(payload.classification) },
          { label: "Review status", valueHtml: renderStatusPill(payload.review_status) },
          { label: "Score", value: formatScore(payload.score) },
          { label: "Source type", value: payload.source_type || "unknown" },
          { label: "Scan timestamp", value: formatTimestamp(payload.scan_timestamp) },
          { label: "Updated", value: formatTimestamp(payload.updated_at) },
        ])}
        <p><strong>Title:</strong> ${escapeHtml(payload.title || "Untitled result")}</p>
        <p><strong>URL:</strong> ${renderExternalLink(payload.url)}</p>
        <p><strong>Snippet:</strong> ${escapeHtml(payload.snippet || "—")}</p>
        <p><strong>Query:</strong> ${escapeHtml(payload.query_used || "—")}</p>
        <p><strong>Notes:</strong> ${escapeHtml(payload.notes || "—")}</p>
        <div class="review-actions">
          <input
            type="text"
            class="evidence-note-input"
            placeholder="Optional note"
            value="${inputValue(payload.notes)}"
          >
          <button type="button" class="evidence-action" data-action="accept" data-evidence-id="${escapeHtml(payload.id)}">Accept</button>
          <button type="button" class="evidence-action" data-action="dismiss" data-evidence-id="${escapeHtml(payload.id)}">Dismiss</button>
          <span class="review-status"></span>
        </div>
      `;
    }

    async function loadEvidenceDetail(evidenceId) {
      selectedEvidenceId = evidenceId;
      renderEvidenceReviewQueue(evidenceQueueRows);
      const panel = document.getElementById("evidence-detail");
      panel.innerHTML = "<p class=\\"muted\\">Loading evidence detail…</p>";
      try {
        const payload = await fetchJson(`/api/evidence/${evidenceId}`);
        panel.innerHTML = renderEvidenceDetail(payload);
        attachEvidenceActionHandlers();
      } catch (error) {
        panel.innerHTML = `<div class="error">${escapeHtml(
          error instanceof Error ? error.message : "Failed to load evidence detail"
        )}</div>`;
      }
    }

    function attachEvidenceActionHandlers() {
      document.querySelectorAll(".evidence-action").forEach((button) => {
        button.addEventListener("click", async () => {
          const panel = button.closest(".detail-panel");
          const noteInput = panel.querySelector(".evidence-note-input");
          const status = panel.querySelector(".review-status");
          const buttons = panel.querySelectorAll(".evidence-action");
          const evidenceId = Number(button.dataset.evidenceId);
          const action = button.dataset.action;

          status.textContent = "Saving...";
          buttons.forEach((item) => {
            item.disabled = true;
          });
          noteInput.disabled = true;

          try {
            await postJson(`/api/evidence/${evidenceId}/${action}`, {
              note: noteInput.value,
            });
            status.textContent = action === "accept" ? "Accepted" : "Dismissed";
            await loadReviewData();
          } catch (error) {
            status.textContent = error instanceof Error ? error.message : "Save failed";
          } finally {
            buttons.forEach((item) => {
              item.disabled = false;
            });
            noteInput.disabled = false;
          }
        });
      });
    }

    function renderOutcomeSuggestionStats(stats) {
      const suggestionBuckets = stats.suggestion_buckets || {};
      document.getElementById("outcome-suggestion-buckets").innerHTML = OUTCOME_SUGGESTION_BUCKETS.map((label) => `
        <div class="stat">
          <div class="muted">${escapeHtml(OUTCOME_SUGGESTION_LABELS[label])}</div>
          <div class="stat-value score-accent">${escapeHtml(formatInteger(suggestionBuckets[label] || 0))}</div>
        </div>
      `).join("");

      const overall = stats.overall || {};
      const backlog = stats.review_backlog || {};
      const backlogItems = [
        { label: "Open predictions", value: overall.open || 0 },
        { label: "Resolved predictions", value: overall.resolved_total || 0 },
        { label: "Predictions needing review", value: backlog.open_predictions_needing_review || 0 },
        { label: "Unreviewed reviewable hits", value: backlog.total_unreviewed_reviewable_evidence_hits || 0 },
        { label: "Accepted support only", value: backlog.open_predictions_with_accepted_support_only || 0 },
        { label: "Accepted contradiction only", value: backlog.open_predictions_with_accepted_contradiction_only || 0 },
        { label: "Accepted conflicting evidence", value: backlog.open_predictions_with_accepted_conflicting_evidence || 0 },
      ];
      document.getElementById("outcome-review-backlog").innerHTML = backlogItems.map((item) => `
        <div class="stat">
          <div class="muted">${escapeHtml(item.label)}</div>
          <div class="stat-value">${escapeHtml(formatInteger(item.value))}</div>
        </div>
      `).join("");
    }

    function renderOutcomeReviewQueue(rows) {
      outcomeQueueRows = rows;
      const container = document.getElementById("outcome-review-queue");
      if (!rows.length) {
        container.innerHTML = "<p class=\\"muted\\">No review-ready predictions found.</p>";
        return;
      }
      const body = rows.map((row) => `
        <tr
          class="review-table-row ${Number(row.id) === selectedOutcomePredictionId ? "is-selected" : ""}"
          data-prediction-id="${escapeHtml(row.id)}"
        >
          <td>${escapeHtml(row.id)}</td>
          <td>${escapeHtml(row.transmission_number)}</td>
          <td>${renderStatusPill(row.outcome_status || "open")}</td>
          <td>${escapeHtml(formatInteger(row.accepted_support_hits || 0))}</td>
          <td>${escapeHtml(formatInteger(row.accepted_contradiction_hits || 0))}</td>
          <td>${escapeHtml(formatInteger(row.unreviewed_reviewable_hits || 0))}</td>
          <td>${renderStatusPill(row.recommendation || "insufficient_evidence")}</td>
          <td>${escapeHtml(row.mechanism_type || "unknown")}</td>
          <td>${escapeHtml(truncateText(predictionSummary(row), 110))}</td>
        </tr>
      `).join("");
      container.innerHTML = `
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>Prediction ID</th>
                <th>Transmission #</th>
                <th>Current outcome</th>
                <th>Support hits</th>
                <th>Contradiction hits</th>
                <th>Unreviewed reviewable hits</th>
                <th>Recommendation</th>
                <th>Mechanism type</th>
                <th>Short prediction</th>
              </tr>
            </thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      `;
      attachOutcomeQueueHandlers();
    }

    function attachOutcomeQueueHandlers() {
      document.querySelectorAll("#outcome-review-queue [data-prediction-id]").forEach((row) => {
        row.addEventListener("click", () => {
          loadOutcomeReviewDetail(Number(row.dataset.predictionId));
        });
      });
    }

    function renderOutcomeHitGroup(title, rows, totalCount) {
      if (!rows.length) {
        return `
          <details open>
            <summary>${escapeHtml(title)} (0 shown of ${totalCount || 0})</summary>
            <p class="muted">None.</p>
          </details>
        `;
      }
      return `
        <details open>
          <summary>${escapeHtml(title)} (${rows.length} shown of ${totalCount || 0})</summary>
          <div class="detail-stack">
            ${rows.map((row) => `
              <div class="detail-card">
                <p><strong>Evidence #${escapeHtml(row.id)}</strong> ${renderStatusPill(row.classification)} ${renderStatusPill(row.review_status)}</p>
                <p><strong>Score:</strong> ${escapeHtml(formatScore(row.score))} | <strong>Scanned:</strong> ${escapeHtml(formatTimestamp(row.scan_timestamp))}</p>
                <p><strong>Title:</strong> ${escapeHtml(row.title || "Untitled result")}</p>
                <p><strong>URL:</strong> ${renderExternalLink(row.url)}</p>
                <p><strong>Snippet:</strong> ${escapeHtml(row.snippet || "—")}</p>
                <p><strong>Query:</strong> ${escapeHtml(row.query_used || "—")}</p>
              </div>
            `).join("")}
          </div>
        </details>
      `;
    }

    function renderOutcomeReviewDetail(payload) {
      const statement = payload.prediction_statement && payload.prediction_statement !== payload.prediction_summary
        ? `<p><strong>Statement:</strong> ${escapeHtml(payload.prediction_statement)}</p>`
        : "";
      return `
        ${renderDetailGrid([
          { label: "Prediction ID", value: payload.id },
          { label: "Transmission #", value: payload.transmission_number },
          { label: "Status", valueHtml: renderStatusPill(payload.status || "unknown") },
          { label: "Outcome", valueHtml: renderStatusPill(payload.outcome_status || "open") },
          { label: "Utility", value: payload.utility_class || "unknown" },
          { label: "Mechanism type", value: payload.mechanism_type || "unknown" },
          { label: "Source domain", value: payload.source_domain || "—" },
          { label: "Target domain", value: payload.target_domain || "—" },
          { label: "Prediction quality", value: formatScore(payload.prediction_quality_score) },
          { label: "Depth score", value: formatScore(payload.depth_score) },
          { label: "Adversarial survival", value: formatScore(payload.adversarial_survival_score) },
          { label: "Recommendation", valueHtml: renderStatusPill(payload.recommendation || "insufficient_evidence") },
        ])}
        <p><strong>Summary:</strong> ${escapeHtml(payload.prediction_summary || "—")}</p>
        ${statement}
        <p><strong>Test summary:</strong> ${escapeHtml(payload.test_summary || "—")}</p>
        <p><strong>Falsification condition:</strong> ${escapeHtml(payload.falsification_condition || "—")}</p>
        <p><strong>Recommendation rationale:</strong> ${escapeHtml(payload.recommendation_rationale || "—")}</p>
        ${renderDetailGrid([
          { label: "Accepted support hits", value: formatInteger(payload.accepted_support_hits || 0) },
          { label: "Accepted contradiction hits", value: formatInteger(payload.accepted_contradiction_hits || 0) },
          { label: "Unreviewed reviewable hits", value: formatInteger(payload.unreviewed_reviewable_hits || 0) },
          { label: "Dismissed reviewable hits", value: formatInteger(payload.dismissed_reviewable_hits || 0) },
          { label: "Accepted unclear hits", value: formatInteger(payload.accepted_unclear_hits || 0) },
          { label: "Total hits", value: formatInteger(payload.total_hits || 0) },
        ])}
        ${renderOutcomeHitGroup(
          "Accepted support hits",
          payload.accepted_support_examples || [],
          payload.accepted_support_hits || 0
        )}
        ${renderOutcomeHitGroup(
          "Accepted contradiction hits",
          payload.accepted_contradiction_examples || [],
          payload.accepted_contradiction_hits || 0
        )}
        ${renderOutcomeHitGroup(
          "Unreviewed reviewable hits",
          payload.unreviewed_reviewable_examples || [],
          payload.unreviewed_reviewable_hits || 0
        )}
      `;
    }

    async function loadOutcomeReviewDetail(predictionId) {
      selectedOutcomePredictionId = predictionId;
      renderOutcomeReviewQueue(outcomeQueueRows);
      const panel = document.getElementById("outcome-review-detail");
      panel.innerHTML = "<p class=\\"muted\\">Loading outcome review detail…</p>";
      try {
        const payload = await fetchJson(`/api/outcome-review/${predictionId}`);
        panel.innerHTML = renderOutcomeReviewDetail(payload);
      } catch (error) {
        panel.innerHTML = `<div class="error">${escapeHtml(
          error instanceof Error ? error.message : "Failed to load outcome review detail"
        )}</div>`;
      }
    }

    function renderStrongRejectionStats(stats) {
      const items = [
        { label: "Total strong rejections", value: stats.total || 0 },
        { label: "Open", value: stats.open || 0, valueClass: "score-accent" },
        { label: "Salvaged", value: stats.salvaged || 0 },
        { label: "Dismissed", value: stats.dismissed || 0 },
        { label: "Avg total score", value: formatScore(stats.average_total_score), valueClass: "score-accent" },
      ];
      document.getElementById("strong-rejection-stats").innerHTML = items.map((item) => `
        <div class="stat">
          <div class="muted">${escapeHtml(item.label)}</div>
          <div class="stat-value ${item.valueClass || ""}">${escapeHtml(item.value)}</div>
        </div>
      `).join("");
    }

    function renderStrongRejectionQueue(rows) {
      strongRejectionRows = rows;
      const container = document.getElementById("strong-rejection-queue");
      if (!rows.length) {
        container.innerHTML = "<p class=\\"muted\\">No strong rejections found.</p>";
        return;
      }
      const body = rows.map((row) => `
        <tr
          class="review-table-row ${Number(row.id) === selectedStrongRejectionId ? "is-selected" : ""}"
          data-strong-rejection-id="${escapeHtml(row.id)}"
        >
          <td>${escapeHtml(row.id)}</td>
          <td>${renderStatusPill(row.status)}</td>
          <td class="score-accent">${escapeHtml(formatScore(row.total_score))}</td>
          <td>${escapeHtml(row.mechanism_type || "unknown")}</td>
          <td>${escapeHtml(row.seed_domain || "—")}</td>
          <td>${escapeHtml(row.target_domain || "—")}</td>
          <td>${escapeHtml(row.rejection_stage || "—")}</td>
          <td>${escapeHtml(truncateText(row.salvage_reason || "—", 90))}</td>
          <td>${escapeHtml(formatTimestamp(row.timestamp))}</td>
        </tr>
      `).join("");
      container.innerHTML = `
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>Rejection ID</th>
                <th>Status</th>
                <th>Total score</th>
                <th>Mechanism type</th>
                <th>Seed domain</th>
                <th>Target domain</th>
                <th>Rejection stage</th>
                <th>Salvage reason</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      `;
      attachStrongRejectionQueueHandlers();
    }

    function attachStrongRejectionQueueHandlers() {
      document.querySelectorAll("#strong-rejection-queue [data-strong-rejection-id]").forEach((row) => {
        row.addEventListener("click", () => {
          loadStrongRejectionDetail(Number(row.dataset.strongRejectionId));
        });
      });
    }

    function renderStrongRejectionDetail(payload) {
      return `
        ${renderDetailGrid([
          { label: "Rejection ID", value: payload.id },
          { label: "Timestamp", value: formatTimestamp(payload.timestamp) },
          { label: "Status", valueHtml: renderStatusPill(payload.status) },
          { label: "Exploration ID", value: payload.exploration_id ?? "—" },
          { label: "Seed domain", value: payload.seed_domain || "—" },
          { label: "Target domain", value: payload.target_domain || "—" },
          { label: "Total score", value: formatScore(payload.total_score) },
          { label: "Novelty score", value: formatScore(payload.novelty_score) },
          { label: "Distance score", value: formatScore(payload.distance_score) },
          { label: "Depth score", value: formatScore(payload.depth_score) },
          { label: "Prediction quality", value: formatScore(payload.prediction_quality_score) },
          { label: "Mechanism type", value: payload.mechanism_type || "—" },
          { label: "Rejection stage", value: payload.rejection_stage || "—" },
        ])}
        <p><strong>Path:</strong> ${escapeHtml(formatPathValue(payload.path))}</p>
        <p><strong>Salvage reason:</strong> ${escapeHtml(payload.salvage_reason || "—")}</p>
        <p><strong>Rejection reasons:</strong></p>
        ${renderReasonList(payload.rejection_reasons)}
        <p><strong>Notes:</strong> ${escapeHtml(payload.notes || "—")}</p>
        <div class="review-actions">
          <input
            type="text"
            class="strong-rejection-note-input"
            placeholder="Optional note"
            value="${inputValue(payload.notes)}"
          >
          <button type="button" class="strong-rejection-action" data-action="salvage" data-strong-rejection-id="${escapeHtml(payload.id)}">Mark salvaged</button>
          <button type="button" class="strong-rejection-action" data-action="dismiss" data-strong-rejection-id="${escapeHtml(payload.id)}">Dismiss</button>
          <span class="review-status"></span>
        </div>
        ${renderJsonDetailSection("Connection payload", payload.connection_payload)}
        ${renderJsonDetailSection("Validation", payload.validation)}
        ${renderJsonDetailSection("Evidence map", payload.evidence_map)}
        ${renderJsonDetailSection("Mechanism typing", payload.mechanism_typing)}
      `;
    }

    async function loadStrongRejectionDetail(rejectionId) {
      selectedStrongRejectionId = rejectionId;
      renderStrongRejectionQueue(strongRejectionRows);
      const panel = document.getElementById("strong-rejection-detail");
      panel.innerHTML = "<p class=\\"muted\\">Loading strong rejection detail…</p>";
      try {
        const payload = await fetchJson(`/api/strong-rejection/${rejectionId}`);
        panel.innerHTML = renderStrongRejectionDetail(payload);
        attachStrongRejectionActionHandlers();
      } catch (error) {
        panel.innerHTML = `<div class="error">${escapeHtml(
          error instanceof Error ? error.message : "Failed to load strong rejection detail"
        )}</div>`;
      }
    }

    function attachStrongRejectionActionHandlers() {
      document.querySelectorAll("#strong-rejection-detail .strong-rejection-action").forEach((button) => {
        button.addEventListener("click", async () => {
          const panel = button.closest(".detail-panel");
          const noteInput = panel.querySelector(".strong-rejection-note-input");
          const status = panel.querySelector(".review-status");
          const buttons = panel.querySelectorAll(".strong-rejection-action");
          const rejectionId = Number(button.dataset.strongRejectionId);
          const action = button.dataset.action;

          status.textContent = "Saving...";
          buttons.forEach((item) => {
            item.disabled = true;
          });
          noteInput.disabled = true;

          try {
            await postJson(`/api/strong-rejection/${rejectionId}/${action}`, {
              note: noteInput.value,
            });
            await loadReviewData();
          } catch (error) {
            status.textContent = error instanceof Error ? error.message : "Save failed";
          } finally {
            buttons.forEach((item) => {
              item.disabled = false;
            });
            noteInput.disabled = false;
          }
        });
      });
    }

    function renderGradeSummary(rows) {
      const summary = document.getElementById("grade-summary");
      const counts = Object.fromEntries(GRADE_OPTIONS.map((grade) => [grade, 0]));
      let graded = 0;
      rows.forEach((row) => {
        if (GRADE_OPTIONS.includes(row.user_rating)) {
          counts[row.user_rating] += 1;
          graded += 1;
        }
      });
      if (!graded) {
        summary.hidden = true;
        summary.textContent = "";
        return;
      }
      summary.hidden = false;
      summary.textContent = GRADE_OPTIONS.map((grade) => `${grade}: ${counts[grade]}`).join(" | ");
    }

    function buildGradeOptions(selectedGrade) {
      const options = ['<option value="">Grade</option>'];
      GRADE_OPTIONS.forEach((grade) => {
        const selected = grade === selectedGrade ? " selected" : "";
        options.push(`<option value="${grade}"${selected}>${grade}</option>`);
      });
      return options.join("");
    }

    function attachGradeHandlers() {
      document.querySelectorAll(".grade-save").forEach((button) => {
        button.addEventListener("click", async () => {
          const controls = button.closest(".grade-controls");
          const select = controls.querySelector(".grade-select");
          const notes = controls.querySelector(".grade-notes");
          const status = controls.querySelector(".grade-status");
          const transmissionId = Number(button.dataset.transmissionId);
          const grade = select.value;

          if (!GRADE_OPTIONS.includes(grade)) {
            status.textContent = "Pick grade";
            return;
          }

          status.textContent = "Saving...";
          button.disabled = true;
          select.disabled = true;
          notes.disabled = true;

          try {
            const response = await fetch(`/api/transmissions/${transmissionId}/grade`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                grade,
                notes: notes.value,
              }),
            });
            const payload = await response.json();
            if (!response.ok || !payload.ok) {
              throw new Error(payload.error || "Save failed");
            }

            const row = transmissionRows.find((item) => Number(item.id) === transmissionId);
            if (row) {
              row.user_rating = grade;
              row.user_notes = notes.value;
            }
            renderGradeSummary(transmissionRows);
            status.textContent = "Saved";
            window.setTimeout(() => {
              if (status.textContent === "Saved") {
                status.textContent = "";
              }
            }, 1500);
          } catch (error) {
            status.textContent = error instanceof Error ? error.message : "Save failed";
          } finally {
            button.disabled = false;
            select.disabled = false;
            notes.disabled = false;
          }
        });
      });
    }

    function attachCopyHandlers() {
      document.querySelectorAll(".copy-markdown").forEach((button) => {
        button.addEventListener("click", async () => {
          const transmissionId = Number(button.dataset.transmissionId);
          const status = button.parentElement.querySelector(".copy-status");
          status.textContent = "Copying...";
          button.disabled = true;

          try {
            const markdown = await fetchText(`/api/transmissions/${transmissionId}/markdown`);
            await navigator.clipboard.writeText(markdown);
            status.textContent = "Copied!";
            window.setTimeout(() => {
              if (status.textContent === "Copied!") {
                status.textContent = "";
              }
            }, 1500);
          } catch (error) {
            status.textContent = error instanceof Error ? error.message : "Copy failed";
          } finally {
            button.disabled = false;
          }
        });
      });
    }

    function renderTransmissions(rows) {
      transmissionRows = rows;
      renderGradeSummary(rows);
      document.getElementById("transmission-count").textContent = `${rows.length} transmissions`;
      if (!rows.length) {
        document.getElementById("transmissions").innerHTML = "<p class=\\"muted\\">No transmissions found.</p>";
        return;
      }
      document.getElementById("transmissions").innerHTML = rows.map((row) => `
        <div class="transmission-item">
          <details>
            <summary>
              Tx #${escapeHtml(row.transmission_number)} | score <span class="score-accent">${escapeHtml(formatScore(row.total_score))}</span> | ${escapeHtml(row.seed_domain)} -> ${escapeHtml(row.jump_target_domain)}
            </summary>
            <pre>${escapeHtml(row.formatted_output)}</pre>
          </details>
          <div class="grade-controls">
            <select class="grade-select" aria-label="Grade for transmission ${escapeHtml(row.transmission_number)}">
              ${buildGradeOptions(GRADE_OPTIONS.includes(row.user_rating) ? row.user_rating : "")}
            </select>
            <input
              class="grade-notes"
              type="text"
              placeholder="Notes (optional)"
              value="${inputValue(row.user_notes)}"
            >
            <button type="button" class="grade-save" data-transmission-id="${escapeHtml(row.id)}">Save</button>
            <span class="grade-status"></span>
            <button type="button" class="copy-markdown" data-transmission-id="${escapeHtml(row.id)}">Copy MD</button>
            <span class="copy-status"></span>
          </div>
        </div>
      `).join("");
      attachGradeHandlers();
      attachCopyHandlers();
    }

    async function loadReviewData() {
      const [
        operatorHome,
        evidenceReviewStats,
        evidenceReviewQueue,
        outcomeReviewQueue,
        outcomeSuggestionStats,
        strongRejectionStats,
        strongRejectionQueue,
      ] = await Promise.all([
        fetchJson("/api/operator-home"),
        fetchJson("/api/evidence-review-stats"),
        fetchJson("/api/evidence-review-queue?limit=25"),
        fetchJson("/api/outcome-review-queue?limit=25"),
        fetchJson("/api/outcome-suggestion-stats"),
        fetchJson("/api/strong-rejection-stats"),
        fetchJson("/api/strong-rejections?limit=25"),
      ]);
      renderOperatorHome(operatorHome);
      renderEvidenceReviewStats(evidenceReviewStats);
      renderEvidenceReviewQueue(evidenceReviewQueue);
      renderOutcomeSuggestionStats(outcomeSuggestionStats);
      renderOutcomeReviewQueue(outcomeReviewQueue);
      renderStrongRejectionStats(strongRejectionStats);
      renderStrongRejectionQueue(strongRejectionQueue);
      if (selectedEvidenceId != null) {
        await loadEvidenceDetail(selectedEvidenceId);
      }
      if (selectedOutcomePredictionId != null) {
        await loadOutcomeReviewDetail(selectedOutcomePredictionId);
      }
      if (selectedStrongRejectionId != null) {
        await loadStrongRejectionDetail(selectedStrongRejectionId);
      }
    }

    function renderError(error) {
      const message = error instanceof Error ? error.message : String(error);
      document.body.insertAdjacentHTML("beforeend", `<section><div class="error">${escapeHtml(message)}</div></section>`);
    }

    document.getElementById("theme-toggle").addEventListener("click", () => {
      isDarkMode = !isDarkMode;
      applyTheme();
    });
    applyTheme();

    async function loadDashboard() {
      const [stats, costs, timeline, topKilled, transmissions, domainGraph] = await Promise.all([
        fetchJson("/api/stats"),
        fetchJson("/api/costs"),
        fetchJson("/api/transmission-timeline"),
        fetchJson("/api/top-killed"),
        fetchJson("/api/transmissions"),
        fetchJson("/api/domain-graph"),
      ]);
      renderStats(stats);
      renderCosts(costs);
      renderDomainGraph(domainGraph);
      renderTransmissionTimeline(timeline);
      renderTopKilled(topKilled);
      renderTransmissions(transmissions);
      await loadReviewData();
    }

    loadDashboard().catch(renderError);
  </script>
</body>
</html>"""


@app.get("/domains")
def domains():
    rows = _get_domain_stats()
    rows_html = []
    for row in rows:
        domain = row.get("seed_domain") or ""
        domain_url = f"/explorations?seed_domain={quote(domain, safe='')}"
        best_target = row.get("best_jump_target_domain") or ""
        best_description = row.get("best_connection_description") or ""
        best_connection = best_description or best_target or "n/a"
        if best_target and best_description:
            best_connection = f"{best_target}: {best_description}"
        rows_html.append(
            "<tr>"
            f'<td data-sort="{escape(domain)}"><a href="{domain_url}">{escape(domain)}</a></td>'
            f'<td data-sort="{int(row.get("exploration_count", 0) or 0)}">{int(row.get("exploration_count", 0) or 0)}</td>'
            f'<td data-sort="{int(row.get("transmitted_count", 0) or 0)}">{int(row.get("transmitted_count", 0) or 0)}</td>'
            f'<td data-sort="{float(row.get("avg_total_score", -1) or -1):.6f}">{escape(_format_score(row.get("avg_total_score")))}</td>'
            f'<td data-sort="{escape(best_connection)}">{escape(best_connection)}</td>'
            "</tr>"
        )

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BlackClaw Domains</title>
  <style>
    :root {
      font-family: Menlo, Monaco, Consolas, monospace;
      color-scheme: dark;
      --bg: #1a1a2e;
      --panel-bg: #16213e;
      --text: #e0e0e0;
      --header-text: #ffffff;
      --link-color: #4fc3f7;
      --border-color: #333;
      --muted: #a9b4c2;
      --panel-alt: #1d2b4d;
    }
    [data-theme="light"] {
      color-scheme: light;
      --bg: #f5f5f5;
      --panel-bg: #ffffff;
      --text: #111;
      --header-text: #000;
      --link-color: #0366d6;
      --border-color: #d7d7d7;
      --muted: #666;
      --panel-alt: #fafafa;
    }
    body {
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
    }
    body,
    section,
    table,
    th,
    td,
    .theme-toggle {
      transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
    section {
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      padding: 16px;
    }
    h1 {
      margin: 0 0 12px;
      color: var(--header-text);
    }
    p {
      margin: 0 0 12px;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
    }
    a {
      color: var(--link-color);
    }
    .theme-toggle {
      position: fixed;
      top: 20px;
      right: 24px;
      z-index: 10;
      font: inherit;
      padding: 8px 12px;
      background: var(--panel-bg);
      color: var(--header-text);
      border: 1px solid var(--border-color);
      border-radius: 999px;
      cursor: pointer;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      text-align: left;
      padding: 8px;
      border-bottom: 1px solid var(--border-color);
      vertical-align: top;
    }
    th button {
      font: inherit;
      background: none;
      border: 0;
      padding: 0;
      cursor: pointer;
      color: var(--header-text);
    }
  </style>
</head>
<body>
  <button id="theme-toggle" class="theme-toggle" type="button">Light Mode</button>
  <h1>Domains</h1>
  <p><a href="/">Back to dashboard</a></p>
  <section>
    <p class="muted">Click a column header to sort. Click a domain to view matching explorations.</p>
    <table id="domains-table">
      <thead>
        <tr>
          <th><button type="button" data-index="0" data-type="text">Domain</button></th>
          <th><button type="button" data-index="1" data-type="number">Explorations</button></th>
          <th><button type="button" data-index="2" data-type="number">Transmitted</button></th>
          <th><button type="button" data-index="3" data-type="number">Avg Score</button></th>
          <th><button type="button" data-index="4" data-type="text">Best Connection</button></th>
        </tr>
      </thead>
      <tbody>
        __ROWS__
      </tbody>
    </table>
  </section>
  <script>
    let isDarkMode = true;
    const table = document.getElementById("domains-table");
    const tbody = table.querySelector("tbody");
    let currentIndex = 1;
    let currentDirection = "desc";

    function applyTheme() {
      document.documentElement.dataset.theme = isDarkMode ? "dark" : "light";
      document.getElementById("theme-toggle").textContent = isDarkMode ? "Light Mode" : "Dark Mode";
    }

    function compareValues(a, b, type) {
      if (type === "number") {
        return Number(a) - Number(b);
      }
      return String(a).localeCompare(String(b));
    }

    table.querySelectorAll("th button").forEach((button) => {
      button.addEventListener("click", () => {
        const index = Number(button.dataset.index);
        const type = button.dataset.type || "text";
        if (currentIndex === index) {
          currentDirection = currentDirection === "asc" ? "desc" : "asc";
        } else {
          currentIndex = index;
          currentDirection = type === "number" ? "desc" : "asc";
        }

        const rows = Array.from(tbody.querySelectorAll("tr"));
        rows.sort((leftRow, rightRow) => {
          const leftValue = leftRow.children[index].dataset.sort || leftRow.children[index].textContent.trim();
          const rightValue = rightRow.children[index].dataset.sort || rightRow.children[index].textContent.trim();
          const result = compareValues(leftValue, rightValue, type);
          return currentDirection === "asc" ? result : -result;
        });
        rows.forEach((row) => tbody.appendChild(row));
      });
    });

    document.getElementById("theme-toggle").addEventListener("click", () => {
      isDarkMode = !isDarkMode;
      applyTheme();
    });
    applyTheme();
  </script>
</body>
</html>""".replace("__ROWS__", "".join(rows_html))


@app.get("/explorations")
def explorations_page():
    seed_domain = (request.args.get("seed_domain") or "").strip() or None
    limit = _parse_positive_int(request.args.get("limit"), DEFAULT_EXPLORATION_LIMIT)
    rows = _get_recent_explorations(limit, seed_domain=seed_domain)

    rows_html = []
    for row in rows:
        exploration_id = int(row.get("id", 0) or 0)
        rows_html.append(
            "<tr>"
            f'<td><a href="/api/explorations/{exploration_id}">{exploration_id}</a></td>'
            f"<td>{escape(row.get('timestamp') or '')}</td>"
            f"<td>{escape(row.get('seed_domain') or '')}</td>"
            f"<td>{escape(row.get('jump_target_domain') or '')}</td>"
            f"<td>{escape('yes' if row.get('transmitted') else 'no')}</td>"
            f"<td>{escape(_format_score(row.get('total_score')))}</td>"
            f"<td>{escape(row.get('connection_description') or '')}</td>"
            "</tr>"
        )

    heading = "Explorations"
    if seed_domain is not None:
        heading = f"Explorations for {seed_domain}"
    summary = f"Showing {len(rows)} exploration(s)"
    if seed_domain is not None:
        summary += f" for {seed_domain}"
    summary += f" (limit {limit})."

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BlackClaw Explorations</title>
  <style>
    :root {
      font-family: Menlo, Monaco, Consolas, monospace;
      color-scheme: dark;
      --bg: #1a1a2e;
      --panel-bg: #16213e;
      --text: #e0e0e0;
      --header-text: #ffffff;
      --link-color: #4fc3f7;
      --border-color: #333;
      --muted: #a9b4c2;
    }
    [data-theme="light"] {
      color-scheme: light;
      --bg: #f5f5f5;
      --panel-bg: #ffffff;
      --text: #111;
      --header-text: #000;
      --link-color: #0366d6;
      --border-color: #d7d7d7;
      --muted: #666;
    }
    body {
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
    }
    body,
    section,
    table,
    th,
    td,
    .theme-toggle {
      transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
    section {
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      padding: 16px;
    }
    h1 {
      margin: 0 0 12px;
      color: var(--header-text);
    }
    p {
      margin: 0 0 12px;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
    }
    a {
      color: var(--link-color);
    }
    .theme-toggle {
      position: fixed;
      top: 20px;
      right: 24px;
      z-index: 10;
      font: inherit;
      padding: 8px 12px;
      background: var(--panel-bg);
      color: var(--header-text);
      border: 1px solid var(--border-color);
      border-radius: 999px;
      cursor: pointer;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      text-align: left;
      padding: 8px;
      border-bottom: 1px solid var(--border-color);
      vertical-align: top;
    }
  </style>
</head>
<body>
  <button id="theme-toggle" class="theme-toggle" type="button">Light Mode</button>
  <h1>__HEADING__</h1>
  <p><a href="/">Back to dashboard</a> | <a href="/domains">Domains</a></p>
  <section>
    <p class="muted">__SUMMARY__</p>
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Timestamp</th>
          <th>Seed</th>
          <th>Target</th>
          <th>Transmitted</th>
          <th>Total Score</th>
          <th>Connection</th>
        </tr>
      </thead>
      <tbody>
        __ROWS__
      </tbody>
    </table>
  </section>
  <script>
    let isDarkMode = true;

    function applyTheme() {
      document.documentElement.dataset.theme = isDarkMode ? "dark" : "light";
      document.getElementById("theme-toggle").textContent = isDarkMode ? "Light Mode" : "Dark Mode";
    }

    document.getElementById("theme-toggle").addEventListener("click", () => {
      isDarkMode = !isDarkMode;
      applyTheme();
    });
    applyTheme();
  </script>
</body>
</html>""".replace("__HEADING__", escape(heading)).replace(
        "__SUMMARY__", escape(summary)
    ).replace("__ROWS__", "".join(rows_html))


if __name__ == "__main__":
    app.run()
