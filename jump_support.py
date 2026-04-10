from __future__ import annotations

import re
from urllib.parse import urlparse


GENERIC_QUERY_TOKENS = {
    "approach",
    "constraint",
    "control",
    "delay",
    "effect",
    "failure",
    "function",
    "level",
    "limit",
    "mechanism",
    "mode",
    "pattern",
    "pressure",
    "process",
    "rate",
    "response",
    "result",
    "signal",
    "stability",
    "state",
    "system",
    "systems",
    "threshold",
}
WEAK_QUERY_TOKENS = {
    "adapt",
    "adaptation",
    "balance",
    "balancing",
    "change",
    "changes",
    "dynamic",
    "dynamics",
    "feedback",
    "optimization",
    "optimize",
    "optimum",
    "regulation",
    "relationship",
}
WEAK_RELATIONAL_QUERY_TOKENS = {
    "against",
    "across",
    "between",
    "during",
    "from",
    "into",
    "through",
    "toward",
    "under",
    "until",
    "via",
    "when",
    "where",
    "with",
    "within",
}
AMBIGUOUS_JUMP_QUERY_TOKENS = {
    "activity",
    "change",
    "event",
    "events",
    "flow",
    "load",
    "phase",
    "range",
    "rate",
    "signal",
    "state",
    "timing",
}
OVERLOADED_JUMP_QUERY_TOKENS = {
    "circuit",
    "channel",
    "flow",
    "signal",
    "switch",
    "switching",
    "traffic",
}
JUMP_TITLE_SIGNATURE_NOISE_TOKENS = {
    "analysis",
    "application",
    "applications",
    "case",
    "characterization",
    "design",
    "effects",
    "method",
    "model",
    "models",
    "paper",
    "review",
    "study",
}
MECHANISM_QUERY_TOKENS = {
    "attenuation",
    "bottleneck",
    "buffering",
    "cascade",
    "comparator",
    "damping",
    "gating",
    "hysteresis",
    "inhibition",
    "latching",
    "limiting",
    "pooling",
    "queueing",
    "rerouting",
    "saturation",
    "segregation",
    "suppression",
    "threshold",
    "throttling",
}
PHRASE_ANCHOR_TAIL_TOKENS = {
    "bottleneck",
    "boundary",
    "cascade",
    "control",
    "cost",
    "gating",
    "inhibition",
    "limiting",
    "pooling",
    "pressure",
    "rate",
    "rerouting",
    "routing",
    "saturation",
    "suppression",
    "throttling",
    "threshold",
}
SOLUTION_EVIDENCE_MARKERS = (
    "adjust",
    "attenuat",
    "control",
    "gating",
    "intervention",
    "isolation",
    "mitigat",
    "operator response",
    "rerout",
    "stabiliz",
    "suppress",
    "throttl",
    "workaround",
)
INTERVENTION_CONTROL_MARKERS = (
    "adjust",
    "bias",
    "control",
    "modulat",
    "setpoint",
    "tune",
    "vary",
)
INTERVENTION_RESPONSE_MARKERS = (
    "attenuat",
    "gate",
    "isolat",
    "rout",
    "stabiliz",
    "suppress",
    "switch",
    "throttl",
)
INTERVENTION_CONDITION_PHRASES = (
    "crosslink density",
    "flow rate",
    "residence time",
)
INTERVENTION_CONDITION_TOKENS = {
    "concentration",
    "configuration",
    "loading",
    "pressure",
    "pulse",
    "startup",
    "temperature",
}
QUERY_PHRASE_STOPWORDS = {
    "a",
    "after",
    "an",
    "and",
    "as",
    "at",
    "before",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "that",
    "this",
    "to",
    "under",
    "until",
    "via",
    "when",
    "where",
    "which",
    "with",
    "without",
}
JUMP_QUERY_CAUSAL_VERB_STEMS = (
    "attenuat",
    "compare",
    "cross",
    "delay",
    "gate",
    "isolat",
    "limit",
    "lock",
    "prevent",
    "redirect",
    "reduce",
    "release",
    "reopen",
    "replenish",
    "reset",
    "restore",
    "rerout",
    "route",
    "stabiliz",
    "suppress",
    "throttl",
    "trigger",
)
JUMP_QUERY_CAUSAL_OUTCOME_HINTS = {
    "access",
    "collapse",
    "competition",
    "dominance",
    "imbalance",
    "interference",
    "monopolization",
    "recovery",
    "regime",
    "response",
    "stability",
    "throughput",
}
JUMP_QUERY_CLAUSE_PREFIXES = (
    "applies to systems where ",
    "applies to any system where ",
    "applies where ",
    "transfers to systems where ",
    "transfers to any system where ",
    "transfers to ",
    "maps to any system where ",
    "maps to systems where ",
    "maps to ",
    "systems where ",
    "any system where ",
    "where ",
)
JUMP_QUERY_FILLER_TOKENS = {
    "a",
    "an",
    "actor",
    "actors",
    "already",
    "any",
    "are",
    "be",
    "because",
    "by",
    "durable",
    "long",
    "most",
    "one",
    "previously",
    "raise",
    "same",
    "term",
    "the",
    "toward",
}
JUMP_SOURCE_LEAKAGE_GENERIC_TOKENS = {
    "declaration",
    "event",
    "initial",
    "response",
    "scale-free",
    "signal",
    "state",
}
JUMP_BROAD_SOURCE_OVERLAP_TOKENS = JUMP_SOURCE_LEAKAGE_GENERIC_TOKENS.union(
    {
        "abundance",
        "aperture",
        "composition",
        "how",
        "limits",
        "lower",
        "stress",
    }
)


def _tokenize_query_terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", (text or "").lower())


def _jump_transferable_source_leakage_terms(
    transferable_tokens: set[str],
    grounded_source_tokens: set[str],
) -> list[str]:
    return sorted(
        token
        for token in transferable_tokens.intersection(grounded_source_tokens)
        if token not in GENERIC_QUERY_TOKENS
        and token not in WEAK_QUERY_TOKENS
        and token not in JUMP_QUERY_FILLER_TOKENS
        and token not in QUERY_PHRASE_STOPWORDS
        and not _is_generic_jump_grounded_source_token(token)
        and len(token) > 2
    )


def _jump_exact_domain_blocker_tokens(*values: str) -> set[str]:
    blockers: set[str] = set()
    for value in values:
        domain_tokens = _tokenize_query_terms(str(value or ""))
        if len(domain_tokens) == 1:
            blockers.add(domain_tokens[0])
    return blockers


def _is_generic_jump_grounded_source_token(token: str) -> bool:
    candidate = str(token or "").strip().lower()
    if not candidate:
        return True
    if (
        candidate in AMBIGUOUS_JUMP_QUERY_TOKENS
        or candidate in JUMP_SOURCE_LEAKAGE_GENERIC_TOKENS
    ):
        return True
    broad_terms = (
        tuple(MECHANISM_QUERY_TOKENS)
        + tuple(INTERVENTION_CONDITION_TOKENS)
        + INTERVENTION_CONTROL_MARKERS
        + INTERVENTION_RESPONSE_MARKERS
        + JUMP_QUERY_CAUSAL_VERB_STEMS
    )
    for term in broad_terms:
        term_token = str(term or "").strip().lower()
        if len(term_token) < 4:
            continue
        if candidate == term_token:
            return True
        if len(term_token) >= 6 and candidate.startswith(term_token[:6]):
            return True
    return False


def _is_strong_jump_source_specific_token(token: str) -> bool:
    candidate = str(token or "").strip().lower()
    if (
        len(candidate) < 3
        or candidate in GENERIC_QUERY_TOKENS
        or candidate in WEAK_QUERY_TOKENS
        or candidate in JUMP_QUERY_FILLER_TOKENS
        or candidate in QUERY_PHRASE_STOPWORDS
        or candidate in OVERLOADED_JUMP_QUERY_TOKENS
        or candidate in AMBIGUOUS_JUMP_QUERY_TOKENS
        or candidate in JUMP_BROAD_SOURCE_OVERLAP_TOKENS
        or _is_generic_jump_grounded_source_token(candidate)
    ):
        return False
    return "-" in candidate or any(char.isdigit() for char in candidate) or len(candidate) >= 6


def _strong_jump_source_terms(terms: list[str]) -> list[str]:
    return sorted(token for token in terms if _is_strong_jump_source_specific_token(token))


def _jump_grounded_source_tokens(grounded: dict) -> set[str]:
    return {
        token
        for token in _tokenize_query_terms(str(grounded.get("source_control", "") or ""))
        + _tokenize_query_terms(str(grounded.get("source_metric", "") or ""))
        if token not in GENERIC_QUERY_TOKENS
        and token not in WEAK_QUERY_TOKENS
        and token not in JUMP_QUERY_FILLER_TOKENS
        and token not in QUERY_PHRASE_STOPWORDS
        and not _is_generic_jump_grounded_source_token(token)
        and len(token) > 2
    }


def _jump_source_shaped_terms(candidate_text: str, pattern: dict) -> list[str]:
    candidate_tokens = {
        token
        for token in _tokenize_query_terms(candidate_text)
        if token not in GENERIC_QUERY_TOKENS
        and token not in WEAK_QUERY_TOKENS
        and token not in JUMP_QUERY_FILLER_TOKENS
        and token not in QUERY_PHRASE_STOPWORDS
        and token not in OVERLOADED_JUMP_QUERY_TOKENS
        and not _is_generic_jump_grounded_source_token(token)
        and len(token) > 2
    }
    grounded = (
        pattern.get("grounded") if isinstance(pattern.get("grounded"), dict) else {}
    )
    source_tokens = _jump_grounded_source_tokens(grounded)
    return sorted(candidate_tokens.intersection(source_tokens))


def _is_specific_jump_query_token(token: str) -> bool:
    if token in WEAK_QUERY_TOKENS:
        return False
    if token in MECHANISM_QUERY_TOKENS:
        return True
    if "-" in token:
        return True
    return len(token) >= 7


def _is_concrete_jump_query_token(token: str) -> bool:
    return _is_specific_jump_query_token(token) and token not in AMBIGUOUS_JUMP_QUERY_TOKENS


def _is_causal_jump_query_token(token: str) -> bool:
    return any(token.startswith(stem) for stem in JUMP_QUERY_CAUSAL_VERB_STEMS)


def _normalize_jump_result_host(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    host = str(parsed.netloc or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _normalize_jump_scope_text(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower())).strip()


def _jump_result_mentions_source_domain(
    title_text: str,
    clean: str,
    url: str,
    normalized_host: str,
    source_domain: str,
) -> bool:
    source_scope = _normalize_jump_scope_text(source_domain)
    if not source_scope:
        return False
    candidate_scope = _normalize_jump_scope_text(
        " ".join(
            part
            for part in (title_text, clean, url, normalized_host)
            if str(part or "").strip()
        )
    )
    if not candidate_scope:
        return False
    source_tokens = source_scope.split()
    candidate_tokens = candidate_scope.split()
    if len(source_tokens) == 1:
        return source_tokens[0] in set(candidate_tokens)
    return f" {source_scope} " in f" {candidate_scope} "


def _host_matches_jump_include_domains(
    host: str,
    include_domains: tuple[str, ...],
) -> bool:
    clean_host = str(host or "").strip().lower()
    if not clean_host:
        return False
    return any(
        clean_host == domain or clean_host.endswith(f".{domain}")
        for domain in include_domains
    )


def _build_jump_title_signature(
    title_text: str,
    blocked_tokens: set[str],
) -> tuple[str, ...]:
    priority_tokens: list[str] = []
    secondary_tokens: list[str] = []
    seen: set[str] = set()
    for token in _tokenize_query_terms(title_text):
        if (
            token in seen
            or token in blocked_tokens
            or token in GENERIC_QUERY_TOKENS
            or token in WEAK_QUERY_TOKENS
            or token in JUMP_TITLE_SIGNATURE_NOISE_TOKENS
            or len(token) <= 2
            or not _is_specific_jump_query_token(token)
        ):
            continue
        seen.add(token)
        if token in MECHANISM_QUERY_TOKENS or "-" in token or len(token) >= 9:
            priority_tokens.append(token)
        else:
            secondary_tokens.append(token)
    signature = tuple((priority_tokens + secondary_tokens)[:4])
    return signature if len(signature) >= 2 else ()


def _jump_solution_marker_count(text: str) -> int:
    lowered = str(text or "").lower()
    return sum(1 for marker in SOLUTION_EVIDENCE_MARKERS if marker in lowered)


def _match_jump_marker_stems(tokens: list[str], stems: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        if any(token.startswith(stem) for stem in stems):
            seen.add(token)
            matches.append(token)
    return matches


def _classify_jump_intervention_evidence(
    title_text: str,
    clean: str,
    *,
    anchor_overlap: int,
    preferred_phrase_match: bool,
    strong_anchor_tokens: set[str],
    solution_marker_count: int,
    specificity_score: int,
) -> dict[str, object]:
    combined_text = " ".join(
        part for part in (str(title_text or ""), str(clean or "")) if part
    ).lower()
    tokens = _tokenize_query_terms(f"{title_text} {clean}")
    token_set = set(tokens)
    matched_control = _match_jump_marker_stems(tokens, INTERVENTION_CONTROL_MARKERS)
    matched_response = _match_jump_marker_stems(tokens, INTERVENTION_RESPONSE_MARKERS)
    matched_condition_phrases = [
        phrase for phrase in INTERVENTION_CONDITION_PHRASES if phrase in combined_text
    ]
    matched_condition_tokens = [
        token for token in tokens if token in INTERVENTION_CONDITION_TOKENS
    ]
    mechanism_context_count = len(
        token_set.intersection(strong_anchor_tokens.union(MECHANISM_QUERY_TOKENS))
    )
    strong_process_context = (
        preferred_phrase_match
        or anchor_overlap >= 2
        or mechanism_context_count >= 3
        or specificity_score >= 7
    )
    has_condition = bool(matched_condition_phrases or matched_condition_tokens)
    intervention_evidence = (
        bool(matched_response)
        and strong_process_context
        and (has_condition or solution_marker_count > 0 or mechanism_context_count >= 3)
    ) or (
        bool(matched_control) and has_condition and strong_process_context
    )
    signal_parts: list[str] = []
    for part in (
        matched_response[:1]
        + matched_control[:1]
        + matched_condition_phrases[:1]
        + matched_condition_tokens[:1]
    ):
        if part and part not in signal_parts:
            signal_parts.append(part)
    return {
        "intervention_marker_count": len(
            set(
                matched_control
                + matched_response
                + matched_condition_phrases
                + matched_condition_tokens
            )
        ),
        "intervention_evidence": intervention_evidence,
        "intervention_signal": ", ".join(signal_parts[:3]),
    }


def _jump_legacy_flat_pattern(pattern: dict) -> dict:
    grounded = (
        pattern.get("grounded") if isinstance(pattern.get("grounded"), dict) else {}
    )
    return {
        "pattern_name": str(pattern.get("pattern_name", "") or "").strip(),
        "abstract_structure": str(pattern.get("abstract_structure", "") or "").strip(),
        "search_query": str(pattern.get("search_query", "") or "").strip(),
        "measurable_signal": str(pattern.get("measurable_signal", "") or "").strip(),
        "control_lever": str(pattern.get("control_lever", "") or "").strip(),
        "transfer_rationale": str(pattern.get("transfer_rationale", "") or "").strip(),
        "grounded": dict(grounded),
    }


def _normalize_jump_query_clause(text: str) -> str:
    clean_text = re.sub(r"\s+", " ", str(text or "").strip()).lower()
    for prefix in JUMP_QUERY_CLAUSE_PREFIXES:
        if clean_text.startswith(prefix):
            return clean_text[len(prefix):].strip()
    return clean_text


def _score_jump_query_clause(
    clause: str,
    blocked_tokens: set[str],
) -> tuple[int, int, int, int, int]:
    tokens = _tokenize_query_terms(clause)
    specific_count = sum(
        1
        for token in tokens
        if token not in blocked_tokens
        and token not in GENERIC_QUERY_TOKENS
        and token not in WEAK_QUERY_TOKENS
        and token not in JUMP_QUERY_FILLER_TOKENS
        and (_is_specific_jump_query_token(token) or _is_causal_jump_query_token(token))
    )
    mechanism_count = sum(token in MECHANISM_QUERY_TOKENS for token in tokens)
    causal_count = sum(_is_causal_jump_query_token(token) for token in tokens)
    outcome_count = sum(token in JUMP_QUERY_CAUSAL_OUTCOME_HINTS for token in tokens)
    return (
        1 if causal_count > 0 else 0,
        causal_count + mechanism_count + outcome_count,
        specific_count,
        outcome_count,
        len(tokens),
    )


def _jump_transferable_query_profile(
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> dict:
    transferable = (
        pattern.get("transferable") if isinstance(pattern.get("transferable"), dict) else {}
    )
    grounded = (
        pattern.get("grounded") if isinstance(pattern.get("grounded"), dict) else {}
    )
    fields = {
        "mechanism": str(transferable.get("mechanism", "") or "").strip(),
        "control_logic": str(transferable.get("control_logic", "") or "").strip(),
        "signal_shape": str(transferable.get("signal_shape", "") or "").strip(),
    }
    raw_backfilled_fields = transferable.get("_backfilled_fields")
    if isinstance(raw_backfilled_fields, list):
        backfilled_fields = [
            str(field_name).strip()
            for field_name in raw_backfilled_fields
            if str(field_name).strip() in fields
        ]
    elif bool(transferable.get("_backfilled")):
        backfilled_fields = [
            field_name for field_name, text in fields.items() if text
        ]
    else:
        backfilled_fields = []
    backfilled_field_set = set(backfilled_fields)
    grounded_source_tokens = _jump_grounded_source_tokens(grounded)
    domain_blocker_tokens = _jump_exact_domain_blocker_tokens(
        source_domain,
        source_category,
    )
    field_token_sets: dict[str, set[str]] = {}
    concerns: list[str] = []
    for field_name, text in fields.items():
        token_set = {
            token
            for token in _tokenize_query_terms(text)
            if (
                token not in GENERIC_QUERY_TOKENS
                and token not in WEAK_QUERY_TOKENS
                and token not in JUMP_QUERY_FILLER_TOKENS
                and token not in QUERY_PHRASE_STOPWORDS
                and len(token) > 2
            )
        }
        field_token_sets[field_name] = token_set
        if field_name in backfilled_field_set:
            continue
        clause_score = _score_jump_query_clause(
            text,
            grounded_source_tokens.union(domain_blocker_tokens),
        )
        if len(token_set) < 2 or (clause_score[1] < 1 and clause_score[2] < 4):
            concerns.append(f"{field_name}_too_generic")

    native_field_token_sets = [
        token_set
        for field_name, token_set in field_token_sets.items()
        if field_name not in backfilled_field_set
    ]
    source_leakage_terms = sorted(
        _jump_transferable_source_leakage_terms(
            set().union(*native_field_token_sets),
            grounded_source_tokens,
        )
        if native_field_token_sets
        else []
    )
    if source_leakage_terms:
        concerns.append("transferable_source_leakage")
    has_transferable_fields = any(fields.values())
    has_native_transferable_fields = any(
        text and field_name not in backfilled_field_set
        for field_name, text in fields.items()
    )
    source_shape_terms = (
        _jump_source_shaped_terms(
            " ".join(
                fields[field_name]
                for field_name in fields
                if field_name not in backfilled_field_set
            ),
            pattern,
        )
        if has_native_transferable_fields
        else []
    )
    if len(source_shape_terms) >= 2:
        concerns.append("transferable_source_shaped")
    strong_source_leakage_terms = _strong_jump_source_terms(source_leakage_terms)
    source_context_tokens = set(grounded_source_tokens)
    source_context_tokens.update(_tokenize_query_terms(source_domain))
    source_context_tokens.update(_tokenize_query_terms(source_category))
    strong_source_shape_terms = _strong_jump_source_terms(
        [token for token in source_shape_terms if token in source_context_tokens]
    )

    overlap_pairs: list[str] = []
    for left_name, right_name in (
        ("mechanism", "control_logic"),
        ("mechanism", "signal_shape"),
        ("control_logic", "signal_shape"),
    ):
        if left_name in backfilled_field_set or right_name in backfilled_field_set:
            continue
        left_tokens = field_token_sets.get(left_name, set())
        right_tokens = field_token_sets.get(right_name, set())
        if len(left_tokens) < 2 or len(right_tokens) < 2:
            continue
        overlap = left_tokens.intersection(right_tokens)
        overlap_ratio = len(overlap) / max(1, min(len(left_tokens), len(right_tokens)))
        if len(overlap) >= 2 and overlap_ratio >= 0.8:
            overlap_pairs.append(f"{left_name}/{right_name}")
    if overlap_pairs:
        concerns.append("transferable_field_overlap")

    blocking_concerns = {"transferable_field_overlap"}
    return {
        "usable": has_native_transferable_fields
        and not any(
            str(concern).endswith("_too_generic") or concern in blocking_concerns
            for concern in concerns
        )
        and not strong_source_leakage_terms
        and not strong_source_shape_terms,
        "backfilled": has_transferable_fields and not has_native_transferable_fields,
        "backfilled_fields": backfilled_fields,
        "has_transferable_fields": has_transferable_fields,
        "concerns": concerns[:4],
        "source_leakage_terms": source_leakage_terms[:4],
        "strong_source_leakage_terms": strong_source_leakage_terms[:4],
        "source_shape_terms": source_shape_terms[:4],
        "strong_source_shape_terms": strong_source_shape_terms[:4],
        "overlap_pairs": overlap_pairs[:3],
        "fields": fields,
    }


def _jump_query_pattern_view(
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> tuple[dict, dict]:
    legacy_pattern = _jump_legacy_flat_pattern(pattern)
    transferable_profile = _jump_transferable_query_profile(
        pattern,
        source_domain,
        source_category,
    )
    if not transferable_profile.get("usable"):
        return legacy_pattern, transferable_profile

    transferable_fields = dict(transferable_profile.get("fields") or {})
    backfilled_field_set = {
        str(field_name).strip()
        for field_name in (transferable_profile.get("backfilled_fields") or [])
        if str(field_name).strip()
    }
    preferred_pattern = dict(legacy_pattern)
    if "mechanism" not in backfilled_field_set:
        preferred_pattern["abstract_structure"] = (
            str(transferable_fields.get("mechanism") or "").strip()
            or preferred_pattern["abstract_structure"]
        )
    if "control_logic" not in backfilled_field_set:
        preferred_pattern["control_lever"] = (
            str(transferable_fields.get("control_logic") or "").strip()
            or preferred_pattern["control_lever"]
        )
    if "signal_shape" not in backfilled_field_set:
        preferred_pattern["measurable_signal"] = (
            str(transferable_fields.get("signal_shape") or "").strip()
            or preferred_pattern["measurable_signal"]
        )
    return preferred_pattern, transferable_profile


def _jump_query_support_context(
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> tuple[set[str], list[str], list[str]]:
    query_pattern, _transferable_profile = _jump_query_pattern_view(
        pattern,
        source_domain,
        source_category,
    )
    blocked_tokens = set(_tokenize_query_terms(source_domain))
    blocked_tokens.update(_tokenize_query_terms(source_category))
    preferred_anchor_phrases = _preferred_jump_query_anchor_phrases(
        query_pattern,
        blocked_tokens,
    )
    support_tokens: list[str] = []
    seen: set[str] = set()
    for text in (
        str(query_pattern.get("control_lever", "") or ""),
        str(query_pattern.get("abstract_structure", "") or ""),
        str(query_pattern.get("measurable_signal", "") or ""),
        str(query_pattern.get("pattern_name", "") or ""),
        str(query_pattern.get("transfer_rationale", "") or ""),
    ):
        for token in _tokenize_query_terms(text):
            if (
                token in seen
                or token in blocked_tokens
                or token in GENERIC_QUERY_TOKENS
                or token in WEAK_QUERY_TOKENS
                or token in OVERLOADED_JUMP_QUERY_TOKENS
                or len(token) <= 2
                or (
                    not _is_specific_jump_query_token(token)
                    and not _is_causal_jump_query_token(token)
                )
            ):
                continue
            seen.add(token)
            support_tokens.append(token)
    return blocked_tokens, preferred_anchor_phrases, support_tokens


def _extract_jump_query_phrases(text: str, blocked_tokens: set[str]) -> list[str]:
    phrases: list[str] = []
    raw_tokens = _tokenize_query_terms(text)
    for index in range(len(raw_tokens) - 1):
        first = raw_tokens[index]
        second = raw_tokens[index + 1]
        second_is_anchor_tail = (
            second in PHRASE_ANCHOR_TAIL_TOKENS
            or second in MECHANISM_QUERY_TOKENS
        )
        if (
            first in blocked_tokens
            or second in blocked_tokens
            or first in GENERIC_QUERY_TOKENS
            or (second in GENERIC_QUERY_TOKENS and not second_is_anchor_tail)
        ):
            continue
        if first in QUERY_PHRASE_STOPWORDS or second in QUERY_PHRASE_STOPWORDS:
            continue
        if len(first) <= 2 or len(second) <= 2:
            continue
        if first in WEAK_QUERY_TOKENS and second in WEAK_QUERY_TOKENS:
            continue
        if first in WEAK_QUERY_TOKENS and not second_is_anchor_tail:
            continue
        if not second_is_anchor_tail:
            continue
        if not (
            _is_specific_jump_query_token(first)
            or _is_specific_jump_query_token(second)
            or second_is_anchor_tail
        ):
            continue
        phrase = f"{first} {second}"
        if phrase not in phrases:
            phrases.append(phrase)
    return phrases


def _preferred_jump_query_anchor_phrases(
    pattern: dict,
    blocked_tokens: set[str],
) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for text in (
        str(pattern.get("pattern_name", "") or ""),
        str(pattern.get("control_lever", "") or ""),
        str(pattern.get("abstract_structure", "") or ""),
    ):
        raw_tokens = _tokenize_query_terms(text)
        for index in range(len(raw_tokens) - 1):
            first = raw_tokens[index]
            second = raw_tokens[index + 1]
            second_is_anchor_tail = (
                second in PHRASE_ANCHOR_TAIL_TOKENS
                or second in MECHANISM_QUERY_TOKENS
            )
            if (
                first in blocked_tokens
                or second in blocked_tokens
                or first in QUERY_PHRASE_STOPWORDS
                or second in QUERY_PHRASE_STOPWORDS
                or first in GENERIC_QUERY_TOKENS
                or (second in GENERIC_QUERY_TOKENS and not second_is_anchor_tail)
            ):
                continue
            if len(first) <= 2 or len(second) <= 2:
                continue
            if first in WEAK_QUERY_TOKENS and second in WEAK_QUERY_TOKENS:
                continue
            if first in WEAK_QUERY_TOKENS and not second_is_anchor_tail:
                continue
            if not (
                _is_specific_jump_query_token(first)
                or _is_specific_jump_query_token(second)
                or second_is_anchor_tail
            ):
                continue
            phrase = f"{first} {second}"
            if phrase in seen:
                continue
            seen.add(phrase)
            phrases.append(phrase)
    return phrases


def _select_best_jump_anchor_phrase(preferred_anchor_phrases: list[str]) -> str:
    best_phrase = ""
    best_rank = (-1, -1, -1, -1)
    for phrase in preferred_anchor_phrases:
        phrase_tokens = _tokenize_query_terms(phrase)
        if len(phrase_tokens) < 2:
            continue
        first, second = phrase_tokens[0], phrase_tokens[1]
        rank = (
            1 if second in PHRASE_ANCHOR_TAIL_TOKENS else 0,
            1 if second in MECHANISM_QUERY_TOKENS else 0,
            1 if "-" not in first and "-" not in second else 0,
            1 if first not in WEAK_QUERY_TOKENS else 0,
        )
        if rank > best_rank:
            best_rank = rank
            best_phrase = phrase
    return best_phrase or (preferred_anchor_phrases[0] if preferred_anchor_phrases else "")
