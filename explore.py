"""
BlackClaw Exploration — Dive + Pattern Extraction
Searches a seed domain and extracts abstract patterns via LLM.
"""
import json
import re
from urllib.parse import urlparse
from tavily import TavilyClient
from config import MODEL, TAVILY_API_KEY
from llm_client import get_llm_client
from sanitize import sanitize, check_llm_output
from store import increment_tavily_calls, increment_llm_calls

_tavily = TavilyClient(api_key=TAVILY_API_KEY)
EXPLORE_MODEL = MODEL
SEED_SEARCH_QUERY_LIMIT = 3
SEED_SEARCH_MAX_RESULTS = 4
SEED_SEARCH_ADVANCED_QUERY_COUNT = 1
SEED_SEARCH_SELECTED_RESULT_LIMIT = 6
SEED_SEARCH_MECHANISM_MARKERS = (
    "accumul",
    "bottleneck",
    "compare",
    "constraint",
    "control",
    "decay",
    "feedback",
    "filter",
    "gate",
    "inhibit",
    "latency",
    "limit",
    "mechanism",
    "queue",
    "regulat",
    "route",
    "saturat",
    "switch",
    "threshold",
)
SEED_SEARCH_INTERVENTION_MARKERS = (
    "adjust",
    "audit",
    "control strateg",
    "intervention",
    "mainten",
    "manual",
    "mitigat",
    "operat",
    "protocol",
    "repair",
    "rerout",
    "schedule",
    "suppres",
    "tuning",
    "workflow",
)
SEED_SEARCH_BROAD_TEXT_MARKERS = (
    "comprehensive survey",
    "fundamental concepts",
    "general background",
    "introduction",
    "overview",
    "review",
    "survey",
    "tutorial",
)
SEED_SEARCH_SCHOLARLY_HOST_MARKERS = (
    "arxiv.org",
    "biorxiv.org",
    "cell.com",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "nature.com",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "sciencedirect.com",
    "springer.com",
)
SEED_SEARCH_REPOSITORY_HOST_MARKERS = (
    "github.com",
    "gitlab.com",
    "wikipedia.org",
)
SEED_SEARCH_OPERATOR_HOST_MARKERS = (
    "cdc.gov",
    "docs.",
    "engineering.",
    "nih.gov",
    "nasa.gov",
    "support.",
)
SEED_SEARCH_SOURCE_TYPE_SCORES = {
    "scholarly_primary": 5,
    "operator_or_technical": 4,
    "patent_or_standard": 3,
    "general_web": 2,
    "repository_or_reference": 1,
    "broad_overview": 0,
}
EXTRACT_PROMPT = """You are a pattern extraction engine. Your job is to extract up to 5 transferable, mechanism-level patterns from a domain.
Domain: {domain}

What counts as a strong pattern:
- It captures a specific relationship, mechanism, or dynamic from the domain.
- It names a recognizable causal process, not a theme or topic area.
- It can be rewritten without domain vocabulary and still preserve structure.
- It is searchable in unrelated fields using neutral terms.
- It would help a later search detect a concrete control, routing, accumulation, threshold, bottleneck, feedback, or phase-change mechanism.
- It contains a concrete driver, an operative mechanism, and a resulting system behavior.
- It names at least one measurable variable or state change and, when possible, one controllable lever or intervention point.
- It would give a later jump stage something an operator could monitor, tune, gate, route, throttle, filter, or compare.

Prefer patterns in families like:
- threshold switching
- accumulation and release
- competition for constrained channels
- spatial routing or bottlenecks
- feedback stabilization or destabilization
- periodic reset under disturbance

Few-shot examples:
GOOD (from "Ant Colony Foraging")
- pattern_name: Pheromone-weighted path reinforcement
- description: Ant traffic amplifies route choice via local pheromone deposition and decay, creating rapid convergence to efficient paths under changing constraints.
- abstract_structure: Agents repeatedly choose among options using a shared, decaying memory field; each traversal increases local preference strength, producing positive feedback tempered by evaporation.
- search_query: decaying reinforcement path selection
Why GOOD: specific mechanism, explicit dynamics (deposit + decay), and transferable control logic.

BAD (from "Ribosome Translation")
- pattern_name: Protein synthesis sequence
- description: Ribosomes read mRNA codons to build proteins.
- abstract_structure: A system reads instructions in order.
- search_query: sequential instruction processing
Why BAD: mostly restates domain facts, abstract form is too generic, and query will retrieve broad computing/education material rather than a specific mechanism.

GOOD (from "Ribosome Translation")
- pattern_name: Triplet-coded error-tolerant decoding
- description: Translation maps fixed-width codon units to amino acids with redundancy that dampens point-mutation impact on resulting proteins.
- abstract_structure: A finite alphabet is decoded in fixed-size chunks through a many-to-one lookup, where neighborhood redundancy reduces output sensitivity to single-symbol perturbations.
- search_query: fixed width redundant decoding robustness
- measurable_signal: output error rate under single-symbol perturbation
- control_lever: chunk width or code redundancy
- transfer_rationale: maps to any system where fixed-size encoded inputs are decoded under noise and robustness depends on redundancy structure
Why GOOD: concrete encoding/decoding structure, measurable robustness property, and non-domain-specific abstract form.

BAD (from "Ant Colony Foraging")
- pattern_name: Collective intelligence
- description: Ants work together and self-organize.
- abstract_structure: Many simple agents produce complex behavior.
- search_query: emergence in multi agent systems
Why BAD: universal principle with no distinctive mechanism.

Rules:
- Avoid universal catch-alls (feedback, emergence, adaptation, optimization, generic networks) unless tightly parameterized and distinctive.
- Prefer explicit process constraints, measurable relationships, and mechanism details.
- Prefer patterns in families like thresholding, gating, accumulation-and-release, queueing, routing, bottlenecking, delayed feedback, inhibition, saturation, switching, selective filtering, or coupled control.
- pattern_name should name the mechanism, not the subject area. Avoid vague names like "adaptive behavior", "balance", "interaction effects", or "coordination".
- description should say what changes, through what process, and what outcome follows.
- abstract_structure should state a driver, the operative mechanism, and the resulting system behavior.
- abstract_structure should include a controllable trigger, bottleneck, comparator, queue, threshold, filter, routing rule, or measurable state variable whenever the source material supports it.
- abstract_structure must use zero domain-specific terminology.
- search_query must be 3-6 words and should avoid terms likely to retrieve the original domain.
- search_query should include the distinctive mechanism tokens that would help retrieve an unrelated analogue, not broad textbook language.
- measurable_signal should name one metric, threshold, rate, count, error mode, load measure, or state variable that a target-domain operator could plausibly check.
- control_lever should name one concrete operator action, tuning knob, gating rule, scheduling choice, routing decision, filter, or intervention implied by the mechanism.
- transfer_rationale should say why the process shape can transfer across domains without reverting to domain-specific nouns.
- transferable.mechanism should restate the domain-neutral causal/process structure with no source implementation nouns.
- transferable.control_logic should state the domain-neutral intervention/control principle, not the source implementation.
- transferable.signal_shape should state the abstract trajectory/topology of the signal (for example, monotonic rise to threshold collapse), not the source metric name.
- grounded.source_control should preserve source-domain implementation details for downstream reporting.
- grounded.source_metric should preserve source-domain metric wording for downstream reporting.
- Reject patterns that collapse to generic statements like "systems adapt to change", "multiple forces interact", or "local averaging occurs".
- Exclude patterns that are merely "things interact", "system adapts", "resources balance", or other broad abstractions without a concrete causal operator.
- Exclude patterns that are descriptive but non-operational: historical summaries, aesthetic motifs, symbolic readings, subject-area overviews, or taxonomic restatements.
- Exclude patterns that are elegant but unusable because they lack a measurable variable, controllable lever, or concrete process operator.
- Fewer patterns is better than weak patterns; return 1-5 patterns only if each one is mechanistically specific and transferable.
- Prefer 1-3 strong patterns over 4-5 mixed patterns.
- If a candidate cannot be expressed as a distinct mechanism with a concrete operator, omit it.
- Return ONLY valid JSON, no markdown, no extra text.

Output schema:
{{
  "patterns": [
    {{
      "pattern_name": "...",
      "description": "...",
      "abstract_structure": "...",
      "search_query": "...",
      "measurable_signal": "...",
      "control_lever": "...",
      "transfer_rationale": "...",
      "transferable": {{
        "mechanism": "...",
        "control_logic": "...",
        "signal_shape": "..."
      }},
      "grounded": {{
        "source_control": "...",
        "source_metric": "..."
      }}
    }}
  ]
}}"""
JSON_RETRY_PROMPT = (
    "Your last response was invalid JSON. Return valid JSON only matching the required schema."
)
PATTERN_QUALITY_HIGH_THRESHOLD = 0.72
PATTERN_QUALITY_MEDIUM_THRESHOLD = 0.5
PATTERN_JUMP_READY_THRESHOLD = 0.64
PATTERN_MAX_RETURNED = 4
LOW_SIGNAL_PATTERN_NAMES = {
    "adaptation",
    "adaptive behavior",
    "balance",
    "collective intelligence",
    "coordination",
    "emergence",
    "feedback",
    "feedback loop",
    "interaction",
    "interaction effects",
    "local averaging",
    "optimization",
    "self-organization",
    "systems adapt to change",
}
LOW_SIGNAL_SEARCH_QUERIES = {
    "adaptive behavior systems",
    "balance in complex systems",
    "collective intelligence systems",
    "emergence in multi agent systems",
    "feedback in complex systems",
    "generic optimization under constraints",
    "interaction effects in systems",
    "local averaging in systems",
    "multiple forces interact",
}
PATTERN_MECHANISM_TERMS = {
    "accumulation",
    "amplification",
    "bottleneck",
    "calibration",
    "cascade",
    "competition",
    "constraint",
    "control",
    "decay",
    "filter",
    "gating",
    "inhibition",
    "latency",
    "load",
    "queue",
    "rate limit",
    "reinforcement",
    "routing",
    "saturation",
    "screen",
    "signal",
    "switching",
    "threshold",
    "timing",
    "triage",
}
PATTERN_MEASURABLE_TERMS = {
    "accuracy",
    "capacity",
    "collision",
    "count",
    "delay",
    "density",
    "dropout",
    "error",
    "failure rate",
    "false positive",
    "false negative",
    "frequency",
    "latency",
    "load",
    "loss",
    "metric",
    "pressure",
    "probability",
    "queue length",
    "rate",
    "response time",
    "score",
    "throughput",
    "threshold",
    "utilization",
    "variance",
    "voltage",
}
PATTERN_CONTROL_TERMS = {
    "allocate",
    "audit",
    "compare",
    "control",
    "filter",
    "gate",
    "intervention",
    "limit",
    "prioritize",
    "rerank",
    "route",
    "schedule",
    "screen",
    "select",
    "shift",
    "throttle",
    "tune",
}
PATTERN_TRANSFER_TERMS = {
    "across domains",
    "any system",
    "cross-domain",
    "independent of substrate",
    "same process shape",
    "same structure",
    "transfer",
    "transferable",
}
PATTERN_GENERIC_TERMS = {
    "adaptation",
    "balance",
    "behavior",
    "complexity",
    "coordination",
    "emergence",
    "generic",
    "general principle",
    "interaction",
    "optimization",
    "organization",
    "pattern",
    "self-organization",
}
PATTERN_AESTHETIC_TERMS = {
    "aesthetic",
    "beauty",
    "composition",
    "expressive",
    "interpretation",
    "narrative",
    "readability",
    "style",
    "symbolic",
    "typography",
}
PATTERN_DESCRIPTIVE_TERMS = {
    "classification",
    "history",
    "overview",
    "subject area",
    "taxonomy",
    "topic",
}
PATTERN_REQUIRED_FIELDS = {
    "pattern_name",
    "description",
    "abstract_structure",
    "search_query",
}
PATTERN_OPTIONAL_FIELDS = (
    "measurable_signal",
    "control_lever",
    "transfer_rationale",
)
PATTERN_TRANSFERABLE_FIELDS = (
    "mechanism",
    "control_logic",
    "signal_shape",
)
PATTERN_GROUNDED_FIELDS = (
    "source_control",
    "source_metric",
)
PATTERN_ANCHOR_STOPWORDS = {
    "across",
    "against",
    "after",
    "before",
    "because",
    "control",
    "controls",
    "domain",
    "general",
    "however",
    "mechanism",
    "mechanisms",
    "operator",
    "operators",
    "process",
    "processes",
    "research",
    "resulting",
    "selected",
    "signal",
    "signals",
    "snippet",
    "source",
    "sources",
    "system",
    "systems",
    "that",
    "this",
    "through",
    "using",
    "when",
    "where",
    "which",
    "raise",
    "with",
    "without",
    "within",
}


def _normalize_text(value: object) -> str:
    """Collapse whitespace in model-returned text fields."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _normalize_pattern_schema(pattern: dict) -> dict:
    """Preserve flat pattern fields while backfilling additive nested schema fields."""
    normalized = {
        key: _normalize_text(pattern.get(key))
        for key in PATTERN_REQUIRED_FIELDS.union(PATTERN_OPTIONAL_FIELDS)
    }
    transferable = pattern.get("transferable") if isinstance(pattern.get("transferable"), dict) else {}
    grounded = pattern.get("grounded") if isinstance(pattern.get("grounded"), dict) else {}
    mechanism = _normalize_text(transferable.get("mechanism"))
    control_logic = _normalize_text(transferable.get("control_logic"))
    signal_shape = _normalize_text(transferable.get("signal_shape"))
    backfilled_fields: list[str] = []
    if not mechanism:
        mechanism = normalized["abstract_structure"]
        backfilled_fields.append("mechanism")
    if not control_logic:
        control_logic = normalized["control_lever"]
        backfilled_fields.append("control_logic")
    if not signal_shape:
        signal_shape = normalized["measurable_signal"]
        backfilled_fields.append("signal_shape")
    normalized["transferable"] = {
        "mechanism": mechanism,
        "control_logic": control_logic,
        "signal_shape": signal_shape,
        "_backfilled_fields": backfilled_fields,
    }
    normalized["grounded"] = {
        "source_control": _normalize_text(grounded.get("source_control"))
        or normalized["control_lever"],
        "source_metric": _normalize_text(grounded.get("source_metric"))
        or normalized["measurable_signal"],
    }
    return normalized


def _match_terms(text: str, terms: set[str]) -> list[str]:
    """Return bounded phrase matches for lightweight pattern scoring."""
    matches = [
        term
        for term in terms
        if re.search(rf"\b{re.escape(term)}\b", text) is not None
    ]
    return sorted(matches, key=lambda value: (len(value), value))[:4]


def _pattern_source_tokens(seed: dict) -> set[str]:
    """Extract source-domain tokens that should not dominate pattern phrasing."""
    blocked = set(
        re.findall(
            r"[a-z0-9]+(?:-[a-z0-9]+)?",
            " ".join(
                [
                    str(seed.get("name", "") or ""),
                    str(seed.get("category", "") or ""),
                ]
            ).lower(),
        )
    )
    blocked.difference_update(
        {
            "and",
            "or",
            "of",
            "in",
            "for",
            "the",
            "science",
            "systems",
            "system",
        }
    )
    return blocked


def _transferable_source_leakage_terms(
    transferable_tokens: set[str],
    source_tokens: set[str],
) -> list[str]:
    """Return only source-specific overlap terms drawn from grounded source fields."""
    return sorted(
        token
        for token in transferable_tokens.intersection(source_tokens)
        if token not in PATTERN_ANCHOR_STOPWORDS
        and token not in PATTERN_GENERIC_TERMS
        and not _is_generic_grounded_source_token(token)
        and len(token) >= 4
    )


def _is_low_signal_pattern(pattern: dict) -> bool:
    """Reject obviously generic patterns that are unlikely to help jump/search."""
    name = _normalize_text(pattern.get("pattern_name")).lower()
    abstract = _normalize_text(pattern.get("abstract_structure")).lower()
    query = _normalize_text(pattern.get("search_query")).lower()
    if not name or not abstract or not query:
        return True
    if name in LOW_SIGNAL_PATTERN_NAMES or query in LOW_SIGNAL_SEARCH_QUERIES:
        return True
    if len(abstract.split()) < 12:
        return True
    generic_markers = (
        "local averaging occurs",
        "multiple forces interact",
        "things interact",
        "system adapts",
        "systems adapt to change",
        "resources balance",
        "many simple agents produce complex behavior",
    )
    return any(marker in abstract for marker in generic_markers)


def _profile_transferable_pattern_quality(pattern: dict, seed: dict) -> dict:
    """Lightly profile nested transferable fields without hard-rejecting them."""
    transferable = pattern.get("transferable") if isinstance(pattern.get("transferable"), dict) else {}
    grounded = pattern.get("grounded") if isinstance(pattern.get("grounded"), dict) else {}
    mechanism = _normalize_text(transferable.get("mechanism"))
    control_logic = _normalize_text(transferable.get("control_logic"))
    signal_shape = _normalize_text(transferable.get("signal_shape"))
    source_tokens = _pattern_grounded_source_tokens(
        grounded.get("source_control"),
        grounded.get("source_metric"),
    )

    field_tokens = {
        "mechanism": _pattern_anchor_tokens(mechanism),
        "control_logic": _pattern_anchor_tokens(control_logic),
        "signal_shape": _pattern_anchor_tokens(signal_shape),
    }
    concerns: list[str] = []
    short_fields = [
        field_name
        for field_name, tokens in field_tokens.items()
        if len(tokens) < 2
    ]
    if short_fields:
        concerns.append("transferable_fields_too_thin")

    transferable_text = " | ".join([mechanism, control_logic, signal_shape]).lower()
    generic_matches = _match_terms(transferable_text, PATTERN_GENERIC_TERMS)
    if generic_matches:
        concerns.append("transferable_fields_too_generic")

    transferable_tokens = set().union(*field_tokens.values()) if field_tokens else set()
    leakage_terms = _transferable_source_leakage_terms(
        transferable_tokens,
        source_tokens,
    )
    if leakage_terms:
        concerns.append("transferable_source_leakage")

    overlap_pairs: list[str] = []
    for left_name, right_name in (
        ("mechanism", "control_logic"),
        ("mechanism", "signal_shape"),
        ("control_logic", "signal_shape"),
    ):
        left_tokens = field_tokens[left_name]
        right_tokens = field_tokens[right_name]
        if len(left_tokens) < 2 or len(right_tokens) < 2:
            continue
        overlap = left_tokens.intersection(right_tokens)
        overlap_ratio = len(overlap) / max(1, min(len(left_tokens), len(right_tokens)))
        if len(overlap) >= 2 and overlap_ratio >= 0.8:
            overlap_pairs.append(f"{left_name}/{right_name}")
    if overlap_pairs:
        concerns.append("transferable_field_overlap")

    return {
        "usable": not concerns,
        "concerns": concerns[:4],
        "source_overlap_terms": leakage_terms[:4],
        "field_token_counts": {
            field_name: len(tokens)
            for field_name, tokens in field_tokens.items()
        },
    }


def _profile_pattern_quality(pattern: dict, seed: dict) -> dict:
    """Score one extracted pattern for mechanism, measurability, and jump readiness."""
    name = _normalize_text(pattern.get("pattern_name"))
    description = _normalize_text(pattern.get("description"))
    abstract = _normalize_text(pattern.get("abstract_structure"))
    query = _normalize_text(pattern.get("search_query"))
    measurable_signal = _normalize_text(pattern.get("measurable_signal"))
    control_lever = _normalize_text(pattern.get("control_lever"))
    transfer_rationale = _normalize_text(pattern.get("transfer_rationale"))
    transferable_quality = _profile_transferable_pattern_quality(pattern, seed)

    corpus = " | ".join(
        [
            name,
            description,
            abstract,
            query,
            measurable_signal,
            control_lever,
            transfer_rationale,
        ]
    ).lower()
    source_tokens = _pattern_source_tokens(seed)
    query_tokens = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", query.lower()))

    mechanism_matches = _match_terms(corpus, PATTERN_MECHANISM_TERMS)
    measurable_matches = _match_terms(corpus, PATTERN_MEASURABLE_TERMS)
    control_matches = _match_terms(corpus, PATTERN_CONTROL_TERMS)
    transfer_matches = _match_terms(corpus, PATTERN_TRANSFER_TERMS)
    generic_matches = _match_terms(corpus, PATTERN_GENERIC_TERMS)
    aesthetic_matches = _match_terms(corpus, PATTERN_AESTHETIC_TERMS)
    descriptive_matches = _match_terms(corpus, PATTERN_DESCRIPTIVE_TERMS)
    source_overlap = sorted(query_tokens.intersection(source_tokens))

    score = 0.28
    strengths: list[str] = []
    concerns: list[str] = []

    if mechanism_matches:
        score += 0.2 + min(0.06, 0.02 * max(0, len(mechanism_matches) - 1))
        strengths.append(f"mechanism-rich via {', '.join(mechanism_matches[:2])}")
    else:
        score -= 0.12
        concerns.append("no concrete mechanism tokens")

    if measurable_matches:
        score += 0.15 + min(0.05, 0.02 * max(0, len(measurable_matches) - 1))
        strengths.append(f"measurable variables via {', '.join(measurable_matches[:2])}")
    elif measurable_signal:
        score += 0.08
        strengths.append("explicit measurable signal")
    else:
        score -= 0.08
        concerns.append("missing measurable variable")

    if control_matches:
        score += 0.15 + min(0.05, 0.02 * max(0, len(control_matches) - 1))
        strengths.append(f"controllable lever via {', '.join(control_matches[:2])}")
    elif control_lever:
        score += 0.08
        strengths.append("explicit control lever")
    else:
        score -= 0.08
        concerns.append("missing controllable lever")

    if transfer_matches:
        score += 0.1
        strengths.append(f"transfer structure via {', '.join(transfer_matches[:2])}")
    elif transfer_rationale:
        score += 0.05
        strengths.append("explicit transfer rationale")

    if 3 <= len(query.split()) <= 6:
        score += 0.04
        strengths.append("tight jump query")
    else:
        score -= 0.06
        concerns.append("search query too broad or malformed")

    if len(abstract.split()) >= 16 and len(description.split()) >= 12:
        score += 0.05
    else:
        score -= 0.07
        concerns.append("pattern description too thin")

    if re.search(r"\b(via|through|using|by|under|when|as)\b", abstract):
        score += 0.04
    else:
        score -= 0.05
        concerns.append("abstract structure lacks an operative driver")

    if re.search(r"\b(resulting in|producing|causing|leading to|triggering)\b", abstract):
        score += 0.04
    else:
        score -= 0.04
        concerns.append("abstract structure lacks a clear outcome")

    if source_overlap:
        score -= min(0.12, 0.04 * len(source_overlap))
        concerns.append(f"query still anchored to source terms: {', '.join(source_overlap[:2])}")

    if generic_matches:
        score -= 0.12 + min(0.05, 0.02 * max(0, len(generic_matches) - 1))
        concerns.append(f"broad framing via {', '.join(generic_matches[:2])}")
    if descriptive_matches:
        score -= 0.1
        concerns.append(f"descriptive rather than operational via {', '.join(descriptive_matches[:2])}")
    if aesthetic_matches:
        score -= 0.14
        concerns.append(f"aesthetic or interpretive via {', '.join(aesthetic_matches[:2])}")
    if not transferable_quality.get("usable"):
        score -= 0.03
        for concern in transferable_quality.get("concerns") or []:
            concerns.append(str(concern))

    score = max(0.05, min(0.97, score))
    if score >= PATTERN_QUALITY_HIGH_THRESHOLD:
        band = "high"
    elif score >= PATTERN_QUALITY_MEDIUM_THRESHOLD:
        band = "medium"
    else:
        band = "weak"

    jump_support_score = score
    if not measurable_matches and not measurable_signal:
        jump_support_score -= 0.06
    if not control_matches and not control_lever:
        jump_support_score -= 0.06
    if source_overlap:
        jump_support_score -= 0.05
    jump_support_score = max(0.05, min(0.97, jump_support_score))
    jump_ready = jump_support_score >= PATTERN_JUMP_READY_THRESHOLD

    return {
        "score": round(score, 3),
        "band": band,
        "jump_support_score": round(jump_support_score, 3),
        "jump_ready": jump_ready,
        "strengths": strengths[:4],
        "concerns": concerns[:4],
        "transferable_quality": transferable_quality,
        "summary": (
            f"{band}-quality pattern ({score:.2f}); "
            f"jump {'ready' if jump_ready else 'weak'} ({jump_support_score:.2f})"
        ),
    }


def _pattern_anchor_tokens(*values: object) -> set[str]:
    """Extract a small set of non-trivial anchor tokens from pattern/source text."""
    tokens: set[str] = set()
    for value in values:
        text = _normalize_text(value).lower()
        if not text:
            continue
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text):
            if len(token) < 4 or token in PATTERN_ANCHOR_STOPWORDS:
                continue
            tokens.add(token)
    return tokens


def _is_generic_grounded_source_token(token: str) -> bool:
    """Filter broad mechanism/control words using existing pattern vocab."""
    candidate = str(token or "").strip().lower()
    if not candidate:
        return True
    for term_group in (
        PATTERN_MECHANISM_TERMS,
        PATTERN_MEASURABLE_TERMS,
        PATTERN_CONTROL_TERMS,
    ):
        for term in term_group:
            for term_token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", term.lower()):
                if len(term_token) < 4:
                    continue
                if candidate == term_token:
                    return True
                if len(term_token) >= 6 and candidate.startswith(term_token[:6]):
                    return True
    return False


def _pattern_grounded_source_tokens(*values: object) -> set[str]:
    """Extract source-specific leakage candidates from grounded source fields only."""
    return {
        token
        for token in _pattern_anchor_tokens(*values)
        if not _is_generic_grounded_source_token(token)
    }


def _seed_source_candidates_from_provenance(provenance: dict) -> list[dict]:
    """Return selected seed sources, falling back to coarse provenance when needed."""
    selected_sources = provenance.get("selected_seed_sources")
    if isinstance(selected_sources, list):
        cleaned_sources = [
            dict(source)
            for source in selected_sources
            if isinstance(source, dict) and str(source.get("clean") or "").strip()
        ]
        if cleaned_sources:
            return cleaned_sources

    fallback_excerpt = _normalize_text(provenance.get("seed_excerpt"))
    fallback_url = _normalize_text(provenance.get("seed_url"))
    if not fallback_excerpt and not fallback_url:
        return []
    return [
        {
            "title_text": "",
            "url": fallback_url,
            "clean": fallback_excerpt,
            "selection_reasons": ["fallback provenance"],
            "source_type": "general_web",
            "likely_primary_or_operator_source": False,
            "mechanism_signal": False,
            "intervention_signal": False,
            "query_index": 0,
            "specificity_score": 0,
        }
    ]


def _attach_pattern_source_anchor(pattern: dict, provenance: dict) -> tuple[dict, int]:
    """Attach one best-effort source anchor to a retained pattern for inspection."""
    sources = _seed_source_candidates_from_provenance(provenance)
    if not sources:
        return pattern, 0

    query_tokens = _pattern_anchor_tokens(pattern.get("search_query"))
    signal_tokens = _pattern_anchor_tokens(
        pattern.get("measurable_signal"),
        pattern.get("control_lever"),
    )
    pattern_tokens = _pattern_anchor_tokens(
        pattern.get("pattern_name"),
        pattern.get("description"),
        pattern.get("abstract_structure"),
        pattern.get("transfer_rationale"),
    )

    best_source: dict | None = None
    best_score = 0
    best_overlap: list[str] = []
    for source in sources:
        source_tokens = _pattern_anchor_tokens(
            source.get("title_text"),
            source.get("clean"),
        )
        query_overlap = sorted(query_tokens.intersection(source_tokens))
        signal_overlap = sorted(signal_tokens.intersection(source_tokens))
        general_overlap = sorted(
            pattern_tokens.intersection(source_tokens)
            - set(query_overlap)
            - set(signal_overlap)
        )
        overlap_tokens = query_overlap + signal_overlap + general_overlap
        overlap_score = (
            len(query_overlap) * 5
            + len(signal_overlap) * 4
            + min(4, len(general_overlap)) * 2
        )
        source_score = (
            overlap_score
            + (3 if source.get("likely_primary_or_operator_source") else 0)
            + (2 if source.get("mechanism_signal") else 0)
            + (1 if source.get("intervention_signal") else 0)
            - (2 if source.get("source_type") == "broad_overview" else 0)
        )
        if not overlap_tokens:
            continue
        if (
            best_source is None
            or (
                source_score,
                -int(source.get("query_index") or 0),
                int(source.get("specificity_score") or 0),
            )
            > (
                best_score,
                -int(best_source.get("query_index") or 0),
                int(best_source.get("specificity_score") or 0),
            )
        ):
            best_source = source
            best_score = source_score
            best_overlap = overlap_tokens[:3]

    if best_source is None or best_score <= 0:
        return pattern, 0

    anchored = dict(pattern)
    source_anchor = {
        "snippet": str(best_source.get("clean") or "").strip()[:500],
    }
    title_text = _normalize_text(best_source.get("title_text"))
    source_url = _normalize_text(best_source.get("url"))
    if title_text:
        source_anchor["title"] = title_text
    if source_url:
        source_anchor["url"] = source_url
    note_parts: list[str] = []
    if best_overlap:
        note_parts.append(f"matched source terms: {', '.join(best_overlap)}")
    selection_reasons = [
        _normalize_text(reason)
        for reason in list(best_source.get("selection_reasons") or [])
        if _normalize_text(reason)
    ]
    if selection_reasons:
        note_parts.append(f"source quality: {', '.join(selection_reasons[:2])}")
    if note_parts:
        source_anchor["note"] = "; ".join(note_parts)
    anchored["source_anchor"] = source_anchor
    return anchored, best_score


def _pattern_diagnostics(
    seed: dict,
    *,
    raw_count: int,
    missing_fields: int,
    low_signal_rejections: int,
    weak_quality_rejections: int,
    retained_patterns: list[dict],
    rejected_profiles: list[dict],
) -> dict:
    """Summarize how extraction quality behaved for one seed."""
    retained_profiles = [
        pattern.get("pattern_quality", {})
        for pattern in retained_patterns
        if isinstance(pattern.get("pattern_quality"), dict)
    ]
    high_count = sum(1 for item in retained_profiles if item.get("band") == "high")
    medium_count = sum(1 for item in retained_profiles if item.get("band") == "medium")
    weak_count = len(rejected_profiles)
    jump_ready_count = sum(1 for item in retained_profiles if item.get("jump_ready"))
    if retained_patterns:
        if high_count <= 0:
            outcome = "no_strong_patterns_found"
        else:
            outcome = "patterns_ready"
    elif raw_count <= 0:
        outcome = "no_patterns_returned"
    elif low_signal_rejections > 0 or weak_quality_rejections > 0:
        outcome = "only_weak_patterns_found"
    else:
        outcome = "no_strong_patterns_found"

    concerns: list[str] = []
    for profile in rejected_profiles:
        for concern in profile.get("concerns", []):
            if concern not in concerns:
                concerns.append(concern)
            if len(concerns) >= 3:
                break
        if len(concerns) >= 3:
            break

    return {
        "seed_name": str(seed.get("name", "") or "").strip(),
        "raw_pattern_count": int(raw_count),
        "retained_pattern_count": len(retained_patterns),
        "high_quality_count": high_count,
        "medium_quality_count": medium_count,
        "weak_quality_count": weak_count,
        "jump_ready_count": jump_ready_count,
        "drop_counts": {
            "missing_required_fields": int(missing_fields),
            "low_signal": int(low_signal_rejections),
            "weak_quality": int(weak_quality_rejections),
        },
        "top_rejection_reasons": concerns,
        "outcome": outcome,
        "summary": (
            f"{outcome}: kept {len(retained_patterns)}/{raw_count} patterns "
            f"(high={high_count}, medium={medium_count}, weak_rejected={weak_count})"
        ),
    }


def _store_pattern_diagnostics(seed: dict, diagnostics: dict) -> None:
    """Attach extraction diagnostics to the mutable seed object."""
    if isinstance(seed, dict):
        seed["pattern_diagnostics"] = diagnostics


def append_jump_attempt_diagnostic(seed: dict, jump_attempt: dict) -> dict | None:
    """Append one lightweight jump-attempt diagnostic to the seed diagnostics."""
    diagnostics = seed.get("pattern_diagnostics")
    if not isinstance(diagnostics, dict) or not diagnostics:
        return None
    if not isinstance(jump_attempt, dict) or not jump_attempt:
        return diagnostics

    jump_attempts = diagnostics.get("jump_attempts")
    if not isinstance(jump_attempts, list):
        jump_attempts = []
    jump_attempts = list(jump_attempts)
    jump_attempts.append(dict(jump_attempt))

    updated = dict(diagnostics)
    updated["jump_attempts"] = jump_attempts
    _store_pattern_diagnostics(seed, updated)
    return updated


def finalize_pattern_diagnostics(seed: dict, connections_found: int) -> dict | None:
    """Update extraction diagnostics with post-jump outcome context."""
    diagnostics = seed.get("pattern_diagnostics")
    if not isinstance(diagnostics, dict) or not diagnostics:
        return None

    final = dict(diagnostics)
    if isinstance(diagnostics.get("jump_attempts"), list):
        final["jump_attempts"] = list(diagnostics.get("jump_attempts") or [])
    final["connections_found"] = int(connections_found)
    if connections_found > 0:
        final["jump_outcome"] = "connection_found"
    elif int(final.get("retained_pattern_count", 0) or 0) <= 0:
        final["jump_outcome"] = str(final.get("outcome") or "no_patterns_returned")
    elif int(final.get("high_quality_count", 0) or 0) <= 0:
        final["jump_outcome"] = "patterns_too_weak_for_jump"
    elif int(final.get("jump_ready_count", 0) or 0) <= 0:
        final["jump_outcome"] = "patterns_too_weak_for_jump"
    else:
        final["jump_outcome"] = "patterns_present_but_no_connection"
    final["summary"] = (
        f"{diagnostics.get('summary', 'pattern diagnostics')}; "
        f"jump_outcome={final['jump_outcome']}"
    )
    _store_pattern_diagnostics(seed, final)
    return final


def _normalize_seed_search_host(url: str) -> str:
    """Return one normalized host for lightweight seed-source typing."""
    parsed = urlparse(str(url or "").strip())
    host = str(parsed.netloc or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _match_seed_search_markers(text: str, markers: tuple[str, ...]) -> list[str]:
    """Return up to a few ordered substring matches for search-result quality signals."""
    lowered = str(text or "").lower()
    matches: list[str] = []
    for marker in markers:
        if marker in lowered and marker not in matches:
            matches.append(marker)
    return matches[:3]


def _classify_seed_search_result(
    title_text: str,
    url: str,
    clean: str,
    query_index: int,
) -> dict:
    """Attach lightweight source and evidence-quality metadata to one seed-search result."""
    host = _normalize_seed_search_host(url)
    title_lower = str(title_text or "").lower()
    reference_text = " ".join(part for part in (title_text, host, url) if part).lower()
    content_text = " ".join(part for part in (title_text, clean) if part).lower()

    mechanism_matches = _match_seed_search_markers(
        content_text,
        SEED_SEARCH_MECHANISM_MARKERS,
    )
    intervention_matches = _match_seed_search_markers(
        content_text,
        SEED_SEARCH_INTERVENTION_MARKERS,
    )
    broad_matches = _match_seed_search_markers(
        content_text,
        SEED_SEARCH_BROAD_TEXT_MARKERS,
    )

    if (
        any(marker in host for marker in SEED_SEARCH_REPOSITORY_HOST_MARKERS)
        or "readme" in reference_text
    ):
        source_type = "repository_or_reference"
    elif "patent" in reference_text or "standard" in title_lower or "rfc" in title_lower:
        source_type = "patent_or_standard"
    elif (
        any(marker in host for marker in SEED_SEARCH_SCHOLARLY_HOST_MARKERS)
        or host.endswith(".edu")
        or host.endswith(".ac.uk")
        or "[pdf]" in title_lower
    ):
        source_type = "scholarly_primary"
    elif (
        host.endswith(".gov")
        or any(marker in host for marker in SEED_SEARCH_OPERATOR_HOST_MARKERS)
        or any(
            marker in title_lower
            for marker in ("guideline", "manual", "operations", "protocol", "workflow")
        )
    ):
        source_type = "operator_or_technical"
    else:
        source_type = "general_web"
        if broad_matches:
            source_type = "broad_overview"

    likely_primary_or_operator_source = source_type in {
        "operator_or_technical",
        "patent_or_standard",
        "scholarly_primary",
    }
    specificity_score = len(
        {
            token
            for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", content_text)
            if len(token) >= 6
            and token
            not in {
                "across",
                "analysis",
                "because",
                "general",
                "overview",
                "review",
                "source",
                "system",
                "systems",
                "through",
                "within",
            }
        }
    )

    score = (
        SEED_SEARCH_SOURCE_TYPE_SCORES[source_type] * 10
        + (8 if likely_primary_or_operator_source else 0)
        + (min(4, len(mechanism_matches)) * 3)
        + (min(3, len(intervention_matches)) * 4)
        + min(8, specificity_score)
    )
    if broad_matches:
        score -= 5
        if not mechanism_matches and not intervention_matches:
            score -= 8
    if source_type == "repository_or_reference":
        score -= 3

    selection_reasons: list[str] = []
    if likely_primary_or_operator_source:
        selection_reasons.append("primary/operator source")
    if mechanism_matches:
        selection_reasons.append("mechanism-rich")
    if intervention_matches:
        selection_reasons.append("intervention-bearing")
    if not selection_reasons and source_type == "broad_overview":
        selection_reasons.append("broad context only")
    if not selection_reasons:
        selection_reasons.append("usable source detail")

    return {
        "title_text": str(title_text or "").strip(),
        "url": str(url or "").strip(),
        "host": host,
        "clean": str(clean or "").strip(),
        "query_index": int(query_index),
        "source_type": source_type,
        "likely_primary_or_operator_source": likely_primary_or_operator_source,
        "mechanism_signal": bool(mechanism_matches),
        "intervention_signal": bool(intervention_matches),
        "mechanism_matches": mechanism_matches,
        "intervention_matches": intervention_matches,
        "broad_matches": broad_matches,
        "specificity_score": specificity_score,
        "score": score,
        "selection_reasons": selection_reasons,
    }


def _search_seed(seed: dict) -> tuple[str, dict]:
    """Run Tavily searches for the seed domain and return combined content + provenance."""
    combined = []
    provenance = {"seed_url": None, "seed_excerpt": None}
    ranked_results: dict[str, dict] = {}
    for index, query in enumerate(seed.get("seed_queries", [])[:SEED_SEARCH_QUERY_LIMIT]):
        query = " ".join(str(query or "").split()).strip()
        if not query:
            continue
        try:
            results = _tavily.search(
                query=query,
                max_results=SEED_SEARCH_MAX_RESULTS,
                include_answer=False,
                search_depth=(
                    "advanced"
                    if index < SEED_SEARCH_ADVANCED_QUERY_COUNT
                    else "basic"
                ),
            )
            increment_tavily_calls(1)
            for result in results.get("results", []):
                content = result.get("content", "")
                if content:
                    # Sanitize BEFORE collecting
                    clean = sanitize(content)
                    if clean:
                        title_text = str(result.get("title", "") or "").strip()
                        url = str(result.get("url", "") or "").strip()
                        ranked_result = _classify_seed_search_result(
                            title_text,
                            url,
                            clean,
                            query_index=index,
                        )
                        dedupe_key = (url or title_text or clean).lower()
                        existing = ranked_results.get(dedupe_key)
                        if existing is None or (
                            ranked_result["score"],
                            -ranked_result["query_index"],
                            ranked_result["specificity_score"],
                        ) > (
                            existing["score"],
                            -existing["query_index"],
                            existing["specificity_score"],
                        ):
                            ranked_results[dedupe_key] = ranked_result
        except Exception as e:
            print(f"  [!] Tavily search failed for '{query}': {e}")
            continue

    selected_results = sorted(
        ranked_results.values(),
        key=lambda item: (
            -int(item.get("score") or 0),
            int(item.get("query_index") or 0),
            -int(item.get("specificity_score") or 0),
            str(item.get("title_text") or ""),
        ),
    )
    strong_results_present = any(
        result.get("likely_primary_or_operator_source")
        or result.get("mechanism_signal")
        or result.get("intervention_signal")
        for result in selected_results
    )
    if strong_results_present:
        selected_results = [
            result
            for result in selected_results
            if not (
                result.get("source_type") in {"broad_overview", "repository_or_reference"}
                and not result.get("likely_primary_or_operator_source")
                and not result.get("intervention_signal")
            )
        ]

    for result in selected_results[:SEED_SEARCH_SELECTED_RESULT_LIMIT]:
        if provenance["seed_excerpt"] is None:
            provenance["seed_excerpt"] = str(result.get("clean") or "")[:500]
        if provenance["seed_url"] is None and str(result.get("url") or "").strip():
            provenance["seed_url"] = str(result.get("url") or "").strip()
        source_type_label = str(result.get("source_type") or "general_web").replace("_", " ")
        selected_because = ", ".join(result.get("selection_reasons") or []) or "usable source detail"
        combined.append(f"Source: {result.get('title_text') or 'Unknown'}")
        combined.append(
            f"Host: {result.get('host') or 'unknown'} | "
            f"Source type: {source_type_label} | "
            f"Selected because: {selected_because}"
        )
        combined.append(f"Snippet: {result.get('clean') or ''}")
        combined.append("")
    if selected_results:
        provenance["selected_seed_sources"] = [
            dict(result) for result in selected_results[:SEED_SEARCH_SELECTED_RESULT_LIMIT]
        ]
    return "\n".join(combined), provenance
def _extract_json_substring(text: str) -> str | None:
    """
    Try to isolate valid JSON from model output.
    1) Parse cleaned full text (after fence stripping)
    2) Parse substring from first '{' to last '}'.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    # Attempt full payload first.
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass
    # Fall back to first-object extraction.
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = cleaned[first:last + 1].strip()
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        return None
def _generate_json_with_retry(full_prompt: str, max_output_tokens: int) -> str | None:
    """Generate JSON with up to two correction retries if parsing fails."""
    raw_responses = []
    prompt = full_prompt
    for attempt in range(3):
        response = get_llm_client().generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json",
            },
        )
        raw_output = getattr(response, "text", "") or ""
        increment_llm_calls(1)
        raw_responses.append(raw_output)
        checked = check_llm_output(raw_output)
        if checked is None:
            raise RuntimeError("LLM output failed safety check.")
        extracted = _extract_json_substring(checked)
        if extracted is None:
            if attempt < 2:
                prompt = f"{JSON_RETRY_PROMPT}\n\n{full_prompt}"
                continue
            break
        try:
            json.loads(extracted)
            return extracted
        except json.JSONDecodeError:
            if attempt < 2:
                prompt = f"{JSON_RETRY_PROMPT}\n\n{full_prompt}"
                continue
    raise RuntimeError(
        "Failed to parse Ollama response as JSON after 3 attempts. "
        f"Raw response: {raw_responses[-1] if raw_responses else '<empty>'}"
    )
def dive(seed: dict) -> list[dict]:
    """
    Dive into a seed domain:
    1. Search the web for information
    2. Extract abstract patterns via LLM
    3. Return list of pattern dicts
    Returns empty list on failure.
    """
    # Step 1: Search
    research, provenance = _search_seed(seed)
    if not research.strip():
        _store_pattern_diagnostics(
            seed,
            {
                "seed_name": str(seed.get("name", "") or "").strip(),
                "raw_pattern_count": 0,
                "retained_pattern_count": 0,
                "high_quality_count": 0,
                "medium_quality_count": 0,
                "weak_quality_count": 0,
                "jump_ready_count": 0,
                "drop_counts": {
                    "missing_required_fields": 0,
                    "low_signal": 0,
                    "weak_quality": 0,
                },
                "top_rejection_reasons": ["no search results"],
                "outcome": "no_search_results",
                "summary": "no_search_results: upstream seed search returned no usable material",
            },
        )
        print(f"  [!] No search results for {seed['name']}")
        return []
    # Step 2: Extract patterns via LLM
    prompt = EXTRACT_PROMPT.format(domain=seed["name"])
    full_prompt = f"{prompt}\n\n--- RESEARCH MATERIAL ---\n\n{research}"
    try:
        extracted_json = _generate_json_with_retry(full_prompt, 4096)
    except Exception as e:
        _store_pattern_diagnostics(
            seed,
            {
                "seed_name": str(seed.get("name", "") or "").strip(),
                "raw_pattern_count": 0,
                "retained_pattern_count": 0,
                "high_quality_count": 0,
                "medium_quality_count": 0,
                "weak_quality_count": 0,
                "jump_ready_count": 0,
                "drop_counts": {
                    "missing_required_fields": 0,
                    "low_signal": 0,
                    "weak_quality": 0,
                },
                "top_rejection_reasons": ["llm extraction failed"],
                "outcome": "llm_extraction_failed",
                "summary": "llm_extraction_failed: pattern extraction did not return usable JSON",
            },
        )
        print(f"  [!] Failed to extract JSON from LLM response: {e}")
        return []
    try:
        data = json.loads(extracted_json)
    except json.JSONDecodeError as e:
        _store_pattern_diagnostics(
            seed,
            {
                "seed_name": str(seed.get("name", "") or "").strip(),
                "raw_pattern_count": 0,
                "retained_pattern_count": 0,
                "high_quality_count": 0,
                "medium_quality_count": 0,
                "weak_quality_count": 0,
                "jump_ready_count": 0,
                "drop_counts": {
                    "missing_required_fields": 0,
                    "low_signal": 0,
                    "weak_quality": 0,
                },
                "top_rejection_reasons": ["json parsing failed"],
                "outcome": "llm_json_parse_failed",
                "summary": "llm_json_parse_failed: extraction returned invalid JSON payload",
            },
        )
        print(f"  [!] Failed to parse LLM response as JSON: {e}")
        return []
    patterns = data.get("patterns", [])
    # Validate each pattern has required fields
    valid = []
    missing_fields = 0
    low_signal_rejections = 0
    weak_quality_rejections = 0
    rejected_profiles: list[dict] = []
    for p in patterns:
        if not PATTERN_REQUIRED_FIELDS.issubset(p.keys()):
            missing_fields += 1
            continue
        normalized = _normalize_pattern_schema(dict(p))
        if _is_low_signal_pattern(normalized):
            low_signal_rejections += 1
            rejected_profiles.append(
                {
                    "concerns": ["generic or low-signal pattern"],
                    "band": "weak",
                    "jump_ready": False,
                }
            )
            continue
        quality = _profile_pattern_quality(normalized, seed)
        normalized["pattern_quality"] = quality
        if quality.get("band") == "weak":
            weak_quality_rejections += 1
            rejected_profiles.append(quality)
            continue
        if provenance.get("seed_url"):
            normalized["seed_url"] = provenance["seed_url"]
        if provenance.get("seed_excerpt"):
            normalized["seed_excerpt"] = provenance["seed_excerpt"]
        anchored_pattern, _anchor_score = _attach_pattern_source_anchor(
            normalized,
            provenance,
        )
        valid.append(anchored_pattern)

    valid.sort(
        key=lambda pattern: (
            float(pattern.get("pattern_quality", {}).get("jump_support_score", 0.0) or 0.0),
            float(pattern.get("pattern_quality", {}).get("score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    valid = valid[:PATTERN_MAX_RETURNED]
    diagnostics = _pattern_diagnostics(
        seed,
        raw_count=len(patterns),
        missing_fields=missing_fields,
        low_signal_rejections=low_signal_rejections,
        weak_quality_rejections=weak_quality_rejections,
        retained_patterns=valid,
        rejected_profiles=rejected_profiles,
    )
    _store_pattern_diagnostics(seed, diagnostics)
    return valid
