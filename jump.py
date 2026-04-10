"""
BlackClaw Lateral Jump
Two-stage process:
1) Detect a real structural signal in another domain.
2) Hypothesize a mechanism-level mapping from that signal.
"""
import copy
import json
import os
import re
from urllib.parse import urlparse
from tavily import TavilyClient
from config import TAVILY_API_KEY
from cycle_budget import CycleBudget, CycleBudgetExhausted
from hypothesis_validation import (
    MECHANISM_TYPE_V1_VOCAB,
    PROCESS_CONNECTORS,
    normalize_edge_analysis,
    normalize_evidence_map,
    normalize_mechanism_typing,
    summarize_edge_usefulness_alignment,
    summarize_evidence_map_provenance,
)
from llm_client import get_llm_client
from sanitize import sanitize, check_llm_output
from store import get_relevant_scars, increment_tavily_calls, increment_llm_calls
from debug_log import log_gemini_output
import jump_support as _jump_support
from jump_pre_stage1 import (
    _build_alternate_jump_search_query,
    _build_jump_search_content,
    _classify_weak_jump_result,
    _has_intervention_query_label,
    _jump_result_anchor_context,
    _should_attempt_alternate_jump_retrieval,
    PreStage1Dependencies,
    augment_pre_stage1_with_query,
    run_pre_stage1,
)
from jump_types import (
    JumpAttemptDiagnostic,
    JumpQueryBuildResult,
    JumpSearchQueryMetadataResult,
    JumpReplaySnapshot,
)

_llm_client = get_llm_client()
_tavily = TavilyClient(api_key=TAVILY_API_KEY)
MECHANISM_VOCAB_TEXT = ", ".join(MECHANISM_TYPE_V1_VOCAB)
ACADEMIC_JUMP_INCLUDE_DOMAINS = (
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
)

DETECT_PROMPT = """Stage 1: detection only.
You are deciding whether there is enough evidence of a real structural parallel to proceed.
ORIGINAL DOMAIN: {source_domain}
ABSTRACT STRUCTURE TO FIND: {abstract_structure}
SEARCH RESULTS FROM OTHER FIELDS:
{search_results}

Strict rules:
- Look for a conserved causal structure, not shared topic words.
- A real structural match usually preserves the same driver -> mechanism -> outcome shape, with similar control logic.
- Strong structural clues include similar threshold behavior, routing, bottlenecks, feedback loops, switching conditions, or gating logic.
- Treat titles, provenance labels, and snippets as evidence, but weight concrete mechanism-bearing snippets more heavily than broad topical overlap or generic titles.
- If SEARCH RESULTS are grouped into candidate clusters, reason cluster-by-cluster and prefer the strongest coherent cluster over isolated snippet overlap.
- Approve only when one candidate domain shows one concrete target-domain process, one concrete shared constraint/mechanism, and one concrete workaround or operating response in the same evidence cluster.
- Approve only when the target domain shows both the shared causal structure and concrete evidence of an already engineered workaround, mitigation, or operating response to that constraint.
- Treat a concrete engineered intervention, operating adjustment, suppression/control response, or manipulated-condition change as valid solution-bearing evidence when it clearly manages the same constraint or failure mode.
- A paper can count as solution-bearing even without the literal word `workaround` if it shows one concrete target-domain intervention that changes the same bottleneck, failure mode, or control problem.
- Reject vague analogies, keyword overlap, and broad theme matches without similar causal organization.
- Reject universal principles that connect everything (generic feedback, emergence, optimization, networks).
- If multiple candidate domains appear, prefer the one with the clearest retrieved workaround or mitigation evidence.
- Do not treat broad process description, descriptive operating context, or mechanism background alone as solution evidence unless it includes one concrete operator-relevant intervention or engineered response.
- If the search results only restate the problem, constraint, or failure mode without concrete workaround evidence, return no_connection.

Return ONLY valid JSON. No markdown.
If no real signal: {{"no_connection": true}}
If yes signal:
{{
  "no_connection": false,
  "target_domain": "specific target field",
  "signal": "1-2 sentence mechanism-level signal",
  "evidence": "specific evidence from search results",
  "solution_evidence": "specific retrieved workaround, mitigation, operating response, or engineered intervention evidence"
}}"""

HYPOTHESIZE_PROMPT = """Stage 2: hypothesis only.
Build a mechanism-first cross-domain hypothesis from an approved Stage 1 signal.
ORIGINAL DOMAIN: {source_domain}
ABSTRACT STRUCTURE: {abstract_structure}
STAGE 1 DETECTION JSON:
{stage_one_json}
SEARCH RESULTS:
{search_results}
RELEVANT PRIOR FAILURE CONSTRAINTS:
{relevant_scars}

Requirements:
- Keep target_domain aligned with Stage 1.
- Ask the core retrieval-grounded question directly: which domain operated under this same constraint and engineered a working solution, what specific workaround or mitigation did it use, and how does that workaround translate into a concrete target-domain lever grounded in SEARCH RESULTS?
- If STAGE 1 DETECTION JSON includes `solution_evidence`, treat it as a required anchor for the working solution or workaround. Reuse that retrieved workaround evidence instead of inventing a different fix.
- If RELEVANT PRIOR FAILURE CONSTRAINTS are provided, treat them as hard cautionary constraints. Do not repeat the same failure mode unless the retrieved target evidence specifically overcomes it.
- Lock onto exactly one primary target-domain causal claim before elaborating the comparison.
- Final candidate wording must read like a concise operator briefing, not an analogy, essay, or literature summary.
- Prefer short, direct sentences. Prefer explicit operators, metrics, comparators, and decisions over abstract connective filler.
- Reduce analogy-style phrasing, decorative transitions, and explanatory padding. State the claim and mechanism directly.
- The primary target-domain claim must be no broader than the retrieved target-domain evidence. Do not generalize beyond what the target snippets directly support.
- If the search results support only a narrow, local, conditional, or partial version of the claim, make that narrower version the core claim.
- Prefer the smaller honest claim over the broader impressive one. Do not reward elegant analogy shells that outrun the retrieved target evidence.
- Keep the workaround, mitigation, or engineered operating response grounded in retrieved target-domain evidence. Do not invent a fix that is not supported in SEARCH RESULTS or STAGE 1 DETECTION JSON.
- That primary claim must name one measurable target-domain operator or operator-driven outcome that can be checked in literature or experiments.
- If the target material does not directly support a concrete target-domain claim at that level, return `no_connection`.
- The first-pass Stage 2 output should already satisfy required-field checks without relying on repair. If any field would be generic, missing, or placeholder-like, rewrite it concretely now or return `no_connection`.
- `connection`, `mechanism`, `prediction`, and `test` must all stay centered on that same primary claim. If they drift to different effects or outcomes, return `no_connection`.
- `edge_analysis.problem_statement`, `edge_analysis.actionable_lever`, `edge_analysis.cheap_test`, and `edge_analysis.edge_if_right` must stay centered on that same primary claim, process, comparator, and metric. Do not let the edge layer introduce a different operator problem or a second target outcome.
- Explain one concrete shared mechanism, not a metaphor.
- Provide variable_mapping with at least 3 mapped variables.
- Order variable_mapping so the first 3 mappings are the strongest-supported critical mappings. If more mappings are included, put weaker or less direct ones after those first 3.
- If the retrieved target evidence only cleanly supports 3 critical mappings, keep variable_mapping to exactly those 3 instead of padding in weaker broad mappings.
- Provide `mechanism_type` using exactly one tag from this controlled v1 vocabulary:
  {mechanism_vocab}
- Provide `mechanism_type_confidence` as a numeric value in the 0.00-1.00 range.
- Optional `secondary_mechanism_types` must be a JSON array and every tag must also come from the same controlled vocabulary.
- Provide evidence_map with claim-level evidence:
  - evidence_map.variable_mappings must cover each critical mapping in variable_mapping (at least 3 entries).
- Each variable mapping entry must include source_variable, target_variable, claim, evidence_snippet, source_reference, and may include support_level.
- The first 3 variable_mapping entries are the critical mappings, so the first 3 evidence_map.variable_mappings entries must be the strongest-supported ones and must align to those same critical mappings.
- For each critical mapping, the claim must closely match the mapped variables, the evidence_snippet must directly support that exact claim, and the source_reference must point to the specific search result containing that snippet.
- For critical mappings, write the claim as a direct restatement of what the evidence_snippet literally supports. Do not let the claim become broader, more abstract, or more mechanistic than the snippet itself.
- For the first 3 critical mappings, keep the claim as a narrow paraphrase of the snippet and reuse concrete target-domain wording from the snippet where possible.
- For critical mappings, prefer direct support over inferential support whenever possible.
- For the first 3 critical mappings, make each mapping narrow, directly supported, aligned to the mapped variables, and stated at the same specificity as the snippet itself.
- For the first 3 critical mappings, the evidence_snippet must be specific enough to stand on its own: prefer 8+ words, at least one or two concrete overlapping terms with the claim/mapped variable, and enough local detail that it does not read like generic background context.
- Prefer exactly 3 strong critical mappings over padded weak mappings. If support is thin, keep the first 3 mappings narrow and well-supported instead of inventing broader weak critical mappings. Non-critical mappings are lower priority.
  - If a snippet supports only a weaker, local correspondence, keep the mapping claim equally weak and local.
  - Write each claim at the same level of specificity as the mapped variables. Do not make the claim broader than the mapping itself.
  - Each variable mapping snippet must directly bear on the mapped target-domain variable, not just the broader target-domain story or a nearby downstream effect.
  - Do not use vague evidence_snippet text that only supports the broader domain, the general story, or the overall mechanism.
  - Do not cite a broad mechanism sentence as support for a narrow variable-level mapping.
  - Do not pad the first 3 mappings with abstract correspondences or nearby-but-not-exact analogs just to reach 3.
  - For the first 3 critical mappings, choose snippets that mention the mapped variable, threshold, process, or operator directly when possible.
  - If exact support is unavailable, weaken or omit the mapping rather than overstating what the snippet proves.
  - If fewer than 3 mappings are directly supportable at that narrow variable level, return `no_connection`.
  - If a snippet only supports the overall causal story but not the exact mapped-variable claim, use it for mechanism_assertions instead of variable_mappings.
  - If a snippet only supports mechanism-level logic without a direct variable-level claim, put it in `mechanism_assertions`, not in `variable_mappings`.
  - evidence_map.mechanism_assertions must include at least 1 entry with mechanism_claim, evidence_snippet, and source_reference.
  - mechanism_assertions must be concise mechanism-support statements tied to the same target-domain process named in `mechanism`, not broad literature-summary assertions.
  - mechanism_assertions must support the actual causal operator or control logic in the mechanism (what triggers, routes, switches, inhibits, amplifies, or accumulates), not just background context about the target domain.
  - At least one target-domain snippet or mechanism_assertion must directly support the named target-domain process or the exact metric/immediate observable consequence used in the test.
  - Treat the evidence_snippet itself as the core proof. Do not let mechanism_claim carry stronger process language than the snippet actually supports.
  - Do not use broad field summaries, adjacent background explanation, or generic literature framing as `mechanism_assertions`.
  - For the core claim, prefer one direct target-domain snippet that explicitly names the same process noun phrase or the same canonical test metric. If you only have adjacent context, background explanation, or broad domain framing, return `no_connection`.
  - If the mechanism cannot be directly supported by a target-domain snippet or a concise mechanism_assertion on that same process, return `no_connection`.
  - Prefer direct core target evidence over broad contextual target evidence. A weaker overview-style snippet should never be the main support if a narrower direct mechanism or metric snippet is available.
  - For the named target-domain process, `test.metric`, and the first 3 critical mappings, prefer scholarly, technical, primary, standards, or otherwise domain-credible target evidence when available.
  - Reject off-domain or generic background target evidence. If a result title or evidence_snippet is not clearly about the target domain, named process, or named metric, treat it as unusable and return `no_connection`.
  - Treat generic blogs, broad explainers, hobbyist pages, or weakly related overviews as weak evidence for the core target-domain process or metric. Do not use them as the main grounding for the core claim if reasonably domain-appropriate evidence is unavailable; return `no_connection` instead.
  - If a target result title or snippet is only loosely related or obviously lower-quality than needed for the named process or metric, treat it as weak evidence and do not anchor the core claim on it.
  - Keep evidence_snippet short and grounded in SEARCH RESULTS. Use a result title or URL for source_reference. Do not invent sources.
- Provide `prediction` as a structured object with these keys:
  observable, time_horizon, direction, magnitude, confidence,
  falsification_condition, utility_rationale, who_benefits.
- Provide a falsifiable test with metric + confirm + falsify.
- A compelling comparison is not sufficient. If you cannot tie the hypothesis to one measurable target-domain operator or operator-driven outcome, return `no_connection`.
- `test.metric` must name one concrete measurable metric explicitly. Use a standard reported metric name where possible, and keep it specific enough that a paper table, figure, or abstract result could report it directly.
- `test.metric` must not use generic outcome placeholders such as `performance`, `efficiency`, `quality`, `improvement`, or `stability`.
- If you cannot ground one specific `test.metric` from retrieved evidence or the strongest target-domain snippet wording, return `no_connection` instead of writing a vague metric.
- `prediction.observable`, `test.metric`, `test.confirm`, and `test.falsify` must all evaluate the same named target-domain operator or its direct measurable outcome, with the same primary comparator.
- `test.confirm` and `test.falsify` must each refer to that same named metric and its explicit comparator. Do not write vague test language like "check whether the effect happens."
- Good confirm/falsify wording names the metric directly. Examples:
  - `collision rate per hyperperiod is lower under filtered scheduling than under sequential scheduling at equal utilization`
  - `mean cascade size per initiating branch failure does not differ between high-load and low-load configurations`
- Bad confirm/falsify wording is generic. Examples:
  - `the effect happens`
  - `results improve`
- Do not pair a broad analogy with a loosely related metric. If the metric only weakly proxies the claimed mechanism, narrow the claim or return `no_connection`.
- The mechanism field must name one specific causal process centered on the single primary causal operator that actually drives the analogy, not a broad analogy or generic system description.
- The mechanism sentence must name a specific target-domain process that is directly evidenced in the retrieved target material or mechanism assertions.
- The first clause of `mechanism` must open with the named target-domain process noun phrase itself, not with a consequence sentence, threshold/result summary, or broad pattern description.
- `mechanism` must open with exactly one target-domain process noun phrase and then follow with one explicit causal chain in target-domain terms: process/operator -> control, trigger, comparator, or bottleneck variable -> resulting measurable change.
- Open `mechanism` with the exact target-domain process noun phrase used in the strongest supporting evidence snippet, or a very close paraphrase of that wording.
- Do not open `mechanism` with broad framing such as `In this domain`, `The system`, `This process`, or `A mechanism where`.
- Do not bridge into the process with wording like `operates by`, `works by`, `functions by`, or `acts by` when the process noun phrase itself is already available in the target evidence.
- Do not use generic similarity wording in `mechanism` such as `mirrors`, `is analogous to`, `resembles`, `similar to`, or `shares dynamics with`.
- Do not rename the target-domain process into a broader abstract label. If the evidence says `offset assignment`, `mode switching`, `atrial event detection`, or `token bucket refill saturation`, start from that wording.
- The first clause should read like: `[specific target-domain process] [acts on/monitors/routes/tests] [control or monitored quantity]`, then state the discrete or measurable change that process causes.
- Name the operative process itself, not an abstract pattern behind it. Do not stop at generic labels like threshold crossing, feedback loop, accumulation, switching, or competition without the target-domain process that performs that action.
- Do not upgrade generic threshold, redundancy, routing, competition, or feedback language into a stronger target-domain mechanism unless the retrieved target evidence directly supports that stronger process claim.
- The mechanism must name the exact target-domain process that `test.metric` is supposed to measure.
- `test.metric`, `test.confirm`, and `test.falsify` must directly measure that named process or its immediate observable consequence, not a distant downstream proxy.
- Prefer a process name or standard causal operator a target-domain paper might use (for example: `SERCA-mediated SR refilling`, `GABAergic lateral inhibition`, `zero-cross switching`, `frictional contact network formation`).
- Prefer a process term already present in target evidence, mechanism assertions, or standard target-domain literature wording.
- Good mechanism openings:
  - `interval-by-interval atrial event detection counts sensed atrial events against the mode-switch cutoff, triggering ventricular tracking suppression`
  - `predicate subsumption test for cache reuse compares the incoming predicate to cached predicate sets, skipping recomputation when containment holds`
- Unacceptable mechanism naming includes generic placeholders such as `a threshold mechanism`, `a gating effect`, `a competitive dynamic`, or `a self-reinforcing process`.
- Unacceptable mechanism openings also include result-first phrasing such as `when a threshold is crossed...`, `feedback causes escalation...`, or `the system transitions to...` before naming the process.
- Bad mechanism openings:
  - `threshold crossing triggers suppression`
  - `a feedback loop increases reuse`
- If you cannot name a process already grounded in target-domain evidence or literature-facing wording, return `no_connection`.
- If the named process cannot be grounded in target evidence or mechanism assertions, return `no_connection`.
- If the target search results do not directly support a concrete target-domain process, return `no_connection` instead of filling the gap with generic mechanism language.
- If you can describe only a pattern, threshold crossing, or transition but cannot name the operative target-domain process in target-domain terms, return `no_connection`.
- In `mechanism`, explicitly state:
  - the operative causal operator,
  - the control, trigger, threshold, comparator, or bottleneck variable,
  - and the resulting state transition, failure mode, or measurable outcome.
- If you cannot name one operative causal process with its control or trigger variable and resulting state transition, return `no_connection` instead of writing a broad analogy-only mechanism.
- If multiple processes are present, choose the dominant operator as the main mechanism and treat other processes as boundary conditions, assumptions, or brief secondary notes. Do not merge background processes into the primary mechanism.
- The mechanism field must use causal language: explain what drives, causes, regulates, inhibits, amplifies, couples, transfers, or converts what. Describe the primary causal operator, not just a resemblance.
- Make `mechanism` process-level and falsifiable. Avoid metaphorical summaries or generic "things interact" language.
- Do not write `mechanism` as only analogy, resemblance, or high-level summary. It must state a concrete process that could be tested against alternatives.
- Do not claim structural identity or a strong mechanism match when the systems only share broad vocabulary, loose dynamics, or superficially similar outcomes.
- Do not elevate a supporting or background dynamic into the primary mechanism. If the analogy depends on a fragile hidden assumption that is likely to fail under adversarial scrutiny, weaken the claim substantially or return `no_connection`.
- Bad mechanism fields include:
  - "both systems involve complex interactions"
  - "both optimize under constraints"
  - "both exhibit adaptation"
  - "both use feedback"
- Bad mechanism matches also include:
  - same threshold vocabulary but different underlying trigger or operator
  - same bottleneck language but different causal limiter
  - same feedback language but different control loop structure
  - same phase-transition language but equilibrium versus driven-transition mismatch
  unless the mechanism also names a specific process and control logic.
- Prefer a smaller, narrower, more defensible mechanism claim over a broad impressive claim that is likely to fail adversarially. Precision of causal correspondence matters more than scope.
- If several mechanism-to-test framings are possible, choose the single framing with the cleanest measurable target-domain operator and the clearest one-result-family test.
- Make `prediction` literature-resolvable: phrase it so a paper abstract or results section could directly support or contradict it.
- Prefer one measurable outcome and one primary comparison condition over multiple coupled outcomes or several linked claims.
- Make the observable explicit and concrete. Name the measurable variable, metric, population, intervention, comparator, or context when those details matter for checking the claim against external evidence.
- Prefer canonical literature-facing metric names already used in the target-domain search results or standard papers. Use common reported terms (for example, false-positive rate, hazard ratio, burst probability, SPL in dB, odds ratio, correlation coefficient) instead of bespoke paraphrases when an established metric exists.
- Make the comparison phrasing explicit and simple. Prefer exactly one primary comparator such as before/after, treatment/control, lower or higher than baseline, or "as X increases, Y decreases."
- State one expected directional outcome using the current schema's directional comparison words (`increase`, `decrease`, `higher`, or `lower`). Even narrower or cleaner predictions must still populate `prediction.direction` with one of those directional terms.
- Phrase the prediction so it reads like a paper abstract result sentence, figure caption, reported trend, correlation, or threshold comparison.
- Prefer predictions that one paper abstract, one figure, or one reported result trend could directly support or contradict on their own.
- Avoid predictions that require several linked observations, multiple distinct subclaims, latent-variable inference, or combined curve-shape assumptions before they count as validated.
- If a prediction could be written either as one direct reported comparison or as a compound story, choose the one direct reported comparison.
- Avoid decorative, elegant, or idiosyncratic wording when a standard measurable phrasing would be more likely to appear in an abstract or results section.
- Avoid overloaded prediction sentences that stack threshold behavior, monotonicity, saturation, timing, and mechanism in one claim unless each part is essential and jointly testable from the same result family.
- Prefer narrower predictions that can be falsified or supported by one literature result family over elegant but broad claims that only retrieve domain-adjacent evidence.
- The prediction must include a measurable observable, a time horizon, a falsification condition, and why the prediction is useful.
- Provide `edge_analysis` as a grounded operator layer tied to the exact same primary target-domain claim as `connection`, `mechanism`, `prediction`, and `test`.
- `edge_analysis.problem_statement` must name exactly one specific target-domain problem, blind spot, hidden failure mode, or missed control point.
- `edge_analysis.problem_statement` must describe exactly one hidden or underexploited operational problem, not a broad summary of the field.
- Tie `edge_analysis.problem_statement` to the same process, the same metric/comparator, and the same operator decision already used in `prediction` / `test`.
- Make `edge_analysis.problem_statement` read like a missed operator problem, not an essay.
- Reject field-summary prose, restatements of the whole domain, and generic `systems are complex` wording in `edge_analysis.problem_statement`.
- `edge_analysis.actionable_lever` must name exactly one concrete operator move, setting change, filter, routing rule, threshold adjustment, replay, audit, or workflow intervention that follows from the mechanism.
- `edge_analysis.actionable_lever` must reuse the current mechanism, metric, or operator context. Do not write advisory phrasing like `consider`, `explore`, `may help`, `investigate`, or other non-operational wording.
- Reject vague levers such as `investigate further`, `optimize process`, `improve monitoring`, or `apply insights`.
- `edge_analysis.cheap_test` must be one cheap operator-facing check on an existing workflow slice and must include setup, metric, confirm, falsify, and time_to_signal. It must be a fast realistic validation path, not a multi-month research program by default.
- `edge_analysis.cheap_test.setup` must read like one real operator move on a narrow slice of the target-domain workflow. Name one real operator move, dataset, simulation, or measurement path, reuse the same process, comparator, and metric from `mechanism`/`prediction`/`test`, and make the setup smaller, cheaper, and more decision-facing than the main test.
- `edge_analysis.cheap_test.metric` must stay aligned with `test.metric`; name the same measurable quantity or a narrow comparator on that same quantity, not a generic proxy.
- `edge_analysis.cheap_test.metric` must not drift into generic validation wording or a broad proxy metric.
- `edge_analysis.cheap_test` must not merely restate `test.data` or say to validate the hypothesis. Avoid generic wording like `run a study`, `validate the hypothesis`, `collect more data`, or `see if the effect appears`. A good cheap test sounds like replaying one queue, filtering one candidate set, toggling one threshold, auditing one failure bucket, or comparing one narrow before/after operator intervention.
- `edge_analysis.edge_if_right` must state one concrete operator advantage if the test confirms the claim. Keep it contingent and scoped to the retrieved evidence.
- `edge_analysis.edge_if_right` must name exactly one operator, one decision change unlocked by the cheap test, and one concrete advantage if confirmed, not just say the result would be useful.
- `edge_analysis.edge_if_right` must say what the operator will do differently if the cheap test confirms, not just that the result has novelty or value.
- `edge_analysis.edge_if_right` must explicitly say who acts, what they do differently, and what concrete advantage they gain if confirmed.
- `edge_analysis.edge_if_right` must stay concise and operator-facing, with no extra generic value framing before or after the operator decision.
- Do not use generic novelty or value phrasing in `edge_analysis.edge_if_right` such as `this could be useful`, `this may provide an edge`, `novel insight`, or `valuable perspective`.
- Keep `edge_analysis.actionable_lever`, `edge_analysis.cheap_test`, and `edge_analysis.edge_if_right` tied to the same operator and the same metric/comparator. Do not let the lever, cheap test, and edge consequence point to different workflow slices or different success criteria.
- `edge_analysis.primary_operator` must name the specific operator who would use the lever.
- `edge_analysis.why_missed` must explain one concrete search, framing, workflow, metric, or discipline-boundary reason the target-domain problem or lever may be undernoticed.
- `edge_analysis.expected_asymmetry` must explain why the lever is plausibly underused rather than already standard target-domain wisdom.
- `edge_analysis.deployment_scope` should name where to try it first.
- Package the edge layer like an operator handoff: one hidden problem, one concrete lever, one cheap test, and one decision change if the cheap test confirms.
- Keep `edge_analysis.problem_statement`, `edge_analysis.actionable_lever`, `edge_analysis.cheap_test`, and `edge_analysis.edge_if_right` tight and operational. Avoid analogy-heavy framing, literature-summary phrasing, and padded connective filler.
- If the retrieved target-domain snippets already state the problem and lever in normal target-domain language as standard practice, best practice, or obvious operator guidance, return `no_connection` instead of wrapping it in cross-domain language.
- Keep underexploitedness claims retrieval-scoped and honest. `rarely searched`, `cross-silo`, `hidden by default workflow`, or `screened out by standard framing` are acceptable. `nobody knows this` or `unpublished` are not.
- Good problem statements name one concrete hidden failure mode tied to the same metric or observable and to one operator decision. Examples:
  - `Dense periodic schedulers may miss collision-free non-sequential offset assignments before greedy slot commitment, inflating collision rate at high utilization.`
  - `Doorway-capacity models may overestimate marginal throughput above the saturation threshold, causing planners to keep dwell-time assumptions that fail under crowding.`
- Bad problem statements are generic or essay-like. Examples:
  - `Complex systems may hide inefficiencies.`
  - `This domain may have an interesting blind spot.`
- Good actionable levers name one concrete operator action or design choice. Examples:
  - `Add a siteswap-style validity filter before greedy slot assignment.`
  - `Switch doorway-capacity planning from linear throughput assumptions to threshold-based capacity rules above the saturation point.`
- Bad actionable levers are vague or advisory. Examples:
  - `Investigate further.`
  - `Use this perspective to think differently about the system.`
- Good cheap tests sound like real operator moves on the same metric. Examples:
  - `Replay one week of dense scheduling logs with the validity filter turned on before slot assignment and compare collision rate per hyperperiod against the existing scheduler on the same workloads.`
  - `Rerank one triage queue with the threshold gate enabled and compare false-positive rate against the current rule on the same cases.`
- Bad cheap tests are generic validation suggestions or full restatements of the main test. Examples:
  - `Run a study to validate whether the hypothesis is true.`
  - `Collect more data and see if the effect appears.`
  - `Use the main experiment described above.`
- Good test metrics name one concrete literature-facing quantity. Examples:
  - `collision rate per hyperperiod`
  - `mean cascade size per initiating branch failure`
  - `false-positive rate`
- Bad test metrics are generic placeholders. Examples:
  - `performance`
  - `overall efficiency`
  - `outcomes`
- Good edge advantages name one concrete operator gain, one decision change, and one concrete advantage. Examples:
  - `A real-time scheduling engineer can keep the validity filter in the scheduler when the replay lowers collision rate, reducing collisions before redesigning the scheduling architecture.`
  - `A transit planner can cap dwell-time assumptions above the saturation point when the audit confirms the plateau, avoiding overestimated doorway throughput in station plans.`
- Bad edge advantages are generic usefulness claims. Examples:
  - `This could be useful.`
  - `This may provide an edge.`
- Good direct core target evidence explicitly names the same process or metric used in `mechanism` or `test.metric`. Examples:
  - `Automatic mode switching compares the detected atrial rate with a programmable cutoff and switches to a non-tracking mode when the cutoff is exceeded.`
  - `Feasible schedules are constructed by assigning offsets that satisfy collision-avoidance constraints across the hyperperiod.`
- Bad direct core target evidence is only adjacent context or broad framing. Examples:
  - `The article discusses how scheduling matters in real-time systems.`
  - `The paper explains why mode switching is important in device management.`
- Good `why_missed` explanations name one concrete reason the idea may be undernoticed. Examples:
  - `Scheduling teams usually search scheduling heuristics within scheduling literature, not combinatorial juggling notation, so this filter is cross-silo.`
  - `Doorway planning workflows often assume smooth throughput scaling and may not explicitly test for a plateau regime.`
- Bad `why_missed` explanations are generic or evasive. Examples:
  - `People may miss this.`
  - `This is underexplored.`
- Good `expected_asymmetry` explanations justify why the lever is underused. Examples:
  - `The lever is plausibly underused because siteswap validity constraints are rarely framed as scheduling candidate filters in real-time systems tooling.`
  - `The edge comes from testing a threshold regime that standard dwell-time planning treats as linear.`
- Bad `expected_asymmetry` explanations either sound generic or admit the idea is already standard. Examples:
  - `This could create an edge.`
  - `This is already standard practice in the target domain.`
- Do not write generic edge language such as `this could help researchers`, `investigate further`, `monitor this`, or `this may provide an edge`.
- Do not claim `nobody knows this`, `this is unpublished`, or similar novelty claims as fact.
- If you cannot supply a specific problem, actionable lever, cheap test, and contingent edge without unsupported extrapolation, return `no_connection`.
- If `mechanism`, `test.metric`, `test.confirm`, `test.falsify`, `edge_analysis.problem_statement`, `edge_analysis.actionable_lever`, `edge_analysis.edge_if_right`, `edge_analysis.why_missed`, `edge_analysis.expected_asymmetry`, or the first 3 critical evidence snippets are only generic placeholders, rewrite them concretely or return `no_connection`.
- Provide at least 2 assumptions and explicit boundary_conditions.

Return ONLY valid JSON. No markdown.
If insufficient evidence now, return: {{"no_connection": true}}
If valid:
{{
  "no_connection": false,
  "source_domain": "{source_domain}",
  "target_domain": "target field from stage 1",
  "connection": "2-4 sentence explanation that starts with an evidence-bounded primary target-domain claim/process, then links the source-domain correspondence without broadening beyond the retrieved target evidence",
  "mechanism": "one directly evidenced operative target-domain causal process opening with exactly one target-domain process noun phrase from the strongest evidence snippet, then naming one target-domain causal chain from operator/process to control/comparator variable to resulting measurable change measured by the test",
  "mechanism_type": "one controlled vocabulary tag",
  "mechanism_type_confidence": 0.82,
  "secondary_mechanism_types": ["optional additional controlled tag"],
  "variable_mapping": {{"a_in_source": "b_in_target", "c_in_source": "d_in_target", "e_in_source": "f_in_target"}},
  "evidence_map": {{
    "variable_mappings": [
      {{
        "source_variable": "a_in_source",
        "target_variable": "b_in_target",
        "claim": "tight claim explaining this exact mapped-variable correspondence",
        "evidence_snippet": "short snippet directly supporting that exact claim",
        "source_reference": "title or URL from search results",
        "support_level": "direct"
      }}
    ],
    "mechanism_assertions": [
      {{
        "mechanism_claim": "the core causal/shared process",
        "evidence_snippet": "short supporting evidence from search results",
        "source_reference": "title or URL from search results"
      }}
    ]
  }},
  "prediction": {{
    "observable": "canonical measurable quantity or event reported in literature",
    "time_horizon": "when the observable should move in the stated context",
    "direction": "increase/decrease/higher/lower for one explicit named comparison",
    "magnitude": "expected effect size, threshold, or bounded null effect that the same result family could report directly",
    "confidence": "low/medium/high or numeric confidence",
    "falsification_condition": "what concrete result would falsify the prediction",
    "utility_rationale": "why this prediction is useful to test or act on",
    "who_benefits": "who can use this prediction"
  }},
  "test": {{"data": "specific dataset or experiment to use", "metric": "one concrete canonical reported metric name", "horizon": "same or compatible time horizon", "confirm": "what result on that metric confirms the hypothesis", "falsify": "what result on that metric falsifies it"}},
  "edge_analysis": {{
    "problem_statement": "one specific target-domain problem, blind spot, or hidden failure mode",
    "why_missed": "why standard framing or workflow may overlook it",
    "actionable_lever": "one concrete action implied by the mechanism",
    "cheap_test": {{
      "setup": "one real operator move, dataset replay, simulation, or measurement path on a narrow workflow slice",
      "metric": "the same named metric as test.metric",
      "confirm": "what result would support the lever",
      "falsify": "what result would kill the lever",
      "time_to_signal": "how quickly the test should produce evidence"
    }},
    "edge_if_right": "one operator, one decision change, and one concrete advantage if confirmed",
    "expected_asymmetry": "why this is plausibly underexploited",
    "primary_operator": "specific operator who would use it",
    "deployment_scope": "where to try it first"
  }},
  "assumptions": ["...", "..."],
  "boundary_conditions": "when this mapping should and should not hold",
  "evidence": "specific evidence from search results"
}}"""

STAGE2_MECHANISM_PROMPT = """Stage 2A: mechanism only.
Build only the target-domain mechanism core from an approved Stage 1 signal.
ORIGINAL DOMAIN: {source_domain}
ABSTRACT STRUCTURE: {abstract_structure}
STAGE 1 DETECTION JSON:
{stage_one_json}
SEARCH RESULTS:
{search_results}
RELEVANT PRIOR FAILURE CONSTRAINTS:
{relevant_scars}

Return ONLY valid JSON. No markdown.
- Keep target_domain aligned with Stage 1.
- Build only the target-domain core: connection, mechanism, mechanism typing, variable mappings, mechanism assertions, assumptions, and boundary_conditions.
- Do not fill prediction, test, or edge_analysis fields in this stage.
- Reuse Stage 1 `solution_evidence` as the workaround anchor when present.
- `evidence_map.variable_mappings` must contain only narrow, directly supported source->target mappings tied to the same target-domain process named in `mechanism`.
- Prefer exactly 3 strong direct mappings over padded weak mappings.
- Keep each mapping claim at the same specificity level on both the source and target sides and no broader than its supporting snippet.
- If a snippet supports the broader mechanism or background process but not a narrow variable-level correspondence, move that support to `evidence_map.mechanism_assertions`, not `evidence_map.variable_mappings`.
- If fewer than 3 direct variable mappings are supportable, return `{{"no_connection": true}}`.
- If the mechanism core cannot be grounded concretely, return `{{"no_connection": true}}`.

If valid:
{{
  "no_connection": false,
  "target_domain": "target field from stage 1",
  "connection": "2-4 sentence evidence-bounded target-domain claim",
  "mechanism": "one process-first causal chain",
  "mechanism_type": "one controlled vocabulary tag from {mechanism_vocab}",
  "mechanism_type_confidence": 0.82,
  "secondary_mechanism_types": ["optional additional controlled tag"],
  "variable_mapping": {{"a_in_source": "b_in_target", "c_in_source": "d_in_target", "e_in_source": "f_in_target"}},
  "evidence_map": {{
    "variable_mappings": [
      {{
        "source_variable": "a_in_source",
        "target_variable": "b_in_target",
        "claim": "tight mapped-variable claim",
        "evidence_snippet": "short direct evidence",
        "source_reference": "title or URL"
      }}
    ],
    "mechanism_assertions": [
      {{
        "mechanism_claim": "concise same-process mechanism support",
        "evidence_snippet": "short direct evidence",
        "source_reference": "title or URL"
      }}
    ]
  }},
  "assumptions": ["...", "..."],
  "boundary_conditions": "when this mapping should and should not hold"
}}"""

STAGE2_PREDICT_PROMPT = """Stage 2B: predict only.
Build only the prediction bundle for the current Stage 2 claim.
ORIGINAL DOMAIN: {source_domain}
ABSTRACT STRUCTURE: {abstract_structure}
STAGE 1 DETECTION JSON:
{stage_one_json}
CURRENT STAGE 2 JSON:
{current_stage_json}
SEARCH RESULTS:
{search_results}

Return ONLY valid JSON. No markdown.
- Consume the current Stage 2 core as fixed context.
- Fill only the `prediction` bundle in this stage.
- Do not rewrite connection, mechanism, test, or edge_analysis fields here.
- If the prediction cannot be grounded on the current claim, return `{{"no_connection": true}}`.

If valid:
{{
  "no_connection": false,
  "prediction": {{
    "observable": "canonical measurable quantity or event reported in literature",
    "time_horizon": "when the observable should move in the stated context",
    "direction": "increase/decrease/higher/lower",
    "magnitude": "expected effect size, threshold, or bounded null effect",
    "confidence": "low/medium/high or numeric confidence",
    "falsification_condition": "what concrete result would falsify the prediction",
    "utility_rationale": "why this prediction is useful to test or act on",
    "who_benefits": "who can use this prediction"
  }}
}}"""

STAGE2_TEST_PROMPT = """Stage 2C: test only.
Build only the falsifiable test bundle for the current Stage 2 claim.
ORIGINAL DOMAIN: {source_domain}
ABSTRACT STRUCTURE: {abstract_structure}
STAGE 1 DETECTION JSON:
{stage_one_json}
CURRENT STAGE 2 JSON:
{current_stage_json}
SEARCH RESULTS:
{search_results}

Return ONLY valid JSON. No markdown.
- Consume the current mechanism and prediction as fixed context.
- Fill only `test` plus `edge_analysis.cheap_test` in this stage.
- Keep `test.*` and `edge_analysis.cheap_test.*` tied to the same metric, comparator, and operator-facing workflow slice.
- `test.metric` must use a canonical literature-facing metric name already used in the target-domain search results or standard papers, not a bespoke paraphrase or generic placeholder.
- `test.confirm` and `test.falsify` must stay on one explicit named metric and one explicit comparator/result family. Do not drift into vague language like `check whether the effect happens`, `results improve`, or a different success criterion.
- `edge_analysis.cheap_test` must be one real cheap operator-facing workflow slice, not a generic validation program.
- `edge_analysis.cheap_test.setup` must name one concrete operator move, dataset replay, simulation, audit, filter, threshold toggle, or measurement path on a narrow existing workflow slice.
- `edge_analysis.cheap_test` must stay smaller, cheaper, and more decision-facing than the main test, while reusing the same process, metric, comparator, and operator context.
- Reject generic cheap-test wording such as `run a study`, `validate the hypothesis`, `collect more data`, or `see if the effect appears`.
- `edge_analysis.cheap_test.metric` must stay tightly aligned to `test.metric`: use the same named measurable quantity or a narrow comparator on that same quantity, not a generic proxy.
- Do not fill other edge_analysis fields in this stage.
- If the test bundle cannot be grounded on the current claim, return `{{"no_connection": true}}`.

If valid:
{{
  "no_connection": false,
  "test": {{
    "data": "specific dataset or experiment to use",
    "metric": "one concrete canonical reported metric name",
    "horizon": "same or compatible time horizon",
    "confirm": "what result on that metric confirms the hypothesis",
    "falsify": "what result on that metric falsifies it"
  }},
  "edge_analysis": {{
    "cheap_test": {{
      "setup": "one real operator move on a narrow workflow slice",
      "metric": "the same named metric as test.metric",
      "confirm": "what result would support the lever",
      "falsify": "what result would kill the lever",
      "time_to_signal": "how quickly the test should produce evidence"
    }}
  }}
}}"""

STAGE2_EDGE_PROMPT = """Stage 2D: edge only.
Build only the edge-analysis layer for the current Stage 2 claim.
ORIGINAL DOMAIN: {source_domain}
ABSTRACT STRUCTURE: {abstract_structure}
STAGE 1 DETECTION JSON:
{stage_one_json}
CURRENT STAGE 2 JSON:
{current_stage_json}
SEARCH RESULTS:
{search_results}

Return ONLY valid JSON. No markdown.
- Consume the current mechanism, prediction, test, and cheap_test bundle as fixed context.
- Fill only edge-layer fields: problem_statement, why_missed, actionable_lever, edge_if_right, expected_asymmetry, primary_operator, and deployment_scope.
- `edge_analysis.problem_statement` must name exactly one hidden operational problem, blind spot, missed control point, or failure mode, not field-summary prose.
- Tie `edge_analysis.problem_statement` to the same process, the same metric/comparator, and the same operator decision already used in the current test bundle.
- `edge_analysis.actionable_lever` must name exactly one concrete operator move, setting change, filter, routing rule, threshold adjustment, replay, audit, or workflow intervention. Do not use advisory filler like `consider`, `explore`, `may help`, or `investigate`.
- `edge_analysis.edge_if_right` must name exactly one operator, what decision or workflow they change if confirmed, and what concrete advantage they gain.
- `edge_analysis.expected_asymmetry` must explain why the lever is plausibly underused, hidden by workflow, or screened out by standard framing, not merely say it creates an edge or has value.
- `edge_analysis.deployment_scope` should say where to try the lever first.
- Keep the whole edge layer as one operator handoff on the current claim, not a literature summary or second target outcome.
- Do not rewrite connection, mechanism, prediction, test, variable_mapping, or evidence_map fields here.
- Do not add or rewrite `edge_analysis.cheap_test` in this stage.
- If the edge layer cannot be grounded on the current claim, return `{{"no_connection": true}}`.

If valid:
{{
  "no_connection": false,
  "edge_analysis": {{
    "problem_statement": "one specific target-domain problem, blind spot, or hidden failure mode",
    "why_missed": "why standard framing or workflow may overlook it",
    "actionable_lever": "one concrete action implied by the mechanism",
    "edge_if_right": "one operator, one decision change, and one concrete advantage if confirmed",
    "expected_asymmetry": "why this is plausibly underexploited",
    "primary_operator": "specific operator who would use it",
    "deployment_scope": "where to try it first"
  }}
}}"""

STAGE2_SUBSTAGE_SEQUENCE = ("mechanism", "predict", "test", "edge")

STAGE2_SUBSTAGE_STAGE_NAMES = {
    "mechanism": "stage2_mechanism",
    "predict": "stage2_predict",
    "test": "stage2_test",
    "edge": "stage2_edge",
}

STAGE2_SUBSTAGE_FIELD_OWNERSHIP = {
    "mechanism": [
        "target_domain",
        "connection",
        "mechanism",
        "mechanism_type",
        "mechanism_type_confidence",
        "secondary_mechanism_types",
        "variable_mapping",
        "evidence_map.variable_mappings",
        "evidence_map.mechanism_assertions",
        "assumptions",
        "boundary_conditions",
    ],
    "predict": [
        "prediction",
    ],
    "test": [
        "test",
        "edge_analysis.cheap_test",
    ],
    "edge": [
        "edge_analysis.problem_statement",
        "edge_analysis.why_missed",
        "edge_analysis.actionable_lever",
        "edge_analysis.edge_if_right",
        "edge_analysis.expected_asymmetry",
        "edge_analysis.primary_operator",
        "edge_analysis.deployment_scope",
    ],
}

STAGE2_SUBSTAGE_PROMPTS = {
    "mechanism": STAGE2_MECHANISM_PROMPT,
    "predict": STAGE2_PREDICT_PROMPT,
    "test": STAGE2_TEST_PROMPT,
    "edge": STAGE2_EDGE_PROMPT,
}

JSON_RETRY_PROMPT = (
    "Your previous response was not valid JSON. Please respond with ONLY valid JSON, "
    "no markdown, no explanation, no trailing commas, no comments. Here is what I need:"
)

STAGE_ONE_SOLUTION_EVIDENCE_PLACEHOLDERS = {
    "n a",
    "na",
    "n/a",
    "none",
    "null",
    "not applicable",
    "not available",
    "no",
    "no evidence",
    "no solution evidence",
    "no workaround evidence",
    "no mitigation evidence",
    "no concrete workaround evidence",
    "unknown",
    "unspecified",
}

STAGE_ONE_SOLUTION_EVIDENCE_GENERIC_TOKENS = {
    "candidate",
    "cluster",
    "domain",
    "evidence",
    "intervention",
    "mechanism",
    "mitigation",
    "operator",
    "response",
    "result",
    "results",
    "retrieved",
    "search",
    "signal",
    "snippet",
    "solution",
    "supporting",
    "title",
    "via",
    "workaround",
}

JUMP_QUERY_PROMPT = """Write one compact technical web search query for cross-domain jump retrieval.

Return JSON only:
{{"query": "short technical search query"}}

SOURCE DOMAIN TO AVOID: {source_domain}
SOURCE CATEGORY TO AVOID: {source_category}
PATTERN NAME: {pattern_name}
ABSTRACT STRUCTURE: {abstract_structure}
MEASURABLE SIGNAL: {measurable_signal}
CONTROL LEVER: {control_lever}
TRANSFER RATIONALE: {transfer_rationale}
HEURISTIC ANCHOR QUERY: {heuristic_query}

Rules:
- Return exactly one short query string in the `query` field.
- The query must read like a real technical search, not a sentence, explanation, or list.
- Keep it compact: 4 to 10 words.
- Preserve at least one exact 2-word mechanism/process/control phrase from `PATTERN NAME`, `CONTROL LEVER`, or `ABSTRACT STRUCTURE` when available.
- Avoid the source domain and source category names.
- Avoid vague generic wording and token soup.
- Reject formal logic/query-language phrasing like `AND`, `OR`, `verification`, or `alignment` when it is not part of a real mechanism phrase anchor.
- Do not include quotes, bullets, URLs, or extra punctuation.
"""

MISSING_FIELDS_REPAIR_PROMPT = (
    "Your output is missing fields: {missing_fields}. Return ONLY corrected JSON with those fields filled. "
    "Do not change the source_domain or target_domain. Do not add a depth field."
)

PHASE6_SALVAGE_PROMPT = """Targeted salvage rewrite for a high-value near-miss.

Repair only the listed fields. Keep `source_domain`, `target_domain`, `connection`, `prediction`, `test`, `variable_mapping`, `evidence_map`, `boundary_conditions`, `assumptions`, and mechanism typing stable unless one of the listed fields must change to satisfy the repair.

Phase 6 rules:
- This candidate already scored well enough to deserve one narrow rescue pass. Do not broaden the claim, add new unsupported mechanisms, or relax evidence discipline.
- Prefer the smallest valid rewrite that clears the listed blockers. A narrow honest repair is better than a broad rewrite that drifts or fails validation.
- Rewrite toward concise operator-briefing prose: short direct sentences, explicit operator/metric/decision language, minimal connective filler.
- Remove analogy-heavy phrasing, literature-summary framing, and decorative explanation. Keep the claim concrete and operational.
- If `mechanism` is listed, rewrite it to open with one exact target-domain process noun phrase pulled from the strongest direct target-domain evidence.
- If `mechanism` is listed, do not open with broad bridges like `operates by`, `works by`, `functions by`, or analogy-heavy framing when a concrete process noun phrase is available.
- If only `mechanism` is listed, keep the edge layer and test language unchanged. Fix the anchor, not the whole story.
- If only `mechanism` is listed, you may return a JSON object containing only the repaired `mechanism` field.
- If any edge-analysis fields are listed, rewrite `edge_analysis.problem_statement`, `edge_analysis.actionable_lever`, `edge_analysis.cheap_test`, and `edge_analysis.edge_if_right` together so they stay on the same claim, process, comparator, and metric already grounded by `mechanism` / `prediction` / `test`.
- When rewriting the edge layer, reuse the same observable, metric, comparator, and operator-decision language already present in `prediction` / `test`. Reduce drift by reusing those exact anchor phrases instead of loose paraphrases.
- When rewriting the edge layer, package `edge_analysis.problem_statement` as one hidden operational problem, `edge_analysis.actionable_lever` as one concrete operator move or design choice, `edge_analysis.cheap_test` as one cheap operator check, and `edge_analysis.edge_if_right` as one operator decision consequence if confirmed.
- Preserve the current claim, process, metric, comparator, and operator while sharpening the wording. Do not broaden the field summary, invent a new lever, or introduce a different benefit axis.
- `edge_analysis.cheap_test` must describe one real operator move on a narrow slice of workflow, not a generic validation suggestion and not a restatement of the full test.
- If you cannot repair the listed fields without inventing unsupported detail, return `{"no_connection": true}`.

Return JSON only.
"""

GENERIC_MECHANISM_FILLERS = (
    "threshold mechanism",
    "gating effect",
    "competitive dynamic",
    "self-reinforcing process",
    "feedback loop",
    "feedback process",
    "things interact",
    "complex interactions",
    "optimize under constraints",
    "both systems involve",
    "both exhibit",
    "both use feedback",
    "both optimize",
)

RESULT_FIRST_MECHANISM_OPENERS = (
    "in ",
    "when ",
    "as ",
    "if ",
    "once ",
    "after ",
    "because ",
    "the system ",
    "both systems ",
    "this system ",
    "these systems ",
)

MECHANISM_BRIDGE_OPENERS = (
    "operates by",
    "works by",
    "functions by",
    "acts by",
    "does so by",
)

PROCESS_PHRASE_VERBS = (
    "allows",
    "allow",
    "causes",
    "cause",
    "compares",
    "compare",
    "confines",
    "confine",
    "constrains",
    "constrain",
    "controls",
    "control",
    "converts",
    "convert",
    "couples",
    "couple",
    "counts",
    "count",
    "detects",
    "detect",
    "determines",
    "determine",
    "dictates",
    "dictate",
    "drives",
    "drive",
    "enables",
    "enable",
    "generates",
    "generate",
    "gates",
    "gate",
    "governs",
    "govern",
    "induces",
    "induce",
    "is",
    "are",
    "limits",
    "limit",
    "mediates",
    "mediate",
    "modulates",
    "modulate",
    "mediates",
    "mediate",
    "monitors",
    "monitor",
    "occurs",
    "occur",
    "prevents",
    "prevent",
    "produces",
    "produce",
    "regulates",
    "regulate",
    "requires",
    "require",
    "routes",
    "route",
    "shifts",
    "shift",
    "suppresses",
    "suppress",
    "transfers",
    "transfer",
    "tests",
    "test",
    "triggers",
    "trigger",
)

GENERIC_TEST_METRIC_FILLERS = (
    "performance",
    "overall performance",
    "efficiency",
    "effect",
    "effects",
    "outcome",
    "outcomes",
    "result",
    "results",
    "improvement",
    "behavior",
)

GENERIC_PROBLEM_FILLERS = (
    "system may hide inefficiencies",
    "complex systems may hide inefficiencies",
    "performance may degrade",
    "this domain may hide",
    "interesting blind spot",
    "hidden opportunity",
    "operators may be missing something",
    "there may be a problem",
)

GENERIC_ACTION_FILLERS = (
    "investigate further",
    "study this",
    "study further",
    "monitor this",
    "monitor it",
    "explore this",
    "consider this",
    "use this perspective",
    "apply this idea",
    "research this",
    "look into this",
)

GENERIC_EDGE_ADVANTAGE_FILLERS = (
    "could be useful",
    "may be useful",
    "may provide an edge",
    "could provide an edge",
    "improves performance",
    "improve performance",
    "optimize performance",
    "offers an advantage",
    "help researchers",
    "useful to explore",
)

GENERIC_UNDEREXPLOITED_FILLERS = (
    "people may miss this",
    "people might miss this",
    "not often noticed",
    "rarely noticed",
    "underexplored",
    "under explored",
    "hidden opportunity",
    "interesting cross-domain insight",
    "this may create an edge",
    "this could create an edge",
)

KNOWNNESS_MARKERS = (
    "well known",
    "widely known",
    "already known",
    "already established",
    "standard practice",
    "common practice",
    "commonly used",
    "widely used",
    "routine practice",
    "textbook",
    "already explicit",
    "already recommended",
)

GENERIC_TEST_DECISION_FILLERS = (
    "the effect happens",
    "the hypothesis is supported",
    "the hypothesis holds",
    "the mechanism is true",
    "results improve",
    "outcomes improve",
    "performance improves",
    "the effect appears",
)

EDGE_PROBLEM_HINTS = (
    "problem",
    "blind spot",
    "failure",
    "fails",
    "miss",
    "missed",
    "bottleneck",
    "threshold",
    "control point",
    "conflict",
    "collision",
    "plateau",
    "drift",
    "underestimate",
    "overestimate",
    "saturation",
)

EDGE_ACTION_HINTS = (
    "add",
    "apply",
    "compare",
    "filter",
    "rank",
    "switch",
    "tune",
    "route",
    "replay",
    "simulate",
    "measure",
    "test",
    "use",
    "deploy",
    "screen",
    "prioritize",
    "constrain",
    "gate",
)

EDGE_ADVANTAGE_HINTS = (
    "advantage",
    "gain",
    "reduce",
    "lower",
    "faster",
    "earlier",
    "improve",
    "better",
    "throughput",
    "cost",
    "latency",
    "warning",
    "quality",
    "efficiency",
    "allocation",
    "collision",
    "error",
    "risk",
    "yield",
)

UNDEREXPLOITEDNESS_HINTS = (
    "cross-silo",
    "cross silo",
    "rarely searched",
    "framed together",
    "retrieval",
    "search",
    "query",
    "silo",
    "workflow",
    "benchmark",
    "default",
    "measurement blind spot",
    "indexing",
    "discipline",
    "literature",
    "tooling",
    "pipeline",
    "naming mismatch",
    "taxonomy",
    "operator habit",
    "screened out",
    "underused",
    "underexploited",
)

GENERIC_QUERY_TOKENS = {
    "change",
    "changes",
    "complex",
    "constraint",
    "constraints",
    "dynamic",
    "dynamics",
    "effect",
    "effects",
    "generic",
    "interaction",
    "interactions",
    "local",
    "multiple",
    "process",
    "processes",
    "structure",
    "structures",
    "system",
    "systems",
}
WEAK_QUERY_TOKENS = {
    "adjust",
    "credibility",
    "deficiency",
    "directed",
    "feedback",
    "recruitment",
    "stabilization",
    "stabilizing",
    "threshold",
    "tune",
    "trigger",
    "triggered",
    "triggers",
}
WEAK_RELATIONAL_QUERY_TOKENS = {
    "adjacent",
    "adjacency",
    "around",
    "between",
    "near",
    "nearby",
    "position",
    "positions",
    "relation",
    "relations",
    "relative",
}
AMBIGUOUS_JUMP_QUERY_TOKENS = {
    "channel",
    "channels",
    "flow",
    "load",
    "promotion",
    "queue",
    "rate",
    "routing",
    "selection",
    "switching",
    "threshold",
}
OVERLOADED_JUMP_QUERY_TOKENS = {
    "arbitration",
    "backtesting",
    "policy",
    "priority",
    "resource",
    "resources",
    "shared",
}
JUMP_TITLE_SIGNATURE_NOISE_TOKENS = {
    "checklist",
    "guide",
    "guides",
    "introduction",
    "modern",
    "overview",
    "paper",
    "papers",
    "review",
    "reviews",
    "study",
    "studies",
    "tutorial",
    "wikipedia",
}
MECHANISM_QUERY_TOKENS = {
    "accumulation",
    "amplification",
    "bottleneck",
    "cascade",
    "channel",
    "channels",
    "competition",
    "constrained",
    "coupled",
    "coupling",
    "decay",
    "destabilization",
    "disturbance",
    "feedback",
    "filtering",
    "gating",
    "inhibition",
    "periodic",
    "propagation",
    "queueing",
    "release",
    "reset",
    "routing",
    "saturation",
    "selective",
    "spatial",
    "stabilization",
    "switching",
    "threshold",
    "throughput",
    "latency",
    "queue",
    "load",
    "rate",
    "screen",
    "triage",
    "tuning",
}
PHRASE_ANCHOR_TAIL_TOKENS = {
    "block",
    "collapse",
    "cost",
    "detector",
    "gating",
    "latency",
    "lock-in",
    "pressure",
    "rate",
    "routing",
    "start",
    "switching",
    "threshold",
}
FORMAL_QUERY_RED_FLAG_TOKENS = {
    "alignment",
    "simultaneous",
    "verification",
}
SOLUTION_EVIDENCE_MARKERS = (
    "workaround",
    "mitigat",
    "correct",
    "bypass",
    "compensat",
    "solution",
    "response",
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
JUMP_QUERY_ACTION_VERB_TOKENS = {
    "adjust",
    "change",
    "move",
    "tighten",
    "tune",
    "vary",
}
JUMP_QUERY_TRAILING_CONNECTORS = {
    "after",
    "because",
    "before",
    "by",
    "for",
    "from",
    "into",
    "that",
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
    "regime",
    "response",
    "recovery",
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
    """Extract lowercase query tokens while preserving hyphenated mechanism words."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", (text or "").lower())


def _jump_transferable_source_leakage_terms(
    transferable_tokens: set[str],
    grounded_source_tokens: set[str],
) -> list[str]:
    """Return only source-specific grounded overlap terms."""
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
    """Block exact one-token domain names without adding broad domain vocabulary."""
    blockers: set[str] = set()
    for value in values:
        domain_tokens = _tokenize_query_terms(str(value or ""))
        if len(domain_tokens) == 1:
            blockers.add(domain_tokens[0])
    return blockers


def _is_generic_jump_grounded_source_token(token: str) -> bool:
    """Filter broad mechanism/control terms using existing query vocab."""
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
    """Return True when an overlap term is specific enough to block fallback use."""
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
    """Keep only source overlap terms that are strong enough to make usable=false."""
    return sorted(
        token
        for token in terms
        if _is_strong_jump_source_specific_token(token)
    )


def _jump_grounded_source_tokens(grounded: dict) -> set[str]:
    """Derive a bounded source-specific term bag from grounded source fields only."""
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
    """Return overlap terms showing transferable text still mirrors source-native labels."""
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
    grounded = pattern.get("grounded") if isinstance(pattern.get("grounded"), dict) else {}
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
    """Normalize one scope or result string for source-domain containment checks."""
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower())).strip()


def _jump_result_mentions_source_domain(
    title_text: str,
    clean: str,
    url: str,
    normalized_host: str,
    source_domain: str,
) -> bool:
    """Reject same-source-domain hits even when the title omits the source domain."""
    source_scope = _normalize_jump_scope_text(source_domain)
    if not source_scope:
        return False
    candidate_scope = _normalize_jump_scope_text(
        " ".join(
            part
            for part in (
                title_text,
                clean,
                url,
                normalized_host,
            )
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


def _host_matches_jump_include_domains(host: str, include_domains: tuple[str, ...]) -> bool:
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
        bool(matched_control)
        and has_condition
        and strong_process_context
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
    grounded = pattern.get("grounded") if isinstance(pattern.get("grounded"), dict) else {}
    return {
        "pattern_name": str(pattern.get("pattern_name", "") or "").strip(),
        "abstract_structure": str(pattern.get("abstract_structure", "") or "").strip(),
        "search_query": str(pattern.get("search_query", "") or "").strip(),
        "measurable_signal": str(pattern.get("measurable_signal", "") or "").strip(),
        "control_lever": str(pattern.get("control_lever", "") or "").strip(),
        "transfer_rationale": str(pattern.get("transfer_rationale", "") or "").strip(),
        "grounded": dict(grounded),
    }


def _jump_transferable_query_profile(
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> dict:
    transferable = pattern.get("transferable") if isinstance(pattern.get("transferable"), dict) else {}
    grounded = pattern.get("grounded") if isinstance(pattern.get("grounded"), dict) else {}
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
            field_name
            for field_name, text in fields.items()
            if text
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
        if len(token_set) < 2 or clause_score[1] < 1 or clause_score[2] < 2:
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
        [
            token
            for token in source_shape_terms
            if token in source_context_tokens
        ]
    )

    overlap_pairs: list[str] = []
    for left_name, right_name in (
        ("mechanism", "control_logic"),
        ("mechanism", "signal_shape"),
        ("control_logic", "signal_shape"),
    ):
        if (
            left_name in backfilled_field_set
            or right_name in backfilled_field_set
        ):
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

    blocking_concerns = {
        "transferable_field_overlap",
    }
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


def _trim_jump_query_connector_tail(query: str) -> str:
    tokens = _tokenize_query_terms(query)
    while tokens and tokens[-1] in JUMP_QUERY_TRAILING_CONNECTORS:
        tokens.pop()
    return " ".join(tokens).strip()


def _looks_like_jump_query_verb_token(token: str) -> bool:
    return (
        _is_causal_jump_query_token(token)
        or token in JUMP_QUERY_ACTION_VERB_TOKENS
        or token.endswith("ing")
        or token.endswith("ed")
        or token.endswith("es")
    ) and token not in MECHANISM_QUERY_TOKENS and token not in JUMP_QUERY_CAUSAL_OUTCOME_HINTS


def _is_compact_natural_language_jump_query(
    query: str,
    blocked_tokens: set[str],
) -> bool:
    clean_query = re.sub(r"\s+", " ", str(query or "").strip())
    candidate_tokens = _tokenize_query_terms(clean_query)
    if len(candidate_tokens) < 4 or len(candidate_tokens) > 12:
        return False
    if _looks_like_formal_jump_query_token_soup(clean_query, candidate_tokens):
        return False

    strong_tokens = [
        token
        for token in candidate_tokens
        if token not in blocked_tokens
        and token not in GENERIC_QUERY_TOKENS
        and token not in WEAK_QUERY_TOKENS
    ]
    if len(strong_tokens) < 3:
        return False

    connector_markers = {
        "after",
        "before",
        "during",
        "under",
        "when",
        "where",
        "with",
        "without",
    }
    if any(token in connector_markers for token in candidate_tokens):
        return True
    return any(_is_causal_jump_query_token(token) for token in candidate_tokens) and len(
        candidate_tokens
    ) <= 5


def _is_compact_keyword_jump_query(
    query: str,
    blocked_tokens: set[str],
) -> bool:
    candidate_tokens = _tokenize_query_terms(
        _trim_jump_query_connector_tail(str(query or "").strip())
    )
    if len(candidate_tokens) < 4 or len(candidate_tokens) > 8:
        return False
    if _looks_like_formal_jump_query_token_soup(str(query or ""), candidate_tokens):
        return False
    strong_tokens = [
        token
        for token in candidate_tokens
        if token not in blocked_tokens
        and token not in GENERIC_QUERY_TOKENS
        and token not in WEAK_QUERY_TOKENS
        and token not in JUMP_QUERY_FILLER_TOKENS
        and token not in QUERY_PHRASE_STOPWORDS
        and token not in OVERLOADED_JUMP_QUERY_TOKENS
    ]
    clause_score = _score_jump_query_clause(" ".join(candidate_tokens), blocked_tokens)
    return (
        len(strong_tokens) >= 3
        and clause_score[1] >= 1
        and clause_score[2] >= 3
    )


def _build_causal_jump_query_fragment(
    text: str,
    blocked_tokens: set[str],
) -> str:
    normalized_text = _normalize_jump_query_clause(text)
    if not normalized_text:
        return ""

    clauses = [
        clause.strip()
        for clause in re.split(r"[.;:,]", normalized_text)
        if clause.strip()
    ]
    if not clauses:
        clauses = [normalized_text]
    best_clause = max(
        clauses,
        key=lambda clause: _score_jump_query_clause(clause, blocked_tokens),
    )
    clause_tokens = _tokenize_query_terms(best_clause)
    if not clause_tokens:
        return ""

    focus_index = next(
        (
            index
            for index, token in enumerate(clause_tokens)
            if _is_causal_jump_query_token(token)
        ),
        next(
            (
                index
                for index, token in enumerate(clause_tokens)
                if token in MECHANISM_QUERY_TOKENS
            ),
            0,
        ),
    )
    start_index = max(0, focus_index - 2)
    selected: list[str] = []
    strong_token_count = 0
    for token in clause_tokens[start_index:]:
        if token in blocked_tokens or token in OVERLOADED_JUMP_QUERY_TOKENS:
            continue
        if token in JUMP_QUERY_FILLER_TOKENS:
            continue
        if (
            token in GENERIC_QUERY_TOKENS
            and not _is_causal_jump_query_token(token)
            and token not in JUMP_QUERY_CAUSAL_OUTCOME_HINTS
        ):
            continue
        if (
            token in WEAK_QUERY_TOKENS
            and not _is_causal_jump_query_token(token)
        ):
            continue
        if (
            token in AMBIGUOUS_JUMP_QUERY_TOKENS
            and strong_token_count >= 3
            and token not in MECHANISM_QUERY_TOKENS
        ):
            continue
        if len(token) <= 2:
            continue
        selected.append(token)
        if (
            _is_specific_jump_query_token(token)
            or _is_causal_jump_query_token(token)
            or token in MECHANISM_QUERY_TOKENS
            or token in JUMP_QUERY_CAUSAL_OUTCOME_HINTS
        ):
            strong_token_count += 1
        if len(selected) >= 7:
            break
    fallback = " ".join(selected[:7]).strip()
    trimmed = _trim_jump_query_connector_tail(fallback)
    if len(_tokenize_query_terms(trimmed)) >= 4:
        return trimmed
    return trimmed or fallback


def _build_jump_keyword_query_fragment(
    pattern: dict,
    raw_query: str,
    blocked_tokens: set[str],
) -> str:
    selected: list[str] = []
    covered_tokens: set[str] = set()
    has_causal_term = False
    has_outcome_term = False
    mechanism_support_tokens: set[str] = set()
    for text in (
        str(pattern.get("control_lever", "") or ""),
        str(pattern.get("abstract_structure", "") or ""),
        str(pattern.get("transfer_rationale", "") or ""),
    ):
        for token in _tokenize_query_terms(text):
            if (
                token in blocked_tokens
                or token in GENERIC_QUERY_TOKENS
                or token in WEAK_QUERY_TOKENS
                or token in WEAK_RELATIONAL_QUERY_TOKENS
                or token in OVERLOADED_JUMP_QUERY_TOKENS
                or token in JUMP_QUERY_FILLER_TOKENS
                or token in QUERY_PHRASE_STOPWORDS
                or len(token) <= 2
            ):
                continue
            if (
                _is_concrete_jump_query_token(token)
                or _is_causal_jump_query_token(token)
                or token in MECHANISM_QUERY_TOKENS
                or token in PHRASE_ANCHOR_TAIL_TOKENS
                or token in JUMP_QUERY_CAUSAL_OUTCOME_HINTS
            ):
                mechanism_support_tokens.add(token)
                mechanism_support_tokens.update(
                    part for part in token.split("-") if len(part) > 2
                )
    source_surface_tokens = {
        token
        for token in _tokenize_query_terms(raw_query)
        if (
            token not in mechanism_support_tokens
            and token not in blocked_tokens
            and token not in GENERIC_QUERY_TOKENS
            and token not in WEAK_QUERY_TOKENS
            and token not in WEAK_RELATIONAL_QUERY_TOKENS
            and token not in OVERLOADED_JUMP_QUERY_TOKENS
            and token not in JUMP_QUERY_FILLER_TOKENS
            and token not in QUERY_PHRASE_STOPWORDS
            and token not in MECHANISM_QUERY_TOKENS
            and token not in PHRASE_ANCHOR_TAIL_TOKENS
            and token not in JUMP_QUERY_CAUSAL_OUTCOME_HINTS
            and not _is_causal_jump_query_token(token)
            and len(token) > 2
        )
    }

    def _token_budget() -> int:
        return 8 - len(_tokenize_query_terms(" ".join(selected)))

    def _is_source_surface_token(token: str) -> bool:
        if token in source_surface_tokens:
            return True
        token_parts = [part for part in str(token or "").split("-") if len(part) > 2]
        return bool(token_parts) and all(part in source_surface_tokens for part in token_parts)

    def _is_keyword_phrase_anchor_token(token: str) -> bool:
        if token in WEAK_RELATIONAL_QUERY_TOKENS:
            return False
        return (
            token in MECHANISM_QUERY_TOKENS
            or token in PHRASE_ANCHOR_TAIL_TOKENS
            or token in JUMP_QUERY_CAUSAL_OUTCOME_HINTS
            or _is_causal_jump_query_token(token)
            or any(
                token.startswith(marker)
                for marker in (
                    SOLUTION_EVIDENCE_MARKERS
                    + INTERVENTION_CONTROL_MARKERS
                    + INTERVENTION_RESPONSE_MARKERS
                )
            )
            or (
                token in mechanism_support_tokens
                and _is_concrete_jump_query_token(token)
                and token.endswith(
                    ("ance", "ence", "ing", "ion", "ment", "sis", "ure")
                )
            )
        )

    def _is_keyword_token(
        token: str,
        *,
        allow_causal: bool = False,
        allow_weak_relational: bool = False,
    ) -> bool:
        if (
            token in blocked_tokens
            or _is_source_surface_token(token)
            or token in GENERIC_QUERY_TOKENS
            or token in WEAK_QUERY_TOKENS
            or (
                token in WEAK_RELATIONAL_QUERY_TOKENS
                and not allow_weak_relational
            )
            or token in OVERLOADED_JUMP_QUERY_TOKENS
            or token in JUMP_QUERY_FILLER_TOKENS
            or token in QUERY_PHRASE_STOPWORDS
            or len(token) <= 2
        ):
            return False
        if (
            token in MECHANISM_QUERY_TOKENS
            or token in PHRASE_ANCHOR_TAIL_TOKENS
            or token in JUMP_QUERY_CAUSAL_OUTCOME_HINTS
        ):
            return True
        if not allow_causal and _looks_like_jump_query_verb_token(token):
            return False
        if _is_concrete_jump_query_token(token):
            return not _is_causal_jump_query_token(token) or allow_causal
        return allow_causal and _is_causal_jump_query_token(token)

    def _is_keyword_phrase(phrase: str) -> bool:
        phrase_tokens = _tokenize_query_terms(phrase)
        if len(phrase_tokens) < 2 or len(phrase_tokens) > 3:
            return False
        first_token = phrase_tokens[0]
        second_token = phrase_tokens[1]
        second_is_anchor_tail = (
            second_token in PHRASE_ANCHOR_TAIL_TOKENS
            or second_token in MECHANISM_QUERY_TOKENS
        )
        if (
            first_token in blocked_tokens
            or first_token in GENERIC_QUERY_TOKENS
            or (first_token in WEAK_QUERY_TOKENS and not second_is_anchor_tail)
            or first_token in JUMP_QUERY_FILLER_TOKENS
            or first_token in QUERY_PHRASE_STOPWORDS
            or (
                first_token in OVERLOADED_JUMP_QUERY_TOKENS
                and not second_is_anchor_tail
            )
            or len(first_token) <= 2
            or (
                _looks_like_jump_query_verb_token(first_token)
                and not second_is_anchor_tail
            )
        ):
            return False
        weak_relational_count = sum(
            token in WEAK_RELATIONAL_QUERY_TOKENS for token in phrase_tokens
        )
        if weak_relational_count > 1:
            return False
        for token in phrase_tokens[1:]:
            if token in WEAK_RELATIONAL_QUERY_TOKENS:
                continue
            if token == second_token and second_is_anchor_tail:
                continue
            if (not _is_keyword_token(token) or _looks_like_jump_query_verb_token(token)):
                return False
        if not any(
            _is_keyword_phrase_anchor_token(token)
            for token in phrase_tokens
            if token not in WEAK_RELATIONAL_QUERY_TOKENS
        ):
            return False
        if weak_relational_count:
            return any(
                _is_keyword_phrase_anchor_token(token)
                for token in phrase_tokens
                if token not in WEAK_RELATIONAL_QUERY_TOKENS
            )
        return second_is_anchor_tail or _is_keyword_token(phrase_tokens[1])

    def _append_part(part: str) -> None:
        nonlocal has_causal_term, has_outcome_term
        normalized = _trim_jump_query_connector_tail(
            re.sub(r"\s+", " ", str(part or "").strip().lower())
        )
        if not normalized or normalized in selected:
            return
        part_tokens = _tokenize_query_terms(normalized)
        if (
            not part_tokens
            or len(part_tokens) > _token_budget()
            or any(token in covered_tokens for token in part_tokens)
        ):
            return
        selected.append(normalized)
        covered_tokens.update(part_tokens)
        has_causal_term = has_causal_term or any(
            _is_causal_jump_query_token(token) for token in part_tokens
        )
        has_outcome_term = has_outcome_term or any(
            token in JUMP_QUERY_CAUSAL_OUTCOME_HINTS for token in part_tokens
        )

    preferred_field_texts = [
        str(pattern.get("control_lever", "") or ""),
        str(pattern.get("measurable_signal", "") or ""),
        str(pattern.get("abstract_structure", "") or ""),
        str(pattern.get("transfer_rationale", "") or ""),
    ]
    fallback_field_texts = [
        str(pattern.get("pattern_name", "") or ""),
        raw_query,
    ]
    phrase_candidates: list[str] = []
    phrase_seen: set[str] = set()
    token_groups: dict[str, list[str]] = {
        "mechanism": [],
        "outcome": [],
        "causal": [],
    }
    token_seen: set[str] = set()

    def _collect_anchor_phrases(anchor_pattern: dict) -> None:
        preferred_anchor_phrases = _preferred_jump_query_anchor_phrases(
            anchor_pattern,
            blocked_tokens,
        )
        for phrase in preferred_anchor_phrases:
            normalized_phrase = _trim_jump_query_connector_tail(phrase)
            if (
                not normalized_phrase
                or normalized_phrase in phrase_seen
                or not _is_keyword_phrase(normalized_phrase)
            ):
                continue
            phrase_seen.add(normalized_phrase)
            phrase_candidates.append(normalized_phrase)

    def _collect_from_texts(field_texts: list[str]) -> None:
        for text in field_texts:
            normalized_text = _normalize_jump_query_clause(text)
            if not normalized_text:
                continue
            for phrase in _extract_jump_query_phrases(normalized_text, blocked_tokens):
                normalized_phrase = _trim_jump_query_connector_tail(phrase)
                if (
                    not normalized_phrase
                    or normalized_phrase in phrase_seen
                    or not _is_keyword_phrase(normalized_phrase)
                ):
                    continue
                phrase_seen.add(normalized_phrase)
                phrase_candidates.append(normalized_phrase)

            text_tokens = _tokenize_query_terms(normalized_text)
            for index in range(len(text_tokens) - 1):
                first = text_tokens[index]
                second = text_tokens[index + 1]
                if not _is_keyword_token(first) or not _is_keyword_token(second):
                    continue
                if _looks_like_jump_query_verb_token(second):
                    continue
                phrase = _trim_jump_query_connector_tail(f"{first} {second}")
                if (
                    phrase
                    and phrase not in phrase_seen
                    and _is_keyword_phrase(phrase)
                ):
                    phrase_seen.add(phrase)
                    phrase_candidates.append(phrase)

            for token in text_tokens:
                if token in token_seen:
                    continue
                if token in JUMP_QUERY_CAUSAL_OUTCOME_HINTS and _is_keyword_token(token):
                    token_groups["outcome"].append(token)
                    token_seen.add(token)
                    continue
                if token in MECHANISM_QUERY_TOKENS and _is_keyword_token(token):
                    token_groups["mechanism"].append(token)
                    token_seen.add(token)
                    continue
                if _is_concrete_jump_query_token(token) and _is_keyword_token(token):
                    token_groups["mechanism"].append(token)
                    token_seen.add(token)
                    continue
                if (
                    _is_keyword_token(token, allow_causal=True)
                    and _is_causal_jump_query_token(token)
                ):
                    token_groups["causal"].append(token)
                    token_seen.add(token)

    def _compose_selected_query() -> str:
        for phrase in phrase_candidates:
            if len(selected) >= 2 or _token_budget() < 2:
                break
            _append_part(phrase)

        for token in token_groups["mechanism"]:
            if _token_budget() <= 0:
                break
            _append_part(token)

        if _token_budget() > 0 and not has_outcome_term:
            for token in token_groups["outcome"]:
                _append_part(token)
                if has_outcome_term or _token_budget() <= 0:
                    break

        if _token_budget() > 0 and not has_causal_term and not has_outcome_term:
            for token in token_groups["causal"]:
                _append_part(token)
                if has_causal_term or _token_budget() <= 0:
                    break

        return _trim_jump_query_connector_tail(" ".join(selected))

    _collect_anchor_phrases(
        {
            "pattern_name": "",
            "control_lever": str(pattern.get("control_lever", "") or ""),
            "abstract_structure": str(pattern.get("abstract_structure", "") or ""),
        }
    )
    _collect_from_texts(preferred_field_texts)
    composed_query = _compose_selected_query()
    if len(_tokenize_query_terms(composed_query)) >= 4:
        return composed_query

    _collect_anchor_phrases(
        {
            "pattern_name": str(pattern.get("pattern_name", "") or ""),
            "control_lever": "",
            "abstract_structure": "",
        }
    )
    _collect_from_texts(fallback_field_texts)
    composed_query = _compose_selected_query()
    if len(_tokenize_query_terms(composed_query)) < 4:
        return ""
    return composed_query


def _jump_query_anchor_support(
    query: str,
    preferred_anchor_phrases: list[str],
    blocked_tokens: set[str],
) -> tuple[bool, list[str]]:
    lowered_query = str(query or "").lower()
    has_preferred_phrase = any(
        phrase and phrase in lowered_query for phrase in preferred_anchor_phrases
    )
    concrete_tokens = [
        token
        for token in _tokenize_query_terms(query)
        if (
            token not in blocked_tokens
            and token not in GENERIC_QUERY_TOKENS
            and token not in WEAK_QUERY_TOKENS
            and token not in OVERLOADED_JUMP_QUERY_TOKENS
            and _is_concrete_jump_query_token(token)
        )
    ]
    return has_preferred_phrase, concrete_tokens


def _has_strong_jump_query_support_beyond_preferred_phrase(
    candidate: str,
    preferred_anchor_phrases: list[str],
    blocked_tokens: set[str],
) -> bool:
    lowered_candidate = str(candidate or "").lower()
    matched_anchor_phrase = _select_best_jump_anchor_phrase(
        [
            phrase
            for phrase in preferred_anchor_phrases
            if phrase and phrase in lowered_candidate
        ]
    )
    matched_phrase_tokens = set(_tokenize_query_terms(matched_anchor_phrase))

    extra_support_tokens = [
        token
        for token in _tokenize_query_terms(candidate)
        if (
            token not in matched_phrase_tokens
            and token not in blocked_tokens
            and token not in GENERIC_QUERY_TOKENS
            and token not in WEAK_QUERY_TOKENS
            and token not in OVERLOADED_JUMP_QUERY_TOKENS
            and token not in JUMP_QUERY_FILLER_TOKENS
            and (
                _is_concrete_jump_query_token(token)
                or _is_causal_jump_query_token(token)
                or token in MECHANISM_QUERY_TOKENS
                or token in JUMP_QUERY_CAUSAL_OUTCOME_HINTS
            )
        )
    ]
    clause_score = _score_jump_query_clause(candidate, blocked_tokens)
    return (
        len(extra_support_tokens) >= 2
        and clause_score[1] >= 2
        and clause_score[2] >= 3
    )


def _has_source_surface_jump_query_overhang(
    candidate_tokens: list[str],
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> bool:
    blocked_tokens, _preferred_anchor_phrases, support_tokens = _jump_query_support_context(
        pattern,
        source_domain,
        source_category,
    )
    support_token_set = set(support_tokens)
    source_surface_tokens = {
        token
        for token in _tokenize_query_terms(str(pattern.get("search_query", "") or ""))
        if (
            token not in support_token_set
            and token not in blocked_tokens
            and token not in GENERIC_QUERY_TOKENS
            and token not in WEAK_QUERY_TOKENS
            and token not in OVERLOADED_JUMP_QUERY_TOKENS
            and token not in MECHANISM_QUERY_TOKENS
            and not _is_causal_jump_query_token(token)
            and len(token) > 3
        )
    }
    if not source_surface_tokens:
        return False

    surface_hit_count = sum(token in source_surface_tokens for token in candidate_tokens)
    if surface_hit_count < 2:
        return False

    transferable_hit_count = sum(
        1
        for token in candidate_tokens
        if (
            token not in source_surface_tokens
            and token not in blocked_tokens
            and token not in GENERIC_QUERY_TOKENS
            and token not in WEAK_QUERY_TOKENS
            and token not in OVERLOADED_JUMP_QUERY_TOKENS
            and token not in JUMP_QUERY_FILLER_TOKENS
            and (
                _is_concrete_jump_query_token(token)
                or _is_causal_jump_query_token(token)
                or token in MECHANISM_QUERY_TOKENS
                or token in JUMP_QUERY_CAUSAL_OUTCOME_HINTS
            )
        )
    )
    return surface_hit_count > transferable_hit_count


def _unsupported_llm_jump_query_tokens(
    candidate_tokens: list[str],
    pattern: dict,
    source_domain: str,
    source_category: str,
    heuristic_query: str,
) -> list[str]:
    blocked_tokens, preferred_anchor_phrases, support_tokens = _jump_query_support_context(
        pattern,
        source_domain,
        source_category,
    )
    connector_tokens = {
        "after",
        "before",
        "during",
        "under",
        "when",
        "where",
        "with",
        "without",
    }
    grounded = pattern.get("grounded") if isinstance(pattern.get("grounded"), dict) else {}
    grounded_source_tokens = _jump_grounded_source_tokens(grounded)
    if grounded_source_tokens:
        return _strong_jump_source_terms(
            _jump_transferable_source_leakage_terms(
                set(candidate_tokens),
                grounded_source_tokens,
            )
        )

    grounded_tokens = set(support_tokens)
    for phrase in preferred_anchor_phrases:
        grounded_tokens.update(_tokenize_query_terms(phrase))
    for text in (
        str(pattern.get("search_query", "") or ""),
        str(pattern.get("control_lever", "") or ""),
        str(pattern.get("abstract_structure", "") or ""),
        str(pattern.get("measurable_signal", "") or ""),
        str(pattern.get("pattern_name", "") or ""),
        str(pattern.get("transfer_rationale", "") or ""),
        heuristic_query,
    ):
        for token in _tokenize_query_terms(text):
            if (
                token in blocked_tokens
                or token in GENERIC_QUERY_TOKENS
                or token in WEAK_QUERY_TOKENS
                or token in OVERLOADED_JUMP_QUERY_TOKENS
                or token in JUMP_QUERY_FILLER_TOKENS
                or token in QUERY_PHRASE_STOPWORDS
                or token in connector_tokens
                or len(token) <= 2
            ):
                continue
            grounded_tokens.add(token)

    def _is_grounded_candidate_token(token: str) -> bool:
        if token in grounded_tokens:
            return True
        if token.endswith("s") and len(token) > 4 and token[:-1] in grounded_tokens:
            return True
        if f"{token}s" in grounded_tokens:
            return True
        return False

    unsupported_tokens: list[str] = []
    seen_tokens: set[str] = set()
    for token in candidate_tokens:
        singular_token = token[:-1] if token.endswith("s") else token
        if (
            token in seen_tokens
            or _is_grounded_candidate_token(token)
            or token in blocked_tokens
            or token in GENERIC_QUERY_TOKENS
            or token in WEAK_QUERY_TOKENS
            or token in OVERLOADED_JUMP_QUERY_TOKENS
            or token in JUMP_QUERY_FILLER_TOKENS
            or token in QUERY_PHRASE_STOPWORDS
            or token in connector_tokens
            or token in MECHANISM_QUERY_TOKENS
            or singular_token in MECHANISM_QUERY_TOKENS
            or token in JUMP_QUERY_CAUSAL_OUTCOME_HINTS
            or singular_token in JUMP_QUERY_CAUSAL_OUTCOME_HINTS
            or _is_causal_jump_query_token(token)
            or any(
                token.startswith(marker)
                for marker in (
                    SOLUTION_EVIDENCE_MARKERS
                    + INTERVENTION_CONTROL_MARKERS
                    + INTERVENTION_RESPONSE_MARKERS
                )
            )
            or not _is_specific_jump_query_token(token)
            or len(token) <= 3
        ):
            continue
        seen_tokens.add(token)
        unsupported_tokens.append(token)
    return unsupported_tokens


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


def _needs_jump_query_collision_guard(
    query: str,
    preferred_anchor_phrases: list[str],
    blocked_tokens: set[str],
) -> bool:
    query_tokens = _tokenize_query_terms(query)
    if not query_tokens or not any(
        token in OVERLOADED_JUMP_QUERY_TOKENS for token in query_tokens
    ):
        return False
    has_preferred_phrase, concrete_tokens = _jump_query_anchor_support(
        query,
        preferred_anchor_phrases,
        blocked_tokens,
    )
    return not (has_preferred_phrase and len(concrete_tokens) >= 2)


def _apply_jump_query_collision_guard(
    query: str,
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> tuple[str, bool]:
    clean_query = re.sub(r"\s+", " ", str(query or "").strip())
    if not clean_query:
        return "", False

    blocked_tokens, preferred_anchor_phrases, support_tokens = _jump_query_support_context(
        pattern,
        source_domain,
        source_category,
    )
    if not _needs_jump_query_collision_guard(
        clean_query,
        preferred_anchor_phrases,
        blocked_tokens,
    ):
        return clean_query, False

    selected: list[str] = []
    covered_tokens: set[str] = set()

    def _append_part(part: str) -> None:
        normalized = str(part or "").strip()
        if not normalized or normalized in selected:
            return
        selected.append(normalized)
        covered_tokens.update(_tokenize_query_terms(normalized))

    best_anchor_phrase = _select_best_jump_anchor_phrase(preferred_anchor_phrases)
    if best_anchor_phrase:
        phrase_tokens = _tokenize_query_terms(best_anchor_phrase)
        if phrase_tokens and not any(token in covered_tokens for token in phrase_tokens):
            _append_part(best_anchor_phrase)

    def _append_tokens(tokens: list[str]) -> None:
        for token in tokens:
            if (
                token in covered_tokens
                or token in blocked_tokens
                or token in GENERIC_QUERY_TOKENS
                or token in WEAK_QUERY_TOKENS
                or token in OVERLOADED_JUMP_QUERY_TOKENS
                or len(token) <= 2
            ):
                continue
            _append_part(token)
            if len(_tokenize_query_terms(" ".join(selected))) >= 6:
                return

    current_tokens = [
        token
        for token in _tokenize_query_terms(clean_query)
        if token not in OVERLOADED_JUMP_QUERY_TOKENS
    ]
    _append_tokens([token for token in current_tokens if _is_concrete_jump_query_token(token)])
    _append_tokens(support_tokens)
    _append_tokens([token for token in current_tokens if token in MECHANISM_QUERY_TOKENS])

    rebuilt_query = " ".join(selected).strip()
    rebuilt_tokens = _tokenize_query_terms(rebuilt_query)
    if len(rebuilt_tokens) < 4:
        _append_tokens(
            [
                token
                for token in current_tokens
                if token not in GENERIC_QUERY_TOKENS and token not in WEAK_QUERY_TOKENS
            ]
        )
        rebuilt_query = " ".join(selected).strip()
        rebuilt_tokens = _tokenize_query_terms(rebuilt_query)

    if len(rebuilt_tokens) > 10:
        rebuilt_query = " ".join(rebuilt_tokens[:10])
    return rebuilt_query or clean_query, True


def _extract_jump_query_phrases(text: str, blocked_tokens: set[str]) -> list[str]:
    phrases: list[str] = []
    raw_tokens = _tokenize_query_terms(text)
    for index in range(len(raw_tokens) - 1):
        first = raw_tokens[index]
        second = raw_tokens[index + 1]
        if (
            first in blocked_tokens
            or second in blocked_tokens
            or first in GENERIC_QUERY_TOKENS
            or second in GENERIC_QUERY_TOKENS
        ):
            continue
        if first in QUERY_PHRASE_STOPWORDS or second in QUERY_PHRASE_STOPWORDS:
            continue
        if len(first) <= 2 or len(second) <= 2:
            continue
        if first in WEAK_QUERY_TOKENS and second in WEAK_QUERY_TOKENS:
            continue
        if first in WEAK_QUERY_TOKENS:
            continue
        if not (
            second in PHRASE_ANCHOR_TAIL_TOKENS
            or second in MECHANISM_QUERY_TOKENS
        ):
            continue
        if not (
            _is_specific_jump_query_token(first)
            or _is_specific_jump_query_token(second)
            or second in PHRASE_ANCHOR_TAIL_TOKENS
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
            if (
                first in blocked_tokens
                or second in blocked_tokens
                or first in QUERY_PHRASE_STOPWORDS
                or second in QUERY_PHRASE_STOPWORDS
                or first in GENERIC_QUERY_TOKENS
                or second in GENERIC_QUERY_TOKENS
            ):
                continue
            if len(first) <= 2 or len(second) <= 2:
                continue
            if first in WEAK_QUERY_TOKENS and second in WEAK_QUERY_TOKENS:
                continue
            if not (
                _is_specific_jump_query_token(first)
                or _is_specific_jump_query_token(second)
                or second in PHRASE_ANCHOR_TAIL_TOKENS
            ):
                continue
            phrase = f"{first} {second}"
            if phrase in seen:
                continue
            seen.add(phrase)
            phrases.append(phrase)
    return phrases


GENERIC_QUERY_TOKENS = _jump_support.GENERIC_QUERY_TOKENS
WEAK_QUERY_TOKENS = _jump_support.WEAK_QUERY_TOKENS
AMBIGUOUS_JUMP_QUERY_TOKENS = _jump_support.AMBIGUOUS_JUMP_QUERY_TOKENS
OVERLOADED_JUMP_QUERY_TOKENS = _jump_support.OVERLOADED_JUMP_QUERY_TOKENS
JUMP_TITLE_SIGNATURE_NOISE_TOKENS = _jump_support.JUMP_TITLE_SIGNATURE_NOISE_TOKENS
MECHANISM_QUERY_TOKENS = _jump_support.MECHANISM_QUERY_TOKENS
PHRASE_ANCHOR_TAIL_TOKENS = _jump_support.PHRASE_ANCHOR_TAIL_TOKENS
SOLUTION_EVIDENCE_MARKERS = _jump_support.SOLUTION_EVIDENCE_MARKERS
INTERVENTION_CONTROL_MARKERS = _jump_support.INTERVENTION_CONTROL_MARKERS
INTERVENTION_RESPONSE_MARKERS = _jump_support.INTERVENTION_RESPONSE_MARKERS
INTERVENTION_CONDITION_PHRASES = _jump_support.INTERVENTION_CONDITION_PHRASES
INTERVENTION_CONDITION_TOKENS = _jump_support.INTERVENTION_CONDITION_TOKENS
QUERY_PHRASE_STOPWORDS = _jump_support.QUERY_PHRASE_STOPWORDS
JUMP_QUERY_CAUSAL_VERB_STEMS = _jump_support.JUMP_QUERY_CAUSAL_VERB_STEMS
JUMP_QUERY_CAUSAL_OUTCOME_HINTS = _jump_support.JUMP_QUERY_CAUSAL_OUTCOME_HINTS
JUMP_QUERY_CLAUSE_PREFIXES = _jump_support.JUMP_QUERY_CLAUSE_PREFIXES
JUMP_QUERY_FILLER_TOKENS = _jump_support.JUMP_QUERY_FILLER_TOKENS
JUMP_SOURCE_LEAKAGE_GENERIC_TOKENS = _jump_support.JUMP_SOURCE_LEAKAGE_GENERIC_TOKENS
JUMP_BROAD_SOURCE_OVERLAP_TOKENS = _jump_support.JUMP_BROAD_SOURCE_OVERLAP_TOKENS

_tokenize_query_terms = _jump_support._tokenize_query_terms
_jump_transferable_source_leakage_terms = (
    _jump_support._jump_transferable_source_leakage_terms
)
_jump_exact_domain_blocker_tokens = _jump_support._jump_exact_domain_blocker_tokens
_is_generic_jump_grounded_source_token = (
    _jump_support._is_generic_jump_grounded_source_token
)
_is_strong_jump_source_specific_token = (
    _jump_support._is_strong_jump_source_specific_token
)
_strong_jump_source_terms = _jump_support._strong_jump_source_terms
_jump_grounded_source_tokens = _jump_support._jump_grounded_source_tokens
_jump_source_shaped_terms = _jump_support._jump_source_shaped_terms
_is_specific_jump_query_token = _jump_support._is_specific_jump_query_token
_is_concrete_jump_query_token = _jump_support._is_concrete_jump_query_token
_is_causal_jump_query_token = _jump_support._is_causal_jump_query_token
_normalize_jump_result_host = _jump_support._normalize_jump_result_host
_normalize_jump_scope_text = _jump_support._normalize_jump_scope_text
_jump_result_mentions_source_domain = _jump_support._jump_result_mentions_source_domain
_host_matches_jump_include_domains = _jump_support._host_matches_jump_include_domains
_build_jump_title_signature = _jump_support._build_jump_title_signature
_jump_solution_marker_count = _jump_support._jump_solution_marker_count
_classify_jump_intervention_evidence = (
    _jump_support._classify_jump_intervention_evidence
)
_jump_legacy_flat_pattern = _jump_support._jump_legacy_flat_pattern
_normalize_jump_query_clause = _jump_support._normalize_jump_query_clause
_score_jump_query_clause = _jump_support._score_jump_query_clause
_jump_transferable_query_profile = _jump_support._jump_transferable_query_profile
_jump_query_pattern_view = _jump_support._jump_query_pattern_view
_jump_query_support_context = _jump_support._jump_query_support_context
_extract_jump_query_phrases = _jump_support._extract_jump_query_phrases
_preferred_jump_query_anchor_phrases = (
    _jump_support._preferred_jump_query_anchor_phrases
)
_select_best_jump_anchor_phrase = _jump_support._select_best_jump_anchor_phrase


def _looks_like_formal_jump_query_token_soup(
    raw_candidate: str,
    candidate_tokens: list[str],
) -> bool:
    if re.search(r"\b(?:AND|OR|NOT|XOR)\b", raw_candidate):
        return True
    return sum(token in FORMAL_QUERY_RED_FLAG_TOKENS for token in candidate_tokens) >= 2


def _needs_jump_query_disambiguation(
    query: str,
    preferred_anchor_phrases: list[str],
) -> bool:
    lowered_query = str(query or "").lower()
    query_tokens = _tokenize_query_terms(query)
    if not query_tokens:
        return False
    has_preferred_phrase = any(phrase in lowered_query for phrase in preferred_anchor_phrases)
    concrete_tokens = [
        token for token in query_tokens if _is_concrete_jump_query_token(token)
    ]
    ambiguous_tokens = [
        token for token in query_tokens if token in AMBIGUOUS_JUMP_QUERY_TOKENS
    ]
    generic_tokens = [
        token
        for token in query_tokens
        if token in GENERIC_QUERY_TOKENS or token in WEAK_QUERY_TOKENS
    ]
    return (
        len(concrete_tokens) < 2
        and len(ambiguous_tokens) >= 2
        and len(ambiguous_tokens) + len(generic_tokens) >= min(len(query_tokens), 3)
        and (not has_preferred_phrase or len(concrete_tokens) < 2)
    )


def _disambiguate_jump_search_query(
    query: str,
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> str:
    clean_query = re.sub(r"\s+", " ", str(query or "").strip())
    if not clean_query:
        return ""

    blocked_tokens = set(_tokenize_query_terms(source_domain))
    blocked_tokens.update(_tokenize_query_terms(source_category))
    preferred_anchor_phrases = _preferred_jump_query_anchor_phrases(pattern, blocked_tokens)
    if not _needs_jump_query_disambiguation(clean_query, preferred_anchor_phrases):
        return clean_query

    selected: list[str] = []
    covered_tokens: set[str] = set()

    def _append_part(part: str) -> None:
        normalized = str(part or "").strip()
        if not normalized or normalized in selected:
            return
        selected.append(normalized)
        covered_tokens.update(_tokenize_query_terms(normalized))

    anchor_phrase = _select_best_jump_anchor_phrase(
        [
            phrase
            for phrase in preferred_anchor_phrases
            if phrase not in clean_query.lower()
        ]
        or preferred_anchor_phrases
    )
    if anchor_phrase:
        _append_part(anchor_phrase)

    def _append_token_group(tokens: list[str]) -> None:
        for token in tokens:
            if (
                token in covered_tokens
                or token in blocked_tokens
                or token in GENERIC_QUERY_TOKENS
                or token in WEAK_QUERY_TOKENS
                or len(token) <= 2
            ):
                continue
            _append_part(token)
            if len(_tokenize_query_terms(" ".join(selected))) >= 6:
                return

    current_tokens = _tokenize_query_terms(clean_query)
    _append_token_group(
        [token for token in current_tokens if _is_concrete_jump_query_token(token)]
    )

    pattern_specific_tokens: list[str] = []
    for text in (
        str(pattern.get("control_lever", "") or ""),
        str(pattern.get("abstract_structure", "") or ""),
        str(pattern.get("measurable_signal", "") or ""),
        str(pattern.get("pattern_name", "") or ""),
        str(pattern.get("transfer_rationale", "") or ""),
    ):
        pattern_specific_tokens.extend(
            token
            for token in _tokenize_query_terms(text)
            if _is_concrete_jump_query_token(token)
        )
    _append_token_group(pattern_specific_tokens)
    _append_token_group(
        [token for token in current_tokens if token in MECHANISM_QUERY_TOKENS]
    )

    refined_query = " ".join(selected)
    return refined_query or clean_query


def _build_jump_search_query_heuristic(
    pattern: dict,
    source_domain: str,
    source_category: str,
    *,
    prefer_transferable: bool = True,
) -> str:
    """Deterministically prefer causal-dynamics phrasing over source-token recombination."""
    query_pattern = (
        _jump_query_pattern_view(pattern, source_domain, source_category)[0]
        if prefer_transferable
        else _jump_legacy_flat_pattern(pattern)
    )
    raw_query = str(query_pattern.get("search_query", "") or "").strip()
    if not raw_query:
        return ""

    blocked_tokens = set(_tokenize_query_terms(source_domain))
    blocked_tokens.update(_tokenize_query_terms(source_category))

    keyword_query = _build_jump_keyword_query_fragment(
        query_pattern,
        raw_query,
        blocked_tokens,
    )
    if keyword_query:
        return keyword_query

    def _filtered_tokens(text: str, *, specific_only: bool = False) -> list[str]:
        out = []
        for token in _tokenize_query_terms(text):
            if (
                token in blocked_tokens
                or token in GENERIC_QUERY_TOKENS
                or token in WEAK_QUERY_TOKENS
            ):
                continue
            if len(token) <= 2:
                continue
            if specific_only and not _is_specific_jump_query_token(token):
                continue
            out.append(token)
        return out

    selected: list[str] = []
    base_specific_tokens = _filtered_tokens(raw_query, specific_only=True)
    base_tokens = _filtered_tokens(raw_query)
    pattern_field_text = [
        str(query_pattern.get("control_lever", "") or ""),
        str(query_pattern.get("abstract_structure", "") or ""),
        str(query_pattern.get("measurable_signal", "") or ""),
        str(query_pattern.get("pattern_name", "") or ""),
        str(query_pattern.get("transfer_rationale", "") or ""),
    ]
    phrase_field_text = [
        str(query_pattern.get("control_lever", "") or ""),
        str(query_pattern.get("abstract_structure", "") or ""),
        str(query_pattern.get("measurable_signal", "") or ""),
        str(query_pattern.get("pattern_name", "") or ""),
        raw_query,
    ]
    phrase_anchors: list[str] = []
    for text in phrase_field_text:
        phrase_anchors.extend(_extract_jump_query_phrases(text, blocked_tokens))
    pattern_specific_tokens: list[str] = []
    pattern_tokens: list[str] = []
    for text in pattern_field_text:
        pattern_specific_tokens.extend(_filtered_tokens(text, specific_only=True))
        pattern_tokens.extend(_filtered_tokens(text))

    covered_tokens: set[str] = set()

    def _append_part(part: str) -> None:
        if part in selected:
            return
        selected.append(part)
        covered_tokens.update(_tokenize_query_terms(part))

    for phrase in phrase_anchors:
        if len(selected) >= 2:
            break
        if covered_tokens.intersection(_tokenize_query_terms(phrase)):
            continue
        _append_part(phrase)

    max_parts = 6
    if len(selected) >= 2:
        max_parts = 4
    elif len(selected) == 1:
        max_parts = 5

    for token_group in (
        base_specific_tokens,
        pattern_specific_tokens,
        base_tokens,
        pattern_tokens,
    ):
        for token in token_group:
            if token in covered_tokens:
                continue
            _append_part(token)
            if len(selected) >= max_parts:
                break
        if len(selected) >= max_parts:
            break

    fallback_query = _trim_jump_query_connector_tail(" ".join(selected[:6]))
    return fallback_query or raw_query


def _is_acceptable_llm_jump_query(
    query: str,
    pattern: dict,
    source_domain: str,
    source_category: str,
    heuristic_query: str,
) -> bool:
    raw_candidate = str(query or "").strip()
    if not raw_candidate or "\n" in raw_candidate or "\r" in raw_candidate:
        return False

    candidate = re.sub(r"\s+", " ", raw_candidate).strip(" .,!?")
    if len(candidate) > 96:
        return False
    if "http://" in candidate.lower() or "https://" in candidate.lower():
        return False
    if any(char in candidate for char in ('{', '}', '[', ']', ':', ';', '"', "`", "|")):
        return False
    if re.search(r"[()]", candidate):
        return False

    blocked_tokens = set(_tokenize_query_terms(source_domain))
    blocked_tokens.update(_tokenize_query_terms(source_category))

    candidate_tokens = _tokenize_query_terms(candidate)
    if len(candidate_tokens) < 3 or len(candidate_tokens) > 12:
        return False
    if any(token in blocked_tokens for token in candidate_tokens):
        return False
    if _looks_like_formal_jump_query_token_soup(raw_candidate, candidate_tokens):
        return False
    if _has_source_surface_jump_query_overhang(
        candidate_tokens,
        pattern,
        source_domain,
        source_category,
    ):
        return False
    unsupported_tokens = _unsupported_llm_jump_query_tokens(
        candidate_tokens,
        pattern,
        source_domain,
        source_category,
        heuristic_query,
    )

    strong_tokens = [
        token
        for token in candidate_tokens
        if token not in GENERIC_QUERY_TOKENS and token not in WEAK_QUERY_TOKENS
    ]
    natural_language_query = _is_compact_natural_language_jump_query(candidate, blocked_tokens)
    if len(strong_tokens) < 3:
        return False
    if not any(_is_specific_jump_query_token(token) for token in strong_tokens):
        return False

    preferred_anchor_phrases = _preferred_jump_query_anchor_phrases(pattern, blocked_tokens)
    lowered_candidate = candidate.lower()
    matched_preferred_anchor_phrase = any(
        phrase in lowered_candidate for phrase in preferred_anchor_phrases
    )
    if preferred_anchor_phrases:
        if matched_preferred_anchor_phrase:
            if not _has_strong_jump_query_support_beyond_preferred_phrase(
                candidate,
                preferred_anchor_phrases,
                blocked_tokens,
            ):
                return False
        else:
            if _needs_jump_query_disambiguation(candidate, preferred_anchor_phrases):
                return False
    elif _needs_jump_query_disambiguation(candidate, preferred_anchor_phrases):
        return False

    anchor_texts = [
        str(pattern.get("control_lever", "") or ""),
        str(pattern.get("abstract_structure", "") or ""),
        str(pattern.get("measurable_signal", "") or ""),
        str(pattern.get("pattern_name", "") or ""),
        str(pattern.get("transfer_rationale", "") or ""),
        heuristic_query,
    ]
    anchor_tokens: set[str] = set()
    anchor_phrases: list[str] = []
    for text in anchor_texts:
        anchor_phrases.extend(_extract_jump_query_phrases(text, blocked_tokens))
        for token in _tokenize_query_terms(text):
            if (
                token in blocked_tokens
                or token in GENERIC_QUERY_TOKENS
                or token in WEAK_QUERY_TOKENS
                or len(token) <= 2
            ):
                continue
            if _is_specific_jump_query_token(token):
                anchor_tokens.add(token)

    if not anchor_tokens and not anchor_phrases:
        return False

    candidate_token_set = set(candidate_tokens)
    anchor_token_overlap = candidate_token_set.intersection(anchor_tokens)
    if unsupported_tokens:
        return False

    clause_score = _score_jump_query_clause(candidate, blocked_tokens)
    if len(anchor_token_overlap) >= 2:
        return True

    if any(phrase in lowered_candidate for phrase in anchor_phrases):
        return (
            not unsupported_tokens
            and natural_language_query
            and clause_score[1] >= 2
            and clause_score[2] >= 3
        )

    return (
        not unsupported_tokens
        and natural_language_query
        and clause_score[1] >= 2
        and clause_score[2] >= 3
    )


def _preserve_jump_query_causal_shape(
    original_query: str,
    rebuilt_query: str,
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> str:
    clean_original = re.sub(r"\s+", " ", str(original_query or "").strip())
    clean_rebuilt = re.sub(r"\s+", " ", str(rebuilt_query or "").strip())
    if not clean_original or not clean_rebuilt or clean_original == clean_rebuilt:
        return clean_rebuilt or clean_original

    blocked_tokens = set(_tokenize_query_terms(source_domain))
    blocked_tokens.update(_tokenize_query_terms(source_category))
    rebuilt_is_natural_language = _is_compact_natural_language_jump_query(
        clean_rebuilt,
        blocked_tokens,
    )
    rebuilt_is_keyword_query = _is_compact_keyword_jump_query(
        clean_rebuilt,
        blocked_tokens,
    )
    original_has_source_overhang = _has_source_surface_jump_query_overhang(
        _tokenize_query_terms(clean_original),
        pattern,
        source_domain,
        source_category,
    )
    if not _is_compact_natural_language_jump_query(clean_original, blocked_tokens):
        return clean_rebuilt
    if rebuilt_is_natural_language or (
        rebuilt_is_keyword_query and original_has_source_overhang
    ):
        return clean_rebuilt

    preferred_anchor_phrases = _preferred_jump_query_anchor_phrases(pattern, blocked_tokens)
    original_has_phrase, original_concrete_tokens = _jump_query_anchor_support(
        clean_original,
        preferred_anchor_phrases,
        blocked_tokens,
    )
    if not original_has_phrase and len(original_concrete_tokens) < 2:
        return clean_rebuilt

    if not rebuilt_is_natural_language and not (
        rebuilt_is_keyword_query and original_has_source_overhang
    ):
        return clean_original

    original_score = _score_jump_query_clause(clean_original, blocked_tokens)
    rebuilt_score = _score_jump_query_clause(clean_rebuilt, blocked_tokens)
    if original_score >= rebuilt_score:
        return clean_original
    return clean_rebuilt


def _generate_llm_jump_search_query(
    pattern: dict,
    source_domain: str,
    source_category: str,
    heuristic_query: str,
) -> str | None:
    prompt = JUMP_QUERY_PROMPT.format(
        source_domain=str(source_domain or "").strip() or "Unknown",
        source_category=str(source_category or "").strip() or "Unknown",
        pattern_name=str(pattern.get("pattern_name", "") or "").strip() or "Unknown",
        abstract_structure=str(pattern.get("abstract_structure", "") or "").strip() or "Unknown",
        measurable_signal=str(pattern.get("measurable_signal", "") or "").strip() or "Unknown",
        control_lever=str(pattern.get("control_lever", "") or "").strip() or "Unknown",
        transfer_rationale=str(pattern.get("transfer_rationale", "") or "").strip() or "Unknown",
        heuristic_query=heuristic_query,
    )
    extracted_json = _generate_json_with_retry(
        prompt,
        "jump_query_builder",
        256,
    )
    if extracted_json is None:
        return None

    try:
        payload = json.loads(extracted_json)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    query = str(payload.get("query", "") or "").strip()
    if not _is_acceptable_llm_jump_query(
        query,
        pattern,
        source_domain,
        source_category,
        heuristic_query,
    ):
        return None
    return re.sub(r"\s+", " ", query).strip(" .,!?")


def _build_jump_search_query(
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> str:
    query_build_result = _coerce_jump_search_query_metadata_result(
        _build_jump_search_query_with_metadata(
            pattern,
            source_domain,
            source_category,
        )
    )
    return query_build_result.built_jump_query


def _default_jump_transferable_query_profile() -> dict[str, object]:
    return {
        "usable": False,
        "backfilled": False,
        "backfilled_fields": [],
        "has_transferable_fields": False,
        "concerns": [],
        "source_leakage_terms": [],
        "overlap_pairs": [],
        "fields": {},
    }


def _coerce_jump_search_query_metadata_result(
    query_build_output: JumpSearchQueryMetadataResult | tuple[str, bool],
) -> JumpSearchQueryMetadataResult:
    if isinstance(query_build_output, JumpSearchQueryMetadataResult):
        return JumpSearchQueryMetadataResult(
            built_jump_query=str(query_build_output.built_jump_query or "").strip(),
            legacy_built_jump_query=str(
                query_build_output.legacy_built_jump_query or ""
            ).strip(),
            transferable_query_profile=dict(
                query_build_output.transferable_query_profile or {}
            ),
            query_collision_guard_applied=bool(
                query_build_output.query_collision_guard_applied
            ),
        )

    built_jump_query, query_collision_guard_applied = query_build_output
    return JumpSearchQueryMetadataResult(
        built_jump_query=str(built_jump_query or "").strip(),
        legacy_built_jump_query="",
        transferable_query_profile=_default_jump_transferable_query_profile(),
        query_collision_guard_applied=bool(query_collision_guard_applied),
    )


def _build_jump_search_query_with_metadata(
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> JumpSearchQueryMetadataResult:
    raw_query = str(pattern.get("search_query", "") or "").strip()
    if not raw_query:
        return JumpSearchQueryMetadataResult(
            built_jump_query="",
            legacy_built_jump_query="",
            transferable_query_profile=_jump_transferable_query_profile(
                pattern,
                source_domain,
                source_category,
            ),
            query_collision_guard_applied=False,
        )

    query_pattern, transferable_profile = _jump_query_pattern_view(
        pattern,
        source_domain,
        source_category,
    )

    heuristic_query = _build_jump_search_query_heuristic(
        query_pattern,
        source_domain,
        source_category,
        prefer_transferable=False,
    )
    heuristic_query = _disambiguate_jump_search_query(
        heuristic_query,
        query_pattern,
        source_domain,
        source_category,
    )
    heuristic_query, heuristic_collision_guard_applied = _apply_jump_query_collision_guard(
        heuristic_query,
        query_pattern,
        source_domain,
        source_category,
    )
    heuristic_query = _preserve_jump_query_causal_shape(
        raw_query,
        heuristic_query,
        query_pattern,
        source_domain,
        source_category,
    )

    legacy_query = _build_jump_search_query_heuristic(
        _jump_legacy_flat_pattern(pattern),
        source_domain,
        source_category,
        prefer_transferable=False,
    )
    legacy_query = _disambiguate_jump_search_query(
        legacy_query,
        _jump_legacy_flat_pattern(pattern),
        source_domain,
        source_category,
    )
    legacy_query, _legacy_collision_guard_applied = _apply_jump_query_collision_guard(
        legacy_query,
        _jump_legacy_flat_pattern(pattern),
        source_domain,
        source_category,
    )
    legacy_query = _preserve_jump_query_causal_shape(
        raw_query,
        legacy_query,
        _jump_legacy_flat_pattern(pattern),
        source_domain,
        source_category,
    )

    llm_query = _generate_llm_jump_search_query(
        query_pattern,
        source_domain,
        source_category,
        heuristic_query,
    )
    original_llm_query = None
    llm_query_adjusted = False
    if llm_query:
        original_llm_query = llm_query
        llm_query = _disambiguate_jump_search_query(
            llm_query,
            query_pattern,
            source_domain,
            source_category,
        )
        llm_query_adjusted = llm_query != original_llm_query
    llm_collision_guard_applied = False
    if llm_query:
        llm_query, llm_collision_guard_applied = _apply_jump_query_collision_guard(
            llm_query,
            query_pattern,
            source_domain,
            source_category,
        )
        llm_query_adjusted = (
            llm_query_adjusted
            or llm_collision_guard_applied
            or llm_query != original_llm_query
        )
        llm_query = _preserve_jump_query_causal_shape(
            original_llm_query,
            llm_query,
            query_pattern,
            source_domain,
            source_category,
        )
        if not _is_acceptable_llm_jump_query(
            llm_query,
            query_pattern,
            source_domain,
            source_category,
            heuristic_query,
        ):
            llm_query_adjusted = True
            llm_query = None
    return JumpSearchQueryMetadataResult(
        built_jump_query=llm_query or heuristic_query,
        legacy_built_jump_query=legacy_query,
        transferable_query_profile=transferable_profile,
        query_collision_guard_applied=(
            heuristic_collision_guard_applied
            or llm_collision_guard_applied
            or llm_query_adjusted
        ),
    )


def _build_jump_query_build_result(
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> JumpQueryBuildResult:
    query_pattern, _transferable_profile = _jump_query_pattern_view(
        pattern,
        source_domain,
        source_category,
    )
    raw_source_query = str(pattern.get("search_query", "") or "").strip()

    def _family_support_terms(*texts: str) -> list[str]:
        blocked_tokens, _preferred_anchor_phrases, _support_tokens = _jump_query_support_context(
            pattern,
            source_domain,
            source_category,
        )
        support_token_set = set(_support_tokens)
        terms: list[str] = []
        seen_terms: set[str] = set()
        base_tokens = set(_tokenize_query_terms(base_query))
        for text in texts:
            clean_text = str(text or "").strip()
            if not clean_text:
                continue
            for phrase in _extract_jump_query_phrases(clean_text, blocked_tokens):
                normalized_phrase = re.sub(r"\s+", " ", phrase).strip()
                phrase_tokens = _tokenize_query_terms(normalized_phrase)
                unseen_tokens = [
                    token for token in phrase_tokens if token not in base_tokens
                ]
                if (
                    not phrase_tokens
                    or not unseen_tokens
                    or all(
                        token in GENERIC_QUERY_TOKENS
                        or token in WEAK_QUERY_TOKENS
                        or token in QUERY_PHRASE_STOPWORDS
                        or len(token) <= 2
                        for token in unseen_tokens
                    )
                    or normalized_phrase in seen_terms
                ):
                    continue
                seen_terms.add(normalized_phrase)
                terms.append(normalized_phrase)
            for token in _tokenize_query_terms(clean_text):
                if (
                    token in seen_terms
                    or token in base_tokens
                    or token in blocked_tokens
                    or token in GENERIC_QUERY_TOKENS
                    or token in WEAK_QUERY_TOKENS
                    or token in OVERLOADED_JUMP_QUERY_TOKENS
                    or len(token) <= 2
                    or not (
                        _is_specific_jump_query_token(token)
                        or _is_causal_jump_query_token(token)
                    )
                    or (
                        clean_text == raw_source_query
                        and token not in support_token_set
                        and token not in MECHANISM_QUERY_TOKENS
                        and not _is_causal_jump_query_token(token)
                    )
                ):
                    continue
                seen_terms.add(token)
                terms.append(token)
        return terms

    def _needs_mechanism_family_support(current_query: str) -> bool:
        blocked_tokens = set(_tokenize_query_terms(source_domain))
        blocked_tokens.update(_tokenize_query_terms(source_category))
        clause_score = _score_jump_query_clause(current_query, blocked_tokens)
        concrete_tokens = [
            token
            for token in _tokenize_query_terms(current_query)
            if (
                token not in blocked_tokens
                and token not in GENERIC_QUERY_TOKENS
                and token not in WEAK_QUERY_TOKENS
                and token not in OVERLOADED_JUMP_QUERY_TOKENS
                and token not in JUMP_QUERY_FILLER_TOKENS
                and (
                    _is_concrete_jump_query_token(token)
                    or _is_causal_jump_query_token(token)
                    or token in MECHANISM_QUERY_TOKENS
                    or token in JUMP_QUERY_CAUSAL_OUTCOME_HINTS
                )
            )
        ]
        return clause_score[1] < 2 or clause_score[2] < 3 or len(concrete_tokens) < 3

    def _build_family_variant(
        family_terms: list[str],
        fallback_terms: tuple[str, ...],
    ) -> str:
        base_tokens = _tokenize_query_terms(base_query)
        selected = [base_query]
        covered_tokens = set(base_tokens)
        remaining_slots = max(0, 12 - len(base_tokens))
        selected_family_term = False
        for term in family_terms:
            term_tokens = _tokenize_query_terms(term)
            unseen_tokens = [
                token for token in term_tokens if token not in covered_tokens
            ]
            if not unseen_tokens or remaining_slots <= 0:
                continue
            if len(unseen_tokens) > remaining_slots:
                continue
            selected.append(term)
            covered_tokens.update(unseen_tokens)
            remaining_slots -= len(unseen_tokens)
            selected_family_term = True
            break
        for term in fallback_terms:
            term_tokens = _tokenize_query_terms(term)
            unseen_tokens = [
                token for token in term_tokens if token not in covered_tokens
            ]
            if not unseen_tokens or len(unseen_tokens) > remaining_slots:
                continue
            selected.append(term)
            covered_tokens.update(unseen_tokens)
            break
        return re.sub(r"\s+", " ", " ".join(selected)).strip()

    query_metadata_result = _coerce_jump_search_query_metadata_result(
        _build_jump_search_query_with_metadata(
            pattern,
            source_domain,
            source_category,
        )
    )
    query = query_metadata_result.built_jump_query
    collision_guard_applied = query_metadata_result.query_collision_guard_applied
    legacy_built_jump_query = query_metadata_result.legacy_built_jump_query
    transferable_query_profile = dict(
        query_metadata_result.transferable_query_profile
    )
    if not query:
        return JumpQueryBuildResult(
            built_jump_query="",
            built_jump_queries=[],
            built_jump_query_labels=[],
            legacy_built_jump_query=legacy_built_jump_query,
            transferable_query_profile=transferable_query_profile,
            query_collision_guard_applied=collision_guard_applied,
        )

    intervention_terms = (
        "workaround",
        "mitigation",
        "correction",
        "bypass",
        "compensation",
        "solution",
    )
    base_query = re.sub(r"\s+", " ", query).strip()
    query_tokens = set(_tokenize_query_terms(base_query))
    intervention_term = next(
        (term for term in intervention_terms if term not in query_tokens),
        intervention_terms[0],
    )
    mechanism_query = base_query
    if _needs_mechanism_family_support(base_query):
        mechanism_query = _build_family_variant(
            _family_support_terms(
                str(query_pattern.get("abstract_structure", "") or ""),
                str(query_pattern.get("pattern_name", "") or ""),
                raw_source_query,
            ),
            ("mechanism",),
        )
    intervention_query = _build_family_variant(
        _family_support_terms(
            str(query_pattern.get("control_lever", "") or ""),
            str(query_pattern.get("transfer_rationale", "") or ""),
        ),
        (intervention_term,),
    )
    operator_query = _build_family_variant(
        _family_support_terms(
            str(query_pattern.get("measurable_signal", "") or ""),
            str(query_pattern.get("search_query", "") or ""),
            str(query_pattern.get("abstract_structure", "") or ""),
        ),
        ("failure",),
    )
    query_entries = [
        ("mechanism-family", mechanism_query),
        ("intervention-family", intervention_query),
        ("operator-family", operator_query),
    ]
    queries: list[str] = []
    query_labels: list[str] = []
    seen_queries: set[str] = set()
    for label, built_query in query_entries:
        normalized_query = re.sub(r"\s+", " ", str(built_query or "").strip())
        if not normalized_query or normalized_query in seen_queries:
            continue
        seen_queries.add(normalized_query)
        queries.append(normalized_query)
        query_labels.append(label)
    return JumpQueryBuildResult(
        built_jump_query=queries[0] if queries else "",
        built_jump_queries=queries,
        built_jump_query_labels=query_labels,
        legacy_built_jump_query=legacy_built_jump_query,
        transferable_query_profile=transferable_query_profile,
        query_collision_guard_applied=collision_guard_applied,
    )


def _build_jump_search_queries(
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> list[str]:
    query_build_result = _build_jump_query_build_result(
        pattern,
        source_domain,
        source_category,
    )
    return list(query_build_result.built_jump_queries)


def _build_pre_stage1_jump_query_build_result(
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> JumpQueryBuildResult:
    legacy_builder = globals().get("_build_jump_search_queries")
    if (
        callable(legacy_builder)
        and legacy_builder is not _ORIGINAL_BUILD_JUMP_SEARCH_QUERIES
    ):
        return legacy_builder(
            pattern,
            source_domain,
            source_category,
        )
    return _build_jump_query_build_result(
        pattern,
        source_domain,
        source_category,
    )

_ORIGINAL_BUILD_JUMP_SEARCH_QUERIES = _build_jump_search_queries

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
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass
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


def _generate_json_with_retry(
    full_prompt: str,
    stage: str,
    max_output_tokens: int,
    cycle_budget: CycleBudget | None = None,
    budget_outcome: str | None = None,
) -> str | None:
    """Generate JSON with one retry if parsing fails."""
    env_key = {
        "stage1_detect": "BLACKCLAW_JUMP_STAGE1_MAX_OUTPUT_TOKENS",
        "stage2_hypothesize": "BLACKCLAW_JUMP_STAGE2_MAX_OUTPUT_TOKENS",
        "stage2_mechanism": "BLACKCLAW_JUMP_STAGE2_MAX_OUTPUT_TOKENS",
        "stage2_predict": "BLACKCLAW_JUMP_STAGE2_MAX_OUTPUT_TOKENS",
        "stage2_test": "BLACKCLAW_JUMP_STAGE2_MAX_OUTPUT_TOKENS",
        "stage2_edge": "BLACKCLAW_JUMP_STAGE2_MAX_OUTPUT_TOKENS",
    }.get(str(stage or "").strip())
    effective_max_output_tokens = max_output_tokens
    if env_key:
        raw_override = str(os.getenv(env_key, "")).strip()
        if raw_override:
            try:
                parsed_override = int(raw_override)
            except ValueError:
                parsed_override = max_output_tokens
            if parsed_override > 0:
                effective_max_output_tokens = parsed_override
    disable_retry = str(
        os.getenv("BLACKCLAW_JUMP_DISABLE_JSON_RETRY", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    try:
        if cycle_budget is not None and budget_outcome:
            cycle_budget.consume_llm(
                outcome=budget_outcome,
                callsite=str(stage or "").strip() or "stage1_detect",
            )
        response = _llm_client.generate_content(
            full_prompt,
            generation_config={
                "max_output_tokens": effective_max_output_tokens,
                "response_mime_type": "application/json",
            },
        )
        log_gemini_output("jump", f"{stage}_initial", response)
        increment_llm_calls(1)
        raw_output = response.text if getattr(response, "text", None) else ""
        checked = check_llm_output(raw_output)
        if checked is None:
            print(f"  [!] Jump {stage} output failed safety check")
            return None
        extracted = _extract_json_substring(checked)
        if extracted is not None:
            return extracted
        if disable_retry:
            return None

        retry_prompt = f"{JSON_RETRY_PROMPT}\n\n{full_prompt}"
        if cycle_budget is not None and budget_outcome:
            cycle_budget.consume_llm(
                outcome=budget_outcome,
                callsite=str(stage or "").strip() or "stage1_detect",
            )
        retry_response = _llm_client.generate_content(
            retry_prompt,
            generation_config={
                "max_output_tokens": effective_max_output_tokens,
                "response_mime_type": "application/json",
            },
        )
        log_gemini_output("jump", f"{stage}_retry", retry_response)
        increment_llm_calls(1)
        retry_raw = retry_response.text if getattr(retry_response, "text", None) else ""
        retry_checked = check_llm_output(retry_raw)
        if retry_checked is None:
            print(f"  [!] Jump {stage} retry output failed safety check")
            return None
        return _extract_json_substring(retry_checked)
    except CycleBudgetExhausted:
        raise
    except Exception as e:
        print(f"  [!] Jump {stage} LLM call failed: {e}")
        return None


def _apply_normalized_mechanism_typing(data: dict) -> dict:
    """Copy normalized mechanism typing back onto the candidate payload."""
    normalized = normalize_mechanism_typing(data)
    out = dict(data)
    out["mechanism_typing"] = normalized
    out["mechanism_type"] = normalized.get("mechanism_type")
    out["mechanism_type_confidence"] = normalized.get(
        "mechanism_type_confidence"
    )
    out["secondary_mechanism_types"] = normalized.get(
        "secondary_mechanism_types", []
    )
    return out


REPAIR_TERM_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "via",
    "with",
}


def _repair_term_list(text: object) -> list[str]:
    cleaned = str(text or "").lower()
    terms: list[str] = []
    for token in re.findall(r"[a-z0-9]+(?:[_/\-][a-z0-9]+)*", cleaned):
        for part in re.split(r"[_/\-]+", token):
            if len(part) < 4 or part.isdigit() or part in REPAIR_TERM_STOPWORDS:
                continue
            if part not in terms:
                terms.append(part)
    return terms


def _repair_term_variants(token: str) -> set[str]:
    variants = {token}
    if len(token) < 5 or token.isdigit():
        return variants

    if token.endswith("ies") and len(token) > 6:
        variants.add(token[:-3] + "y")

    for suffix in ("ing", "ed", "ions", "ion", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            variants.add(token[: -len(suffix)])
            break

    return {
        value
        for value in variants
        if len(value) >= 4 and not value.isdigit()
    }


def _repair_term_set(text: object) -> set[str]:
    terms: set[str] = set()
    for token in _repair_term_list(text):
        terms.update(_repair_term_variants(token))
    return terms


MECHANISM_ANCHOR_PROCESS_TERMS = {
    "analysis",
    "accumulat",
    "accumulation",
    "anneal",
    "annealing",
    "assign",
    "assignment",
    "avoidance",
    "channel",
    "channels",
    "collision",
    "compar",
    "compare",
    "compares",
    "consolidation",
    "count",
    "counts",
    "cutoff",
    "dependency",
    "dependencies",
    "detect",
    "detection",
    "enforc",
    "enforce",
    "enforces",
    "excitability",
    "failure",
    "failures",
    "filter",
    "filtering",
    "gating",
    "heat",
    "heating",
    "hyperperiod",
    "inhibit",
    "inhibition",
    "memory",
    "monitor",
    "plasticity",
    "prevent",
    "prevents",
    "rate",
    "relax",
    "relaxation",
    "refill",
    "residual",
    "redistribution",
    "route",
    "routing",
    "sample",
    "sampling",
    "saturation",
    "slot",
    "subsystem",
    "subsystems",
    "synaptic",
    "switch",
    "switching",
    "threshold",
    "trigger",
    "triggers",
}

PROCESS_PHRASE_LEAD_INS = (
    "one kind of ",
    "a kind of ",
    "one form of ",
    "a form of ",
    "the process of ",
    "a process of ",
    "process of ",
)

MECHANISM_REPORTING_PREFIX_PATTERNS = (
    r"^(?:[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)?\s+\d{4}[a-z]?\s+)"
    r"(?:directly\s+)?(?:demonstrates|shows|finds|reports|establishes|identifies|describes|documents|indicates)\s+that\s+",
    r"^(?:the|this)\s+(?:paper|study|article|report|pdf|work)\s+"
    r"(?:directly\s+)?(?:demonstrates|shows|finds|reports|establishes|identifies|describes|documents|indicates)\s+that\s+",
    r"^(?:retrieved|target)\s+(?:evidence|literature|paper|study|pdf|report)\s+"
    r"(?:directly\s+)?(?:demonstrates|shows|finds|reports|establishes|identifies|describes|documents|indicates)\s+that\s+",
)

PROCESS_PHRASE_BREAK_WORDS = (
    "which",
    "that",
    "when",
    "where",
    "while",
    "must",
    "may",
    "can",
)

GERUND_TO_FINITE_CONNECTORS = {
    "allowing": "allows",
    "amplifying": "amplifies",
    "causing": "causes",
    "comparing": "compares",
    "confining": "confines",
    "controlling": "controls",
    "converting": "converts",
    "counting": "counts",
    "detecting": "detects",
    "determining": "determines",
    "driving": "drives",
    "enabling": "enables",
    "generating": "generates",
    "governing": "governs",
    "inducing": "induces",
    "inhibiting": "inhibits",
    "limiting": "limits",
    "mediating": "mediates",
    "modulating": "modulates",
    "producing": "produces",
    "propagating": "propagates",
    "regulating": "regulates",
    "requiring": "requires",
    "suppressing": "suppresses",
    "transferring": "transfers",
    "triggering": "triggers",
}

BARE_TO_FINITE_CONNECTORS = {
    "allow": "allows",
    "amplify": "amplifies",
    "cause": "causes",
    "compare": "compares",
    "constrain": "constrains",
    "control": "controls",
    "convert": "converts",
    "count": "counts",
    "detect": "detects",
    "determine": "determines",
    "drive": "drives",
    "enable": "enables",
    "generate": "generates",
    "govern": "governs",
    "induce": "induces",
    "inhibit": "inhibits",
    "limit": "limits",
    "mediate": "mediates",
    "modulate": "modulates",
    "produce": "produces",
    "propagate": "propagates",
    "regulate": "regulates",
    "require": "requires",
    "suppress": "suppresses",
    "transfer": "transfers",
    "trigger": "triggers",
}


def _process_anchor_score(text: object) -> int:
    cleaned = str(text or "").strip()
    if not cleaned:
        return 0

    lowered = cleaned.lower()
    if any(phrase in lowered for phrase in GENERIC_MECHANISM_FILLERS):
        return 0

    terms = _repair_term_set(cleaned)
    if len(terms) < 3:
        return 0

    process_hits = len(terms & MECHANISM_ANCHOR_PROCESS_TERMS)
    if process_hits == 0:
        return 0

    score = (process_hits * 3) + min(len(terms), 6)
    if re.search(
        r"\b(?:against|across|before|during|under|within|through|via)\b",
        lowered,
    ):
        score += 1
    if len(cleaned.split()) >= 8:
        score += 1
    return score


def _mechanism_anchor_rank(candidate: dict) -> tuple[int, int, float, int]:
    return (
        int(candidate.get("process_score") or 0),
        int(candidate.get("source_priority") or 0),
        float(candidate.get("provenance_score") or 0.0),
        len(str(candidate.get("text") or "")),
    )


def _phase3_repair_context(data: dict | None) -> dict:
    payload = data if isinstance(data, dict) else {}
    evidence_map = normalize_evidence_map(payload.get("evidence_map"))
    provenance = summarize_evidence_map_provenance(
        {
            **payload,
            "evidence_map": evidence_map,
        }
    )

    mechanism_candidates: list[dict] = []

    def _add_candidate(
        *,
        text: object,
        snippet: object,
        source_reference: object,
        provenance_score: float,
        source_priority: int,
        require_process_level: bool,
    ) -> None:
        clean_text = str(text or "").strip()
        if not clean_text:
            return
        process_score = _process_anchor_score(clean_text)
        if require_process_level and process_score <= 0:
            return
        mechanism_candidates.append(
            {
                "text": clean_text,
                "snippet": str(snippet or "").strip(),
                "source_reference": str(source_reference or "").strip(),
                "provenance_score": provenance_score,
                "process_score": process_score,
                "source_priority": source_priority,
            }
        )

    core_target_anchor = (
        provenance.get("best_core_target_evidence")
        if isinstance(provenance.get("best_core_target_evidence"), dict)
        else {}
    )
    if core_target_anchor:
        score = float(
            (((core_target_anchor.get("provenance_score") or {}).get("overall")) or 0.0)
        )
        _add_candidate(
            text=core_target_anchor.get("evidence_snippet"),
            snippet=core_target_anchor.get("evidence_snippet"),
            source_reference=core_target_anchor.get("source_reference"),
            provenance_score=score,
            source_priority=5,
            require_process_level=True,
        )

    for entry in provenance.get("scored_mechanism_assertions") or []:
        if not isinstance(entry, dict):
            continue
        score = float(((entry.get("provenance_score") or {}).get("overall")) or 0.0)
        _add_candidate(
            text=entry.get("mechanism_claim"),
            snippet=entry.get("evidence_snippet"),
            source_reference=entry.get("source_reference"),
            provenance_score=score,
            source_priority=4,
            require_process_level=False,
        )
        _add_candidate(
            text=entry.get("evidence_snippet"),
            snippet=entry.get("evidence_snippet"),
            source_reference=entry.get("source_reference"),
            provenance_score=score,
            source_priority=3,
            require_process_level=False,
        )
    for entry in evidence_map.get("variable_mappings", [])[:3]:
        if not isinstance(entry, dict):
            continue
        _add_candidate(
            text=entry.get("claim"),
            snippet=entry.get("evidence_snippet"),
            source_reference=entry.get("source_reference"),
            provenance_score=0.0,
            source_priority=2,
            require_process_level=True,
        )
        _add_candidate(
            text=entry.get("evidence_snippet"),
            snippet=entry.get("evidence_snippet"),
            source_reference=entry.get("source_reference"),
            provenance_score=0.0,
            source_priority=1,
            require_process_level=True,
        )
    if str(payload.get("evidence") or "").strip():
        _add_candidate(
            text=payload.get("evidence"),
            snippet="",
            source_reference="",
            provenance_score=0.0,
            source_priority=0,
            require_process_level=False,
        )

    mechanism_anchor = max(
        (
            candidate
            for candidate in mechanism_candidates
            if candidate.get("text")
        ),
        key=_mechanism_anchor_rank,
        default=None,
    )

    critical_failures: list[dict] = []
    for detail in provenance.get("failure_details") or []:
        if not isinstance(detail, dict):
            continue
        if str(detail.get("kind") or "").strip() != "variable_mapping":
            continue
        source_variable = str(detail.get("source_variable") or "").strip()
        target_variable = str(detail.get("target_variable") or "").strip()
        pair = " -> ".join(part for part in (source_variable, target_variable) if part)
        critical_failures.append(
            {
                "pair": pair,
                "reason_codes": [
                    str(code).strip()
                    for code in (detail.get("reason_codes") or [])
                    if str(code).strip()
                ],
                "claim": str(detail.get("claim") or "").strip(),
                "evidence_snippet": str(detail.get("evidence_snippet") or "").strip(),
                "source_reference": str(detail.get("source_reference") or "").strip(),
            }
        )

    missing_critical_pairs = []
    for item in provenance.get("missing_critical_mappings") or []:
        if not isinstance(item, dict):
            continue
        source_variable = str(item.get("source_variable") or "").strip()
        target_variable = str(item.get("target_variable") or "").strip()
        pair = " -> ".join(part for part in (source_variable, target_variable) if part)
        if pair:
            missing_critical_pairs.append(pair)

    mechanism_text = str(payload.get("mechanism") or "").strip()
    mechanism_terms = set()
    for token in _repair_term_list(mechanism_text)[:5]:
        mechanism_terms.update(_repair_term_variants(token))
    mechanism_overlap = 0
    for candidate in mechanism_candidates:
        candidate_terms = _repair_term_set(
            " ".join(
                part
                for part in (
                    candidate.get("text"),
                    candidate.get("snippet"),
                    candidate.get("source_reference"),
                )
                if part
            )
        )
        mechanism_overlap = max(
            mechanism_overlap,
            len(mechanism_terms & candidate_terms),
        )

    return {
        "provenance": provenance,
        "mechanism_anchor": mechanism_anchor,
        "core_target_anchor": core_target_anchor,
        "core_target_evidence_strength": str(
            provenance.get("core_target_evidence_strength") or ""
        ).strip(),
        "core_target_reasons": [
            str(reason).strip()
            for reason in (provenance.get("core_target_reasons") or [])
            if str(reason).strip()
        ],
        "critical_failures": critical_failures,
        "missing_critical_pairs": missing_critical_pairs,
        "mechanism_overlap": mechanism_overlap,
    }


def _display_target_evidence_rank(summary: dict | None) -> tuple[float, ...]:
    payload = summary if isinstance(summary, dict) else {}
    best_entry = (
        payload.get("best_core_target_evidence")
        if isinstance(payload.get("best_core_target_evidence"), dict)
        else {}
    )
    provenance_score = (
        best_entry.get("provenance_score")
        if isinstance(best_entry.get("provenance_score"), dict)
        else {}
    )
    evidence_snippet = str(best_entry.get("evidence_snippet") or "").strip()
    source_reference = str(best_entry.get("source_reference") or "").strip()
    snippet_word_count = len(evidence_snippet.split())
    concise_snippet = 1.0 if 8 <= snippet_word_count <= 28 else 0.0
    suspicious_reference = 0.0
    lowered_reference = source_reference.lower()
    if any(
        marker in lowered_reference
        for marker in (
            "blog",
            "guide",
            "overview",
            "introduction",
            "magazine",
            "marketing",
            "vendor",
            "company",
            "product",
            "solution",
        )
    ):
        suspicious_reference = 1.0
    strength_rank = {
        "strong_direct": 3.0,
        "weak_direct": 2.0,
        "contextual_only": 1.0,
        "none": 0.0,
    }.get(str(best_entry.get("strength") or "").strip(), 0.0)
    entry_type_rank = {
        "mechanism_assertion": 2.0,
        "variable_mapping": 1.0,
        "top_level_target_evidence": 0.0,
    }.get(str(best_entry.get("entry_type") or "").strip(), 0.0)
    return (
        strength_rank,
        1.0 if best_entry.get("direct_process_match") else 0.0,
        1.0 if best_entry.get("direct_metric_match") else 0.0,
        float(best_entry.get("mechanism_overlap") or 0.0),
        float(best_entry.get("metric_overlap") or 0.0),
        float(best_entry.get("observable_overlap") or 0.0),
        1.0 if not best_entry.get("weak_source") else 0.0,
        1.0 if not best_entry.get("broad_page") else 0.0,
        1.0 if not best_entry.get("generic_signal") else 0.0,
        1.0 if not best_entry.get("mismatch_signal") else 0.0,
        1.0 if not suspicious_reference else 0.0,
        concise_snippet,
        float(provenance_score.get("snippet_specificity") or 0.0),
        float(provenance_score.get("overall") or 0.0),
        entry_type_rank,
    )


def _display_source_evidence_rank(
    text: object,
    *,
    source_terms: set[str],
    mapped_source_terms: set[str],
    field_priority: int,
) -> tuple[int, int, int, int, int]:
    candidate_terms = _repair_term_set(text)
    candidate_text = re.sub(r"\s+", " ", str(text or "")).strip()
    return (
        len(candidate_terms & mapped_source_terms),
        len(candidate_terms & source_terms),
        min(len(candidate_text.split()), 24),
        field_priority,
        len(candidate_text),
    )


def _select_display_source_evidence(
    pattern: dict | None,
    data: dict | None,
    source_domain: str,
) -> tuple[str | None, str | None]:
    source_pattern = pattern if isinstance(pattern, dict) else {}
    payload = data if isinstance(data, dict) else {}
    evidence_map = normalize_evidence_map(payload.get("evidence_map"))
    seed_url = str(source_pattern.get("seed_url") or "").strip() or None

    source_terms = _repair_term_set(source_domain)
    source_terms.update(_repair_term_set(source_pattern.get("pattern_name")))
    source_terms.update(_repair_term_set(source_pattern.get("measurable_signal")))
    source_terms.update(_repair_term_set(source_pattern.get("control_lever")))

    mapped_source_terms: set[str] = set()
    for entry in evidence_map.get("variable_mappings", [])[:3]:
        if not isinstance(entry, dict):
            continue
        mapped_source_terms.update(_repair_term_set(entry.get("source_variable")))

    candidates: list[dict] = []
    for field_name, field_priority in (
        ("seed_excerpt", 3),
        ("description", 2),
        ("abstract_structure", 1),
    ):
        candidate_text = re.sub(
            r"\s+",
            " ",
            str(source_pattern.get(field_name) or ""),
        ).strip()
        if not candidate_text:
            continue
        candidates.append(
            {
                "excerpt": candidate_text,
                "url": seed_url,
                "rank": _display_source_evidence_rank(
                    candidate_text,
                    source_terms=source_terms,
                    mapped_source_terms=mapped_source_terms,
                    field_priority=field_priority,
                ),
            }
        )

    if not candidates:
        return seed_url, (
            re.sub(r"\s+", " ", str(source_pattern.get("seed_excerpt") or "")).strip()
            or None
        )

    best_candidate = max(candidates, key=lambda item: item["rank"])
    return best_candidate.get("url"), best_candidate.get("excerpt")


def _select_display_target_evidence(
    data: dict | None,
    raw_target_candidates: list[dict],
) -> tuple[str | None, str | None]:
    payload = data if isinstance(data, dict) else {}
    evidence_map = normalize_evidence_map(payload.get("evidence_map"))
    candidate_pool = raw_target_candidates or [{"target_excerpt": None, "target_url": None}]
    best_selection: dict | None = None

    for candidate in candidate_pool:
        working_payload = dict(payload)
        working_payload["evidence_map"] = evidence_map

        candidate_excerpt = str(candidate.get("target_excerpt") or "").strip() or None
        candidate_url = str(candidate.get("target_url") or "").strip() or None
        evaluation_source_reference = (
            str(candidate.get("evaluation_source_reference") or "").strip()
            or candidate_url
        )
        if candidate_excerpt:
            working_payload["target_excerpt"] = candidate_excerpt
        else:
            working_payload.pop("target_excerpt", None)
        if evaluation_source_reference:
            working_payload["target_url"] = evaluation_source_reference
        else:
            working_payload.pop("target_url", None)

        provenance_summary = summarize_evidence_map_provenance(working_payload)
        rank = _display_target_evidence_rank(provenance_summary)
        best_entry = (
            provenance_summary.get("best_core_target_evidence")
            if isinstance(provenance_summary.get("best_core_target_evidence"), dict)
            else {}
        )

        selected_excerpt = candidate_excerpt
        selected_url = candidate_url
        if str(best_entry.get("entry_type") or "").strip() != "top_level_target_evidence":
            best_excerpt = str(best_entry.get("evidence_snippet") or "").strip()
            best_source = str(best_entry.get("source_reference") or "").strip()
            if best_excerpt:
                selected_excerpt = best_excerpt
            if best_source:
                selected_url = best_source

        if best_selection is None or rank > best_selection["rank"]:
            best_selection = {
                "rank": rank,
                "target_excerpt": selected_excerpt,
                "target_url": selected_url,
            }

    if best_selection is not None:
        return best_selection.get("target_excerpt"), best_selection.get("target_url")

    first_candidate = raw_target_candidates[0] if raw_target_candidates else {}
    fallback_excerpt = str(first_candidate.get("target_excerpt") or "").strip() or None
    fallback_url = str(first_candidate.get("target_url") or "").strip() or None
    return fallback_excerpt, fallback_url


def _mechanism_word_tokens(text: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if token
    }


def _has_mechanism_connector(text: object) -> bool:
    return any(token in PROCESS_CONNECTORS for token in _mechanism_word_tokens(text))


def _strip_mechanism_reporting_prefix(text: object) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n\"'`")
    if not cleaned:
        return ""
    rewritten = cleaned
    while True:
        updated = rewritten
        for pattern in MECHANISM_REPORTING_PREFIX_PATTERNS:
            candidate = re.sub(pattern, "", updated, count=1, flags=re.IGNORECASE).strip()
            if candidate != updated:
                updated = candidate
                break
        if updated == rewritten:
            return rewritten
        rewritten = updated


def _extract_process_phrase(text: object) -> str:
    cleaned = _strip_mechanism_reporting_prefix(text)
    if not cleaned:
        return ""

    lower = cleaned.lower()
    for prefix in PROCESS_PHRASE_LEAD_INS:
        if lower.startswith(prefix):
            cleaned = cleaned[len(prefix):].lstrip()
            lower = cleaned.lower()
            break

    end = len(cleaned)
    for verb in PROCESS_PHRASE_VERBS:
        match = re.search(rf"\b{re.escape(verb)}\b", lower)
        if match is not None and len(cleaned[: match.start()].split()) >= 2:
            end = min(end, match.start())
    for opener in MECHANISM_BRIDGE_OPENERS:
        match = re.search(rf"\b{re.escape(opener)}\b", lower)
        if match is not None:
            end = min(end, match.start())
    for word in PROCESS_PHRASE_BREAK_WORDS:
        match = re.search(rf"\b{re.escape(word)}\b", lower)
        if match is not None:
            end = min(end, match.start())

    phrase = cleaned[:end].strip(" ,.;:-")
    phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
    if len(phrase.split()) < 2 or len(phrase.split()) > 12:
        return ""
    if len(_repair_term_set(phrase)) < 2:
        return ""
    return phrase


def _normalize_mechanism_clause(text: object) -> str:
    clause = re.sub(r"\s+", " ", str(text or "")).strip(" ,.;:-")
    if not clause:
        return ""

    match = re.match(r"^(?P<verb>[A-Za-z][A-Za-z\-]*)(?P<rest>\b.*)$", clause)
    if match is None:
        return clause

    verb = match.group("verb")
    rest = match.group("rest")
    lowered = verb.lower()
    if lowered in GERUND_TO_FINITE_CONNECTORS:
        clause = GERUND_TO_FINITE_CONNECTORS[lowered] + rest
    elif lowered in BARE_TO_FINITE_CONNECTORS:
        clause = BARE_TO_FINITE_CONNECTORS[lowered] + rest
    elif not _has_mechanism_connector(clause):
        clause = f"via {clause}"
    return clause


def _mechanism_precision_candidate_rank(candidate: dict) -> tuple[int, int, int, int, int, int, int, int, int]:
    phrase = str(candidate.get("phrase") or "")
    phrase_terms = _repair_term_set(phrase)
    return (
        1 if int(candidate.get("relevance_score") or 0) > 0 else 0,
        1 if candidate.get("mechanism_source") else 0,
        1
        if int(candidate.get("measurable_hits") or 0) > 0
        and int(candidate.get("control_hits") or 0) > 0
        else 0,
        int(candidate.get("phrase_process_hits") or 0),
        int(candidate.get("process_score") or 0),
        int(candidate.get("measurable_hits") or 0),
        int(candidate.get("control_hits") or 0),
        int(candidate.get("source_priority") or 0),
        len(phrase_terms),
    )


def _best_mechanism_process_anchor(data: dict, repair_context: dict) -> dict | None:
    candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()
    evidence_map = normalize_evidence_map(data.get("evidence_map"))
    context_terms = _repair_term_set(data.get("target_domain"))
    context_terms.update(_repair_term_set(data.get("mechanism")))
    prediction = data.get("prediction") if isinstance(data.get("prediction"), dict) else {}
    test_payload = data.get("test") if isinstance(data.get("test"), dict) else {}
    context_terms.update(_repair_term_set(prediction.get("observable")))
    context_terms.update(_repair_term_set(test_payload.get("metric")))
    context_terms.update(_repair_term_set(test_payload.get("confirm")))
    measurable_terms = _repair_term_set(prediction.get("observable"))
    measurable_terms.update(_repair_term_set(test_payload.get("metric")))
    control_terms = _repair_term_set(test_payload.get("confirm"))
    control_terms.update(_repair_term_set(test_payload.get("falsify")))
    for entry in evidence_map.get("variable_mappings", [])[:3]:
        if not isinstance(entry, dict):
            continue
        control_terms.update(_repair_term_set(entry.get("target_variable")))
    control_terms.difference_update(measurable_terms)

    def _add_candidate(
        text: object,
        source_priority: int,
        *,
        mechanism_source: bool = False,
    ) -> None:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            return
        phrase = _extract_process_phrase(cleaned)
        if not phrase:
            return
        key = (phrase.lower(), cleaned.lower())
        if key in seen:
            return
        seen.add(key)
        phrase_process_hits = len(_repair_term_set(phrase) & MECHANISM_ANCHOR_PROCESS_TERMS)
        if phrase_process_hits == 0:
            return
        process_score = max(_process_anchor_score(cleaned), _process_anchor_score(phrase))
        if process_score <= 0 and len(_repair_term_set(phrase)) < 3:
            return
        candidate_terms = _repair_term_set(cleaned)
        relevance_score = len(_repair_term_set(phrase) & context_terms)
        candidates.append(
            {
                "text": cleaned,
                "phrase": phrase,
                "mechanism_source": mechanism_source,
                "measurable_hits": len(candidate_terms & measurable_terms),
                "control_hits": len(candidate_terms & control_terms),
                "phrase_process_hits": phrase_process_hits,
                "process_score": process_score,
                "relevance_score": relevance_score,
                "source_priority": source_priority,
            }
        )

    mechanism_anchor = (
        repair_context.get("mechanism_anchor")
        if isinstance(repair_context.get("mechanism_anchor"), dict)
        else {}
    )
    core_target_anchor = (
        repair_context.get("core_target_anchor")
        if isinstance(repair_context.get("core_target_anchor"), dict)
        else {}
    )
    for entry in evidence_map.get("mechanism_assertions", [])[:3]:
        if not isinstance(entry, dict):
            continue
        _add_candidate(entry.get("mechanism_claim"), 8, mechanism_source=True)
        _add_candidate(entry.get("evidence_snippet"), 5, mechanism_source=True)

    _add_candidate(data.get("mechanism"), 6)

    for entry in evidence_map.get("variable_mappings", [])[:3]:
        if not isinstance(entry, dict):
            continue
        _add_candidate(entry.get("claim"), 7)
        _add_candidate(entry.get("evidence_snippet"), 4)

    _add_candidate(core_target_anchor.get("claim"), 4)
    _add_candidate(core_target_anchor.get("evidence_snippet"), 3)
    _add_candidate(mechanism_anchor.get("text"), 2)
    _add_candidate(mechanism_anchor.get("snippet"), 1)

    _add_candidate(data.get("target_domain"), 0)

    return max(candidates, key=_mechanism_precision_candidate_rank, default=None)


def _extract_bridge_clause(mechanism: object) -> str:
    text = re.sub(r"\s+", " ", str(mechanism or "")).strip()
    if not text:
        return ""
    match = re.search(
        r"\b(?:operates by|works by|functions by|acts by|does so by)\b\s*(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return ""
    return _normalize_mechanism_clause(match.group(1))


def _extract_connector_clause(mechanism: object) -> str:
    text = re.sub(r"\s+", " ", str(mechanism or "")).strip()
    if not text:
        return ""
    connector_pattern = "|".join(sorted((re.escape(token) for token in PROCESS_CONNECTORS), key=len, reverse=True))
    match = re.search(rf"\b(?:{connector_pattern})\b.*$", text, flags=re.IGNORECASE)
    if match is None:
        return ""
    return _normalize_mechanism_clause(match.group(0))


def _rewrite_anchor_sentence_with_phrase(anchor_phrase: str, anchor_text: object) -> str:
    cleaned = _strip_mechanism_reporting_prefix(anchor_text).strip(" \t\r\n.;")
    if not cleaned:
        return ""
    extracted_phrase = _extract_process_phrase(cleaned)
    if not extracted_phrase:
        return ""
    remainder = cleaned[len(extracted_phrase):].strip(" ,.;:-")
    normalized_remainder = _normalize_mechanism_clause(remainder)
    if not normalized_remainder or not _has_mechanism_connector(normalized_remainder):
        return ""
    return f"{anchor_phrase} {normalized_remainder}".strip()


def _mechanism_needs_precision_rewrite(mechanism: object) -> bool:
    text = str(mechanism or "").strip().lower()
    if not text:
        return False
    opening_phrase = _extract_process_phrase(mechanism)
    if not opening_phrase or _process_anchor_score(opening_phrase) <= 0:
        return True
    if not _has_mechanism_connector(mechanism):
        return True
    if any(phrase in text for phrase in MECHANISM_BRIDGE_OPENERS):
        return True
    if text.startswith(RESULT_FIRST_MECHANISM_OPENERS):
        return True
    if _strip_mechanism_reporting_prefix(mechanism).lower() != text:
        return True
    return (
        text.startswith("the retrieved evidence")
        or text.startswith("retrieved evidence")
        or text.startswith("literature ")
        or text.startswith("studies ")
    )


def _apply_mechanism_naming_precision(data: dict) -> dict:
    if not isinstance(data, dict):
        return data

    mechanism = str(data.get("mechanism") or "").strip()
    if not mechanism or not _mechanism_needs_precision_rewrite(mechanism):
        return data

    repair_context = _phase3_repair_context(data)
    anchor = _best_mechanism_process_anchor(data, repair_context)
    if not isinstance(anchor, dict):
        return data

    anchor_phrase = str(anchor.get("phrase") or "").strip()
    anchor_text = str(anchor.get("text") or "").strip()
    if not anchor_phrase:
        return data

    rewritten_options = [
        f"{anchor_phrase} {_extract_bridge_clause(mechanism)}".strip(),
        f"{anchor_phrase} {_extract_connector_clause(mechanism)}".strip(),
        _rewrite_anchor_sentence_with_phrase(anchor_phrase, anchor_text),
    ]
    mechanism_anchor = (
        repair_context.get("mechanism_anchor")
        if isinstance(repair_context.get("mechanism_anchor"), dict)
        else {}
    )
    rewritten_options.append(
        _rewrite_anchor_sentence_with_phrase(
            anchor_phrase,
            mechanism_anchor.get("snippet") or mechanism_anchor.get("text"),
        )
    )

    for rewritten in rewritten_options:
        cleaned = re.sub(r"\s+", " ", str(rewritten or "")).strip(" \t\r\n.;")
        if not cleaned:
            continue
        if not cleaned.lower().startswith(anchor_phrase.lower()):
            continue
        if len(cleaned.split()) < 8:
            continue
        if not _has_mechanism_connector(cleaned):
            continue
        if cleaned == mechanism:
            continue
        out = dict(data)
        out["mechanism"] = cleaned
        return out

    return data


def _missing_required_fields(data: dict) -> list[str]:
    def _is_non_empty(value: object) -> bool:
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, dict, tuple, set)):
            return len(value) > 0
        return value is not None

    def _mapping_count(variable_mapping: object) -> int:
        if isinstance(variable_mapping, dict):
            return sum(
                1
                for k, v in variable_mapping.items()
                if str(k).strip() and str(v).strip()
            )
        if isinstance(variable_mapping, list):
            count = 0
            for item in variable_mapping:
                if isinstance(item, dict):
                    if any(str(v).strip() for v in item.values()):
                        count += 1
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    if str(item[0]).strip() and str(item[1]).strip():
                        count += 1
                elif isinstance(item, str):
                    cleaned = item.strip()
                    if cleaned and any(sep in cleaned for sep in ("->", "=>", ":", "=")):
                        count += 1
            return count
        if isinstance(variable_mapping, str):
            return sum(
                1
                for part in variable_mapping.split(",")
                if part.strip() and any(sep in part for sep in ("->", "=>", ":", "="))
            )
        return 0

    def _assumptions_count(assumptions: object) -> int:
        if isinstance(assumptions, list):
            return sum(1 for item in assumptions if str(item).strip())
        if isinstance(assumptions, str):
            return len([p for p in assumptions.replace("\n", ";").split(";") if p.strip()])
        return 0

    def _test_has_metric_confirm_falsify(test: object) -> bool:
        if isinstance(test, dict):
            has_metric = _is_non_empty(test.get("metric")) or _is_non_empty(test.get("metrics"))
            has_confirm = any(
                _is_non_empty(test.get(key))
                for key in ("confirm", "confirms", "confirmed_if", "supports")
            )
            has_falsify = any(
                _is_non_empty(test.get(key))
                for key in ("falsify", "falsifies", "falsified_if", "refutes")
            )
            return has_metric and has_confirm and has_falsify
        if isinstance(test, str):
            lower = test.lower()
            has_metric = "metric" in lower
            has_confirm = any(
                k in lower for k in ("confirm", "support", "validated", "true")
            )
            has_falsify = any(
                k in lower for k in ("falsif", "refut", "reject", "false")
            )
            return has_metric and has_confirm and has_falsify
        return False

    def _contains_any_phrase(text: object, phrases: tuple[str, ...]) -> bool:
        cleaned = str(text or "").strip().lower()
        return any(phrase in cleaned for phrase in phrases)

    def _meaningful_terms(text: object) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9_]+", str(text or "").lower())
            if len(token) >= 4 and not token.isdigit()
        }

    def _mechanism_needs_repair(mechanism: object) -> bool:
        text = str(mechanism or "").strip()
        if not text:
            return True
        lower = text.lower()
        if _contains_any_phrase(lower, GENERIC_MECHANISM_FILLERS):
            return True
        if _contains_any_phrase(lower, MECHANISM_BRIDGE_OPENERS):
            return True
        if lower.startswith(RESULT_FIRST_MECHANISM_OPENERS):
            return True
        if lower.startswith("the retrieved evidence") or lower.startswith("retrieved evidence"):
            return True
        if lower.startswith("literature ") or lower.startswith("studies "):
            return True
        if len(text.split()) < 8:
            return True
        return False

    def _test_metric_needs_repair(test: object) -> bool:
        if not isinstance(test, dict):
            return True
        metric = str(test.get("metric") or test.get("metrics") or "").strip()
        if not metric:
            return True
        lower = metric.lower()
        if lower in GENERIC_TEST_METRIC_FILLERS or _contains_any_phrase(
            lower, GENERIC_TEST_METRIC_FILLERS
        ):
            return True
        if len(metric.split()) < 2:
            return True
        return False

    def _test_decision_needs_repair(test: object) -> bool:
        if not isinstance(test, dict):
            return True
        metric = str(test.get("metric") or test.get("metrics") or "").strip()
        confirm = str(
            test.get("confirm")
            or test.get("confirms")
            or test.get("confirmed_if")
            or test.get("supports")
            or ""
        ).strip()
        falsify = str(
            test.get("falsify")
            or test.get("falsifies")
            or test.get("falsified_if")
            or test.get("refutes")
            or ""
        ).strip()
        if not metric or not confirm or not falsify:
            return True
        metric_terms = _meaningful_terms(metric)
        if not metric_terms:
            return True
        if _contains_any_phrase(confirm, GENERIC_TEST_DECISION_FILLERS) or _contains_any_phrase(
            falsify, GENERIC_TEST_DECISION_FILLERS
        ):
            return True
        if len(metric_terms & _meaningful_terms(confirm)) == 0:
            return True
        if len(metric_terms & _meaningful_terms(falsify)) == 0:
            return True
        return False

    def _problem_statement_needs_repair(text: object) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        lower = value.lower()
        if len(value.split()) < 7:
            return True
        if _contains_any_phrase(lower, GENERIC_PROBLEM_FILLERS):
            return True
        if not any(hint in lower for hint in EDGE_PROBLEM_HINTS):
            return True
        return False

    def _actionable_lever_needs_repair(text: object) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        lower = value.lower()
        if _contains_any_phrase(lower, GENERIC_ACTION_FILLERS):
            return True
        if not any(hint in lower for hint in EDGE_ACTION_HINTS):
            return True
        if len(value.split()) < 4:
            return True
        return False

    def _edge_advantage_needs_repair(text: object) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        lower = value.lower()
        if _contains_any_phrase(lower, GENERIC_EDGE_ADVANTAGE_FILLERS):
            return True
        if not any(hint in lower for hint in EDGE_ADVANTAGE_HINTS):
            return True
        if len(value.split()) < 6:
            return True
        return False

    def _underexploitedness_needs_repair(text: object) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        lower = value.lower()
        if _contains_any_phrase(lower, KNOWNNESS_MARKERS):
            return True
        if _contains_any_phrase(lower, GENERIC_UNDEREXPLOITED_FILLERS):
            return True
        if len(value.split()) < 5:
            return True
        if not any(hint in lower for hint in UNDEREXPLOITEDNESS_HINTS):
            return True
        return False

    def _critical_evidence_snippets_need_repair(evidence_map: dict) -> bool:
        variable_mappings = (
            evidence_map.get("variable_mappings")
            if isinstance(evidence_map.get("variable_mappings"), list)
            else []
        )
        for entry in variable_mappings[:3]:
            if not isinstance(entry, dict):
                return True
            claim = str(entry.get("claim") or "").strip()
            snippet = str(entry.get("evidence_snippet") or "").strip()
            if len(snippet.split()) < 8:
                return True
            if len(_meaningful_terms(snippet)) < 4:
                return True
            if len(_meaningful_terms(claim) & _meaningful_terms(snippet)) == 0:
                return True
        return False

    def _mechanism_anchor_needs_repair(repair_context: dict) -> bool:
        mechanism = str(data.get("mechanism") or "").strip()
        if not mechanism:
            return True
        if _mechanism_needs_repair(mechanism):
            return True
        mechanism_terms = _repair_term_list(mechanism)[:5]
        if len(mechanism_terms) < 2:
            return True
        return int(repair_context.get("mechanism_overlap") or 0) == 0

    def _prediction_missing_fields(prediction: object) -> list[str]:
        if not isinstance(prediction, dict):
            return [
                "prediction.observable",
                "prediction.time_horizon",
                "prediction.direction",
                "prediction.magnitude",
                "prediction.confidence",
                "prediction.falsification_condition",
                "prediction.utility_rationale",
                "prediction.who_benefits",
            ]
        missing = []
        for field in (
            "observable",
            "time_horizon",
            "direction",
            "magnitude",
            "confidence",
            "falsification_condition",
            "utility_rationale",
            "who_benefits",
        ):
            if not _is_non_empty(prediction.get(field)):
                missing.append(f"prediction.{field}")
        return missing

    def _edge_analysis_missing_fields(edge_analysis: object) -> list[str]:
        if not isinstance(edge_analysis, dict):
            return [
                "edge_analysis.problem_statement",
                "edge_analysis.actionable_lever",
                "edge_analysis.cheap_test",
                "edge_analysis.edge_if_right",
                "edge_analysis.primary_operator",
            ]
        missing = []
        if not _is_non_empty(edge_analysis.get("problem_statement")):
            missing.append("edge_analysis.problem_statement")
        if not _is_non_empty(edge_analysis.get("actionable_lever")):
            missing.append("edge_analysis.actionable_lever")
        cheap_test = edge_analysis.get("cheap_test")
        if not isinstance(cheap_test, dict) or not all(
            _is_non_empty(cheap_test.get(key))
            for key in ("setup", "metric", "confirm", "falsify")
        ):
            missing.append("edge_analysis.cheap_test")
        if not _is_non_empty(edge_analysis.get("edge_if_right")):
            missing.append("edge_analysis.edge_if_right")
        if not _is_non_empty(edge_analysis.get("primary_operator")):
            missing.append("edge_analysis.primary_operator")
        return missing

    missing: list[str] = []
    for field in ("source_domain", "target_domain", "connection"):
        if field not in data or not _is_non_empty(data.get(field)):
            missing.append(field)
    if not _is_non_empty(data.get("mechanism")) or _mechanism_needs_repair(
        data.get("mechanism")
    ):
        missing.append("mechanism")
    normalized_mechanism_typing = normalize_mechanism_typing(data)
    if not _is_non_empty(normalized_mechanism_typing.get("mechanism_type")):
        missing.append("mechanism_type")
    if normalized_mechanism_typing.get("mechanism_type_confidence") is None:
        missing.append("mechanism_type_confidence")
    if _mapping_count(data.get("variable_mapping")) < 3:
        missing.append("variable_mapping")
    missing.extend(_prediction_missing_fields(data.get("prediction")))
    if not _test_has_metric_confirm_falsify(data.get("test")):
        missing.append("test")
    elif _test_metric_needs_repair(data.get("test")):
        missing.extend(["test.metric", "test.confirm", "test.falsify"])
    elif _test_decision_needs_repair(data.get("test")):
        missing.extend(["test.confirm", "test.falsify"])
    missing.extend(_edge_analysis_missing_fields(data.get("edge_analysis")))
    edge_analysis = (
        data.get("edge_analysis") if isinstance(data.get("edge_analysis"), dict) else {}
    )
    edge_alignment = summarize_edge_usefulness_alignment(edge_analysis, data)
    if _problem_statement_needs_repair(edge_analysis.get("problem_statement")):
        missing.append("edge_analysis.problem_statement")
    elif not edge_alignment.get("problem_aligned"):
        missing.append("edge_analysis.problem_statement")
    if _actionable_lever_needs_repair(edge_analysis.get("actionable_lever")):
        missing.append("edge_analysis.actionable_lever")
    elif not edge_alignment.get("actionable_lever_aligned"):
        missing.append("edge_analysis.actionable_lever")
    if "edge_analysis.cheap_test" not in missing and not edge_alignment.get(
        "cheap_test_real_operator_move"
    ):
        missing.append("edge_analysis.cheap_test")
    if _edge_advantage_needs_repair(edge_analysis.get("edge_if_right")):
        missing.append("edge_analysis.edge_if_right")
    elif not edge_alignment.get("edge_advantage_aligned"):
        missing.append("edge_analysis.edge_if_right")
    if _underexploitedness_needs_repair(edge_analysis.get("why_missed")):
        missing.append("edge_analysis.why_missed")
    if _underexploitedness_needs_repair(edge_analysis.get("expected_asymmetry")):
        missing.append("edge_analysis.expected_asymmetry")
    if _assumptions_count(data.get("assumptions")) < 2:
        missing.append("assumptions")
    if not _is_non_empty(data.get("boundary_conditions")):
        missing.append("boundary_conditions")
    if not _is_non_empty(data.get("evidence")):
        missing.append("evidence")
    evidence_map = normalize_evidence_map(data.get("evidence_map"))
    repair_context = _phase3_repair_context(
        {
            **data,
            "evidence_map": evidence_map,
        }
    )
    provenance = (
        repair_context.get("provenance")
        if isinstance(repair_context.get("provenance"), dict)
        else {}
    )
    critical_failure_codes = {
        code
        for failure in (repair_context.get("critical_failures") or [])
        if isinstance(failure, dict)
        for code in (failure.get("reason_codes") or [])
        if str(code).strip()
    }
    core_target_evidence_strength = str(
        provenance.get("core_target_evidence_strength") or ""
    ).strip()
    if len(evidence_map.get("variable_mappings", [])) < 3:
        missing.append("evidence_map.variable_mappings")
    elif (
        _critical_evidence_snippets_need_repair(evidence_map)
        or int(provenance.get("supported_critical_mapping_count") or 0) < 3
        or bool(repair_context.get("missing_critical_pairs"))
        or bool(
            critical_failure_codes
            & {
                "claim_snippet_mismatch",
                "vague_snippet",
                "missing_source_reference",
                "low_overall_provenance_quality",
            }
        )
    ):
        missing.append("evidence_map.variable_mappings")
    if len(evidence_map.get("mechanism_assertions", [])) < 1:
        missing.append("evidence_map.mechanism_assertions")
    elif int(provenance.get("supported_mechanism_assertion_count") or 0) < int(
        provenance.get("required_mechanism_assertion_count") or 0
    ):
        missing.append("evidence_map.mechanism_assertions")
    elif core_target_evidence_strength != "strong_direct":
        missing.append("evidence_map.mechanism_assertions")
        missing.append("mechanism")
    if _mechanism_anchor_needs_repair(repair_context):
        missing.append("mechanism")
    deduped_missing: list[str] = []
    for field in missing:
        if field not in deduped_missing:
            deduped_missing.append(field)
    return deduped_missing


def _repair_guidance_for_missing_fields(
    missing_fields: list[str],
    *,
    original_data: dict | None = None,
) -> str:
    guidance: list[str] = []
    missing_field_set = set(missing_fields)
    repair_context = _phase3_repair_context(original_data)
    edge_alignment = summarize_edge_usefulness_alignment(
        original_data or {},
        original_data if isinstance(original_data, dict) else None,
    )
    normalized_edge = normalize_edge_analysis(original_data or {})
    evidence_map = normalize_evidence_map(
        original_data.get("evidence_map")
        if isinstance(original_data, dict)
        else None
    )
    prediction = (
        original_data.get("prediction")
        if isinstance(original_data, dict) and isinstance(original_data.get("prediction"), dict)
        else {}
    )
    test_payload = (
        original_data.get("test")
        if isinstance(original_data, dict) and isinstance(original_data.get("test"), dict)
        else {}
    )
    mechanism_anchor = (
        repair_context.get("mechanism_anchor")
        if isinstance(repair_context.get("mechanism_anchor"), dict)
        else {}
    )
    core_target_anchor = (
        repair_context.get("core_target_anchor")
        if isinstance(repair_context.get("core_target_anchor"), dict)
        else {}
    )
    usefulness_bottleneck_fields = {
        "edge_analysis.problem_statement",
        "edge_analysis.actionable_lever",
        "edge_analysis.cheap_test",
        "edge_analysis.edge_if_right",
        "edge_analysis.primary_operator",
    }
    coherent_edge_package_fields = {
        "edge_analysis.problem_statement",
        "edge_analysis.actionable_lever",
        "edge_analysis.cheap_test",
        "edge_analysis.edge_if_right",
        "edge_analysis.expected_asymmetry",
    }
    coherent_evidence_package_fields = {
        "evidence_map.variable_mappings",
        "evidence_map.mechanism_assertions",
    }
    current_mechanism = (
        str(original_data.get("mechanism") or "").strip()
        if isinstance(original_data, dict)
        else ""
    )
    connection_anchor = (
        str(original_data.get("connection") or "").strip()
        if isinstance(original_data, dict)
        else ""
    )
    metric_anchor = str(test_payload.get("metric") or test_payload.get("metrics") or "").strip()
    observable_anchor = str(prediction.get("observable") or "").strip()
    mechanism_claim_anchor = ""
    for entry in evidence_map.get("mechanism_assertions", []):
        if not isinstance(entry, dict):
            continue
        mechanism_claim_anchor = str(entry.get("mechanism_claim") or "").strip()
        if mechanism_claim_anchor:
            break
    confirm_anchor = str(test_payload.get("confirm") or "").strip()
    falsify_anchor = str(
        prediction.get("falsification_condition")
        or test_payload.get("falsify")
        or ""
    ).strip()
    cheap_test_payload = (
        normalized_edge.get("cheap_test")
        if isinstance(normalized_edge.get("cheap_test"), dict)
        else {}
    )
    cheap_test_setup_anchor = str(cheap_test_payload.get("setup") or "").strip()
    cheap_test_metric_anchor = str(cheap_test_payload.get("metric") or "").strip()
    cheap_test_confirm_anchor = str(cheap_test_payload.get("confirm") or "").strip()
    cheap_test_falsify_anchor = str(cheap_test_payload.get("falsify") or "").strip()
    problem_statement_anchor = str(normalized_edge.get("problem_statement") or "").strip()
    lever_anchor = str(normalized_edge.get("actionable_lever") or "").strip()
    edge_if_right_anchor = str(normalized_edge.get("edge_if_right") or "").strip()
    operator_anchor = str(normalized_edge.get("primary_operator") or "").strip()
    solution_evidence_anchor = (
        str(original_data.get("solution_evidence") or "").strip()
        if isinstance(original_data, dict)
        else ""
    )
    if solution_evidence_anchor:
        guidance.append(
            f"- Reuse the current workaround or operating-response anchor instead of inventing a different intervention: `{solution_evidence_anchor}`."
        )
    target_evidence_anchor = str(core_target_anchor.get("evidence_snippet") or "").strip()
    target_evidence_source = str(core_target_anchor.get("source_reference") or "").strip()
    if not target_evidence_anchor:
        for entry in evidence_map.get("mechanism_assertions", [])[:3]:
            if not isinstance(entry, dict):
                continue
            target_evidence_anchor = str(entry.get("evidence_snippet") or "").strip()
            target_evidence_source = str(entry.get("source_reference") or "").strip()
            if target_evidence_anchor:
                break
    if not target_evidence_anchor:
        for entry in evidence_map.get("variable_mappings", [])[:3]:
            if not isinstance(entry, dict):
                continue
            target_evidence_anchor = str(entry.get("evidence_snippet") or "").strip()
            target_evidence_source = str(entry.get("source_reference") or "").strip()
            if target_evidence_anchor:
                break
    coherent_edge_missing = [
        field for field in missing_fields if field in coherent_edge_package_fields
    ]
    coherent_evidence_missing = [
        field for field in missing_fields if field in coherent_evidence_package_fields
    ]
    coherent_package_mode = len(coherent_edge_missing) + len(coherent_evidence_missing) >= 3 and (
        len(coherent_edge_missing) >= 2
        or bool(coherent_edge_missing and coherent_evidence_missing)
    )
    if coherent_package_mode:
        guidance.append(
            "- Multi-field coherent repair mode: repair the affected edge/evidence layer as one coordinated grounded package instead of patching each listed field independently."
        )
        guidance.append(
            "- Decision rule: either repair a coherent grounded package anchored to the existing claim, mechanism, target evidence, metric, and comparator, or return `{\"no_connection\": true}` if that support is too thin."
        )
        guidance.append(
            "- Treat `edge_analysis.problem_statement`, `edge_analysis.actionable_lever`, `edge_analysis.cheap_test`, `edge_analysis.edge_if_right`, `edge_analysis.expected_asymmetry`, `evidence_map.variable_mappings`, and `evidence_map.mechanism_assertions` as coupled support for the same target-domain claim rather than independent fill-in-the-blank fields."
        )
        guidance.append(
            "- Rebuild only the affected layer around the current grounded core: same target claim, same mechanism, same operator decision, same test metric/comparator, and the same best available target-domain evidence already present in the payload."
        )
        claim_anchor = observable_anchor or connection_anchor
        if claim_anchor:
            guidance.append(
                f"- Keep the package tied to the current target claim anchor: `{claim_anchor}`."
            )
        if current_mechanism:
            guidance.append(
                f"- Keep the package tied to the current mechanism: `{current_mechanism}`."
            )
        if target_evidence_anchor:
            guidance.append(
                "- Rebuild around this current target-domain evidence snippet instead of inventing new support: "
                f"`{target_evidence_anchor}`"
                + (f" (source: `{target_evidence_source}`)." if target_evidence_source else ".")
            )
        if metric_anchor:
            guidance.append(
                f"- Keep the package tied to the current test metric: `{metric_anchor}`."
            )
        if confirm_anchor:
            guidance.append(
                f"- Reuse the current confirm-side comparator wording where possible: `{confirm_anchor}`."
            )
        if falsify_anchor:
            guidance.append(
                f"- Reuse the current falsify-side decision wording where possible: `{falsify_anchor}`."
            )
        guidance.append(
            "- Package the repair coherently: one concrete hidden operator problem, one concrete operator lever, one cheap operator check on the same metric/comparator, one operator consequence if confirmed, and only the minimum mapping/mechanism-support entries needed to ground that same story."
        )
        guidance.append(
            "- Reuse current claim/process/operator/metric language wherever possible. Do not broaden the claim, do not add a new mechanism, and do not rewrite unrelated parts of the payload if the current grounded core is still usable."
        )
        guidance.append(
            "- If the current payload plus retrieved evidence do not support a concrete operator problem, lever, cheap test, and mapping/mechanism-support set without unsupported extrapolation, return `{\"no_connection\": true}`."
        )
        guidance.append(
            "- Prefer grounded repair or `{\"no_connection\": true}`. Do not invent a lever, operator advantage, variable mapping, or mechanism assertion just to satisfy required fields."
        )
        guidance.append(
            "- If support-layer fields cannot be concretely grounded from the current payload and evidence, prefer an explicit `{\"no_connection\": true}` path over malformed partial JSON, placeholder text, or generic filler."
        )
    if any(field in usefulness_bottleneck_fields for field in missing_fields):
        guidance.append(
            "- Phase 5 usefulness-alignment bottleneck: keep `connection`, `mechanism`, `prediction`, `test`, and `evidence_map` stable unless they are empty. Rewrite the edge layer so it points to the exact same core claim, process, comparator, and metric already named elsewhere."
        )
        guidance.append(
            "- Treat this as a repair-quality pass, not a reframing pass. Preserve the original target-domain claim and evidence grounding instead of drifting to a different problem, mechanism, metric, comparator, or stakeholder."
        )
        if metric_anchor:
            guidance.append(
                f"- Keep the edge layer tied to the current core metric: `{metric_anchor}`."
            )
        if observable_anchor:
            guidance.append(
                f"- Keep the edge layer tied to the current observable/claim anchor: `{observable_anchor}`."
            )
        if confirm_anchor:
            guidance.append(
                f"- Reuse the existing confirm-side comparator language instead of paraphrasing it: `{confirm_anchor}`."
            )
        if falsify_anchor:
            guidance.append(
                f"- Reuse the existing falsify-side decision language so the edge layer stays tied to the same operator decision: `{falsify_anchor}`."
            )
        if lever_anchor:
            guidance.append(
                f"- Keep the cheap test focused on the current lever unless it is rewritten for specificity: `{lever_anchor}`."
            )
        if operator_anchor:
            guidance.append(
                f"- Keep the edge framed as a decision for this operator: `{operator_anchor}`."
            )
        guidance.append(
            "- Prefer light-touch edge rewrites that reuse the same observable noun phrase, metric name, and comparison wording already present in `prediction` and `test`."
        )
    if {"edge_analysis.actionable_lever", "edge_analysis.cheap_test"}.issubset(
        missing_field_set
    ):
        guidance.append(
            "- Treat `edge_analysis.actionable_lever` + `edge_analysis.cheap_test` as one coordinated operator package. The lever should name the operator move, and the cheap test should be the cheapest grounded way to try, replay, filter, or audit that same move on the same workflow slice."
        )
        guidance.append(
            "- Complete only the missing lever/test package around one shared target-domain process, operator decision, metric, comparator, and workflow context. Do not let the lever describe one move while the cheap test evaluates a different intervention."
        )
        if lever_anchor:
            guidance.append(
                f"- Preserve the current grounded lever wording if usable and make the cheap test the smallest check of that same move: `{lever_anchor}`."
            )
        if cheap_test_setup_anchor:
            guidance.append(
                f"- Preserve the current grounded cheap-test move if usable and make `edge_analysis.actionable_lever` name that same move directly: `{cheap_test_setup_anchor}`."
            )
        if current_mechanism:
            guidance.append(
                f"- Keep the lever/test package tied to the current mechanism wording: `{current_mechanism}`."
            )
        if metric_anchor:
            guidance.append(
                f"- Keep both fields tied to the current metric and decision boundary: `{metric_anchor}`."
            )
        if solution_evidence_anchor:
            guidance.append(
                f"- Reuse the current workaround or operating-response anchor instead of inventing a different intervention: `{solution_evidence_anchor}`."
            )
        guidance.append(
            "- If the current payload and evidence do not support both the lever and cheap test as one grounded operator package, return `{\"no_connection\": true}` instead of fabricating the missing half."
        )
    if {
        "edge_analysis.actionable_lever",
        "edge_analysis.cheap_test",
        "edge_analysis.edge_if_right",
    }.issubset(missing_field_set):
        guidance.append(
            "- Treat `edge_analysis.actionable_lever` + `edge_analysis.cheap_test` + `edge_analysis.edge_if_right` as one coordinated edge package: one operator move, one cheap check of that move on the same workflow slice, and one decision change plus concrete advantage if it confirms."
        )
        guidance.append(
            "- Preserve the already grounded target-domain process, operator, metric, and comparator. Complete only the missing edge package and do not rewrite `mechanism`, `prediction`, `test`, or unrelated evidence when the current core is usable."
        )
        if operator_anchor:
            guidance.append(
                f"- Keep the full edge package tied to the current operator: `{operator_anchor}`."
            )
        if metric_anchor:
            guidance.append(
                f"- Keep the full edge package tied to the current metric/comparator boundary: `{metric_anchor}`."
            )
        if current_mechanism:
            guidance.append(
                f"- Keep the full edge package tied to the current target-domain process wording: `{current_mechanism}`."
            )
        if lever_anchor:
            guidance.append(
                f"- Preserve any grounded lever wording already present and make the cheap test and `edge_if_right` inherit that same move: `{lever_anchor}`."
            )
        if cheap_test_setup_anchor:
            guidance.append(
                f"- Preserve any grounded cheap-test move already present and make the lever and `edge_if_right` describe that same move directly: `{cheap_test_setup_anchor}`."
            )
        guidance.append(
            "- If the current payload and evidence cannot support the full edge package as one coherent operator-metric story, prefer `{\"no_connection\": true}` over generic edge filler or malformed partial JSON."
        )
    if {"test.confirm", "test.falsify"}.issubset(missing_field_set):
        guidance.append(
            "- Treat `test.confirm` + `test.falsify` as one coordinated decision-boundary pair. Rewrite them together so both sentences use the same `test.metric`, the same comparator or direction, and opposite outcomes for the same operator decision."
        )
        guidance.append(
            "- Make `test.confirm` the positive side of the boundary and `test.falsify` the kill condition for that exact same metric/comparator pair. Do not let them drift to different metrics, baselines, or failure criteria."
        )
        if metric_anchor:
            guidance.append(
                f"- Keep the confirm/falsify pair tied to the current metric wording: `{metric_anchor}`."
            )
        if confirm_anchor:
            guidance.append(
                f"- Preserve any usable confirm-side comparator wording already present: `{confirm_anchor}`."
            )
        if falsify_anchor:
            guidance.append(
                f"- Preserve any usable falsify-side decision wording already present: `{falsify_anchor}`."
            )
        guidance.append(
            "- If you cannot support both sides of the same metric boundary from the current payload and evidence, prefer `{\"no_connection\": true}` over unsupported confirm/falsify wording."
        )
    if {
        "test.confirm",
        "test.falsify",
        "edge_analysis.actionable_lever",
        "edge_analysis.cheap_test",
    }.issubset(missing_field_set):
        guidance.append(
            "- Treat `test.confirm` + `test.falsify` + `edge_analysis.actionable_lever` + `edge_analysis.cheap_test` as one coordinated operator-metric package: one operator move, one cheap check of that move, and one shared metric boundary repeated consistently across all four fields."
        )
        guidance.append(
            "- Preserve any grounded pieces already present and complete only the missing coupled fields. Do not rewrite `mechanism`, `prediction`, or unrelated evidence when the current grounded core is already usable."
        )
        if operator_anchor:
            guidance.append(
                f"- Keep the full operator-metric package tied to the current operator: `{operator_anchor}`."
            )
        if lever_anchor:
            guidance.append(
                f"- Keep the package centered on the current operator move where possible: `{lever_anchor}`."
            )
        if cheap_test_setup_anchor:
            guidance.append(
                f"- Keep the cheap-test portion centered on the current workflow move where possible: `{cheap_test_setup_anchor}`."
            )
        if metric_anchor:
            guidance.append(
                f"- Keep all four fields on the same metric/comparator boundary: `{metric_anchor}`."
            )
        guidance.append(
            "- If the current payload and evidence cannot support all four fields as the same operator-metric package, return `{\"no_connection\": true}` instead of mixing partial guesses from different stories."
        )
    if {
        "edge_analysis.problem_statement",
        "edge_analysis.edge_if_right",
    }.issubset(missing_field_set) and (
        lever_anchor
        or cheap_test_setup_anchor
        or cheap_test_metric_anchor
        or cheap_test_confirm_anchor
        or operator_anchor
        or metric_anchor
    ):
        guidance.append(
            "- Treat `edge_analysis.problem_statement` + `edge_analysis.edge_if_right` as one coordinated operator-consequence package around the existing lever / cheap-test core. The problem should name the hidden miss on the current metric, and `edge_if_right` should say what that same operator does differently if the current cheap test confirms."
        )
        guidance.append(
            "- Reuse the current operator, lever, cheap-test, metric, and comparator anchors rather than inventing a new workflow, stakeholder, or advantage axis for these two fields."
        )
        if problem_statement_anchor:
            guidance.append(
                f"- Preserve any grounded problem wording already present and tighten only what is generic: `{problem_statement_anchor}`."
            )
        if edge_if_right_anchor:
            guidance.append(
                f"- Preserve any grounded operator-advantage wording already present and keep it tied to the same operator decision: `{edge_if_right_anchor}`."
            )
        guidance.append(
            "- If the existing operator/cheap-test core does not support both the hidden problem and the operator consequence cleanly, return `{\"no_connection\": true}` instead of inventing a broader story."
        )
    if any(field == "mechanism" for field in missing_fields):
        guidance.append(
            "- Rewrite `mechanism` as one process-first sentence that opens with the exact target-domain process noun phrase, then names the operator, monitored/control variable, and resulting measurable change. Do not start with `when`, `as`, `if`, or a result summary."
        )
        guidance.append(
            "- Do not start `mechanism` with broad framing like `In this domain`, `The system`, `This process`, or `A mechanism where`. Open with the concrete process noun phrase itself."
        )
        guidance.append(
            "- Preserve the original target-domain claim/process and repair only the unsupported opening or process wording. Do not drift into a different problem framing, metric, comparator, or alternate mechanism."
        )
        if missing_field_set == {"mechanism"}:
            guidance.append(
                "- This is a mechanism-only rescue pass. Keep `edge_analysis`, `prediction`, `test`, and `evidence_map` wording fixed unless a referenced anchor is empty."
            )
            guidance.append(
                "- Prefer the smallest wording change that restores direct process anchoring and passes schema/validation."
            )
            guidance.append(
                "- Prefer the smallest valid rewrite to the opening/process phrasing rather than rewriting the whole causal story."
            )
            guidance.append(
                "- If the rest of the payload is already sound, return only `{\"mechanism\": ...}` instead of regenerating the full candidate."
            )
        if current_mechanism:
            guidance.append(
                f"- Keep the current mechanism assertion as intact as possible and replace only the unsupported opening/process wording: `{current_mechanism}`."
            )
        anchor_text = str(mechanism_anchor.get("text") or "").strip()
        if anchor_text:
            guidance.append(
                "- Pull the opening noun phrase of `mechanism` directly from target-domain evidence wording. Best available anchor: "
                f"`{anchor_text}`."
            )
        if mechanism_claim_anchor:
            guidance.append(
                "- Reuse the strongest current `evidence_map.mechanism_assertions` wording instead of inventing a broader process label: "
                f"`{mechanism_claim_anchor}`."
            )
        if observable_anchor:
            guidance.append(
                f"- Keep the repaired mechanism tied to the current prediction observable: `{observable_anchor}`."
            )
        if metric_anchor:
            guidance.append(
                f"- Keep the repaired mechanism tied to the current test metric: `{metric_anchor}`."
            )
        if confirm_anchor:
            guidance.append(
                f"- Preserve the current confirm-side comparator wording when you name the measurable consequence: `{confirm_anchor}`."
            )
        if falsify_anchor:
            guidance.append(
                f"- Preserve the current falsify-side decision wording so the rescue stays on the same operator check: `{falsify_anchor}`."
            )
        if mechanism_claim_anchor:
            guidance.append(
                "- Reuse existing process wording from `mechanism` / `evidence_map.mechanism_assertions` wherever it is already specific and evidence-grounded; do not swap in a new process label just to make the sentence sound broader."
            )
        core_strength = str(repair_context.get("core_target_evidence_strength") or "").strip()
        core_reasons = [
            str(reason).strip()
            for reason in (repair_context.get("core_target_reasons") or [])
            if str(reason).strip()
        ]
        direct_anchor = str(core_target_anchor.get("evidence_snippet") or "").strip()
        direct_source = str(core_target_anchor.get("source_reference") or "").strip()
        if core_strength and core_strength != "strong_direct":
            guidance.append(
                "- Narrow `mechanism` to the strongest direct target-domain evidence. Do not preserve broader process wording when the current support is only contextual, generic, or weak."
            )
            if direct_anchor:
                guidance.append(
                    "- Best available direct core target snippet: "
                    f"`{direct_anchor}`"
                    + (f" (source: `{direct_source}`)." if direct_source else ".")
                )
            if core_reasons:
                guidance.append(
                    "- Current core-target-evidence weakness: "
                    + "; ".join(core_reasons[:3])
                    + "."
                )
        guidance.append(
            "- If you cannot ground the opening process noun phrase in current target evidence or mechanism assertions, prefer `{\"no_connection\": true}` over generic mechanism filler."
        )
    if any(field in {"test", "test.metric"} for field in missing_fields):
        guidance.append(
            "- Rewrite `test` so `metric` names one concrete literature-facing quantity, not placeholders like `performance`, `efficiency`, `quality`, `improvement`, `stability`, or `outcomes`. Make `confirm` and `falsify` explicitly refer to that same metric."
        )
        guidance.append(
            "- If you cannot ground a specific literature-facing metric from the current payload and evidence, prefer `{\"no_connection\": true}` over a vague `test.metric`."
        )
    if any(field in {"test.confirm", "test.falsify"} for field in missing_fields):
        guidance.append(
            "- Rewrite `test.confirm` and `test.falsify` so each sentence literally names the same metric used in `test.metric` and states the explicit comparator or direction for that metric."
        )
    if "edge_analysis.problem_statement" in missing_fields:
        guidance.append(
            "- Rewrite `edge_analysis.problem_statement` so it names exactly one specific hidden target-domain failure mode, bottleneck, blind spot, or measurable miss tied to the same observable or metric as the test, not a broad field summary."
        )
        guidance.append(
            "- Tie `edge_analysis.problem_statement` to the same process, the same measurable quantity/comparator, and the same operator decision already implied by the current claim, metric, or comparator."
        )
        guidance.append(
            "- Rewrite only `edge_analysis.problem_statement` or the minimal coupled edge layer needed to keep it coherent. Preserve the current metric, operator, cheap-test, and `edge_analysis.edge_if_right` anchors."
        )
        guidance.append(
            "- Make it read like one hidden decision-relevant operator problem on that same measurable quantity. Reject field-summary prose, broad domain restatements, and generic `systems are complex` filler."
        )
        guidance.append(
            "- If the current payload and evidence cannot ground one concrete hidden operator problem on the same metric/decision boundary, prefer `{\"no_connection\": true}` over generic support-layer filler."
        )
    if "edge_analysis.actionable_lever" in missing_fields:
        guidance.append(
            "- Rewrite `edge_analysis.actionable_lever` so it names exactly one concrete operator move, setting change, filter, routing rule, threshold adjustment, replay, audit, workflow intervention, design choice, or decision rule that reuses the current mechanism, metric, or operator context. Reject vague filler like `investigate further`, `optimize process`, `improve monitoring`, `apply insights`, `study this`, or `consider this`."
        )
    if "edge_analysis.cheap_test" in missing_fields:
        if missing_field_set == {"edge_analysis.cheap_test"}:
            guidance.append(
                "- This is an `edge_analysis.cheap_test`-only completion pass. Keep `connection`, `mechanism`, `prediction`, `test`, `variable_mapping`, `evidence_map`, and the rest of `edge_analysis` stable."
            )
            guidance.append(
                "- If only this field is missing, prefer returning only `{\"edge_analysis\": {\"cheap_test\": {\"setup\": ..., \"metric\": ..., \"confirm\": ..., \"falsify\": ..., \"time_to_signal\": ...}}}` instead of rewriting the full candidate."
            )
        guidance.append(
            "- Rewrite `edge_analysis.cheap_test` so it includes `setup`, `metric`, `confirm`, `falsify`, and `time_to_signal`. It must be one cheap operator-facing check on an existing workflow slice, and `setup` must name one cheap operator move on a narrow slice of the workflow, not a generic validation suggestion and not a restatement of the main test."
        )
        guidance.append(
            "- Make `setup` one concrete operator move, replay, simulation, filter, audit, or measurement path inside the current operator workflow. Do not turn the repair into a generic validation study, broader research program, or different workflow."
        )
        guidance.append(
            "- Preserve the same current mechanism, target-domain process, operator decision, and target claim already grounded elsewhere in the payload. Keep the cheap test tied to the existing workflow context."
        )
        guidance.append(
            "- Reuse the same metric/comparator wording as the current `test.metric`, and make `confirm`/`falsify` name that same metric explicitly."
        )
        guidance.append(
            "- Keep `edge_analysis.cheap_test.metric` on the same measurable quantity as `test.metric`; only narrow to a comparator on that same quantity, and do not drift into generic validation wording or a broad proxy metric."
        )
        if cheap_test_setup_anchor:
            guidance.append(
                f"- Reuse any already-grounded cheap-test setup wording where possible and only complete what is missing: `{cheap_test_setup_anchor}`."
            )
        if str(test_payload.get("metric") or "").strip():
            guidance.append(
                "- Keep `edge_analysis.cheap_test.metric` identical to `test.metric` when possible."
            )
        if cheap_test_metric_anchor:
            guidance.append(
                f"- Reuse the current cheap-test metric wording directly if it already matches the main test metric: `{cheap_test_metric_anchor}`."
            )
        elif metric_anchor:
            guidance.append(
                f"- Reuse the current metric wording directly: `{metric_anchor}`."
            )
        if cheap_test_confirm_anchor:
            guidance.append(
                f"- Reuse the current cheap-test confirm wording closely if it already tracks the same decision boundary: `{cheap_test_confirm_anchor}`."
            )
        if cheap_test_falsify_anchor:
            guidance.append(
                f"- Reuse the current cheap-test falsify wording closely if it already tracks the same decision boundary: `{cheap_test_falsify_anchor}`."
            )
        if confirm_anchor:
            guidance.append(
                "- Reuse the existing `test.confirm` comparator wording as closely as possible so the cheap test checks the same decision boundary."
            )
        if falsify_anchor:
            guidance.append(
                "- Reuse the existing `test.falsify` decision wording as closely as possible so the cheap test kills the same claim if it fails."
            )
        if metric_anchor:
            guidance.append(
                f"- Keep the cheap test on the current metric and decision boundary: `{metric_anchor}`."
            )
        if operator_anchor:
            guidance.append(
                f"- Keep the cheap test scoped to the current operator workflow: `{operator_anchor}`."
            )
        if lever_anchor:
            guidance.append(
                f"- Keep the cheap test tied to the current operator move or lever context: `{lever_anchor}`."
            )
        if current_mechanism:
            guidance.append(
                f"- Keep the cheap test anchored to the current mechanism wording instead of inventing a new causal story: `{current_mechanism}`."
            )
        if observable_anchor:
            guidance.append(
                f"- Keep the cheap test tied to the current target claim anchor: `{observable_anchor}`."
            )
        guidance.append(
            "- If the rest of the candidate is already sound, complete only `edge_analysis.cheap_test` rather than rewriting unrelated fields."
        )
        guidance.append(
            "- If the current payload and evidence cannot support a concrete cheap-test metric on the same quantity as `test.metric`, prefer `{\"no_connection\": true}` over generic cheap-test filler."
        )
        if edge_alignment.get("cheap_test_generic_validation"):
            guidance.append(
                "- The current cheap test sounds like generic validation rather than an operator move. Replace wording like `validate whether`, `run a study`, or `collect more data` with a concrete replay/filter/rerank/toggle/audit action."
            )
        if edge_alignment.get("cheap_test_restates_main_test"):
            guidance.append(
                "- The current cheap test is too close to `test.data`. Make it smaller and more operational so it informs one near-term operator decision before committing to the full test."
            )
    if "edge_analysis.edge_if_right" in missing_fields:
        if missing_field_set == {"edge_analysis.edge_if_right"}:
            guidance.append(
                "- This is an `edge_analysis.edge_if_right`-only completion pass. Keep `connection`, `mechanism`, `prediction`, `test`, `variable_mapping`, `evidence_map`, and the rest of `edge_analysis` stable."
            )
            guidance.append(
                "- If only this field is missing, prefer returning only `{\"edge_analysis\": {\"edge_if_right\": ...}}` instead of rewriting the full candidate."
            )
        guidance.append(
            "- Rewrite `edge_analysis.edge_if_right` so it states one concrete operator gain such as lower collision rate, earlier warning, lower cost, higher throughput, or reduced false positives. Reject generic usefulness language and name the decision or workflow advantage unlocked if the cheap test confirms."
        )
        guidance.append(
            "- Write exactly one operator, one decision change unlocked by confirmation, and one concrete measurable or workflow advantage if confirmed. Explicitly say who acts, what they do differently, and what concrete advantage they gain if the cheap test confirms. Do not add a new stakeholder, KPI, roadmap claim, or strategic narrative."
        )
        guidance.append(
            "- Keep the same operator, the same decision unlocked by the cheap test, and the same measured advantage family already implied by the current metric/comparator. Do not introduce a new benefit axis, stakeholder, or unrelated KPI."
        )
        guidance.append(
            "- Keep `edge_analysis.edge_if_right` concise and operator-facing. If the current payload and evidence do not support one operator, one decision change, and one concrete advantage, prefer `{\"no_connection\": true}` over generic value language."
        )
        if operator_anchor:
            guidance.append(
                f"- Reuse the current `edge_analysis.primary_operator` wording directly: `{operator_anchor}`."
            )
        current_edge_metric_anchor = cheap_test_metric_anchor or metric_anchor
        if current_edge_metric_anchor:
            guidance.append(
                f"- Keep `edge_analysis.edge_if_right` tied to the current metric and decision boundary: `{current_edge_metric_anchor}`."
            )
        current_edge_confirm_anchor = cheap_test_confirm_anchor or confirm_anchor
        if current_edge_confirm_anchor:
            guidance.append(
                f"- Reuse the current cheap-test / confirm-side wording as closely as possible so the operator advantage stays on the same decision boundary: `{current_edge_confirm_anchor}`."
            )
        if cheap_test_setup_anchor:
            guidance.append(
                f"- Preserve the current cheap-test workflow move and make `edge_analysis.edge_if_right` describe what changes if that same move confirms: `{cheap_test_setup_anchor}`."
            )
        if confirm_anchor:
            guidance.append(
                "- Keep `edge_analysis.edge_if_right` tied to the same decision unlocked by the current confirm condition, not to a new loosely related benefit."
            )
    if "edge_analysis.why_missed" in missing_fields:
        guidance.append(
            "- Rewrite `edge_analysis.why_missed` so it names one concrete search, framing, workflow, metric, or discipline-boundary reason this problem or lever may be undernoticed. Reject lines like `people may miss this`."
        )
    if "edge_analysis.expected_asymmetry" in missing_fields:
        guidance.append(
            "- Rewrite `edge_analysis.expected_asymmetry` so it explains why the lever is plausibly underused rather than already standard target-domain wisdom. Reject `widely known`, `standard practice`, or generic novelty claims."
        )
    if "evidence_map.variable_mappings" in missing_fields:
        if missing_field_set == {"evidence_map.variable_mappings"}:
            guidance.append(
                "- This is a variable-mapping completion pass. Keep `target_domain`, `connection`, `mechanism`, `mechanism_type`, `mechanism_type_confidence`, `secondary_mechanism_types`, `prediction`, `test`, `variable_mapping`, `edge_analysis`, and `evidence_map.mechanism_assertions` stable."
            )
            guidance.append(
                "- If only this field is missing, prefer returning only `{\"evidence_map\": {\"variable_mappings\": [...]}}` instead of rewriting the full candidate."
            )
            guidance.append(
                "- Reconstruct only the mapping bundle. Preserve the already grounded mechanism core and any existing valid `evidence_map.mechanism_assertions`."
            )
        if (
            current_mechanism
            or mechanism_claim_anchor
            or metric_anchor
            or observable_anchor
            or lever_anchor
            or cheap_test_setup_anchor
        ):
            guidance.append(
                "- Treat `evidence_map.variable_mappings` as one coordinated mapping package for the current mechanism/operator/metric story, not as permission to invent a broader remap."
            )
            guidance.append(
                "- Treat this as a narrow direct-support repair. Preserve the current claim, mechanism, test/operator anchors, target-domain process, observable, metric, comparator, and strongest current target evidence while rewriting only the first 3 critical mappings."
            )
            guidance.append(
                "- Prefer narrow direct-support reconstruction over filler mappings. If direct support is not there, return `{\"no_connection\": true}` instead of broadening the mapping story."
            )
            if current_mechanism:
                guidance.append(
                    f"- Keep the mapping package tied to the current mechanism wording: `{current_mechanism}`."
                )
            if lever_anchor:
                guidance.append(
                    f"- Keep the mapping package tied to the current operator move where relevant: `{lever_anchor}`."
                )
            if metric_anchor:
                guidance.append(
                    f"- Keep the mapping package tied to the current metric/comparator story: `{metric_anchor}`."
                )
            if target_evidence_anchor:
                guidance.append(
                    "- Use the current strongest target-domain evidence as the default remap anchor instead of broadening the claim: "
                    f"`{target_evidence_anchor}`"
                    + (f" (source: `{target_evidence_source}`)." if target_evidence_source else ".")
                )
            if solution_evidence_anchor:
                guidance.append(
                    f"- Reuse the current workaround/solution anchor where it already supports the remap: `{solution_evidence_anchor}`."
                )
            guidance.append(
                "- If the current grounded mechanism/operator/metric core still cannot support 3 critical mappings directly, return `{\"no_connection\": true}` instead of broadening the claim or inventing extra mapped variables."
            )
        guidance.append(
            "- Rewrite the first 3 `evidence_map.variable_mappings` entries so each `evidence_snippet` is at least one self-contained technical sentence or clause with concrete overlapping terms from the claim or mapped variable. Do not use vague background snippets."
        )
        guidance.append(
            "- Repair the critical mappings before touching non-critical ones. Keep exactly 3 critical mappings if support is thin, and make those the first 3 `variable_mapping` plus `evidence_map.variable_mappings` entries."
        )
        guidance.append(
            "- Complete the missing critical variable mappings from the current payload one supported entry at a time. Reuse the existing source-variable / target-variable pairs, target claim wording, and target evidence wording wherever they are already grounded."
        )
        guidance.append(
            "- Prioritize only the first 3 critical mappings. Prefer exactly 3 strong mappings over padded weak ones. Do not invent extra mappings, broaden the mechanism, or expand beyond the current grounded claim."
        )
        guidance.append(
            "- Rebuild only the first 3 critical mappings. Keep each repaired claim narrowly aligned to its source_variable/target_variable pair, tied to the same target-domain process, and at the same specificity as the supporting snippet on both sides of the mapping."
        )
        guidance.append(
            "- Do not pad with abstract correspondences, nearby downstream effects, or mechanism-level filler. If a snippet supports only the mechanism story, background process, or broader target-domain claim, move that support to `evidence_map.mechanism_assertions` instead of forcing it into `evidence_map.variable_mappings`."
        )
        guidance.append(
            "- If only 1 or 2 critical mappings can be directly supported from the current payload and evidence, prefer `{\"no_connection\": true}` over weak padding or malformed partial JSON."
        )
        current_variable_mapping = (
            original_data.get("variable_mapping")
            if isinstance(original_data, dict) and isinstance(original_data.get("variable_mapping"), dict)
            else {}
        )
        current_mapping_entries: dict[str, dict] = {}
        for entry in evidence_map.get("variable_mappings", []):
            if not isinstance(entry, dict):
                continue
            source_variable = str(entry.get("source_variable") or "").strip()
            target_variable = str(entry.get("target_variable") or "").strip()
            if not source_variable or not target_variable:
                continue
            current_mapping_entries[f"{source_variable} -> {target_variable}"] = entry
        for source_variable, target_variable in list(current_variable_mapping.items())[:3]:
            source_text = str(source_variable).strip()
            target_text = str(target_variable).strip()
            if not source_text or not target_text:
                continue
            pair = f"{source_text} -> {target_text}"
            guidance.append(
                f"- Keep the critical pair wording exactly aligned to the current payload: `{pair}`."
            )
            current_entry = current_mapping_entries.get(pair) or {}
            current_claim = str(current_entry.get("claim") or "").strip()
            current_snippet = str(current_entry.get("evidence_snippet") or "").strip()
            if current_claim:
                guidance.append(
                    f"- Reuse this current mapping claim as the starting point and narrow it only if needed: `{current_claim}`."
                )
            if current_snippet:
                guidance.append(
                    f"- Reuse this current evidence wording where possible and keep the repaired claim as a narrow paraphrase of it: `{current_snippet}`."
                )
        for failure in (repair_context.get("critical_failures") or [])[:3]:
            if not isinstance(failure, dict):
                continue
            pair = str(failure.get("pair") or "").strip()
            codes = ", ".join(
                str(code).strip()
                for code in (failure.get("reason_codes") or [])
                if str(code).strip()
            )
            claim = str(failure.get("claim") or "").strip()
            snippet = str(failure.get("evidence_snippet") or "").strip()
            if pair:
                guidance.append(
                    f"- Critical mapping to rewrite first: `{pair}`"
                    + (f" (`{codes}`)." if codes else ".")
                )
            if claim:
                guidance.append(f"  Current claim: `{claim}`.")
            if snippet:
                guidance.append(f"  Current snippet: `{snippet}`.")
        missing_pairs = [
            str(pair).strip()
            for pair in (repair_context.get("missing_critical_pairs") or [])
            if str(pair).strip()
        ]
        if missing_pairs:
            guidance.append(
                "- Missing critical mapping support that must be restored first: "
                + ", ".join(f"`{pair}`" for pair in missing_pairs[:3])
                + "."
            )
        guidance.append(
            "- For each repaired critical mapping, make the claim a narrow paraphrase of the snippet. If a mapping cannot be supported directly, weaken it or move/drop it instead of keeping it in the first 3."
        )
    if "evidence_map.mechanism_assertions" in missing_fields:
        if missing_field_set == {"evidence_map.mechanism_assertions"}:
            guidance.append(
                "- This is a mechanism-assertion completion pass. Keep `mechanism`, `prediction`, `test`, and the first 3 `evidence_map.variable_mappings` entries stable unless a referenced anchor is empty."
            )
            guidance.append(
                "- If only this field is missing, prefer returning only `{\"evidence_map\": {\"mechanism_assertions\": [...]}}` instead of rewriting the full candidate."
            )
        guidance.append(
            "- Produce at least one specific `evidence_map.mechanism_assertions` entry using already grounded target-domain evidence from the current payload. Do not invent a new mechanism, broaden the target claim, or swap in a different process."
        )
        guidance.append(
            "- Rewrite `evidence_map.mechanism_assertions` so at least one concise entry uses a direct target-domain snippet that explicitly names the same process noun phrase as `mechanism` or the same canonical metric as `test.metric`."
        )
        guidance.append(
            "- Keep each `mechanism_claim` concise and tied to the same target-domain process. Do not let `mechanism_claim` carry stronger process wording than the `evidence_snippet` itself. Prefer direct process or metric support over broad contextual target evidence."
        )
        guidance.append(
            "- Prefer filling one missing mechanism-assertion entry from the strongest current target-domain snippet already in the payload. Reuse exact or near-exact target-domain wording where available instead of rewriting the whole evidence map."
        )
        guidance.append(
            "- Do not use broad literature-summary assertions, background context, or adjacent field framing as `evidence_map.mechanism_assertions`."
        )
        if current_mechanism:
            guidance.append(
                f"- Keep the repaired mechanism assertion tied to the current `mechanism`: `{current_mechanism}`."
            )
        if observable_anchor:
            guidance.append(
                f"- Keep the repaired mechanism assertion tied to the current target claim in `prediction.observable`: `{observable_anchor}`."
            )
        elif metric_anchor:
            guidance.append(
                f"- Keep the repaired mechanism assertion tied to the current target claim anchor in `test.metric`: `{metric_anchor}`."
            )
        direct_anchor = str(core_target_anchor.get("evidence_snippet") or "").strip()
        direct_source = str(core_target_anchor.get("source_reference") or "").strip()
        if direct_anchor:
            guidance.append(
                "- Best current core target evidence to rewrite around: "
                f"`{direct_anchor}`"
                + (f" (source: `{direct_source}`)." if direct_source else ".")
            )
            guidance.append(
                "- Use that strongest current target snippet as the default `evidence_snippet` anchor for at least one repaired mechanism assertion entry unless another existing snippet in the payload is even more direct."
            )
        guidance.append(
            "- If the current payload and evidence cannot directly support the named mechanism with a concise same-process assertion, prefer `{\"no_connection\": true}` over vague mechanism-summary filler."
        )
        core_reasons = [
            str(reason).strip()
            for reason in (repair_context.get("core_target_reasons") or [])
            if str(reason).strip()
        ]
        if core_reasons:
            guidance.append(
                "- Current core-target-evidence weakness: "
                + "; ".join(core_reasons[:3])
                + "."
            )
    if not guidance:
        return ""
    return "\nExtra repair rules:\n" + "\n".join(guidance)


def _build_repair_prompt(
    full_prompt: str,
    original_json: str,
    missing_fields: list[str],
    *,
    original_data: dict | None = None,
) -> str:
    repair_prompt = MISSING_FIELDS_REPAIR_PROMPT.format(
        missing_fields=", ".join(missing_fields)
    )
    repair_guidance = _repair_guidance_for_missing_fields(
        missing_fields,
        original_data=original_data,
    )
    return (
        f"{repair_prompt}{repair_guidance}\n\nOriginal instruction:\n{full_prompt}\n\nOriginal JSON:\n{original_json}"
    )


def _get_nested_repair_value(payload: object, field_path: str) -> tuple[bool, object]:
    current = payload
    for part in str(field_path or "").split("."):
        if not part:
            return False, None
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current.get(part)
    return True, current


def _set_nested_repair_value(payload: dict, field_path: str, value: object) -> None:
    current = payload
    parts = [part for part in str(field_path or "").split(".") if part]
    if not parts:
        return
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = copy.deepcopy(value)


def _merge_targeted_salvage_fields(
    original_data: dict,
    repaired: dict,
    missing_fields: list[str],
) -> dict:
    merged = copy.deepcopy(original_data)
    applied_any = False
    seen_fields: set[str] = set()
    for field_path in missing_fields:
        clean_field = str(field_path).strip()
        if not clean_field or clean_field in seen_fields:
            continue
        seen_fields.add(clean_field)
        found, value = _get_nested_repair_value(repaired, clean_field)
        if not found:
            continue
        _set_nested_repair_value(merged, clean_field, value)
        applied_any = True
    return merged if applied_any else copy.deepcopy(repaired)


def _requested_salvage_fields_still_missing(
    candidate: dict,
    missing_fields: list[str],
) -> list[str]:
    remaining_missing = {
        str(field).strip()
        for field in _missing_required_fields(candidate)
        if str(field).strip()
    }
    unresolved: list[str] = []
    for field in missing_fields:
        clean_field = str(field).strip()
        if clean_field and clean_field in remaining_missing and clean_field not in unresolved:
            unresolved.append(clean_field)
    return unresolved


def _benchmark_salvage_guidance(benchmark_profile: dict | None) -> str:
    """Add narrow operator-edge guidance for replay benchmark candidates."""
    profile = benchmark_profile if isinstance(benchmark_profile, dict) else {}
    if not profile.get("benchmark_edge_candidate"):
        return ""

    operator_value_shape = str(profile.get("operator_value_shape") or "").strip()
    blocker_category = str(profile.get("remaining_blocker_category") or "").strip()
    guidance = [
        "Benchmark conversion priority:",
        "- This replay candidate already appears to contain real operator edge. Preserve that edge rather than broadening the claim into a generic research framing.",
    ]
    if operator_value_shape:
        guidance.append(
            f"- Current operator value shape: `{operator_value_shape}`."
        )
    if blocker_category:
        guidance.append(
            f"- Current blocker category: `{blocker_category}`. Repair only the packaging/alignment needed to clear that blocker."
        )
    guidance.append(
        "- Keep the same target-domain operator, metric, comparator, and workflow slice already grounded by `prediction`, `test`, and `edge_analysis`."
    )
    if blocker_category in {"operator_edge_packaging", "benchmark_packaging"}:
        guidance.append(
            "- When rewriting the edge layer, keep `cheap_test` as one narrow operator move such as a replay, audit, filter, or controlled compare on an existing workflow slice."
        )
    if blocker_category in {"mechanism_packaging", "benchmark_packaging"}:
        guidance.append(
            "- When rewriting `mechanism`, keep the same measurable target-domain process and do not drift into broader explanatory prose."
        )
    if operator_value_shape == "threshold tuning":
        guidance.append(
            "- Keep the edge framed as threshold calibration or tuning on the existing switching metric, not as a generic materials-study hypothesis."
        )
    elif operator_value_shape == "normalization audit":
        guidance.append(
            "- Keep the edge framed as a normalization or decision-threshold audit for the same borderline-value workflow, not as a generic informatics quality-improvement claim."
        )
    return "\n".join(guidance) + "\n"


def _repair_missing_fields(
    full_prompt: str,
    original_json: str,
    missing_fields: list[str],
    *,
    original_data: dict | None = None,
) -> dict | None:
    repair_prompt = _build_repair_prompt(
        full_prompt,
        original_json,
        missing_fields,
        original_data=original_data,
    )
    repair_max_output_tokens = 4096
    raw_override = str(os.getenv("BLACKCLAW_JUMP_REPAIR_MAX_OUTPUT_TOKENS", "")).strip()
    if raw_override:
        try:
            parsed_override = int(raw_override)
        except ValueError:
            parsed_override = 4096
        if parsed_override > 0:
            repair_max_output_tokens = parsed_override
    try:
        response = _llm_client.generate_content(
            repair_prompt,
            generation_config={
                "max_output_tokens": repair_max_output_tokens,
                "response_mime_type": "application/json",
            },
        )
        log_gemini_output("jump", "stage2_repair", response)
        increment_llm_calls(1)
        raw_output = response.text if getattr(response, "text", None) else ""
        checked = check_llm_output(raw_output)
        if checked is None:
            print("  [!] Jump stage2 repair output failed safety check")
            return None
        extracted = _extract_json_substring(checked)
        if extracted is None:
            return None
        data = json.loads(extracted)
        if isinstance(data, dict):
            return data
        return None
    except Exception as e:
        print(f"  [!] Jump stage2 repair call failed: {e}")
        return None


def salvage_high_value_candidate(
    original_data: dict,
    missing_fields: list[str],
    *,
    failure_reasons: list[str] | None = None,
    benchmark_profile: dict | None = None,
) -> dict | None:
    """Run one selective Phase 6 salvage rewrite for a strong near-miss."""
    if not isinstance(original_data, dict) or not missing_fields:
        return None
    guidance_prompt = PHASE6_SALVAGE_PROMPT
    normalized_reasons = [
        str(reason).strip()
        for reason in (failure_reasons or [])
        if str(reason).strip()
    ]
    if normalized_reasons:
        guidance_prompt += (
            "\nCurrent blockers:\n- " + "\n- ".join(normalized_reasons[:6]) + "\n"
        )
    guidance_prompt += _benchmark_salvage_guidance(benchmark_profile)
    repaired = _repair_missing_fields(
        guidance_prompt,
        json.dumps(original_data, ensure_ascii=False, sort_keys=True),
        missing_fields,
        original_data=original_data,
    )
    if repaired is None or repaired.get("no_connection", False):
        return None
    repaired_candidate = _merge_targeted_salvage_fields(
        original_data,
        repaired,
        missing_fields,
    )
    if not isinstance(repaired_candidate, dict):
        return None
    repaired_candidate = _apply_mechanism_naming_precision(repaired_candidate)
    repaired_candidate = _apply_normalized_mechanism_typing(repaired_candidate)
    if _requested_salvage_fields_still_missing(repaired_candidate, missing_fields):
        return None
    repaired_candidate["evidence_map"] = normalize_evidence_map(
        repaired_candidate.get("evidence_map")
    )
    return repaired_candidate


def _normalize_stage_one_solution_evidence_text(value: object) -> str:
    """Normalize Stage 1 solution evidence for placeholder and grounding checks."""
    return " ".join(_tokenize_query_terms(str(value or ""))).strip()


def _stage_one_solution_evidence_is_placeholder(solution_evidence: str) -> bool:
    """Return True for empty or placeholder-only solution evidence payloads."""
    normalized = _normalize_stage_one_solution_evidence_text(solution_evidence)
    if not normalized:
        return True
    if normalized in STAGE_ONE_SOLUTION_EVIDENCE_PLACEHOLDERS:
        return True
    return normalized.startswith("no ") and any(
        token in normalized
        for token in (
            "evidence",
            "intervention",
            "mitigation",
            "solution",
            "workaround",
        )
    )


def _stage_one_search_result_grounding_tokens(search_results: str) -> set[str]:
    """Extract non-metadata tokens from Stage 1 title/snippet lines."""
    evidence_lines: list[str] = []
    for raw_line in str(search_results or "").splitlines():
        clean_line = " ".join(str(raw_line or "").split()).strip()
        if not clean_line:
            continue
        lower_line = clean_line.lower()
        for prefix in ("title:", "snippet:"):
            if lower_line.startswith(prefix):
                evidence_lines.append(clean_line[len(prefix):].strip())
                break
    corpus = " ".join(evidence_lines).strip() or str(search_results or "")
    return {
        token
        for token in _tokenize_query_terms(corpus)
        if token not in GENERIC_QUERY_TOKENS
        and token not in WEAK_QUERY_TOKENS
        and token not in JUMP_QUERY_FILLER_TOKENS
        and token not in QUERY_PHRASE_STOPWORDS
        and token not in STAGE_ONE_SOLUTION_EVIDENCE_GENERIC_TOKENS
        and len(token) > 2
    }


def _stage_one_solution_evidence_is_grounded(
    solution_evidence: str,
    search_results: str,
) -> bool:
    """Check that Stage 1 solution evidence is concrete and grounded in retrieved text."""
    if _stage_one_solution_evidence_is_placeholder(solution_evidence):
        return False
    normalized_solution_evidence = " ".join(
        str(solution_evidence or "").split()
    ).strip()
    if not normalized_solution_evidence:
        return False
    if normalized_solution_evidence.lower() in str(search_results or "").lower():
        return True

    grounding_tokens = _stage_one_search_result_grounding_tokens(search_results)
    solution_tokens = {
        token
        for token in _tokenize_query_terms(normalized_solution_evidence)
        if token not in GENERIC_QUERY_TOKENS
        and token not in WEAK_QUERY_TOKENS
        and token not in JUMP_QUERY_FILLER_TOKENS
        and token not in QUERY_PHRASE_STOPWORDS
        and token not in STAGE_ONE_SOLUTION_EVIDENCE_GENERIC_TOKENS
        and len(token) > 2
    }
    if not solution_tokens:
        return False
    return len(solution_tokens.intersection(grounding_tokens)) >= min(
        2,
        len(solution_tokens),
    )


def _stage_one_detect_with_diagnostics(
    source_domain: str,
    abstract_structure: str,
    search_results: str,
    cycle_budget: CycleBudget | None = None,
) -> tuple[dict | None, str | None]:
    prompt = DETECT_PROMPT.format(
        source_domain=source_domain,
        abstract_structure=abstract_structure,
        search_results=search_results,
    )
    if cycle_budget is None:
        extracted_json = _generate_json_with_retry(prompt, "stage1_detect", 2048)
    else:
        extracted_json = _generate_json_with_retry(
            prompt,
            "stage1_detect",
            2048,
            cycle_budget=cycle_budget,
            budget_outcome="budget_exhausted_stage1",
        )
    if extracted_json is None:
        return None, "generation_failed"
    try:
        data = json.loads(extracted_json)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(data, dict):
        return None, "invalid_payload"
    if data.get("no_connection", True):
        return None, "no_connection"
    target_domain = str(data.get("target_domain", "")).strip()
    signal = str(data.get("signal", "")).strip()
    evidence = str(data.get("evidence", "")).strip()
    solution_evidence = str(data.get("solution_evidence", "")).strip()
    if not target_domain or not signal or not evidence:
        return None, "invalid_payload"
    data["target_domain"] = target_domain
    data["signal"] = signal
    data["evidence"] = evidence
    if not _stage_one_solution_evidence_is_grounded(
        solution_evidence,
        search_results,
    ):
        data.pop("solution_evidence", None)
        return data, "missing_solution_evidence"
    data["solution_evidence"] = solution_evidence
    return data, None


def _stage_one_detect(
    source_domain: str,
    abstract_structure: str,
    search_results: str,
) -> dict | None:
    data, _failure_hint = _stage_one_detect_with_diagnostics(
        source_domain=source_domain,
        abstract_structure=abstract_structure,
        search_results=search_results,
    )
    if _failure_hint is not None:
        return None
    return data


def _is_stage_one_weak_signal(stage_one: dict | None, failure_hint: str | None) -> bool:
    if failure_hint != "missing_solution_evidence" or not isinstance(stage_one, dict):
        return False
    return all(
        str(stage_one.get(field, "") or "").strip()
        for field in ("target_domain", "signal", "evidence")
    )


def _classify_stage_one_outcome(
    stage_one: dict | None,
    failure_hint: str | None,
) -> str:
    if _is_stage_one_weak_signal(stage_one, failure_hint):
        return "weak_signal"
    if isinstance(stage_one, dict) and failure_hint is None:
        return "detect_signal"
    if failure_hint == "no_connection":
        return "detect_no_signal"
    return "no_results"


def _build_stage_one_soft_gate_query(
    built_jump_query: str,
    target_domain: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        (
            f"{str(built_jump_query or '').strip()} "
            f"{str(target_domain or '').strip()} workaround mitigation intervention control"
        ),
    ).strip()


def _short_stage_two_failure_hint(payload: object) -> str | None:
    """Extract one compact Stage 2 no-connection hint when the model provides one."""
    if not isinstance(payload, dict):
        return None
    for key in ("failure_hint", "reason", "why", "message", "note"):
        value = str(payload.get(key, "") or "").strip()
        if value:
            return " ".join(value.split())[:120]
    return None


STAGE2_SUBSTAGE_MAX_OUTPUT_TOKENS = {
    "mechanism": 4096,
    "predict": 4096,
    "test": 4096,
    "edge": 4096,
}

STAGE2_SUBSTAGE_EXTRA_REQUIRED_FIELDS = {
    "test": ["edge_analysis.cheap_test.time_to_signal"],
    "edge": ["edge_analysis.deployment_scope"],
}


def _initialize_stage_two_candidate(source_domain: str, stage_one: dict) -> dict:
    candidate = {
        "no_connection": False,
        "source_domain": str(source_domain or "").strip(),
        "target_domain": str(stage_one.get("target_domain") or "").strip(),
        "evidence": str(stage_one.get("evidence") or "").strip(),
    }
    solution_evidence = str(stage_one.get("solution_evidence") or "").strip()
    if solution_evidence:
        candidate["solution_evidence"] = solution_evidence
    return candidate


def _stage_two_field_owner(field_path: str) -> str | None:
    clean_field = str(field_path or "").strip()
    if not clean_field:
        return None
    for substage in STAGE2_SUBSTAGE_SEQUENCE:
        for owned_field in STAGE2_SUBSTAGE_FIELD_OWNERSHIP.get(substage, []):
            if clean_field == owned_field:
                return substage
            if clean_field.startswith(f"{owned_field}."):
                return substage
    return None


def _stage_two_field_has_value(payload: dict, field_path: str) -> bool:
    found, value = _get_nested_repair_value(payload, field_path)
    if not found:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return value is not None


def _merge_stage_two_selected_fields(
    original_data: dict,
    updated_data: dict,
    field_paths: list[str],
) -> dict:
    merged = copy.deepcopy(original_data)
    for field_path in field_paths:
        found, value = _get_nested_repair_value(updated_data, field_path)
        if not found:
            continue
        _set_nested_repair_value(merged, field_path, value)
    return merged


def _normalize_stage_two_candidate(data: dict) -> dict:
    normalized = copy.deepcopy(data)
    normalized = _apply_mechanism_naming_precision(normalized)
    normalized = _apply_normalized_mechanism_typing(normalized)
    normalized["evidence_map"] = normalize_evidence_map(
        normalized.get("evidence_map")
    )
    normalized.pop("depth", None)
    return normalized


def _missing_fields_for_stage_two_substage(data: dict, substage: str) -> list[str]:
    missing_fields = [
        field
        for field in _missing_required_fields(data)
        if _stage_two_field_owner(field) == substage
    ]
    if substage == "test":
        missing_fields = [
            field for field in missing_fields if field == "test" or field.startswith("test.")
        ]
        edge_analysis = (
            data.get("edge_analysis") if isinstance(data.get("edge_analysis"), dict) else {}
        )
        cheap_test = (
            edge_analysis.get("cheap_test")
            if isinstance(edge_analysis.get("cheap_test"), dict)
            else {}
        )
        setup = str(cheap_test.get("setup") or "").strip().lower()
        if (
            not isinstance(cheap_test, dict)
            or not str(cheap_test.get("setup") or "").strip()
            or not str(cheap_test.get("metric") or "").strip()
            or not str(cheap_test.get("confirm") or "").strip()
            or not str(cheap_test.get("falsify") or "").strip()
            or "run a study" in setup
            or "validate the hypothesis" in setup
            or "collect more data" in setup
            or "effect appears" in setup
        ):
            if "edge_analysis.cheap_test" not in missing_fields:
                missing_fields.append("edge_analysis.cheap_test")
    for field_path in STAGE2_SUBSTAGE_EXTRA_REQUIRED_FIELDS.get(substage, []):
        if (
            not _stage_two_field_has_value(data, field_path)
            and field_path not in missing_fields
        ):
            missing_fields.append(field_path)
    return missing_fields


def _build_stage_two_substage_prompt(
    substage: str,
    *,
    source_domain: str,
    abstract_structure: str,
    stage_one: dict,
    current_candidate: dict,
    search_results: str,
    relevant_scars: str,
) -> str:
    return STAGE2_SUBSTAGE_PROMPTS[substage].format(
        source_domain=source_domain,
        abstract_structure=abstract_structure,
        stage_one_json=json.dumps(stage_one, ensure_ascii=False, sort_keys=True),
        current_stage_json=json.dumps(
            current_candidate, ensure_ascii=False, sort_keys=True
        ),
        search_results=search_results,
        relevant_scars=relevant_scars,
        mechanism_vocab=MECHANISM_VOCAB_TEXT,
    )


def _run_stage_two_substage(
    substage: str,
    *,
    source_domain: str,
    abstract_structure: str,
    stage_one: dict,
    current_candidate: dict,
    search_results: str,
    relevant_scars: str,
) -> tuple[dict | None, str | None, list[str] | None]:
    prompt = _build_stage_two_substage_prompt(
        substage,
        source_domain=source_domain,
        abstract_structure=abstract_structure,
        stage_one=stage_one,
        current_candidate=current_candidate,
        search_results=search_results,
        relevant_scars=relevant_scars,
    )
    extracted_json = _generate_json_with_retry(
        prompt,
        STAGE2_SUBSTAGE_STAGE_NAMES[substage],
        STAGE2_SUBSTAGE_MAX_OUTPUT_TOKENS[substage],
    )
    if extracted_json is None:
        return None, "generation_failed", None
    try:
        data = json.loads(extracted_json)
    except json.JSONDecodeError:
        return None, "invalid_json", None
    if not isinstance(data, dict):
        return None, "invalid_payload", None
    if data.get("no_connection", False):
        return None, _short_stage_two_failure_hint(data) or "returned_no_connection", None

    owned_fields = STAGE2_SUBSTAGE_FIELD_OWNERSHIP[substage]
    candidate = _merge_stage_two_selected_fields(
        current_candidate,
        data,
        owned_fields,
    )
    candidate = _normalize_stage_two_candidate(candidate)
    missing_fields = _missing_fields_for_stage_two_substage(candidate, substage)
    if not missing_fields:
        return candidate, None, None

    repaired = _repair_missing_fields(
        prompt,
        extracted_json,
        missing_fields,
        original_data=candidate,
    )
    if repaired is None:
        return None, "repair_failed", None
    if repaired.get("no_connection", False):
        return None, _short_stage_two_failure_hint(repaired) or "returned_no_connection", None

    candidate = _merge_stage_two_selected_fields(
        candidate,
        repaired,
        missing_fields,
    )
    candidate = _normalize_stage_two_candidate(candidate)
    incomplete_fields = _missing_fields_for_stage_two_substage(candidate, substage)
    if incomplete_fields:
        return None, "repair_incomplete", incomplete_fields
    return candidate, None, None


def _format_relevant_scars_for_prompt(
    target_domain: str,
    abstract_structure: str,
    limit: int = 4,
) -> str:
    clean_target = str(target_domain or "").strip()
    clean_abstract = str(abstract_structure or "").strip()
    if not clean_target or not clean_abstract:
        return "None."
    try:
        abstract_structure_embedding = _llm_client.embed_content(clean_abstract)
        scars = get_relevant_scars(
            clean_target,
            abstract_structure_embedding,
            limit=limit,
        )
    except Exception:
        return "None."
    if not scars:
        return "None."

    blocks = []
    for index, scar in enumerate(scars, start=1):
        blocks.append(
            "\n".join(
                [
                    f"{index}. constraint_rule: {str(scar.get('constraint_rule') or '').strip()}",
                    f"applies_when: {str(scar.get('applies_when') or '').strip()}",
                    f"why_it_failed: {str(scar.get('why_it_failed') or '').strip()}",
                    "does_not_apply_when: "
                    f"{str(scar.get('does_not_apply_when') or 'n/a').strip()}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _stage_two_hypothesize_with_diagnostics(
    source_domain: str,
    abstract_structure: str,
    stage_one: dict,
    search_results: str,
) -> tuple[dict | None, str | None, list[str] | None]:
    setattr(_stage_two_hypothesize_with_diagnostics, "last_failed_at", None)
    relevant_scars = _format_relevant_scars_for_prompt(
        str(stage_one.get("target_domain") or ""),
        abstract_structure,
    )
    candidate = _initialize_stage_two_candidate(source_domain, stage_one)
    for substage in STAGE2_SUBSTAGE_SEQUENCE:
        candidate, failure_hint, incomplete_fields = _run_stage_two_substage(
            substage,
            source_domain=source_domain,
            abstract_structure=abstract_structure,
            stage_one=stage_one,
            current_candidate=candidate,
            search_results=search_results,
            relevant_scars=relevant_scars,
        )
        if candidate is None:
            setattr(_stage_two_hypothesize_with_diagnostics, "last_failed_at", substage)
            return None, failure_hint, incomplete_fields

    candidate = _normalize_stage_two_candidate(candidate)
    incomplete_fields = _missing_required_fields(candidate)
    if incomplete_fields:
        setattr(
            _stage_two_hypothesize_with_diagnostics,
            "last_failed_at",
            next(
                (
                    _stage_two_field_owner(field)
                    for field in incomplete_fields
                    if _stage_two_field_owner(field) is not None
                ),
                None,
            ),
        )
        return None, "repair_incomplete", incomplete_fields
    return candidate, None, None


def _stage_two_hypothesize(
    source_domain: str,
    abstract_structure: str,
    stage_one: dict,
    search_results: str,
) -> dict | None:
    stage_two_result = _stage_two_hypothesize_with_diagnostics(
        source_domain=source_domain,
        abstract_structure=abstract_structure,
        stage_one=stage_one,
        search_results=search_results,
    )
    data = stage_two_result[0] if isinstance(stage_two_result, tuple) and stage_two_result else None
    return data


def _make_pre_stage1_dependencies() -> PreStage1Dependencies:
    return PreStage1Dependencies(
        build_jump_search_queries=_build_pre_stage1_jump_query_build_result,
        tavily_search=_tavily.search,
        increment_tavily_calls=increment_tavily_calls,
        academic_jump_include_domains=ACADEMIC_JUMP_INCLUDE_DOMAINS,
    )


def _apply_pre_stage1_diagnostics(
    diagnostic: JumpAttemptDiagnostic,
    pre_stage1_diagnostics,
) -> None:
    diagnostic.update(pre_stage1_diagnostics.to_dict())


def _apply_stage_one_diagnostic(
    diagnostic: JumpAttemptDiagnostic,
    stage_one: dict | None,
    stage_one_outcome: str,
    stage_one_failure_hint: str | None,
) -> None:
    diagnostic["stage1_outcome"] = stage_one_outcome
    diagnostic["stage1_target_domain"] = (
        str(stage_one.get("target_domain", "") or "").strip()
        if isinstance(stage_one, dict)
        else None
    ) or None
    diagnostic["stage1_failure_hint"] = (
        None if stage_one_outcome == "detect_signal" else stage_one_failure_hint
    )


def _return_runtime_terminal_result(
    diagnostic: JumpAttemptDiagnostic,
    *,
    stage_one_outcome: str,
    stage_one_failure_hint: str | None,
    stage_one: dict | None = None,
    budget_stop: dict | None = None,
) -> tuple[None, JumpAttemptDiagnostic]:
    _apply_stage_one_diagnostic(
        diagnostic,
        stage_one,
        str(stage_one_outcome or "").strip() or "no_results",
        stage_one_failure_hint,
    )
    if budget_stop is not None:
        diagnostic["budget_stop"] = budget_stop
    return None, diagnostic


def _apply_runtime_result_context(
    diagnostic: JumpAttemptDiagnostic,
    *,
    pattern: dict | None = None,
    source_domain: str | None = None,
    raw_target_candidates: list[dict] | None = None,
    data: dict | None = None,
    stage_one: dict | None = None,
    search_results: str | None = None,
) -> dict | None:
    benchmark_snapshot = diagnostic.get("benchmark_snapshot")
    if isinstance(benchmark_snapshot, dict):
        if search_results is not None:
            benchmark_snapshot["search_results"] = search_results
        if isinstance(stage_one, dict):
            benchmark_snapshot["stage_one_success"] = dict(stage_one)
        diagnostic["benchmark_snapshot"] = benchmark_snapshot

    if data is None or pattern is None or source_domain is None:
        return data

    data["abstract_structure"] = str(pattern.get("abstract_structure", "") or "").strip()
    target_excerpt, target_url = _select_display_target_evidence(
        data,
        raw_target_candidates or [],
    )
    seed_url, seed_excerpt = _select_display_source_evidence(
        pattern,
        data,
        source_domain,
    )
    if target_url:
        data["target_url"] = target_url
    if target_excerpt:
        data["target_excerpt"] = target_excerpt
    if seed_url:
        data["seed_url"] = seed_url
        data["source_url"] = seed_url
    if seed_excerpt:
        data["seed_excerpt"] = seed_excerpt
        data["source_excerpt"] = seed_excerpt
    return data


def _run_post_pre_stage1_flow(
    *,
    source_domain: str,
    abstract_structure: str,
    search_results: str,
    diagnostic: JumpAttemptDiagnostic,
    cycle_budget: CycleBudget | None = None,
    stage_one_success: dict | None = None,
) -> tuple[dict | None, dict | None, str, str | None]:
    stage_one = copy.deepcopy(stage_one_success) if isinstance(stage_one_success, dict) else None
    stage_one_failure_hint = None

    if stage_one is None:
        stage_one, stage_one_failure_hint = _stage_one_detect_with_diagnostics(
            source_domain=source_domain,
            abstract_structure=abstract_structure,
            search_results=search_results,
            cycle_budget=cycle_budget,
        )
        stage_one_outcome = _classify_stage_one_outcome(
            stage_one,
            stage_one_failure_hint,
        )
    else:
        stage_one_outcome = "detect_signal"

    _apply_stage_one_diagnostic(
        diagnostic,
        stage_one,
        stage_one_outcome,
        stage_one_failure_hint,
    )
    if stage_one_outcome != "detect_signal" or not isinstance(stage_one, dict):
        return None, stage_one, stage_one_outcome, stage_one_failure_hint

    stage_two_result = _stage_two_hypothesize_with_diagnostics(
        source_domain=source_domain,
        abstract_structure=abstract_structure,
        stage_one=stage_one,
        search_results=search_results,
    )
    data = (
        stage_two_result[0]
        if isinstance(stage_two_result, tuple) and len(stage_two_result) >= 1
        else None
    )
    stage_two_failure_hint = (
        stage_two_result[1]
        if isinstance(stage_two_result, tuple) and len(stage_two_result) >= 2
        else None
    )
    stage2_incomplete_fields = (
        stage_two_result[2]
        if isinstance(stage_two_result, tuple) and len(stage_two_result) >= 3
        else None
    )
    stage2_failed_at = getattr(
        _stage_two_hypothesize_with_diagnostics,
        "last_failed_at",
        None,
    )
    if data is None:
        diagnostic["stage2_outcome"] = "stage2_no_connection"
        diagnostic["stage2_failure_hint"] = (
            stage_two_failure_hint or "returned_no_connection"
        )
        diagnostic["stage2_failed_at"] = (
            str(stage2_failed_at).strip() or None
            if stage2_failed_at is not None
            else None
        )
        if (
            diagnostic["stage2_failure_hint"] == "repair_incomplete"
            and isinstance(stage2_incomplete_fields, list)
        ):
            cleaned_incomplete_fields = [
                str(field).strip()
                for field in stage2_incomplete_fields
                if str(field).strip()
            ]
            if cleaned_incomplete_fields:
                diagnostic["stage2_incomplete_fields"] = cleaned_incomplete_fields
        return None, stage_one, stage_one_outcome, stage_one_failure_hint

    diagnostic["stage2_outcome"] = "connection_found"
    diagnostic["stage2_target_domain"] = (
        str(data.get("target_domain", "") or "").strip() or None
    )
    return data, stage_one, stage_one_outcome, stage_one_failure_hint


def lateral_jump_with_diagnostics(
    pattern: dict,
    source_domain: str,
    source_category: str,
    cycle_budget: CycleBudget | None = None,
) -> tuple[dict | None, JumpAttemptDiagnostic]:
    """
    Attempt a lateral jump and return lightweight diagnostics describing where it died.
    """
    raw_search_query = str(pattern.get("search_query", "") or "").strip()
    diagnostic = JumpAttemptDiagnostic(
        pattern_name=str(pattern.get("pattern_name", "") or "").strip() or "Unknown",
        abstract_structure=str(pattern.get("abstract_structure", "") or "").strip(),
        raw_search_query=raw_search_query,
    )

    pre_stage1_dependencies = _make_pre_stage1_dependencies()
    pre_stage1_result = run_pre_stage1(
        pattern,
        source_domain,
        source_category,
        cycle_budget=cycle_budget,
        deps=pre_stage1_dependencies,
    )
    _apply_pre_stage1_diagnostics(diagnostic, pre_stage1_result.diagnostics)
    if pre_stage1_result.stage1_outcome is not None:
        return _return_runtime_terminal_result(
            diagnostic,
            stage_one_outcome=pre_stage1_result.stage1_outcome,
            stage_one_failure_hint=pre_stage1_result.stage1_failure_hint,
            budget_stop=pre_stage1_result.budget_stop,
        )

    pre_stage1_state = pre_stage1_result.state
    if pre_stage1_state is None or pre_stage1_state.packet is None:
        return _return_runtime_terminal_result(
            diagnostic,
            stage_one_outcome="no_results",
            stage_one_failure_hint="no_usable_results",
        )

    query = str(pre_stage1_state.query_plan.built_jump_query or "").strip()
    combined = pre_stage1_state.packet.search_content
    raw_target_candidates = pre_stage1_state.packet.raw_target_candidates

    try:
        data, stage_one, stage_one_outcome, stage_one_failure_hint = _run_post_pre_stage1_flow(
            source_domain=source_domain,
            abstract_structure=str(pattern.get("abstract_structure", "") or ""),
            search_results=combined,
            diagnostic=diagnostic,
            cycle_budget=cycle_budget,
        )
    except CycleBudgetExhausted as exhausted:
        return _return_runtime_terminal_result(
            diagnostic,
            stage_one_outcome="budget_exhausted_stage1",
            stage_one_failure_hint="cycle_budget_exhausted",
            budget_stop=exhausted.to_diagnostic(),
        )
    initial_stage_one = stage_one
    initial_stage_one_failure_hint = stage_one_failure_hint
    initial_stage_one_outcome = stage_one_outcome

    if stage_one_outcome == "weak_signal":
        diagnostic["stage1_soft_gate_attempted"] = True
        recovery_query = _build_stage_one_soft_gate_query(
            query,
            str(stage_one.get("target_domain", "") or "").strip(),
        )
        if recovery_query:
            recovery_result = augment_pre_stage1_with_query(
                pre_stage1_state,
                pre_stage1_result.diagnostics,
                query=recovery_query,
                general_query_label="soft-gate",
                academic_query_label="soft-gate-academic",
                general_callsite="stage1_soft_gate_search",
                academic_callsite="stage1_soft_gate_academic_search",
                general_failure_message=(
                    "  [!] Tavily soft-gate search failed for jump query "
                    f"'{recovery_query}': {{error}}"
                ),
                academic_failure_message=(
                    "[!] Tavily academic soft-gate search failed for jump query "
                    f"'{recovery_query}': {{error}}"
                ),
                cycle_budget=cycle_budget,
                deps=pre_stage1_dependencies,
            )
            _apply_pre_stage1_diagnostics(diagnostic, recovery_result.diagnostics)
            if recovery_result.stage1_outcome is not None:
                return _return_runtime_terminal_result(
                    diagnostic,
                    stage_one_outcome=recovery_result.stage1_outcome,
                    stage_one_failure_hint=recovery_result.stage1_failure_hint,
                    budget_stop=recovery_result.budget_stop,
                )
            if recovery_result.state is not None and recovery_result.state.packet is not None:
                pre_stage1_state = recovery_result.state
                combined = pre_stage1_state.packet.search_content
                raw_target_candidates = pre_stage1_state.packet.raw_target_candidates
                _apply_runtime_result_context(
                    diagnostic,
                    search_results=combined,
                )

        try:
            data, recovered_stage_one, recovered_stage_one_outcome, recovered_stage_one_failure_hint = _run_post_pre_stage1_flow(
                source_domain=source_domain,
                abstract_structure=str(pattern.get("abstract_structure", "") or ""),
                search_results=combined,
                diagnostic=diagnostic,
                cycle_budget=cycle_budget,
            )
        except CycleBudgetExhausted as exhausted:
            return _return_runtime_terminal_result(
                diagnostic,
                stage_one_outcome="budget_exhausted_stage1",
                stage_one_failure_hint="cycle_budget_exhausted",
                budget_stop=exhausted.to_diagnostic(),
            )
        if recovered_stage_one_outcome == "detect_signal":
            stage_one = recovered_stage_one
            stage_one_failure_hint = recovered_stage_one_failure_hint
            stage_one_outcome = recovered_stage_one_outcome
            diagnostic["stage1_soft_gate_recovered"] = True
        else:
            return _return_runtime_terminal_result(
                diagnostic,
                stage_one_outcome=initial_stage_one_outcome,
                stage_one_failure_hint=initial_stage_one_failure_hint,
                stage_one=initial_stage_one,
            )

    if stage_one_outcome != "detect_signal":
        return None, diagnostic

    if data is None:
        return None, diagnostic
    data = _apply_runtime_result_context(
        diagnostic,
        pattern=pattern,
        source_domain=source_domain,
        raw_target_candidates=raw_target_candidates,
        data=data,
        stage_one=stage_one,
    )
    return data, diagnostic


def replay_jump_attempt(
    snapshot: JumpReplaySnapshot,
) -> tuple[dict | None, JumpAttemptDiagnostic]:
    """Replay one captured jump attempt through Stage 1 and Stage 2 only."""
    replay_snapshot = JumpReplaySnapshot.from_dict(snapshot) or JumpReplaySnapshot()
    source_domain = str(replay_snapshot.source_domain or "").strip() or "Unknown"
    abstract_structure = str(replay_snapshot.abstract_structure or "").strip()
    search_results = str(replay_snapshot.search_results or "").strip()
    diagnostic = JumpAttemptDiagnostic(
        pattern_name=str(replay_snapshot.pattern_name or "").strip() or "Unknown",
        abstract_structure=abstract_structure,
        built_jump_query=str(replay_snapshot.built_jump_query or "").strip() or None,
        replay_snapshot=replay_snapshot,
    )

    data, _stage_one, _stage_one_outcome, _stage_one_failure_hint = _run_post_pre_stage1_flow(
        source_domain=source_domain,
        abstract_structure=abstract_structure,
        search_results=search_results,
        diagnostic=diagnostic,
        stage_one_success=(
            replay_snapshot.stage_one_success
            if isinstance(replay_snapshot.stage_one_success, dict)
            else None
        ),
    )
    return data, diagnostic


def lateral_jump(
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> dict | None:
    """
    Attempt a lateral jump:
    1. Search for the abstract pattern in other domains
    2. Stage 1 detect if a real structural signal exists
    3. Stage 2 hypothesize a mechanism-first mapping
    4. Return connection dict or None
    """
    data, _diagnostic = lateral_jump_with_diagnostics(
        pattern,
        source_domain,
        source_category,
    )
    return data
