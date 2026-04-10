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

const state = {
  isDarkMode: true,
  activeWorkspace: "frontier",
  activeFrontierMode: "regions",
  dashboardStats: null,
  dashboardCosts: null,
  frontier: null,
  operatorHome: null,
  domainGraph: null,
  currentClusterOverview: null,
  currentDomainGraph: null,
  currentDomainGraphLayout: null,
  transmissionRows: [],
  evidenceQueueRows: [],
  outcomeQueueRows: [],
  strongRejectionRows: [],
  selectedEvidenceId: null,
  selectedOutcomePredictionId: null,
  selectedStrongRejectionId: null,
  selectedGraphClusterId: null,
  selectedGraphNodeId: null,
  selectedGraphLinkKey: null,
  hoveredGraphClusterId: null,
  hoveredGraphNodeId: null,
  hoveredGraphLinkKey: null,
  graphSceneMode: "overview",
  graphViewport: {
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    targetScale: 1,
    targetOffsetX: 0,
    targetOffsetY: 0,
    response: 0.18,
    targetResponse: 0.18,
  },
  graphMotion: {
    pointerParallaxX: 0,
    pointerParallaxY: 0,
    targetParallaxX: 0,
    targetParallaxY: 0,
    currentParallaxX: 0,
    currentParallaxY: 0,
  },
  graphPointer: {
    isPanning: false,
    didPan: false,
    startX: 0,
    startY: 0,
    originOffsetX: 0,
    originOffsetY: 0,
    lastX: 0,
    lastY: 0,
    lastTime: 0,
    velocityX: 0,
    velocityY: 0,
    dragType: null,
    dragId: null,
    dragOffsetX: 0,
    dragOffsetY: 0,
  },
  domainGraphControlsInitialized: false,
  domainGraphSearchTimer: null,
  domainGraphResizeTimer: null,
  domainGraphLayoutCache: new Map(),
  clusterGraphLayoutCache: new Map(),
  graphBackdropCache: null,
  graphRipples: [],
  graphAnimationFrameId: null,
  graphAnimationLastFrame: 0,
  graphAnimationTime: 0,
  graphSceneTransition: null,
};

function currentPageKey() {
  return document.body.dataset.page || "home";
}

function canRenderGraphHere() {
  return Boolean(document.getElementById("domain-web-canvas"));
}

function navigateToMapWithFocus(params = {}) {
  const url = new URL("/map", window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value != null && value !== "") {
      url.searchParams.set(key, value);
    }
  });
  window.location.href = url.toString();
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
  return escapeHtml(String(value ?? "").replaceAll("\n", " "));
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

function applyTheme() {
  document.documentElement.dataset.theme = state.isDarkMode ? "dark" : "light";
  const toggle = document.getElementById("theme-toggle");
  if (toggle) {
    toggle.textContent = state.isDarkMode ? "Light Mode" : "Dark Mode";
  }
  if (state.domainGraph) {
    renderDomainGraphCanvas();
    renderDomainGraphDetail();
  }
}

function setActiveWorkspace(workspace) {
  state.activeWorkspace = workspace;
  document.querySelectorAll(".workspace-tab").forEach((button) => {
    const isActive = button.dataset.workspace === workspace;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
  });
  document.querySelectorAll(".workspace-view").forEach((panel) => {
    const isActive = panel.dataset.workspacePanel === workspace;
    panel.classList.toggle("is-active", isActive);
    panel.hidden = !isActive;
  });
}

function initializeWorkspaceTabs() {
  document.querySelectorAll(".workspace-tab").forEach((button) => {
    button.addEventListener("click", () => {
      setActiveWorkspace(button.dataset.workspace || "frontier");
    });
  });
}

function renderHeroSummary() {
  const container = document.getElementById("hero-summary");
  if (!container) {
    return;
  }
  if (!state.dashboardStats && !state.dashboardCosts && !state.operatorHome && !state.domainGraph) {
    container.innerHTML = "<p class=\"muted\">Loading live snapshot…</p>";
    return;
  }

  const graphStats = state.domainGraph ? state.domainGraph.stats || {} : {};
  const operatorCounts = state.operatorHome ? state.operatorHome.counts || {} : {};
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
      label: "Transmitted bridges",
      value: graphStats.transmitted_connections != null ? formatInteger(graphStats.transmitted_connections) : "…",
      valueClass: "score-accent",
    },
    {
      label: "Unreviewed evidence",
      value: state.operatorHome ? formatInteger(operatorCounts.unreviewed_evidence_hits || 0) : "…",
    },
    {
      label: "Open strong rejections",
      value: state.operatorHome ? formatInteger(operatorCounts.open_strong_rejections || 0) : "…",
    },
    {
      label: "Cost per transmission",
      value: state.dashboardCosts && state.dashboardCosts.available
        ? formatCurrency(state.dashboardCosts.cost_per_transmission)
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

function renderFrontierStats(stats) {
  const container = document.getElementById("frontier-stats");
  if (!container) {
    return;
  }
  const items = [
    { label: "High-signal seeds", value: formatInteger(stats.high_signal_seeds || 0), valueClass: "score-accent" },
    { label: "High-signal bridges", value: formatInteger(stats.high_signal_bridges || 0), valueClass: "score-accent" },
    { label: "Open salvage candidates", value: formatInteger(stats.open_salvage_candidates || 0) },
  ];
  container.innerHTML = items.map((item) => `
    <div class="stat">
      <div class="muted">${escapeHtml(item.label)}</div>
      <div class="stat-value ${item.valueClass || ""}">${escapeHtml(item.value)}</div>
    </div>
  `).join("");
}

function clusterLabelForId(clusterId) {
  const clusters = Array.isArray(state.domainGraph?.clusters) ? state.domainGraph.clusters : [];
  return clusters.find((cluster) => cluster.id === clusterId)?.label || "Unknown cluster";
}

function setFrontierMoveMode(mode) {
  state.activeFrontierMode = mode;
  document.querySelectorAll(".frontier-mode-button").forEach((button) => {
    const isActive = button.dataset.frontierMode === mode;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
  });
  const regionsView = document.getElementById("frontier-next-regions");
  const seedsView = document.getElementById("frontier-next-seeds");
  if (regionsView) {
    regionsView.hidden = mode !== "regions";
  }
  if (seedsView) {
    seedsView.hidden = mode !== "seeds";
  }
}

function renderNextMoveCards(rows, mode) {
  if (!rows.length) {
    return `<p class="muted">No ${mode === "regions" ? "regions" : "seed domains"} are ranking yet.</p>`;
  }
  return `
    <div class="next-moves-grid">
      ${rows.map((row) => {
        const isRegion = mode === "regions";
        const title = isRegion ? row.label : row.domain;
        const metaItems = isRegion
          ? [
              `${formatInteger(row.node_count || 0)} domains`,
              `${formatInteger(row.link_count || 0)} mapped bridges`,
              `${formatInteger(row.transmitted_link_count || 0)} transmitting`,
            ]
          : [
              `${formatInteger(row.exploration_count || 0)} explorations`,
              `${formatInteger(row.transmitted_count || 0)} transmissions`,
              clusterLabelForId(row.cluster_id),
            ];
        const buttonAttributes = isRegion
          ? `data-focus-cluster="${escapeHtml(row.cluster_id)}"`
          : `data-focus-domain="${escapeHtml(row.domain)}"`;
        const buttonLabel = isRegion ? "Open Region" : "Focus Seed";
        const domainChips = isRegion && Array.isArray(row.top_domains) && row.top_domains.length
          ? `
              <div class="frontier-cluster-domains">
                ${row.top_domains.slice(0, 4).map((domain) => `
                  <span class="frontier-domain-chip">${escapeHtml(truncateText(domain, 30))}</span>
                `).join("")}
              </div>
            `
          : "";
        return `
          <article class="next-move-card">
            <div class="next-move-top">
              <div>
                <h3>${escapeHtml(title || "Unknown")}</h3>
                <div class="frontier-card-meta">
                  ${metaItems.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
                </div>
              </div>
              <div class="next-move-score">
                <strong>${escapeHtml(formatAverage(row.frontier_score))}</strong>
                <span>frontier</span>
              </div>
            </div>
            <div class="pill-row">
              ${(Array.isArray(row.why) ? row.why : []).map((reason) => `
                <span class="frontier-pill">${escapeHtml(reason)}</span>
              `).join("")}
            </div>
            ${domainChips}
            <div class="frontier-card-actions">
              <button type="button" class="graph-focus-button" ${buttonAttributes}>${buttonLabel}</button>
            </div>
          </article>
        `;
      }).join("")}
    </div>
  `;
}

function renderFrontierLaunchpads(rows) {
  const container = document.getElementById("frontier-launchpads");
  if (!container) {
    return;
  }
  if (!rows.length) {
    container.innerHTML = "<p class=\"muted\">No launchpads available yet.</p>";
    return;
  }
  container.innerHTML = `
    <div class="frontier-card-list">
      ${rows.map((row) => `
        <article class="frontier-card">
          <div class="frontier-card-top">
            <div>
              <h3>${escapeHtml(row.domain || "Unknown domain")}</h3>
              <div class="frontier-card-meta">
                <span>${escapeHtml(formatInteger(row.exploration_count || 0))} explorations</span>
                <span>${escapeHtml(formatInteger(row.transmitted_count || 0))} transmitted</span>
                <span>avg ${escapeHtml(formatScore(row.avg_score))}</span>
                <span>frontier ${escapeHtml(formatAverage(row.frontier_score))}</span>
              </div>
            </div>
            <div class="frontier-card-actions">
              <button type="button" class="graph-focus-button" data-focus-domain="${escapeHtml(row.domain)}">Focus In Graph</button>
            </div>
          </div>
          <p class="muted">Latest activity: ${escapeHtml(formatTimestamp(row.latest_timestamp))}</p>
        </article>
      `).join("")}
    </div>
  `;
}

function renderFrontierBridges(rows) {
  const container = document.getElementById("frontier-bridges");
  if (!container) {
    return;
  }
  if (!rows.length) {
    container.innerHTML = "<p class=\"muted\">No strong bridges available yet.</p>";
    return;
  }
  const body = rows.map((row) => `
    <tr>
      <td><button type="button" class="graph-focus-button" data-focus-domain="${escapeHtml(row.source)}">${escapeHtml(truncateText(row.source, 36))}</button></td>
      <td><button type="button" class="graph-focus-button" data-focus-domain="${escapeHtml(row.target)}">${escapeHtml(truncateText(row.target, 36))}</button></td>
      <td>${escapeHtml(formatInteger(row.count || 0))}</td>
      <td>${escapeHtml(formatInteger(row.transmitted_count || 0))}</td>
      <td class="score-accent">${escapeHtml(formatScore(row.max_score))}</td>
      <td>${escapeHtml(formatTimestamp(row.latest_timestamp))}</td>
    </tr>
  `).join("");
  container.innerHTML = `
    <div class="table-shell">
      <table>
        <thead>
          <tr>
            <th>Seed</th>
            <th>Target</th>
            <th>Count</th>
            <th>Transmitted</th>
            <th>Best score</th>
            <th>Latest</th>
          </tr>
        </thead>
        <tbody>${body}</tbody>
      </table>
    </div>
  `;
}

function renderSalvageCandidates(rows, targetId) {
  const container = document.getElementById(targetId);
  if (!container) {
    return;
  }
  if (!rows.length) {
    container.innerHTML = "<p class=\"muted\">No open salvage candidates.</p>";
    return;
  }
  container.innerHTML = `
    <div class="frontier-card-list">
      ${rows.map((row) => `
        <article class="frontier-card">
          <div class="frontier-card-top">
            <div>
              <h3>${escapeHtml(row.source || "—")} → ${escapeHtml(row.target || "—")}</h3>
              <div class="frontier-card-meta">
                <span class="score-accent">score ${escapeHtml(formatScore(row.total_score))}</span>
                <span>${escapeHtml(row.status || "open")}</span>
                <span>${escapeHtml(formatTimestamp(row.timestamp))}</span>
              </div>
            </div>
            <div class="frontier-card-actions">
              ${row.source ? `<button type="button" class="graph-focus-button" data-focus-domain="${escapeHtml(row.source)}">Focus Source</button>` : ""}
            </div>
          </div>
          <p class="muted">${escapeHtml(row.reason || "No salvage reason recorded.")}</p>
        </article>
      `).join("")}
    </div>
  `;
}

function renderFrontier(payload) {
  state.frontier = payload;
  const regionsContainer = document.getElementById("frontier-next-regions");
  const seedsContainer = document.getElementById("frontier-next-seeds");
  if (regionsContainer) {
    regionsContainer.innerHTML = renderNextMoveCards(payload.next_regions || [], "regions");
  }
  if (seedsContainer) {
    seedsContainer.innerHTML = renderNextMoveCards(payload.next_seeds || [], "seeds");
  }
  renderFrontierStats(payload.stats || {});
  renderFrontierLaunchpads(payload.launchpads || []);
  renderFrontierBridges(payload.strong_bridges || []);
  renderSalvageCandidates(payload.salvage_candidates || [], "frontier-salvage");
  renderSalvageCandidates(payload.salvage_candidates || [], "failure-salvage");
  setFrontierMoveMode(state.activeFrontierMode || "regions");
  renderHeroSummary();
}

function renderStats(stats) {
  state.dashboardStats = stats;
  renderHeroSummary();
  const container = document.getElementById("stats");
  if (!container) {
    return;
  }
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
  container.innerHTML = items.map((item) => `
    <div class="stat">
      <div class="muted">${escapeHtml(item.label)}</div>
      <div class="stat-value ${item.valueClass || ""}">${escapeHtml(item.value)}</div>
    </div>
  `).join("");
}

function renderCosts(costs) {
  state.dashboardCosts = costs;
  renderHeroSummary();
  const container = document.getElementById("costs");
  if (!container) {
    return;
  }
  if (!costs.available) {
    container.innerHTML = "<p class=\"muted\">No cost data available</p>";
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

function renderTransmissionTimeline(rows) {
  const container = document.getElementById("transmission-timeline");
  if (!container) {
    return;
  }
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
    container.innerHTML = "<p class=\"muted\">No scored transmission data yet.</p>";
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
  const xTicks = timeRange <= 1 ? [minTime] : [minTime, minTime + timeRange / 2, maxTime];
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
    return `<text x="${x}" y="${height - 22}" fill="var(--muted)" font-size="12" text-anchor="${anchor}">${escapeHtml(formatTimelineDate(new Date(tick)))}</text>`;
  }).join("");

  const circles = points.map((point) => {
    const color = point.transmitted ? "var(--accent)" : "var(--kill-high)";
    const label = point.transmitted && point.transmissionNumber != null
      ? `Tx #${point.transmissionNumber}`
      : "Not transmitted";
    const title = `${formatTimelineDate(point.date, true)} | total_score ${point.score.toFixed(3)} | ${label}`;
    return `
      <circle cx="${xFor(point.time).toFixed(2)}" cy="${yFor(point.score).toFixed(2)}" r="${point.transmitted ? 4 : 3.5}" fill="${color}" opacity="${point.transmitted ? 0.95 : 0.75}">
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
        detail.innerHTML = `<div class="error">${escapeHtml(error instanceof Error ? error.message : "Failed to load adversarial detail")}</div>`;
      }
    });
  });
}

function renderTopKilled(rows) {
  const container = document.getElementById("top-killed");
  if (!container) {
    return;
  }
  if (!rows.length) {
    container.innerHTML = "<p class=\"muted\">No non-transmitted explorations found.</p>";
    return;
  }
  const body = rows.map((row) => `
    <tr class="top-killed-row" data-exploration-id="${escapeHtml(row.id)}" aria-expanded="false">
      <td>${escapeHtml(row.id)}</td>
      <td class="score-accent">${escapeHtml(formatScore(row.total_score))}</td>
      <td>${escapeHtml(row.seed_domain)}</td>
      <td>${escapeHtml(row.jump_target_domain)}</td>
      <td>${escapeHtml(truncateText(row.connection_description, 80))}</td>
    </tr>
    <tr class="top-killed-detail-row" hidden>
      <td colspan="5" class="adversarial-cell">
        <div class="adversarial-detail">Click row to load adversarial detail.</div>
      </td>
    </tr>
  `).join("");
  container.innerHTML = `
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

function renderEvidenceReviewStats(stats) {
  const statsContainer = document.getElementById("evidence-review-stats");
  const breakdownContainer = document.getElementById("evidence-review-breakdown");
  if (!statsContainer || !breakdownContainer) {
    return;
  }
  const byReviewStatus = stats.by_review_status || {};
  const summaryItems = [
    { label: "Total hits", value: stats.total_hits || 0, valueClass: "score-accent" },
    { label: "Unreviewed", value: byReviewStatus.unreviewed || 0 },
    { label: "Accepted", value: byReviewStatus.accepted || 0, valueClass: "score-accent" },
    { label: "Dismissed", value: byReviewStatus.dismissed || 0 },
    { label: "Predictions needing review", value: stats.predictions_needing_review || 0 },
  ];
  statsContainer.innerHTML = summaryItems.map((item) => `
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

  breakdownContainer.innerHTML = `
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
  state.operatorHome = snapshot;
  renderHeroSummary();
  const summaryContainer = document.getElementById("operator-home-summary");
  if (!summaryContainer) {
    return;
  }
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

  summaryContainer.innerHTML = summaryItems.map((item) => `
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
  if (!container) {
    return;
  }
  if (!rows.length) {
    container.innerHTML = "<p class=\"muted\">No unreviewed evidence hits.</p>";
    return;
  }
  const body = rows.map((row) => `
    <tr class="review-table-row ${Number(row.id) === state.selectedEvidenceId ? "is-selected" : ""}" data-home-evidence-id="${escapeHtml(row.id)}">
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
      setActiveWorkspace("review");
      scrollToPanel("evidence-detail");
    });
  });
}

function renderOperatorHomeOutcomes(rows) {
  const container = document.getElementById("operator-home-outcomes");
  if (!container) {
    return;
  }
  if (!rows.length) {
    container.innerHTML = "<p class=\"muted\">No open outcome candidates right now.</p>";
    return;
  }
  const body = rows.map((row) => `
    <tr class="review-table-row ${Number(row.id) === state.selectedOutcomePredictionId ? "is-selected" : ""}" data-home-prediction-id="${escapeHtml(row.id)}">
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
      setActiveWorkspace("review");
      scrollToPanel("outcome-review-detail");
    });
  });
}

function renderOperatorHomeStrongRejections(rows) {
  const container = document.getElementById("operator-home-strong-rejections");
  if (!container) {
    return;
  }
  if (!rows.length) {
    container.innerHTML = "<p class=\"muted\">No open strong rejections.</p>";
    return;
  }
  const body = rows.map((row) => `
    <tr class="review-table-row ${Number(row.id) === state.selectedStrongRejectionId ? "is-selected" : ""}" data-home-strong-rejection-id="${escapeHtml(row.id)}">
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
      setActiveWorkspace("review");
      scrollToPanel("strong-rejection-detail");
    });
  });
}

function renderEvidenceReviewQueue(rows) {
  state.evidenceQueueRows = rows;
  const container = document.getElementById("evidence-review-queue");
  if (!container) {
    return;
  }
  if (!rows.length) {
    container.innerHTML = "<p class=\"muted\">No unreviewed evidence hits found.</p>";
    return;
  }
  const body = rows.map((row) => `
    <tr class="review-table-row ${Number(row.id) === state.selectedEvidenceId ? "is-selected" : ""}" data-evidence-id="${escapeHtml(row.id)}">
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
      <input type="text" class="evidence-note-input" placeholder="Optional note" value="${inputValue(payload.notes)}">
      <button type="button" class="evidence-action" data-action="accept" data-evidence-id="${escapeHtml(payload.id)}">Accept</button>
      <button type="button" class="evidence-action" data-action="dismiss" data-evidence-id="${escapeHtml(payload.id)}">Dismiss</button>
      <span class="review-status"></span>
    </div>
  `;
}

async function loadEvidenceDetail(evidenceId) {
  state.selectedEvidenceId = evidenceId;
  renderEvidenceReviewQueue(state.evidenceQueueRows);
  const panel = document.getElementById("evidence-detail");
  if (!panel) {
    return;
  }
  panel.innerHTML = "<p class=\"muted\">Loading evidence detail…</p>";
  try {
    const payload = await fetchJson(`/api/evidence/${evidenceId}`);
    panel.innerHTML = renderEvidenceDetail(payload);
    attachEvidenceActionHandlers();
  } catch (error) {
    panel.innerHTML = `<div class="error">${escapeHtml(error instanceof Error ? error.message : "Failed to load evidence detail")}</div>`;
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
        await postJson(`/api/evidence/${evidenceId}/${action}`, { note: noteInput.value });
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
  const bucketsContainer = document.getElementById("outcome-suggestion-buckets");
  const backlogContainer = document.getElementById("outcome-review-backlog");
  if (!bucketsContainer || !backlogContainer) {
    return;
  }
  bucketsContainer.innerHTML = OUTCOME_SUGGESTION_BUCKETS.map((label) => `
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
  backlogContainer.innerHTML = backlogItems.map((item) => `
    <div class="stat">
      <div class="muted">${escapeHtml(item.label)}</div>
      <div class="stat-value">${escapeHtml(formatInteger(item.value))}</div>
    </div>
  `).join("");
}

function renderOutcomeReviewQueue(rows) {
  state.outcomeQueueRows = rows;
  const container = document.getElementById("outcome-review-queue");
  if (!container) {
    return;
  }
  if (!rows.length) {
    container.innerHTML = "<p class=\"muted\">No review-ready predictions found.</p>";
    return;
  }
  const body = rows.map((row) => `
    <tr class="review-table-row ${Number(row.id) === state.selectedOutcomePredictionId ? "is-selected" : ""}" data-prediction-id="${escapeHtml(row.id)}">
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
    ${renderOutcomeHitGroup("Accepted support hits", payload.accepted_support_examples || [], payload.accepted_support_hits || 0)}
    ${renderOutcomeHitGroup("Accepted contradiction hits", payload.accepted_contradiction_examples || [], payload.accepted_contradiction_hits || 0)}
    ${renderOutcomeHitGroup("Unreviewed reviewable hits", payload.unreviewed_reviewable_examples || [], payload.unreviewed_reviewable_hits || 0)}
  `;
}

async function loadOutcomeReviewDetail(predictionId) {
  state.selectedOutcomePredictionId = predictionId;
  renderOutcomeReviewQueue(state.outcomeQueueRows);
  const panel = document.getElementById("outcome-review-detail");
  if (!panel) {
    return;
  }
  panel.innerHTML = "<p class=\"muted\">Loading outcome review detail…</p>";
  try {
    const payload = await fetchJson(`/api/outcome-review/${predictionId}`);
    panel.innerHTML = renderOutcomeReviewDetail(payload);
  } catch (error) {
    panel.innerHTML = `<div class="error">${escapeHtml(error instanceof Error ? error.message : "Failed to load outcome review detail")}</div>`;
  }
}

function renderStrongRejectionStats(stats) {
  const container = document.getElementById("strong-rejection-stats");
  if (!container) {
    return;
  }
  const items = [
    { label: "Total strong rejections", value: stats.total || 0 },
    { label: "Open", value: stats.open || 0, valueClass: "score-accent" },
    { label: "Salvaged", value: stats.salvaged || 0 },
    { label: "Dismissed", value: stats.dismissed || 0 },
    { label: "Avg total score", value: formatScore(stats.average_total_score), valueClass: "score-accent" },
  ];
  container.innerHTML = items.map((item) => `
    <div class="stat">
      <div class="muted">${escapeHtml(item.label)}</div>
      <div class="stat-value ${item.valueClass || ""}">${escapeHtml(item.value)}</div>
    </div>
  `).join("");
}

function renderStrongRejectionQueue(rows) {
  state.strongRejectionRows = rows;
  const container = document.getElementById("strong-rejection-queue");
  if (!container) {
    return;
  }
  if (!rows.length) {
    container.innerHTML = "<p class=\"muted\">No strong rejections found.</p>";
    return;
  }
  const body = rows.map((row) => `
    <tr class="review-table-row ${Number(row.id) === state.selectedStrongRejectionId ? "is-selected" : ""}" data-strong-rejection-id="${escapeHtml(row.id)}">
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
      <input type="text" class="strong-rejection-note-input" placeholder="Optional note" value="${inputValue(payload.notes)}">
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
  state.selectedStrongRejectionId = rejectionId;
  renderStrongRejectionQueue(state.strongRejectionRows);
  const panel = document.getElementById("strong-rejection-detail");
  if (!panel) {
    return;
  }
  panel.innerHTML = "<p class=\"muted\">Loading strong rejection detail…</p>";
  try {
    const payload = await fetchJson(`/api/strong-rejection/${rejectionId}`);
    panel.innerHTML = renderStrongRejectionDetail(payload);
    attachStrongRejectionActionHandlers();
  } catch (error) {
    panel.innerHTML = `<div class="error">${escapeHtml(error instanceof Error ? error.message : "Failed to load strong rejection detail")}</div>`;
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
        await postJson(`/api/strong-rejection/${rejectionId}/${action}`, { note: noteInput.value });
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
  if (!summary) {
    return;
  }
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
          body: JSON.stringify({ grade, notes: notes.value }),
        });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
          throw new Error(payload.error || "Save failed");
        }

        const row = state.transmissionRows.find((item) => Number(item.id) === transmissionId);
        if (row) {
          row.user_rating = grade;
          row.user_notes = notes.value;
        }
        renderGradeSummary(state.transmissionRows);
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
  state.transmissionRows = rows;
  renderGradeSummary(rows);
  const count = document.getElementById("transmission-count");
  const container = document.getElementById("transmissions");
  if (!count || !container) {
    return;
  }
  count.textContent = `${rows.length} transmissions`;
  if (!rows.length) {
    container.innerHTML = "<p class=\"muted\">No transmissions found.</p>";
    return;
  }
  container.innerHTML = rows.map((row) => `
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
        <input class="grade-notes" type="text" placeholder="Notes (optional)" value="${inputValue(row.user_notes)}">
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

function graphClusterColor(cluster, alpha = 1) {
  const transmittedCount = Number(cluster.visible_transmitted_link_count ?? cluster.transmitted_link_count ?? 0);
  if (document.documentElement.dataset.theme === "light") {
    return transmittedCount > 0
      ? `rgba(232, 134, 72, ${alpha})`
      : `rgba(73, 126, 168, ${alpha})`;
  }
  return transmittedCount > 0
    ? `rgba(252, 185, 104, ${alpha})`
    : `rgba(122, 180, 232, ${alpha})`;
}

function rgbaString(channels, alpha = 1) {
  return `rgba(${channels[0]}, ${channels[1]}, ${channels[2]}, ${alpha})`;
}

function graphClusterPalette(cluster) {
  const palettes = document.documentElement.dataset.theme === "light"
    ? [
        { core: [226, 135, 72], glow: [255, 185, 113], mist: [255, 233, 188], edge: [255, 247, 228], label: [35, 49, 56] },
        { core: [64, 130, 179], glow: [127, 201, 231], mist: [210, 241, 249], edge: [240, 251, 255], label: [25, 44, 55] },
        { core: [43, 143, 110], glow: [116, 216, 167], mist: [213, 247, 232], edge: [239, 255, 246], label: [19, 46, 40] },
        { core: [199, 98, 95], glow: [241, 156, 132], mist: [250, 226, 218], edge: [255, 244, 240], label: [53, 34, 33] },
      ]
    : [
        { core: [250, 184, 111], glow: [255, 121, 94], mist: [251, 196, 141], edge: [255, 244, 224], label: [246, 248, 252] },
        { core: [113, 191, 236], glow: [88, 222, 255], mist: [70, 121, 158], edge: [226, 247, 255], label: [242, 249, 255] },
        { core: [103, 224, 167], glow: [61, 184, 145], mist: [60, 117, 96], edge: [228, 255, 242], label: [238, 253, 248] },
        { core: [241, 136, 110], glow: [255, 94, 94], mist: [126, 72, 70], edge: [255, 235, 233], label: [255, 242, 240] },
      ];
  const clusterId = cluster?.id || cluster?.label || "cluster";
  const base = palettes[hashString(clusterId) % palettes.length];
  const transmittedCount = Number(cluster?.visible_transmitted_link_count ?? cluster?.transmitted_link_count ?? 0);
  const intensity = transmittedCount > 0 ? 1 : 0.72;
  return {
    core: rgbaString(base.core, 0.82 + intensity * 0.14),
    coreSoft: rgbaString(base.core, 0.22 + intensity * 0.08),
    glow: rgbaString(base.glow, 0.4 + intensity * 0.18),
    mist: rgbaString(base.mist, 0.14 + intensity * 0.08),
    edge: rgbaString(base.edge, 0.64 + intensity * 0.18),
    label: rgbaString(base.label, 0.96),
  };
}

function graphNodePalette(role, alpha = 1) {
  if (document.documentElement.dataset.theme === "light") {
    if (role === "seed") {
      return {
        fill: rgbaString([221, 134, 41], alpha),
        glow: rgbaString([255, 190, 117], alpha * 0.75),
        ring: rgbaString([255, 244, 224], alpha * 0.95),
      };
    }
    if (role === "target") {
      return {
        fill: rgbaString([34, 151, 201], alpha),
        glow: rgbaString([139, 218, 247], alpha * 0.78),
        ring: rgbaString([234, 250, 255], alpha * 0.95),
      };
    }
    return {
      fill: rgbaString([28, 143, 103], alpha),
      glow: rgbaString([132, 228, 186], alpha * 0.76),
      ring: rgbaString([233, 255, 246], alpha * 0.95),
    };
  }
  if (role === "seed") {
    return {
      fill: rgbaString([255, 183, 77], alpha),
      glow: rgbaString([255, 122, 82], alpha * 0.82),
      ring: rgbaString([255, 241, 214], alpha * 0.95),
    };
  }
  if (role === "target") {
    return {
      fill: rgbaString([103, 213, 255], alpha),
      glow: rgbaString([59, 182, 248], alpha * 0.82),
      ring: rgbaString([223, 248, 255], alpha * 0.95),
    };
  }
  return {
    fill: rgbaString([89, 211, 154], alpha),
    glow: rgbaString([48, 177, 132], alpha * 0.82),
    ring: rgbaString([222, 255, 237], alpha * 0.95),
  };
}

function getGraphBackdropCache(width, height) {
  const key = `${width}x${height}`;
  if (state.graphBackdropCache?.key === key) {
    return state.graphBackdropCache;
  }
  const starCount = Math.max(110, Math.round((width * height) / 7500));
  const stars = Array.from({ length: starCount }, (_, index) => {
    const seed = hashString(`${key}:${index}`);
    return {
      x: ((seed % 10000) / 10000) * width,
      y: ((((seed >> 1) % 10000) + 17) / 10000) * height,
      radius: 0.45 + (((seed >> 3) % 1000) / 1000) * 1.75,
      alpha: 0.12 + (((seed >> 5) % 1000) / 1000) * 0.48,
      speed: 0.35 + (((seed >> 7) % 1000) / 1000) * 0.7,
      phase: (((seed >> 9) % 628) / 100),
    };
  });
  state.graphBackdropCache = { key, stars };
  return state.graphBackdropCache;
}

function drawRoundedRectPath(context, x, y, width, height, radius) {
  const safeRadius = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + safeRadius, y);
  context.lineTo(x + width - safeRadius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + safeRadius);
  context.lineTo(x + width, y + height - safeRadius);
  context.quadraticCurveTo(x + width, y + height, x + width - safeRadius, y + height);
  context.lineTo(x + safeRadius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - safeRadius);
  context.lineTo(x, y + safeRadius);
  context.quadraticCurveTo(x, y, x + safeRadius, y);
  context.closePath();
}

function drawSoftEllipse(context, x, y, radiusX, radiusY, rotation, innerColor, outerColor) {
  const maxRadius = Math.max(radiusX, radiusY);
  context.save();
  context.translate(x, y);
  context.rotate(rotation);
  context.scale(1, radiusY / Math.max(radiusX, 0.001));
  const gradient = context.createRadialGradient(0, 0, maxRadius * 0.12, 0, 0, maxRadius);
  gradient.addColorStop(0, innerColor);
  gradient.addColorStop(1, outerColor);
  context.fillStyle = gradient;
  context.beginPath();
  context.arc(0, 0, maxRadius, 0, Math.PI * 2);
  context.fill();
  context.restore();
}

function startGraphAnimationLoop() {
  if (state.graphAnimationFrameId || !canRenderGraphHere()) {
    return;
  }
  const tick = (time) => {
    state.graphAnimationFrameId = window.requestAnimationFrame(tick);
    if (document.hidden) {
      return;
    }
    if (time - state.graphAnimationLastFrame < 46) {
      return;
    }
    state.graphAnimationLastFrame = time;
    state.graphAnimationTime = time;
    updateGraphMotionFrame();
    if (state.domainGraph) {
      renderDomainGraphCanvas();
    }
  };
  state.graphAnimationFrameId = window.requestAnimationFrame(tick);
}

function addGraphRipple(x, y, options = {}) {
  state.graphRipples.push({
    x,
    y,
    createdAt: state.graphAnimationTime || performance.now(),
    duration: options.duration ?? 640,
    color: options.color || "rgba(143, 231, 255, 0.9)",
    size: options.size ?? 96,
    width: options.width ?? 2.4,
  });
}

function setGraphViewport(scale, offsetX, offsetY, immediate = false) {
  state.graphViewport.targetScale = scale;
  state.graphViewport.targetOffsetX = offsetX;
  state.graphViewport.targetOffsetY = offsetY;
  if (immediate) {
    state.graphViewport.scale = scale;
    state.graphViewport.offsetX = offsetX;
    state.graphViewport.offsetY = offsetY;
  }
}

function setGraphParallaxTargets(x, y) {
  state.graphMotion.pointerParallaxX = x;
  state.graphMotion.pointerParallaxY = y;
}

function applyGraphParallaxStyles() {
  const graphPanel = document.querySelector(".graph-panel");
  if (!graphPanel) {
    return;
  }
  graphPanel.style.setProperty("--graph-parallax-x", `${state.graphMotion.currentParallaxX.toFixed(2)}px`);
  graphPanel.style.setProperty("--graph-parallax-y", `${state.graphMotion.currentParallaxY.toFixed(2)}px`);
  graphPanel.style.setProperty("--graph-tilt-x", `${(-state.graphMotion.currentParallaxY * 0.045).toFixed(2)}deg`);
  graphPanel.style.setProperty("--graph-tilt-y", `${(state.graphMotion.currentParallaxX * 0.05).toFixed(2)}deg`);
}

function boostGraphCamera(response = 0.24) {
  state.graphViewport.targetResponse = Math.max(state.graphViewport.targetResponse || 0.18, response);
}

function startGraphSceneTransition(type, anchor = {}, duration = 900) {
  state.graphSceneTransition = {
    type,
    anchor,
    startTime: state.graphAnimationTime || performance.now(),
    duration,
  };
}

function getGraphSceneTransitionState() {
  const transition = state.graphSceneTransition;
  if (!transition) {
    return null;
  }
  const elapsed = Math.max(0, (state.graphAnimationTime || performance.now()) - transition.startTime);
  const progress = clamp(elapsed / Math.max(transition.duration || 1, 1), 0, 1);
  const eased = 1 - Math.pow(1 - progress, 3);
  if (progress >= 1) {
    state.graphSceneTransition = null;
  }
  return {
    ...transition,
    progress,
    eased,
    intensity: Math.sin(progress * Math.PI),
  };
}

function updateGraphMotionFrame() {
  const now = state.graphAnimationTime || performance.now();
  state.graphRipples = state.graphRipples.filter((ripple) => now - ripple.createdAt <= ripple.duration);
  state.graphViewport.response += ((state.graphViewport.targetResponse || 0.18) - (state.graphViewport.response || 0.18)) * 0.22;
  state.graphViewport.targetResponse += (0.18 - (state.graphViewport.targetResponse || 0.18)) * 0.12;
  if (Math.abs((state.graphViewport.targetResponse || 0.18) - 0.18) < 0.002) {
    state.graphViewport.targetResponse = 0.18;
  }
  if (Math.abs((state.graphViewport.response || 0.18) - (state.graphViewport.targetResponse || 0.18)) < 0.001) {
    state.graphViewport.response = state.graphViewport.targetResponse || 0.18;
  }

  const viewportDeltaScale = state.graphViewport.targetScale - state.graphViewport.scale;
  const viewportDeltaX = state.graphViewport.targetOffsetX - state.graphViewport.offsetX;
  const viewportDeltaY = state.graphViewport.targetOffsetY - state.graphViewport.offsetY;
  const viewportResponse = state.graphViewport.response || 0.18;
  state.graphViewport.scale += viewportDeltaScale * viewportResponse;
  state.graphViewport.offsetX += viewportDeltaX * viewportResponse;
  state.graphViewport.offsetY += viewportDeltaY * viewportResponse;
  if (Math.abs(viewportDeltaScale) < 0.0008) {
    state.graphViewport.scale = state.graphViewport.targetScale;
  }
  if (Math.abs(viewportDeltaX) < 0.18) {
    state.graphViewport.offsetX = state.graphViewport.targetOffsetX;
  }
  if (Math.abs(viewportDeltaY) < 0.18) {
    state.graphViewport.offsetY = state.graphViewport.targetOffsetY;
  }

  const idleParallaxX = canRenderGraphHere() && !state.graphPointer.isPanning && !state.graphPointer.dragType
    ? Math.sin((state.graphAnimationTime || 0) / 2400) * 6
    : 0;
  const idleParallaxY = canRenderGraphHere() && !state.graphPointer.isPanning && !state.graphPointer.dragType
    ? Math.cos((state.graphAnimationTime || 0) / 3100) * 4
    : 0;
  state.graphMotion.targetParallaxX = state.graphMotion.pointerParallaxX + idleParallaxX;
  state.graphMotion.targetParallaxY = state.graphMotion.pointerParallaxY + idleParallaxY;

  state.graphMotion.currentParallaxX += (state.graphMotion.targetParallaxX - state.graphMotion.currentParallaxX) * 0.18;
  state.graphMotion.currentParallaxY += (state.graphMotion.targetParallaxY - state.graphMotion.currentParallaxY) * 0.18;
  if (Math.abs(state.graphMotion.targetParallaxX - state.graphMotion.currentParallaxX) < 0.08) {
    state.graphMotion.currentParallaxX = state.graphMotion.targetParallaxX;
  }
  if (Math.abs(state.graphMotion.targetParallaxY - state.graphMotion.currentParallaxY) < 0.08) {
    state.graphMotion.currentParallaxY = state.graphMotion.targetParallaxY;
  }

  applyGraphParallaxStyles();
  getGraphSceneTransitionState();
}

function getGraphControls() {
  return {
    mode: document.getElementById("graph-mode")?.value || "all",
    minScore: Number(document.getElementById("graph-score")?.value || 0),
    query: document.getElementById("graph-search")?.value.trim().toLowerCase() || "",
  };
}

function filterGraphLinks(baseLinks, controls) {
  return baseLinks.filter((link) => {
    const maxScore = Number(link.max_score ?? link.avg_score ?? 0);
    if (maxScore < controls.minScore) {
      return false;
    }
    if (controls.mode === "successful") {
      return Number(link.transmitted_count || 0) > 0;
    }
    if (controls.mode === "strong") {
      return Number(link.transmitted_count || 0) > 0 || maxScore >= Math.max(controls.minScore, 0.8);
    }
    return true;
  });
}

function resetGraphViewport() {
  setGraphViewport(1, 0, 0, true);
  state.graphViewport.response = 0.18;
  state.graphViewport.targetResponse = 0.18;
}

function graphToScreenPoint(x, y, width, height) {
  const scale = state.graphViewport.scale || 1;
  return {
    x: (x - width / 2) * scale + width / 2 + state.graphViewport.offsetX,
    y: (y - height / 2) * scale + height / 2 + state.graphViewport.offsetY,
  };
}

function screenToGraphPoint(x, y, width, height) {
  const scale = state.graphViewport.scale || 1;
  return {
    x: (x - width / 2 - state.graphViewport.offsetX) / scale + width / 2,
    y: (y - height / 2 - state.graphViewport.offsetY) / scale + height / 2,
  };
}

function graphRadiusToScreen(radius) {
  return radius * (state.graphViewport.scale || 1);
}

function graphAmbientScreenOffset(id, amplitudeX, amplitudeY) {
  const phase = (state.graphAnimationTime || 0) / 1000;
  const seed = hashString(id || "graph-item");
  const speed = 0.45 + (((seed >> 3) % 1000) / 1000) * 0.5;
  const phaseShift = ((seed % 628) / 100);
  const x = Math.cos(phase * speed + phaseShift) * amplitudeX
    + Math.sin(phase * speed * 0.57 + phaseShift * 0.8) * amplitudeX * 0.36;
  const y = Math.sin(phase * speed * 1.08 + phaseShift * 1.2) * amplitudeY
    + Math.cos(phase * speed * 0.44 + phaseShift * 0.5) * amplitudeY * 0.28;
  return { x, y };
}

function graphClusterScreenPoint(cluster, width, height) {
  const base = graphToScreenPoint(cluster.x, cluster.y, width, height);
  const isDragging = state.graphPointer.dragType === "cluster" && state.graphPointer.dragId === cluster.id;
  if (isDragging) {
    return base;
  }
  const emphasis = cluster.id === state.selectedGraphClusterId || cluster.id === state.hoveredGraphClusterId ? 1.3 : cluster.isMatch ? 1.15 : 1;
  const drift = graphAmbientScreenOffset(`cluster:${cluster.id}`, 7.5 * emphasis, 6 * emphasis);
  return {
    x: base.x + drift.x,
    y: base.y + drift.y,
  };
}

function graphNodeScreenPoint(node, width, height) {
  const base = graphToScreenPoint(node.x, node.y, width, height);
  const isDragging = state.graphPointer.dragType === "node" && state.graphPointer.dragId === node.id;
  if (isDragging) {
    return base;
  }
  const isActive = node.id === state.selectedGraphNodeId || node.id === state.hoveredGraphNodeId;
  const emphasis = isActive ? 1.5 : node.isMatch || Number(node.transmitted_count || 0) > 0 ? 1.2 : 1;
  const drift = graphAmbientScreenOffset(`node:${node.id}`, 2.6 * emphasis, 2.1 * emphasis);
  return {
    x: base.x + drift.x,
    y: base.y + drift.y,
  };
}

function graphLinkScreenPoints(link, width, height) {
  const sourcePoint = graphNodeScreenPoint(link.sourceNode, width, height);
  const targetPoint = graphNodeScreenPoint(link.targetNode, width, height);
  const controlPoint = graphCurveControlPoint(sourcePoint, targetPoint, graphLinkKey(link));
  return { sourcePoint, targetPoint, controlPoint };
}

function getGraphTransitionAnchorPoint(transition, width, height) {
  const anchor = transition?.anchor || {};
  if (anchor.nodeId && state.currentDomainGraphLayout?.sceneType === "cluster") {
    const node = state.currentDomainGraphLayout.nodes.find((item) => item.id === anchor.nodeId);
    if (node) {
      return {
        point: graphNodeScreenPoint(node, width, height),
        palette: graphNodePalette(node.role, 1),
      };
    }
  }

  if (anchor.clusterId) {
    if (state.currentDomainGraphLayout?.sceneType === "overview") {
      const cluster = state.currentDomainGraphLayout.clusters.find((item) => item.id === anchor.clusterId);
      if (cluster) {
        return {
          point: graphClusterScreenPoint(cluster, width, height),
          palette: graphClusterPalette(cluster),
        };
      }
    }
    const clusterMeta = (state.domainGraph?.clusters || []).find((item) => item.id === anchor.clusterId);
    if (clusterMeta && state.currentDomainGraphLayout?.sceneType === "cluster" && state.selectedGraphClusterId === anchor.clusterId) {
      const nodes = state.currentDomainGraphLayout.nodes || [];
      if (nodes.length) {
        const point = nodes.reduce((accumulator, node) => {
          const screenPoint = graphNodeScreenPoint(node, width, height);
          accumulator.x += screenPoint.x;
          accumulator.y += screenPoint.y;
          return accumulator;
        }, { x: 0, y: 0 });
        point.x /= nodes.length;
        point.y /= nodes.length;
        return {
          point,
          palette: graphClusterPalette(clusterMeta),
        };
      }
    }
    if (clusterMeta) {
      return {
        point: { x: width / 2, y: height / 2 },
        palette: graphClusterPalette(clusterMeta),
      };
    }
  }

  return {
    point: { x: width / 2, y: height / 2 },
    palette: graphClusterPalette({ id: transition?.type || "transition" }),
  };
}

function renderGraphSceneTransitionOverlay(context, width, height) {
  const transition = getGraphSceneTransitionState();
  if (!transition || transition.intensity <= 0.001) {
    return;
  }
  const { point, palette } = getGraphTransitionAnchorPoint(transition, width, height);
  const radius = Math.min(width, height) * (0.18 + transition.eased * 0.56);
  const overlay = context.createRadialGradient(point.x, point.y, radius * 0.08, point.x, point.y, radius);
  overlay.addColorStop(0, palette.edge.replace(/rgba\(([^,]+), ([^,]+), ([^,]+), [^)]+\)/, "rgba($1, $2, $3, 0.28)"));
  overlay.addColorStop(0.36, palette.coreSoft);
  overlay.addColorStop(1, "rgba(0, 0, 0, 0)");
  context.save();
  context.fillStyle = overlay;
  context.fillRect(0, 0, width, height);

  const ringAlpha = transition.type === "enter-region" ? 0.38 : transition.type === "return-atlas" ? 0.22 : 0.3;
  for (let index = 0; index < 2; index += 1) {
    context.beginPath();
    context.arc(point.x, point.y, radius * (0.44 + index * 0.18), 0, Math.PI * 2);
    context.lineWidth = 2.4 - index * 0.6;
    context.strokeStyle = palette.edge.replace(/rgba\(([^,]+), ([^,]+), ([^,]+), [^)]+\)/, `rgba($1, $2, $3, ${Math.max(0, ringAlpha - index * 0.14) * transition.intensity})`);
    context.stroke();
  }
  context.restore();
}

function renderTravelingEnergy(context, sourcePoint, controlPoint, targetPoint, color, seed, size, alphaScale = 1) {
  const phase = (state.graphAnimationTime || 0) / 1000;
  const base = hashString(seed || "energy");
  const speed = 0.28 + (((base >> 3) % 1000) / 1000) * 0.45;
  const offset = ((base % 1000) / 1000);
  for (let index = 0; index < 2; index += 1) {
    const t = (phase * speed + offset + index * 0.37) % 1;
    const point = quadraticPoint(t, sourcePoint.x, sourcePoint.y, controlPoint.x, controlPoint.y, targetPoint.x, targetPoint.y);
    const radius = size * (index === 0 ? 1.15 : 0.8);
    context.save();
    context.beginPath();
    context.arc(point.x, point.y, radius * 1.9, 0, Math.PI * 2);
    context.fillStyle = color.replace(/rgba\(([^,]+), ([^,]+), ([^,]+), [^)]+\)/, `rgba($1, $2, $3, ${0.18 * alphaScale})`);
    context.shadowBlur = 18;
    context.shadowColor = color;
    context.fill();
    context.restore();

    context.beginPath();
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fillStyle = color.replace(/rgba\(([^,]+), ([^,]+), ([^,]+), [^)]+\)/, `rgba($1, $2, $3, ${0.9 * alphaScale})`);
    context.fill();
  }
}

function renderGraphRipples(context) {
  if (!state.graphRipples.length) {
    return;
  }
  const now = state.graphAnimationTime || performance.now();
  state.graphRipples.forEach((ripple) => {
    const progress = clamp((now - ripple.createdAt) / Math.max(ripple.duration, 1), 0, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    const radius = 12 + ripple.size * ease;
    const alpha = 1 - progress;
    context.save();
    context.beginPath();
    context.arc(ripple.x, ripple.y, radius, 0, Math.PI * 2);
    context.lineWidth = ripple.width * (1 - progress * 0.35);
    context.strokeStyle = ripple.color.replace(/rgba\(([^,]+), ([^,]+), ([^,]+), [^)]+\)/, `rgba($1, $2, $3, ${alpha * 0.9})`);
    context.shadowBlur = 18;
    context.shadowColor = ripple.color;
    context.stroke();
    context.restore();
  });
}

function graphRippleColorForHit(hit) {
  if (!hit) {
    return document.documentElement.dataset.theme === "light"
      ? "rgba(51, 125, 161, 0.82)"
      : "rgba(143, 231, 255, 0.9)";
  }
  if (hit.type === "cluster") {
    return graphClusterColor(hit.item, 0.92);
  }
  if (hit.type === "node") {
    return graphNodeColor(hit.item.role, 0.92);
  }
  if (hit.type === "link" && Number(hit.item.transmitted_count || 0) > 0) {
    return document.documentElement.dataset.theme === "light"
      ? "rgba(28, 143, 103, 0.88)"
      : "rgba(89, 211, 154, 0.9)";
  }
  if (hit.type === "link" && Number(hit.item.max_score ?? hit.item.avg_score ?? 0) >= 0.85) {
    return document.documentElement.dataset.theme === "light"
      ? "rgba(41, 138, 176, 0.88)"
      : "rgba(103, 213, 255, 0.92)";
  }
  return document.documentElement.dataset.theme === "light"
    ? "rgba(108, 128, 138, 0.8)"
    : "rgba(166, 189, 203, 0.88)";
}

function screenToGraphPointWithViewport(x, y, width, height, viewport) {
  const scale = viewport.scale || 1;
  return {
    x: (x - width / 2 - viewport.offsetX) / scale + width / 2,
    y: (y - height / 2 - viewport.offsetY) / scale + height / 2,
  };
}

function zoomGraphAtPoint(scaleFactor, screenX, screenY, width, height) {
  const viewport = {
    scale: state.graphViewport.targetScale ?? state.graphViewport.scale ?? 1,
    offsetX: state.graphViewport.targetOffsetX ?? state.graphViewport.offsetX ?? 0,
    offsetY: state.graphViewport.targetOffsetY ?? state.graphViewport.offsetY ?? 0,
  };
  const worldPoint = screenToGraphPointWithViewport(screenX, screenY, width, height, viewport);
  const nextScale = clamp((viewport.scale || 1) * scaleFactor, 0.55, 3.8);
  const nextOffsetX = screenX - width / 2 - (worldPoint.x - width / 2) * nextScale;
  const nextOffsetY = screenY - height / 2 - (worldPoint.y - height / 2) * nextScale;
  setGraphViewport(nextScale, nextOffsetX, nextOffsetY);
}

function getCurrentGraphBounds(layout) {
  if (!layout) {
    return null;
  }
  const items = state.graphSceneMode === "overview" ? layout.clusters : layout.nodes;
  if (!Array.isArray(items) || !items.length) {
    return null;
  }
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  items.forEach((item) => {
    const radius = Number(item.radius || 0);
    minX = Math.min(minX, item.x - radius);
    maxX = Math.max(maxX, item.x + radius);
    minY = Math.min(minY, item.y - radius);
    maxY = Math.max(maxY, item.y + radius);
  });
  if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
    return null;
  }
  return { minX, minY, maxX, maxY };
}

function zoomGraphToBounds(bounds, width, height, padding = 54) {
  if (!bounds) {
    resetGraphViewport();
    return;
  }
  const worldWidth = Math.max(120, bounds.maxX - bounds.minX);
  const worldHeight = Math.max(120, bounds.maxY - bounds.minY);
  const nextScale = clamp(
    Math.min((width - padding * 2) / worldWidth, (height - padding * 2) / worldHeight),
    0.55,
    3.8
  );
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerY = (bounds.minY + bounds.maxY) / 2;
  setGraphViewport(
    nextScale,
    -((centerX - width / 2) * nextScale),
    -((centerY - height / 2) * nextScale)
  );
}

function centerGraphOnSelection(boost = 0) {
  if (state.graphSceneMode !== "cluster" || !state.currentDomainGraphLayout) {
    return;
  }
  const preparedCanvas = prepareDomainGraphCanvas();
  if (!preparedCanvas) {
    return;
  }
  const { width, height } = preparedCanvas;
  const viewportScale = state.graphViewport.targetScale ?? state.graphViewport.scale ?? 1;

  if (state.selectedGraphNodeId) {
    const node = state.currentDomainGraphLayout.nodes.find((item) => item.id === state.selectedGraphNodeId);
    if (!node) {
      return;
    }
    const nextScale = clamp(
      Math.max(viewportScale, 1.5 + Math.min(0.55, Math.log1p(Number(node.connection_count || 1)) * 0.2) + boost),
      0.7,
      3.2
    );
    setGraphViewport(
      nextScale,
      -((node.x - width / 2) * nextScale),
      -((node.y - height / 2) * nextScale)
    );
    return;
  }

  if (state.selectedGraphLinkKey) {
    const link = state.currentDomainGraphLayout.links.find((item) => graphLinkKey(item) === state.selectedGraphLinkKey);
    if (!link) {
      return;
    }
    const worldX = (link.sourceNode.x + link.targetNode.x) / 2;
    const worldY = (link.sourceNode.y + link.targetNode.y) / 2;
    const nextScale = clamp(Math.max(viewportScale, 1.28 + boost), 0.7, 3.2);
    setGraphViewport(
      nextScale,
      -((worldX - width / 2) * nextScale),
      -((worldY - height / 2) * nextScale)
    );
  }
}

function buildClusterOverview(payload) {
  const controls = getGraphControls();
  const baseNodes = Array.isArray(payload?.nodes) ? payload.nodes : [];
  const baseLinks = Array.isArray(payload?.links) ? payload.links : [];
  const baseClusters = Array.isArray(payload?.clusters) ? payload.clusters : [];
  const nodeById = new Map(baseNodes.map((node) => [node.id, node]));
  const filteredLinks = filterGraphLinks(baseLinks, controls);
  const clusterMap = new Map(
    baseClusters.map((cluster) => [
      cluster.id,
      {
        ...cluster,
        visible_link_count: 0,
        visible_transmitted_link_count: 0,
        visible_explorations: 0,
        visible_score_total: 0,
        visible_score_weight: 0,
        matched_nodes: [],
        label_match: false,
      },
    ])
  );

  if (controls.query) {
    baseNodes.forEach((node) => {
      if (String(node.id || "").toLowerCase().includes(controls.query) && clusterMap.has(node.cluster_id)) {
        clusterMap.get(node.cluster_id).matched_nodes.push(node.id);
      }
    });
    baseClusters.forEach((cluster) => {
      const topDomains = Array.isArray(cluster.top_domains) ? cluster.top_domains : [];
      const labelMatch = String(cluster.label || "").toLowerCase().includes(controls.query)
        || topDomains.some((domain) => String(domain || "").toLowerCase().includes(controls.query));
      if (labelMatch && clusterMap.has(cluster.id)) {
        clusterMap.get(cluster.id).label_match = true;
      }
    });
  }

  filteredLinks.forEach((link) => {
    const sourceNode = nodeById.get(link.source);
    const targetNode = nodeById.get(link.target);
    const clusterId = sourceNode?.cluster_id || targetNode?.cluster_id;
    if (!clusterId || !clusterMap.has(clusterId)) {
      return;
    }
    const cluster = clusterMap.get(clusterId);
    cluster.visible_link_count += 1;
    cluster.visible_explorations += Number(link.count || 0);
    if (Number(link.transmitted_count || 0) > 0) {
      cluster.visible_transmitted_link_count += 1;
    }
    const score = Number(link.max_score ?? link.avg_score);
    if (Number.isFinite(score)) {
      const weight = Math.max(1, Number(link.count || 0));
      cluster.visible_score_total += score * weight;
      cluster.visible_score_weight += weight;
    }
  });

  const clusters = [...clusterMap.values()]
    .filter((cluster) => cluster.visible_link_count > 0 || cluster.matched_nodes.length > 0 || cluster.label_match)
    .map((cluster) => ({
      ...cluster,
      visible_avg_score: cluster.visible_score_weight > 0
        ? cluster.visible_score_total / cluster.visible_score_weight
        : cluster.avg_score,
      matched_node_count: cluster.matched_nodes.length,
      isMatch: cluster.matched_nodes.length > 0 || cluster.label_match,
    }))
    .sort((left, right) => (
      Number(right.isMatch) - Number(left.isMatch)
      || Number(right.visible_transmitted_link_count || 0) - Number(left.visible_transmitted_link_count || 0)
      || Number(right.node_count || 0) - Number(left.node_count || 0)
      || Number(right.visible_avg_score || 0) - Number(left.visible_avg_score || 0)
      || String(left.label || "").localeCompare(String(right.label || ""))
    ));

  let focus = null;
  if (controls.query) {
    const visibleClusterIds = new Set(clusters.map((cluster) => cluster.id));
    const exactMatches = baseNodes.filter((node) => String(node.id || "").toLowerCase() === controls.query && visibleClusterIds.has(node.cluster_id));
    if (exactMatches.length === 1) {
      focus = {
        clusterId: exactMatches[0].cluster_id,
        label: clusters.find((cluster) => cluster.id === exactMatches[0].cluster_id)?.label || exactMatches[0].cluster_id,
        exactNodeId: exactMatches[0].id,
      };
    } else if (clusters.length) {
      const [bestCluster] = clusters;
      focus = {
        clusterId: bestCluster.id,
        label: bestCluster.label,
        exactNodeId: null,
      };
    }
  }

  return {
    controls,
    filteredLinks,
    clusters,
    focus,
  };
}

function buildClusterDetailGraph(payload, clusterId, filteredLinks, query) {
  const baseNodes = Array.isArray(payload?.nodes)
    ? payload.nodes.filter((node) => node.cluster_id === clusterId)
    : [];
  const nodeById = new Map(baseNodes.map((node) => [node.id, node]));
  const links = Array.isArray(filteredLinks)
    ? filteredLinks.filter((link) => nodeById.has(link.source) && nodeById.has(link.target))
    : [];
  const matchedIds = new Set();
  if (query) {
    baseNodes.forEach((node) => {
      if (String(node.id || "").toLowerCase().includes(query)) {
        matchedIds.add(node.id);
      }
    });
  }
  const nodes = baseNodes.map((node) => ({
    ...node,
    degree: 0,
    radius: 7,
    isMatch: matchedIds.has(node.id),
  }));
  const detailNodeMap = new Map(nodes.map((node) => [node.id, node]));
  let totalScoredConnections = 0;
  let scoreSum = 0;
  let visibleExplorations = 0;

  links.forEach((link) => {
    visibleExplorations += Number(link.count || 0);
    const sourceNode = detailNodeMap.get(link.source);
    const targetNode = detailNodeMap.get(link.target);
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
      4 + Math.sqrt(Math.max(1, node.degree || node.connection_count || 1)) * 1.6 + (Number(node.transmitted_count || 0) > 0 ? 1.5 : 0),
      5,
      18
    );
  });

  return {
    clusterId,
    nodes,
    links,
    matchedIds,
    query,
    stats: {
      visible_domains: nodes.length,
      visible_connections: links.length,
      visible_transmitted_connections: links.filter((link) => Number(link.transmitted_count || 0) > 0).length,
      visible_explorations: visibleExplorations,
      avg_score: totalScoredConnections > 0 ? scoreSum / totalScoredConnections : null,
    },
  };
}

function renderDomainGraph(payload) {
  state.domainGraph = payload;
  renderHeroSummary();
  initializeDomainGraphControls();
  updateDomainGraph();
  startGraphAnimationLoop();
}

function initializeDomainGraphControls() {
  if (state.domainGraphControlsInitialized) {
    return;
  }
  const modeInput = document.getElementById("graph-mode");
  const scoreInput = document.getElementById("graph-score");
  const scoreValue = document.getElementById("graph-score-value");
  const searchInput = document.getElementById("graph-search");
  const zoomFitButton = document.getElementById("graph-zoom-fit");
  const resetButton = document.getElementById("graph-reset");
  const canvas = document.getElementById("domain-web-canvas");
  if (!modeInput || !scoreInput || !scoreValue || !searchInput || !zoomFitButton || !resetButton || !canvas) {
    return;
  }

  scoreValue.textContent = Number(scoreInput.value || 0).toFixed(2);
  modeInput.addEventListener("change", () => {
    state.selectedGraphClusterId = null;
    state.selectedGraphNodeId = null;
    state.selectedGraphLinkKey = null;
    updateDomainGraph();
  });
  scoreInput.addEventListener("input", () => {
    scoreValue.textContent = Number(scoreInput.value || 0).toFixed(2);
    state.selectedGraphClusterId = null;
    state.selectedGraphLinkKey = null;
    updateDomainGraph();
  });
  searchInput.addEventListener("input", () => {
    window.clearTimeout(state.domainGraphSearchTimer);
    state.domainGraphSearchTimer = window.setTimeout(() => {
      state.selectedGraphClusterId = state.selectedGraphClusterId || null;
      state.selectedGraphNodeId = null;
      state.selectedGraphLinkKey = null;
      updateDomainGraph();
    }, 120);
  });
  zoomFitButton.addEventListener("click", () => {
    renderDomainGraphCanvas({ autoFit: true });
  });
  resetButton.addEventListener("click", () => {
    modeInput.value = "all";
    scoreInput.value = "0";
    scoreValue.textContent = "0.00";
    searchInput.value = "";
    state.selectedGraphClusterId = null;
    state.selectedGraphNodeId = null;
    state.selectedGraphLinkKey = null;
    state.hoveredGraphClusterId = null;
    state.hoveredGraphNodeId = null;
    state.hoveredGraphLinkKey = null;
    updateDomainGraph();
  });
  canvas.addEventListener("wheel", handleDomainGraphCanvasWheel, { passive: false });
  canvas.addEventListener("mousedown", handleDomainGraphCanvasDown);
  canvas.addEventListener("mousemove", handleDomainGraphCanvasMove);
  canvas.addEventListener("mouseleave", () => {
    if (state.graphPointer.isPanning || state.graphPointer.dragType) {
      return;
    }
    state.hoveredGraphClusterId = null;
    state.hoveredGraphNodeId = null;
    state.hoveredGraphLinkKey = null;
    setGraphParallaxTargets(0, 0);
    canvas.style.cursor = "grab";
    renderDomainGraphCanvas();
    renderDomainGraphDetail();
  });
  canvas.addEventListener("click", handleDomainGraphCanvasClick);
  canvas.addEventListener("dblclick", handleDomainGraphCanvasDoubleClick);
  window.addEventListener("mousemove", handleDomainGraphCanvasMove);
  window.addEventListener("mouseup", handleDomainGraphCanvasUp);
  window.addEventListener("resize", () => {
    window.clearTimeout(state.domainGraphResizeTimer);
    state.domainGraphResizeTimer = window.setTimeout(() => {
      renderDomainGraphCanvas({ autoFit: true });
      renderDomainGraphDetail();
    }, 120);
  });

  state.domainGraphControlsInitialized = true;
}

function renderDomainGraphSummary() {
  const container = document.getElementById("domain-web-summary");
  if (!container) {
    return;
  }
  const overall = state.domainGraph ? state.domainGraph.stats || {} : {};
  const clusters = Array.isArray(state.domainGraph?.clusters) ? state.domainGraph.clusters : [];
  const selectedCluster = clusters.find((cluster) => cluster.id === state.selectedGraphClusterId);
  const controls = state.currentClusterOverview?.controls || getGraphControls();
  const items = [
    {
      label: "Clusters",
      value: formatInteger(overall.cluster_count || 0),
      valueClass: "score-accent",
    },
    {
      label: "Largest cluster",
      value: `${formatInteger(overall.largest_cluster_size || 0)} domains`,
    },
    {
      label: "Transmitting clusters",
      value: formatInteger(clusters.filter((cluster) => Number(cluster.transmitted_link_count || 0) > 0).length),
      valueClass: "score-accent",
    },
    {
      label: "Search focus",
      value: controls.query
        ? truncateText(controls.query, 28)
        : selectedCluster
          ? truncateText(selectedCluster.label, 28)
          : "entire map",
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

function computeClusterOverviewLayout(overview, width, height) {
  const centerX = width / 2;
  const centerY = height / 2;
  const spread = Math.min(width, height) * 0.34;
  const clusters = overview.clusters.map((cluster) => {
    const cached = state.clusterGraphLayoutCache.get(cluster.id);
    const radius = clamp(
      16 + Math.sqrt(Math.max(1, Number(cluster.node_count || 1))) * 7 + Number(cluster.visible_transmitted_link_count || 0) * 1.5,
      18,
      58
    );
    if (cached && Number.isFinite(cached.x) && Number.isFinite(cached.y)) {
      return {
        ...cluster,
        radius,
        x: clamp(cached.x, 40, width - 40),
        y: clamp(cached.y, 40, height - 40),
        vx: 0,
        vy: 0,
      };
    }
    const seed = hashString(cluster.id);
    const angle = ((seed % 3600) / 3600) * Math.PI * 2;
    const ring = 0.16 + (((seed >> 2) % 1000) / 1000) * 0.84;
    return {
      ...cluster,
      radius,
      x: centerX + Math.cos(angle) * spread * ring,
      y: centerY + Math.sin(angle) * spread * ring,
      vx: 0,
      vy: 0,
    };
  });

  if (clusters.length > 1) {
    const ticks = Math.min(220, Math.max(90, 80 + Math.round(clusters.length / 2)));
    for (let tick = 0; tick < ticks; tick += 1) {
      const alpha = 1 - tick / ticks;
      for (let index = 0; index < clusters.length; index += 1) {
        const cluster = clusters[index];
        cluster.vx += (centerX - cluster.x) * 0.0009 * alpha;
        cluster.vy += (centerY - cluster.y) * 0.0009 * alpha;
        for (let sample = index + 1; sample < clusters.length; sample += 1) {
          const other = clusters[sample];
          let dx = cluster.x - other.x;
          let dy = cluster.y - other.y;
          let distanceSquared = dx * dx + dy * dy;
          if (distanceSquared < 0.5) {
            dx = 0.12;
            dy = 0.12;
            distanceSquared = dx * dx + dy * dy;
          }
          const distance = Math.sqrt(distanceSquared);
          const minDistance = cluster.radius + other.radius + 22;
          if (distance < minDistance) {
            const overlap = (minDistance - distance) / Math.max(distance, 0.001);
            const pushX = dx * overlap * 0.03;
            const pushY = dy * overlap * 0.03;
            cluster.vx += pushX;
            cluster.vy += pushY;
            other.vx -= pushX;
            other.vy -= pushY;
          }
          const repulsion = (2200 + (cluster.node_count + other.node_count) * 25) / (distanceSquared + 100);
          cluster.vx += dx * repulsion * 0.00028;
          cluster.vy += dy * repulsion * 0.00028;
          other.vx -= dx * repulsion * 0.00028;
          other.vy -= dy * repulsion * 0.00028;
        }
      }

      clusters.forEach((cluster) => {
        cluster.vx *= 0.86;
        cluster.vy *= 0.86;
        cluster.x = clamp(cluster.x + cluster.vx, cluster.radius + 24, width - cluster.radius - 24);
        cluster.y = clamp(cluster.y + cluster.vy, cluster.radius + 24, height - cluster.radius - 24);
      });
    }
  }

  clusters.forEach((cluster) => {
    state.clusterGraphLayoutCache.set(cluster.id, { x: cluster.x, y: cluster.y });
  });
  return { clusters, width, height };
}

function computeDomainGraphLayout(graph, width, height) {
  const centerX = width / 2;
  const centerY = height / 2;
  const spread = Math.min(width, height) * 0.36;
  const nodes = graph.nodes.map((node) => {
    const cached = state.domainGraphLayoutCache.get(node.id);
    if (cached && Number.isFinite(cached.x) && Number.isFinite(cached.y)) {
      return { ...node, x: clamp(cached.x, 26, width - 26), y: clamp(cached.y, 26, height - 26), vx: 0, vy: 0 };
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
  const links = graph.links.map((link) => ({
    ...link,
    sourceNode: nodeById.get(link.source),
    targetNode: nodeById.get(link.target),
  })).filter((link) => link.sourceNode && link.targetNode);

  if (nodes.length > 1) {
    const repulsionSamples = Math.min(24, Math.max(8, Math.round(Math.sqrt(nodes.length) * 1.4)));
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
    state.domainGraphLayoutCache.set(node.id, { x: node.x, y: node.y });
  });
  return { nodes, links, width, height };
}

function drawGraphBackdrop(context, width, height) {
  const phase = (state.graphAnimationTime || 0) / 1000;
  const isLight = document.documentElement.dataset.theme === "light";
  const baseGradient = context.createLinearGradient(0, 0, width, height);
  baseGradient.addColorStop(0, isLight ? "rgba(250, 253, 250, 0.98)" : "rgba(4, 12, 19, 0.98)");
  baseGradient.addColorStop(0.56, isLight ? "rgba(240, 246, 242, 0.98)" : "rgba(6, 16, 25, 0.98)");
  baseGradient.addColorStop(1, isLight ? "rgba(233, 240, 236, 0.99)" : "rgba(3, 10, 16, 1)");
  context.fillStyle = baseGradient;
  context.fillRect(0, 0, width, height);

  const veils = isLight
    ? [
        { x: width * 0.2, y: height * 0.12, radius: width * 0.42, color: "rgba(119, 201, 226, 0.14)" },
        { x: width * 0.82, y: height * 0.2, radius: width * 0.34, color: "rgba(122, 212, 163, 0.11)" },
        { x: width * 0.58, y: height * 0.82, radius: width * 0.48, color: "rgba(239, 173, 122, 0.08)" },
      ]
    : [
        { x: width * 0.16, y: height * 0.12, radius: width * 0.46, color: "rgba(67, 173, 212, 0.16)" },
        { x: width * 0.82, y: height * 0.18, radius: width * 0.34, color: "rgba(74, 193, 145, 0.12)" },
        { x: width * 0.62, y: height * 0.84, radius: width * 0.52, color: "rgba(235, 136, 92, 0.08)" },
      ];
  veils.forEach((veil, index) => {
    const wobble = Math.sin(phase * (0.18 + index * 0.06) + index * 1.4) * 16;
    const gradient = context.createRadialGradient(veil.x + wobble, veil.y, 0, veil.x, veil.y, veil.radius);
    gradient.addColorStop(0, veil.color);
    gradient.addColorStop(1, "rgba(0, 0, 0, 0)");
    context.fillStyle = gradient;
    context.fillRect(0, 0, width, height);
  });

  const starfield = getGraphBackdropCache(width, height);
  starfield.stars.forEach((star) => {
    const twinkle = 0.58 + 0.42 * Math.sin(phase * star.speed + star.phase);
    context.beginPath();
    context.fillStyle = isLight
      ? `rgba(60, 99, 119, ${star.alpha * twinkle})`
      : `rgba(216, 240, 255, ${star.alpha * twinkle})`;
    context.arc(star.x, star.y, star.radius * (0.88 + twinkle * 0.26), 0, Math.PI * 2);
    context.fill();
  });

  context.save();
  context.strokeStyle = isLight ? "rgba(46, 83, 96, 0.08)" : "rgba(141, 193, 222, 0.09)";
  context.lineWidth = 1;
  [0.68, 0.88, 1.08].forEach((scale, index) => {
    context.beginPath();
    context.arc(width * 0.54, height * 0.56, Math.min(width, height) * 0.18 * scale, Math.PI * (0.18 + index * 0.12), Math.PI * (1.6 + index * 0.1));
    context.stroke();
  });
  context.restore();
}

function distanceToSegment(pointX, pointY, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  if (dx === 0 && dy === 0) {
    return Math.hypot(pointX - x1, pointY - y1);
  }
  const t = clamp(((pointX - x1) * dx + (pointY - y1) * dy) / (dx * dx + dy * dy), 0, 1);
  const closestX = x1 + dx * t;
  const closestY = y1 + dy * t;
  return Math.hypot(pointX - closestX, pointY - closestY);
}

function quadraticPoint(t, x1, y1, cx, cy, x2, y2) {
  const inverse = 1 - t;
  return {
    x: inverse * inverse * x1 + 2 * inverse * t * cx + t * t * x2,
    y: inverse * inverse * y1 + 2 * inverse * t * cy + t * t * y2,
  };
}

function graphCurveControlPoint(sourcePoint, targetPoint, key) {
  const dx = targetPoint.x - sourcePoint.x;
  const dy = targetPoint.y - sourcePoint.y;
  const distance = Math.hypot(dx, dy) || 1;
  const normalX = -dy / distance;
  const normalY = dx / distance;
  const direction = hashString(key) % 2 === 0 ? 1 : -1;
  const bend = clamp(distance * 0.16, 14, 56) * direction;
  return {
    x: (sourcePoint.x + targetPoint.x) / 2 + normalX * bend,
    y: (sourcePoint.y + targetPoint.y) / 2 + normalY * bend,
  };
}

function distanceToQuadraticCurve(pointX, pointY, x1, y1, cx, cy, x2, y2, segments = 18) {
  let minDistance = Number.POSITIVE_INFINITY;
  let previous = { x: x1, y: y1 };
  for (let index = 1; index <= segments; index += 1) {
    const current = quadraticPoint(index / segments, x1, y1, cx, cy, x2, y2);
    minDistance = Math.min(
      minDistance,
      distanceToSegment(pointX, pointY, previous.x, previous.y, current.x, current.y)
    );
    previous = current;
  }
  return minDistance;
}

function drawClusterHull(context, nodes, width, height, palette) {
  if (!Array.isArray(nodes) || nodes.length < 2) {
    return;
  }
  const screenNodes = nodes.map((node) => {
    const point = graphNodeScreenPoint(node, width, height);
    return {
      x: point.x,
      y: point.y,
      radius: graphRadiusToScreen(node.radius),
    };
  });
  const centerX = screenNodes.reduce((sum, node) => sum + node.x, 0) / screenNodes.length;
  const centerY = screenNodes.reduce((sum, node) => sum + node.y, 0) / screenNodes.length;
  const points = screenNodes
    .map((node, index) => {
      const angle = Math.atan2(node.y - centerY, node.x - centerX);
      const distance = Math.hypot(node.x - centerX, node.y - centerY) + node.radius * 1.9 + 28 + (index % 3) * 4;
      return {
        angle,
        x: centerX + Math.cos(angle) * distance,
        y: centerY + Math.sin(angle) * distance,
      };
    })
    .sort((left, right) => left.angle - right.angle);
  if (points.length < 2) {
    return;
  }

  context.save();
  context.beginPath();
  const firstMidpointX = (points[0].x + points[points.length - 1].x) / 2;
  const firstMidpointY = (points[0].y + points[points.length - 1].y) / 2;
  context.moveTo(firstMidpointX, firstMidpointY);
  points.forEach((point, index) => {
    const next = points[(index + 1) % points.length];
    const midpointX = (point.x + next.x) / 2;
    const midpointY = (point.y + next.y) / 2;
    context.quadraticCurveTo(point.x, point.y, midpointX, midpointY);
  });
  context.closePath();
  context.fillStyle = palette.mist;
  context.shadowBlur = 42;
  context.shadowColor = palette.glow;
  context.fill();
  context.lineWidth = 1.4;
  context.strokeStyle = palette.coreSoft;
  context.stroke();
  context.restore();

  drawSoftEllipse(
    context,
    centerX,
    centerY,
    Math.max(120, Math.hypot(points[0].x - centerX, points[0].y - centerY) * 1.18),
    Math.max(84, Math.hypot(points[Math.floor(points.length / 2)].x - centerX, points[Math.floor(points.length / 2)].y - centerY) * 0.76),
    Math.PI / 7,
    palette.mist,
    "rgba(0, 0, 0, 0)"
  );
}

function findDomainGraphHit(pointX, pointY) {
  if (!state.currentDomainGraphLayout) {
    return null;
  }
  const { width, height } = state.currentDomainGraphLayout;
  if (state.graphSceneMode === "overview") {
    const clusterHit = [...state.currentDomainGraphLayout.clusters]
      .sort((left, right) => right.radius - left.radius)
      .find((cluster) => {
        const point = graphClusterScreenPoint(cluster, width, height);
        return Math.hypot(pointX - point.x, pointY - point.y) <= graphRadiusToScreen(cluster.radius) + 4;
      });
    if (clusterHit) {
      return { type: "cluster", item: clusterHit };
    }
    return null;
  }

  const nodeHit = [...state.currentDomainGraphLayout.nodes]
    .sort((left, right) => right.radius - left.radius)
    .find((node) => {
      const point = graphNodeScreenPoint(node, width, height);
      return Math.hypot(pointX - point.x, pointY - point.y) <= graphRadiusToScreen(node.radius) + 3;
    });
  if (nodeHit) {
    return { type: "node", item: nodeHit };
  }
  const linkHit = state.currentDomainGraphLayout.links.find((link) => {
    const { sourcePoint, targetPoint, controlPoint } = graphLinkScreenPoints(link, width, height);
    const threshold = 4 + Math.min(4, Number(link.count || 0)) + Math.max(0, state.graphViewport.scale - 1);
    return distanceToQuadraticCurve(
      pointX,
      pointY,
      sourcePoint.x,
      sourcePoint.y,
      controlPoint.x,
      controlPoint.y,
      targetPoint.x,
      targetPoint.y
    ) <= threshold;
  });
  if (linkHit) {
    return { type: "link", item: linkHit };
  }
  return null;
}

function clearGraphDragState() {
  state.graphPointer.dragType = null;
  state.graphPointer.dragId = null;
  state.graphPointer.dragOffsetX = 0;
  state.graphPointer.dragOffsetY = 0;
}

function isGraphDragHit(hit) {
  if (!hit) {
    return false;
  }
  return (state.graphSceneMode === "overview" && hit.type === "cluster")
    || (state.graphSceneMode === "cluster" && hit.type === "node");
}

function startDraggingGraphItem(hit, pointX, pointY, width, height) {
  if (!isGraphDragHit(hit)) {
    clearGraphDragState();
    return false;
  }
  const worldPoint = screenToGraphPoint(pointX, pointY, width, height);
  state.graphPointer.isPanning = false;
  state.graphPointer.dragType = hit.type;
  state.graphPointer.dragId = hit.item.id;
  state.graphPointer.dragOffsetX = hit.item.x - worldPoint.x;
  state.graphPointer.dragOffsetY = hit.item.y - worldPoint.y;
  state.graphPointer.velocityX = 0;
  state.graphPointer.velocityY = 0;
  if (hit.type === "cluster") {
    state.hoveredGraphClusterId = hit.item.id;
    state.hoveredGraphNodeId = null;
    state.hoveredGraphLinkKey = null;
  } else {
    state.hoveredGraphNodeId = hit.item.id;
    state.hoveredGraphClusterId = null;
    state.hoveredGraphLinkKey = null;
  }
  addGraphRipple(pointX, pointY, {
    color: graphRippleColorForHit(hit),
    size: hit.type === "cluster" ? 118 : 76,
    duration: 720,
    width: 2.8,
  });
  return true;
}

function updateDraggedGraphItem(pointX, pointY, width, height) {
  if (!state.graphPointer.dragType || !state.graphPointer.dragId || !state.currentDomainGraphLayout) {
    return false;
  }
  const worldPoint = screenToGraphPoint(pointX, pointY, width, height);
  const collection = state.graphPointer.dragType === "cluster"
    ? state.currentDomainGraphLayout.clusters
    : state.currentDomainGraphLayout.nodes;
  const item = collection.find((entry) => entry.id === state.graphPointer.dragId);
  if (!item) {
    clearGraphDragState();
    return false;
  }
  const padding = state.graphPointer.dragType === "cluster" ? 24 : 26;
  item.x = clamp(worldPoint.x + state.graphPointer.dragOffsetX, item.radius + padding, width - item.radius - padding);
  item.y = clamp(worldPoint.y + state.graphPointer.dragOffsetY, item.radius + padding, height - item.radius - padding);
  if (state.graphPointer.dragType === "cluster") {
    state.clusterGraphLayoutCache.set(item.id, { x: item.x, y: item.y });
  } else {
    state.domainGraphLayoutCache.set(item.id, { x: item.x, y: item.y });
  }
  return true;
}

function renderGraphNeighbors(neighbors) {
  if (!neighbors.length) {
    return "<p class=\"muted\">No adjacent domains under the current filters.</p>";
  }
  return `
    <div class="graph-neighbors">
      ${neighbors.map((item) => `
        <div class="graph-neighbor">
          <div>
            <strong>${escapeHtml(item.domain)}</strong>
            <div class="muted">
              ${escapeHtml(formatInteger(item.count))} explorations •
              ${escapeHtml(formatInteger(item.transmitted_count))} transmitted •
              best ${escapeHtml(formatScore(item.max_score))}
            </div>
          </div>
          <div class="graph-neighbor-actions">
            <button type="button" class="graph-focus-button" data-focus-domain="${escapeHtml(item.domain)}">Focus</button>
          </div>
        </div>
      `).join("")}
    </div>
  `;
}

function renderClusterHighlights(clusters) {
  if (!clusters.length) {
    return "<p class=\"muted\">No visible constellations under the current filters.</p>";
  }
  return `
    <div class="graph-neighbors">
      ${clusters.map((cluster) => `
        <div class="graph-neighbor">
          <div>
            <strong>${escapeHtml(cluster.label)}</strong>
            <div class="muted">
              ${escapeHtml(formatInteger(cluster.node_count || 0))} domains •
              ${escapeHtml(formatInteger(cluster.link_count || 0))} bridges •
              ${escapeHtml(formatInteger(cluster.open_salvage_count || 0))} salvage •
              avg ${escapeHtml(formatScore(cluster.avg_score))}
            </div>
          </div>
          <div class="graph-neighbor-actions">
            <button type="button" class="graph-focus-button" data-focus-cluster="${escapeHtml(cluster.id)}">Open region</button>
          </div>
        </div>
      `).join("")}
    </div>
  `;
}

function renderClusterBridgeHighlights(links) {
  if (!links.length) {
    return "<p class=\"muted\">No bridges are visible inside this region under the current filters.</p>";
  }
  return `
    <div class="graph-neighbors">
      ${links.map((link) => `
        <div class="graph-neighbor">
          <div>
            <strong>${escapeHtml(truncateText(link.source, 28))} → ${escapeHtml(truncateText(link.target, 28))}</strong>
            <div class="muted">
              ${escapeHtml(formatInteger(link.count || 0))} explorations •
              ${escapeHtml(formatInteger(link.transmitted_count || 0))} transmitted •
              best ${escapeHtml(formatScore(link.max_score))}
            </div>
          </div>
          <div class="graph-neighbor-actions">
            <button type="button" class="graph-focus-button" data-focus-domain="${escapeHtml(link.source)}">Focus source</button>
          </div>
        </div>
      `).join("")}
    </div>
  `;
}

function renderInspectorHeader(eyebrow, title, lead) {
  return `
    <div class="inspector-header">
      <p class="inspector-eyebrow">${escapeHtml(eyebrow)}</p>
      <h3 class="inspector-title">${escapeHtml(title)}</h3>
      <p class="muted inspector-lead">${escapeHtml(lead)}</p>
    </div>
  `;
}

function renderInspectorChipRow(items) {
  const rows = items.filter(Boolean);
  if (!rows.length) {
    return "";
  }
  return `
    <div class="inspector-chip-row">
      ${rows.map((item) => `<span class="inspector-chip">${escapeHtml(item)}</span>`).join("")}
    </div>
  `;
}

function renderDomainGraphDetail() {
  const panel = document.getElementById("domain-web-detail");
  if (!panel) {
    return;
  }
  if (!state.domainGraph) {
    panel.innerHTML = `
      ${renderInspectorHeader("Standby", "Graph Unavailable", "The map is waiting on live graph data before it can render territory.")}
    `;
    return;
  }

  if (state.graphSceneMode === "overview") {
    const hoveredOverviewCluster = Array.isArray(state.currentClusterOverview?.clusters)
      ? state.currentClusterOverview.clusters.find((cluster) => cluster.id === state.hoveredGraphClusterId)
      : null;
    if (hoveredOverviewCluster) {
      const hoveredLaunchpads = (state.frontier?.launchpads || [])
        .filter((row) => row.cluster_id === hoveredOverviewCluster.id)
        .sort((left, right) => Number(right.frontier_score || 0) - Number(left.frontier_score || 0) || Number(right.transmitted_count || 0) - Number(left.transmitted_count || 0))
        .slice(0, 4)
        .map((row) => ({
          domain: row.domain,
          count: Number(row.exploration_count || 0),
          transmitted_count: Number(row.transmitted_count || 0),
          max_score: row.avg_score,
        }));
      panel.innerHTML = `
        ${renderInspectorHeader(
          "Hover Region",
          hoveredOverviewCluster.label,
          "This is a live preview of the constellation under your pointer. Drag it around the atlas or click to dive into its internal web."
        )}
        ${renderInspectorChipRow([
          `${formatInteger(hoveredOverviewCluster.node_count || 0)} domains`,
          `${formatInteger(hoveredOverviewCluster.visible_link_count || hoveredOverviewCluster.link_count || 0)} visible bridges`,
          Number(hoveredOverviewCluster.visible_transmitted_link_count || hoveredOverviewCluster.transmitted_link_count || 0) > 0
            ? `${formatInteger(hoveredOverviewCluster.visible_transmitted_link_count || hoveredOverviewCluster.transmitted_link_count || 0)} transmitting`
            : "no transmissions yet",
        ])}
        <div class="inspector-section">
          ${renderDetailGrid([
            { label: "Mapped bridges", value: formatInteger(hoveredOverviewCluster.link_count || 0) },
            { label: "Visible bridges", value: formatInteger(hoveredOverviewCluster.visible_link_count || 0) },
            { label: "Open salvage", value: formatInteger(hoveredOverviewCluster.open_salvage_count || 0) },
            { label: "Average score", value: formatScore(hoveredOverviewCluster.visible_avg_score ?? hoveredOverviewCluster.avg_score) },
            { label: "Latest activity", value: formatTimestamp(hoveredOverviewCluster.latest_timestamp) },
          ])}
        </div>
        <div class="inspector-section">
          <h4>Anchor Domains</h4>
          <div class="frontier-cluster-domains">
            ${(Array.isArray(hoveredOverviewCluster.top_domains) ? hoveredOverviewCluster.top_domains : []).slice(0, 5).map((domain) => `
              <span class="frontier-domain-chip">${escapeHtml(truncateText(domain, 32))}</span>
            `).join("")}
          </div>
        </div>
        <div class="inspector-section">
          <h4>Top Launchpads</h4>
          ${renderGraphNeighbors(hoveredLaunchpads)}
        </div>
      `;
      return;
    }

    const overviewClusters = Array.isArray(state.currentClusterOverview?.clusters)
      ? state.currentClusterOverview.clusters.slice(0, 6)
      : [];
    const controls = state.currentClusterOverview?.controls || {};
    panel.innerHTML = `
      ${renderInspectorHeader(
        "Atlas View",
        "Constellation Overview",
        "Read the territory at the region level first. The strongest visible constellations are surfaced here before you drill into their internal web."
      )}
      ${renderInspectorChipRow([
        `${formatInteger(overviewClusters.length)} visible constellations`,
        controls.mode === "successful" ? "transmitted filter" : controls.mode === "strong" ? "high-signal filter" : "full territory",
        controls.query ? `focus: ${truncateText(controls.query, 24)}` : "atlas scan",
      ])}
      <div class="inspector-section">
        <h4>Spotlight Regions</h4>
        ${renderClusterHighlights(overviewClusters)}
      </div>
    `;
    return;
  }

  if (!state.currentDomainGraph || !state.currentDomainGraph.nodes.length) {
    panel.innerHTML = `
      ${renderInspectorHeader(
        "Hidden Region",
        "No Visible Cluster Detail",
        "This region is selected, but the current filters are hiding its internal bridges. Reset the view or lower the score threshold to reveal it again."
      )}
    `;
    return;
  }

  const activeLinkKey = state.selectedGraphLinkKey || (!state.selectedGraphNodeId ? state.hoveredGraphLinkKey : null);
  if (activeLinkKey) {
    const link = state.currentDomainGraph.links.find((item) => graphLinkKey(item) === activeLinkKey);
    if (link) {
      const isPreview = !state.selectedGraphLinkKey;
      panel.innerHTML = `
        ${renderInspectorHeader(
          isPreview ? "Hover Route" : "Bridge Route",
          `${link.source} → ${link.target}`,
          isPreview
            ? "You are hovering a live bridge preview. Click to lock onto the route or drag nearby nodes to reshape the region around it."
            : "A directed BlackClaw jump from seed domain to target domain. Brighter, warmer arcs represent more repeated or transmitted routes."
        )}
        ${renderInspectorChipRow([
          Number(link.transmitted_count || 0) > 0 ? "transmitting route" : "untransmitted route",
          Number(link.count || 0) > 1 ? "repeated bridge" : "single bridge",
          Number(link.max_score || 0) >= 0.85 ? "high-signal" : "",
        ])}
        <div class="inspector-section">
          ${renderDetailGrid([
            { label: "Explorations", value: formatInteger(link.count) },
            { label: "Transmitted", value: formatInteger(link.transmitted_count) },
            { label: "Average score", value: formatScore(link.avg_score) },
            { label: "Best score", value: formatScore(link.max_score) },
            { label: "Last explored", value: formatTimestamp(link.latest_timestamp) },
          ])}
        </div>
        <div class="inspector-section">
          <h4>Route Actions</h4>
          <div class="review-actions">
            <button type="button" class="graph-focus-button" data-focus-domain="${escapeHtml(link.source)}">Focus source</button>
            <button type="button" class="graph-focus-button" data-focus-domain="${escapeHtml(link.target)}">Focus target</button>
          </div>
        </div>
      `;
      return;
    }
  }

  const activeNodeId = state.selectedGraphNodeId || (!state.selectedGraphLinkKey ? state.hoveredGraphNodeId : null);
  if (activeNodeId) {
    const node = state.currentDomainGraph.nodes.find((item) => item.id === activeNodeId);
    if (node) {
      const isPreview = !state.selectedGraphNodeId;
      const neighbors = state.currentDomainGraph.links
        .filter((link) => link.source === node.id || link.target === node.id)
        .map((link) => ({
          domain: link.source === node.id ? link.target : link.source,
          count: Number(link.count || 0),
          transmitted_count: Number(link.transmitted_count || 0),
          max_score: link.max_score,
        }))
        .sort((left, right) => right.transmitted_count - left.transmitted_count || Number(right.max_score || 0) - Number(left.max_score || 0) || right.count - left.count)
        .slice(0, 8);
      const cluster = (state.domainGraph.clusters || []).find((item) => item.id === node.cluster_id);

      panel.innerHTML = `
        ${renderInspectorHeader(
          isPreview ? "Hover Node" : "Domain Node",
          node.id,
          isPreview
            ? `You are hovering a ${domainRoleLabel(node.role)} node. Grab it to pull the local web around or click to keep it selected in the inspector.`
            : `This domain appears as ${domainRoleLabel(node.role)} in the visible map. Use it to understand whether the region treats this domain like a launchpad, landing zone, or bridge.`
        )}
        ${renderInspectorChipRow([
          domainRoleLabel(node.role),
          cluster ? truncateText(cluster.label, 26) : "",
          Number(node.transmitted_count || 0) > 0 ? `${formatInteger(node.transmitted_count)} transmitted touch${Number(node.transmitted_count || 0) === 1 ? "" : "es"}` : "",
        ])}
        <div class="inspector-section">
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
        </div>
        <div class="inspector-section">
          <h4>Strongest Adjacent Domains</h4>
          ${renderGraphNeighbors(neighbors)}
        </div>
      `;
      return;
    }
  }

  const selectedCluster = (state.domainGraph.clusters || []).find((cluster) => cluster.id === state.selectedGraphClusterId);
  const clusterLaunchpads = (state.frontier?.launchpads || [])
    .filter((row) => row.cluster_id === state.selectedGraphClusterId)
    .sort((left, right) => Number(right.frontier_score || 0) - Number(left.frontier_score || 0) || Number(right.transmitted_count || 0) - Number(left.transmitted_count || 0))
    .slice(0, 5)
    .map((row) => ({
      domain: row.domain,
      count: Number(row.exploration_count || 0),
      transmitted_count: Number(row.transmitted_count || 0),
      max_score: row.avg_score,
    }));
  const strongestBridges = state.currentDomainGraph.links
    .slice()
    .sort((left, right) => Number(right.transmitted_count || 0) - Number(left.transmitted_count || 0) || Number(right.max_score || 0) - Number(left.max_score || 0) || Number(right.count || 0) - Number(left.count || 0))
    .slice(0, 5);

  panel.innerHTML = `
    ${renderInspectorHeader(
      "Region Focus",
      selectedCluster?.label || "Cluster region",
      "This region is one connected component in BlackClaw’s explored territory. Treat it like a navigable neighborhood rather than a flat list of domains."
    )}
    ${renderInspectorChipRow([
      `${formatInteger(selectedCluster?.node_count || 0)} domains`,
      `${formatInteger(selectedCluster?.transmitted_link_count || 0)} transmitting bridges`,
      Number(selectedCluster?.open_salvage_count || 0) > 0 ? `${formatInteger(selectedCluster?.open_salvage_count || 0)} salvage leads` : "no open salvage",
    ])}
    <div class="inspector-section">
      ${renderDetailGrid([
        { label: "Domains", value: formatInteger(selectedCluster?.node_count || 0) },
        { label: "Mapped bridges", value: formatInteger(selectedCluster?.link_count || 0) },
        { label: "Transmitting bridges", value: formatInteger(selectedCluster?.transmitted_link_count || 0) },
        { label: "Explorations", value: formatInteger(selectedCluster?.exploration_count || 0) },
        { label: "Open salvage", value: formatInteger(selectedCluster?.open_salvage_count || 0) },
        { label: "Average score", value: formatScore(selectedCluster?.avg_score) },
        { label: "Latest activity", value: formatTimestamp(selectedCluster?.latest_timestamp) },
      ])}
    </div>
    <div class="inspector-section">
      <h4>Anchor Domains</h4>
      <div class="frontier-cluster-domains">
        ${(Array.isArray(selectedCluster?.top_domains) ? selectedCluster.top_domains : []).slice(0, 5).map((domain) => `
          <span class="frontier-domain-chip">${escapeHtml(truncateText(domain, 32))}</span>
        `).join("")}
      </div>
    </div>
    <div class="inspector-section">
      <h4>Top Launchpads</h4>
      ${renderGraphNeighbors(clusterLaunchpads)}
    </div>
    <div class="inspector-section">
      <h4>Strongest Bridges</h4>
      ${renderClusterBridgeHighlights(strongestBridges)}
    </div>
  `;
}

function renderClusterOverviewCanvas(context, width, height) {
  const phase = (state.graphAnimationTime || 0) / 1000;
  state.currentDomainGraphLayout.clusters
    .slice()
    .sort((left, right) => left.radius - right.radius)
    .forEach((cluster) => {
      const point = graphClusterScreenPoint(cluster, width, height);
      const radius = graphRadiusToScreen(cluster.radius);
      const isSelected = cluster.id === state.selectedGraphClusterId;
      const isHovered = cluster.id === state.hoveredGraphClusterId;
      const alpha = isSelected || isHovered ? 0.98 : cluster.isMatch ? 0.96 : 0.82;
      const palette = graphClusterPalette(cluster);
      const pulse = 0.86 + Math.sin(phase * 0.8 + (hashString(cluster.id) % 31)) * 0.06;

      drawSoftEllipse(
        context,
        point.x - radius * 0.14,
        point.y + radius * 0.05,
        radius * (1.72 + pulse * 0.06),
        radius * (1.08 + pulse * 0.04),
        (hashString(cluster.id) % 360) / 180,
        palette.mist,
        "rgba(0, 0, 0, 0)"
      );
      drawSoftEllipse(
        context,
        point.x + radius * 0.16,
        point.y - radius * 0.12,
        radius * (1.28 + pulse * 0.04),
        radius * (0.84 + pulse * 0.03),
        (hashString(`${cluster.id}:mist`) % 360) / 160,
        palette.glow,
        "rgba(0, 0, 0, 0)"
      );

      context.save();
      context.beginPath();
      context.arc(point.x, point.y, radius * 1.12, 0, Math.PI * 2);
      context.fillStyle = palette.coreSoft;
      context.shadowBlur = 34 + Number(cluster.visible_transmitted_link_count || 0) * 10;
      context.shadowColor = palette.glow;
      context.fill();
      context.restore();

      const clusterGradient = context.createRadialGradient(
        point.x - radius * 0.24,
        point.y - radius * 0.28,
        radius * 0.18,
        point.x,
        point.y,
        radius
      );
      clusterGradient.addColorStop(0, palette.edge);
      clusterGradient.addColorStop(0.42, palette.core);
      clusterGradient.addColorStop(1, palette.glow);
      context.beginPath();
      context.arc(point.x, point.y, radius, 0, Math.PI * 2);
      context.fillStyle = clusterGradient;
      context.fill();

      context.beginPath();
      context.arc(point.x, point.y, radius * 0.58, 0, Math.PI * 2);
      context.fillStyle = document.documentElement.dataset.theme === "light"
        ? `rgba(255, 255, 255, ${0.48 * alpha})`
        : `rgba(255, 251, 240, ${0.24 * alpha})`;
      context.fill();

      context.beginPath();
      context.arc(point.x, point.y, radius, 0, Math.PI * 2);
      context.lineWidth = isSelected || isHovered ? 2.8 : 1.35;
      context.strokeStyle = isSelected || isHovered ? palette.edge : palette.core;
      context.stroke();

      if (isSelected || isHovered) {
        for (let ring = 0; ring < 2; ring += 1) {
          context.beginPath();
          context.arc(point.x, point.y, radius * (1.2 + ring * 0.18 + pulse * 0.05), 0, Math.PI * 2);
          context.lineWidth = 1.6 - ring * 0.35;
          context.strokeStyle = ring === 0 ? palette.edge : palette.glow;
          context.globalAlpha = 0.28 - ring * 0.11;
          context.stroke();
          context.globalAlpha = 1;
        }
      }

      const orbitPoint = quadraticPoint(
        (phase * 0.12 + (hashString(cluster.id) % 1000) / 1000) % 1,
        point.x - radius * 0.84,
        point.y,
        point.x,
        point.y - radius * 1.12,
        point.x + radius * 0.84,
        point.y
      );
      context.beginPath();
      context.arc(orbitPoint.x, orbitPoint.y, Math.max(1.8, radius * 0.08), 0, Math.PI * 2);
      context.fillStyle = palette.edge;
      context.shadowBlur = 10;
      context.shadowColor = palette.glow;
      context.fill();

      const label = truncateText(cluster.label, 34);
      const sublabel = `${formatInteger(cluster.node_count || 0)} domains`;
      const labelFont = Math.max(13, Math.min(20, radius * 0.22));
      const sublabelFont = Math.max(10, Math.min(12, radius * 0.12));
      context.font = `600 ${labelFont}px "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif`;
      const labelWidth = context.measureText(label).width;
      const plateWidth = Math.max(labelWidth + 36, radius * 1.2);
      const plateHeight = 42;
      drawRoundedRectPath(context, point.x - plateWidth / 2, point.y - 18, plateWidth, plateHeight, 18);
      context.fillStyle = document.documentElement.dataset.theme === "light"
        ? "rgba(255, 255, 255, 0.76)"
        : "rgba(4, 16, 22, 0.72)";
      context.fill();
      context.lineWidth = 1;
      context.strokeStyle = document.documentElement.dataset.theme === "light"
        ? "rgba(51, 83, 96, 0.12)"
        : "rgba(255, 255, 255, 0.08)";
      context.stroke();

      context.textAlign = "center";
      context.textBaseline = "middle";
      context.font = `600 ${labelFont}px "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif`;
      context.fillStyle = palette.label;
      context.fillText(label, point.x, point.y - 7);
      context.font = `${sublabelFont}px "SFMono-Regular", Menlo, Monaco, Consolas, monospace`;
      context.fillStyle = document.documentElement.dataset.theme === "light"
        ? "rgba(15, 34, 41, 0.72)"
        : "rgba(244, 251, 255, 0.74)";
      context.fillText(sublabel, point.x, point.y + 10);
    });
}

function renderClusterDetailCanvas(context, width, height) {
  const activeNodeIds = new Set();
  if (state.selectedGraphNodeId) {
    activeNodeIds.add(state.selectedGraphNodeId);
    state.currentDomainGraph.links.forEach((link) => {
      if (link.source === state.selectedGraphNodeId || link.target === state.selectedGraphNodeId) {
        activeNodeIds.add(link.source);
        activeNodeIds.add(link.target);
      }
    });
  }
  if (state.selectedGraphLinkKey) {
    const selectedLink = state.currentDomainGraph.links.find((link) => graphLinkKey(link) === state.selectedGraphLinkKey);
    if (selectedLink) {
      activeNodeIds.add(selectedLink.source);
      activeNodeIds.add(selectedLink.target);
    }
  }
  if (state.hoveredGraphNodeId) {
    activeNodeIds.add(state.hoveredGraphNodeId);
    state.currentDomainGraph.links.forEach((link) => {
      if (link.source === state.hoveredGraphNodeId || link.target === state.hoveredGraphNodeId) {
        activeNodeIds.add(link.source);
        activeNodeIds.add(link.target);
      }
    });
  }

  const selectedCluster = (state.domainGraph?.clusters || []).find((cluster) => cluster.id === state.selectedGraphClusterId);
  const clusterPalette = graphClusterPalette(selectedCluster || { id: state.selectedGraphClusterId || "cluster" });
  drawClusterHull(context, state.currentDomainGraphLayout.nodes, width, height, clusterPalette);

  state.currentDomainGraphLayout.links.forEach((link) => {
    const { sourcePoint, targetPoint, controlPoint } = graphLinkScreenPoints(link, width, height);
    const key = graphLinkKey(link);
    const isSelected = key === state.selectedGraphLinkKey;
    const isHovered = key === state.hoveredGraphLinkKey;
    const touchesSelectedNode = state.selectedGraphNodeId && (link.source === state.selectedGraphNodeId || link.target === state.selectedGraphNodeId);
    const touchesHoveredNode = state.hoveredGraphNodeId && (link.source === state.hoveredGraphNodeId || link.target === state.hoveredGraphNodeId);
    const isEnergized = isSelected || isHovered || touchesSelectedNode || touchesHoveredNode;
    const isDimmed = (state.selectedGraphNodeId && !touchesSelectedNode) || (state.selectedGraphLinkKey && !isSelected);
    const score = Number(link.max_score ?? link.avg_score ?? 0);
    const baseAlpha = clamp(0.16 + score * 0.36 + Number(link.transmitted_count || 0) * 0.18, 0.12, 0.88);
    const alpha = isSelected || isHovered ? 0.98 : isEnergized ? Math.max(baseAlpha, 0.76) : isDimmed ? 0.08 : baseAlpha;
    const lineWidth = (isSelected || isHovered
      ? 3.9
      : isEnergized
        ? clamp(1.4 + Number(link.count || 0) * 0.6 + Number(link.transmitted_count || 0) * 0.8, 1.4, 4.8)
      : clamp(0.9 + Number(link.count || 0) * 0.55 + Number(link.transmitted_count || 0) * 0.72, 0.9, 4.4))
      * Math.max(0.85, state.graphViewport.scale);
    let lineColor;
    let glowColor;
    if (Number(link.transmitted_count || 0) > 0) {
      lineColor = document.documentElement.dataset.theme === "light"
        ? `rgba(28, 143, 103, ${alpha})`
        : `rgba(89, 211, 154, ${alpha})`;
      glowColor = document.documentElement.dataset.theme === "light"
        ? `rgba(125, 223, 180, ${alpha * 0.56})`
        : `rgba(110, 244, 184, ${alpha * 0.54})`;
    } else if (score >= 0.85) {
      lineColor = document.documentElement.dataset.theme === "light"
        ? `rgba(41, 138, 176, ${alpha})`
        : `rgba(103, 213, 255, ${alpha})`;
      glowColor = document.documentElement.dataset.theme === "light"
        ? `rgba(142, 210, 235, ${alpha * 0.52})`
        : `rgba(84, 201, 255, ${alpha * 0.48})`;
    } else {
      lineColor = document.documentElement.dataset.theme === "light"
        ? `rgba(89, 112, 123, ${alpha})`
        : `rgba(154, 178, 190, ${alpha})`;
      glowColor = document.documentElement.dataset.theme === "light"
        ? `rgba(173, 191, 202, ${alpha * 0.38})`
        : `rgba(121, 151, 169, ${alpha * 0.34})`;
    }

    context.save();
    context.beginPath();
    context.moveTo(sourcePoint.x, sourcePoint.y);
    context.quadraticCurveTo(controlPoint.x, controlPoint.y, targetPoint.x, targetPoint.y);
    context.lineWidth = lineWidth * 2.6;
    context.strokeStyle = glowColor;
    context.shadowBlur = isSelected || isHovered ? 28 : isEnergized ? 24 : 16;
    context.shadowColor = glowColor;
    context.stroke();
    context.restore();

    context.beginPath();
    context.moveTo(sourcePoint.x, sourcePoint.y);
    context.quadraticCurveTo(controlPoint.x, controlPoint.y, targetPoint.x, targetPoint.y);
    context.lineWidth = lineWidth;
    context.strokeStyle = lineColor;
    context.stroke();

    if (Number(link.transmitted_count || 0) > 0 || isEnergized) {
      renderTravelingEnergy(
        context,
        sourcePoint,
        controlPoint,
        targetPoint,
        Number(link.transmitted_count || 0) > 0 ? glowColor : lineColor,
        `${key}:${Number(link.transmitted_count || 0) > 0 ? "tx" : "hover"}`,
        isSelected || isHovered ? 4 : 3,
        isSelected || isHovered ? 1 : isEnergized ? 0.78 : 0.42
      );
    }
  });

  const emphasizedNodes = new Set(activeNodeIds);
  if (state.hoveredGraphNodeId) {
    emphasizedNodes.add(state.hoveredGraphNodeId);
  }
  state.currentDomainGraphLayout.nodes
    .slice()
    .sort((left, right) => left.radius - right.radius)
    .forEach((node) => {
      const point = graphNodeScreenPoint(node, width, height);
      const radius = graphRadiusToScreen(node.radius);
      const isSelected = node.id === state.selectedGraphNodeId;
      const isHovered = node.id === state.hoveredGraphNodeId;
      const isActive = !state.selectedGraphNodeId && !state.selectedGraphLinkKey ? true : emphasizedNodes.has(node.id);
      const alpha = isSelected || isHovered ? 1 : isActive ? 0.92 : 0.24;
      const palette = graphNodePalette(node.role, alpha);

      context.save();
      context.beginPath();
      context.arc(point.x, point.y, radius * 1.5, 0, Math.PI * 2);
      context.fillStyle = palette.glow;
      context.shadowBlur = isSelected || isHovered ? 32 : isActive ? 22 : 18;
      context.shadowColor = palette.glow;
      context.fill();
      context.restore();

      const nodeGradient = context.createRadialGradient(
        point.x - radius * 0.24,
        point.y - radius * 0.28,
        radius * 0.18,
        point.x,
        point.y,
        radius
      );
      nodeGradient.addColorStop(0, palette.ring);
      nodeGradient.addColorStop(0.5, palette.fill);
      nodeGradient.addColorStop(1, palette.glow);
      context.beginPath();
      context.arc(point.x, point.y, radius, 0, Math.PI * 2);
      context.fillStyle = nodeGradient;
      context.fill();

      context.beginPath();
      context.arc(point.x, point.y, radius * 0.38, 0, Math.PI * 2);
      context.fillStyle = document.documentElement.dataset.theme === "light"
        ? `rgba(255, 255, 255, ${alpha * 0.7})`
        : `rgba(255, 252, 244, ${alpha * 0.36})`;
      context.fill();

      context.lineWidth = isSelected || isHovered ? 2.6 : 1.2;
      context.strokeStyle = Number(node.transmitted_count || 0) > 0
        ? palette.ring
        : (document.documentElement.dataset.theme === "light" ? "rgba(16, 39, 49, 0.42)" : "rgba(244, 251, 255, 0.26)");
      context.stroke();
    });

  const labelCandidates = state.currentDomainGraphLayout.nodes
    .filter((node) => node.isMatch || node.id === state.selectedGraphNodeId || node.id === state.hoveredGraphNodeId || Number(node.transmitted_count || 0) > 0 || Number(node.connection_count || 0) >= 3)
    .sort((left, right) => Number(right.isMatch) - Number(left.isMatch) || Number(right.transmitted_count || 0) - Number(left.transmitted_count || 0) || Number(right.connection_count || 0) - Number(left.connection_count || 0))
    .slice(0, state.currentDomainGraph.query ? 28 : 18);

  context.font = '12px "SFMono-Regular", Menlo, Monaco, Consolas, monospace';
  context.textBaseline = "middle";
  context.textAlign = "left";
  labelCandidates.forEach((node) => {
    const point = graphNodeScreenPoint(node, width, height);
    const radius = graphRadiusToScreen(node.radius);
    const label = truncateText(node.id, 34);
    const textX = point.x + radius + 8;
    const textY = point.y;
    const labelWidth = context.measureText(label).width + 18;
    drawRoundedRectPath(context, textX - 6, textY - 12, labelWidth, 24, 12);
    context.fillStyle = document.documentElement.dataset.theme === "light"
      ? "rgba(255, 255, 255, 0.82)"
      : "rgba(7, 19, 23, 0.72)";
    context.fill();
    context.lineWidth = 1;
    context.strokeStyle = document.documentElement.dataset.theme === "light"
      ? "rgba(72, 104, 118, 0.12)"
      : "rgba(255, 255, 255, 0.08)";
    context.stroke();
    context.fillStyle = document.documentElement.dataset.theme === "light"
      ? "rgba(16, 32, 41, 0.92)"
      : "rgba(244, 251, 255, 0.92)";
    context.fillText(label, textX + 2, textY);
  });
}

function renderDomainGraphCanvas(options = {}) {
  const preparedCanvas = prepareDomainGraphCanvas();
  if (!preparedCanvas) {
    return;
  }
  const { context, width, height } = preparedCanvas;
  const emptyState = document.getElementById("domain-web-empty");
  if (!emptyState) {
    return;
  }
  const hasOverview = state.graphSceneMode === "overview" && Array.isArray(state.currentClusterOverview?.clusters) && state.currentClusterOverview.clusters.length;
  const hasDetail = state.graphSceneMode === "cluster" && state.currentDomainGraph && state.currentDomainGraph.nodes.length;
  if (!hasOverview && !hasDetail) {
    context.clearRect(0, 0, width, height);
    emptyState.hidden = false;
    return;
  }
  emptyState.hidden = true;

  const layoutNeedsRefresh = !state.currentDomainGraphLayout
    || state.currentDomainGraphLayout.width !== width
    || state.currentDomainGraphLayout.height !== height
    || (state.graphSceneMode === "overview"
      ? state.currentDomainGraphLayout.sceneType !== "overview"
        || state.currentDomainGraphLayout.clusters.length !== state.currentClusterOverview.clusters.length
      : state.currentDomainGraphLayout.sceneType !== "cluster"
        || state.currentDomainGraphLayout.nodes.length !== state.currentDomainGraph.nodes.length
        || state.currentDomainGraphLayout.links.length !== state.currentDomainGraph.links.length);

  if (layoutNeedsRefresh) {
    if (state.graphSceneMode === "overview") {
      state.currentDomainGraphLayout = {
        ...computeClusterOverviewLayout(state.currentClusterOverview, width, height),
        sceneType: "overview",
      };
    } else {
      state.currentDomainGraphLayout = {
        ...computeDomainGraphLayout(state.currentDomainGraph, width, height),
        sceneType: "cluster",
      };
    }
  }

  if (options.autoFit) {
    zoomGraphToBounds(getCurrentGraphBounds(state.currentDomainGraphLayout), width, height);
    updateGraphMotionFrame();
  }

  context.clearRect(0, 0, width, height);
  drawGraphBackdrop(context, width, height);

  if (state.graphSceneMode === "overview") {
    renderClusterOverviewCanvas(context, width, height);
  } else {
    renderClusterDetailCanvas(context, width, height);
  }

  const vignette = context.createRadialGradient(width / 2, height / 2, Math.min(width, height) * 0.18, width / 2, height / 2, Math.max(width, height) * 0.68);
  vignette.addColorStop(0, "rgba(0, 0, 0, 0)");
  vignette.addColorStop(1, document.documentElement.dataset.theme === "light" ? "rgba(32, 57, 64, 0.08)" : "rgba(2, 8, 13, 0.34)");
  context.fillStyle = vignette;
  context.fillRect(0, 0, width, height);
  renderGraphSceneTransitionOverlay(context, width, height);
  renderGraphRipples(context);
}

function updateDomainGraph() {
  if (!state.domainGraph) {
    return;
  }
  const previousSceneMode = state.graphSceneMode;
  const previousClusterId = state.selectedGraphClusterId;
  state.currentClusterOverview = buildClusterOverview(state.domainGraph);
  const queryFocus = state.currentClusterOverview.focus;

  if (state.currentClusterOverview.controls.query) {
    if (queryFocus) {
      state.selectedGraphClusterId = queryFocus.clusterId;
    } else {
      state.selectedGraphClusterId = null;
      state.selectedGraphNodeId = null;
      state.selectedGraphLinkKey = null;
    }
  } else if (state.selectedGraphClusterId && !(state.domainGraph.clusters || []).some((cluster) => cluster.id === state.selectedGraphClusterId)) {
    state.selectedGraphClusterId = null;
  }

  if (!state.selectedGraphClusterId) {
    state.graphSceneMode = "overview";
    state.currentDomainGraph = null;
    state.selectedGraphNodeId = null;
    state.selectedGraphLinkKey = null;
  } else {
    state.graphSceneMode = "cluster";
    state.currentDomainGraph = buildClusterDetailGraph(
      state.domainGraph,
      state.selectedGraphClusterId,
      state.currentClusterOverview.filteredLinks,
      state.currentClusterOverview.controls.query
    );

    if (queryFocus?.exactNodeId && state.currentDomainGraph.nodes.some((node) => node.id === queryFocus.exactNodeId)) {
      state.selectedGraphNodeId = queryFocus.exactNodeId;
      state.selectedGraphLinkKey = null;
    } else if (state.selectedGraphNodeId && !state.currentDomainGraph.nodes.some((node) => node.id === state.selectedGraphNodeId)) {
      state.selectedGraphNodeId = null;
    }

    if (state.selectedGraphLinkKey && !state.currentDomainGraph.links.some((link) => graphLinkKey(link) === state.selectedGraphLinkKey)) {
      state.selectedGraphLinkKey = null;
    }
  }

  renderDomainGraphSummary();
  state.currentDomainGraphLayout = null;
  state.hoveredGraphClusterId = null;
  state.hoveredGraphNodeId = null;
  state.hoveredGraphLinkKey = null;
  if (previousSceneMode !== state.graphSceneMode) {
    if (state.graphSceneMode === "cluster" && state.selectedGraphClusterId) {
      startGraphSceneTransition("enter-region", { clusterId: state.selectedGraphClusterId }, 980);
      boostGraphCamera(0.26);
    } else if (previousSceneMode === "cluster") {
      startGraphSceneTransition("return-atlas", { clusterId: previousClusterId }, 860);
      boostGraphCamera(0.24);
    }
  } else if (state.graphSceneMode === "cluster" && previousClusterId !== state.selectedGraphClusterId && state.selectedGraphClusterId) {
    startGraphSceneTransition("shift-region", { clusterId: state.selectedGraphClusterId }, 760);
    boostGraphCamera(0.24);
  }
  renderDomainGraphCanvas({ autoFit: true });
  if (state.selectedGraphNodeId || state.selectedGraphLinkKey) {
    centerGraphOnSelection();
  }
  renderDomainGraphDetail();
}

function handleDomainGraphCanvasWheel(event) {
  if (!state.currentDomainGraphLayout) {
    return;
  }
  event.preventDefault();
  const preparedCanvas = prepareDomainGraphCanvas();
  if (!preparedCanvas) {
    return;
  }
  const { canvas, width, height } = preparedCanvas;
  const rect = canvas.getBoundingClientRect();
  const screenX = event.clientX - rect.left;
  const screenY = event.clientY - rect.top;
  const factor = event.deltaY < 0 ? 1.18 : 0.86;
  zoomGraphAtPoint(factor, screenX, screenY, width, height);
  boostGraphCamera(0.28);
  updateGraphMotionFrame();
  renderDomainGraphCanvas();
}

function handleDomainGraphCanvasDown(event) {
  if (event.button !== 0 || !state.currentDomainGraphLayout) {
    return;
  }
  const preparedCanvas = prepareDomainGraphCanvas();
  if (!preparedCanvas) {
    return;
  }
  event.preventDefault();
  const { canvas, width, height } = preparedCanvas;
  const rect = canvas.getBoundingClientRect();
  const pointX = event.clientX - rect.left;
  const pointY = event.clientY - rect.top;
  const hit = findDomainGraphHit(pointX, pointY);
  state.graphPointer.isPanning = false;
  state.graphPointer.didPan = false;
  state.graphPointer.startX = event.clientX;
  state.graphPointer.startY = event.clientY;
  state.graphPointer.originOffsetX = state.graphViewport.offsetX;
  state.graphPointer.originOffsetY = state.graphViewport.offsetY;
  state.graphPointer.lastX = event.clientX;
  state.graphPointer.lastY = event.clientY;
  state.graphPointer.lastTime = event.timeStamp || performance.now();
  state.graphPointer.velocityX = 0;
  state.graphPointer.velocityY = 0;
  setGraphParallaxTargets(0, 0);
  if (startDraggingGraphItem(hit, pointX, pointY, width, height)) {
    canvas.style.cursor = "grabbing";
    renderDomainGraphCanvas();
    renderDomainGraphDetail();
    return;
  }
  clearGraphDragState();
  state.graphPointer.isPanning = true;
  canvas.style.cursor = "grabbing";
  addGraphRipple(pointX, pointY, {
    color: graphRippleColorForHit(hit),
    size: hit ? 74 : 52,
    duration: 520,
    width: 2.2,
  });
}

function handleDomainGraphCanvasUp() {
  const preparedCanvas = prepareDomainGraphCanvas();
  if (state.graphPointer.dragType) {
    state.graphPointer.isPanning = false;
    clearGraphDragState();
    if (preparedCanvas) {
      preparedCanvas.canvas.style.cursor = "grab";
    }
    renderDomainGraphCanvas();
    renderDomainGraphDetail();
    return;
  }
  if (state.graphPointer.isPanning) {
    const inertiaOffsetX = state.graphViewport.offsetX + state.graphPointer.velocityX * 280;
    const inertiaOffsetY = state.graphViewport.offsetY + state.graphPointer.velocityY * 280;
    setGraphViewport(state.graphViewport.targetScale ?? state.graphViewport.scale ?? 1, inertiaOffsetX, inertiaOffsetY);
  }
  state.graphPointer.isPanning = false;
  if (preparedCanvas) {
    preparedCanvas.canvas.style.cursor = "grab";
  }
}

function handleDomainGraphCanvasMove(event) {
  const preparedCanvas = prepareDomainGraphCanvas();
  if (!preparedCanvas) {
    return;
  }
  const { canvas, width, height } = preparedCanvas;
  if (!state.currentDomainGraphLayout) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const isWithinCanvas = event.clientX >= rect.left
    && event.clientX <= rect.right
    && event.clientY >= rect.top
    && event.clientY <= rect.bottom;
  const pointX = event.clientX - rect.left;
  const pointY = event.clientY - rect.top;
  if (!isWithinCanvas && !state.graphPointer.isPanning && !state.graphPointer.dragType) {
    return;
  }
  if (state.graphPointer.dragType) {
    const deltaX = event.clientX - state.graphPointer.startX;
    const deltaY = event.clientY - state.graphPointer.startY;
    if (Math.abs(deltaX) > 2 || Math.abs(deltaY) > 2) {
      state.graphPointer.didPan = true;
    }
    updateDraggedGraphItem(pointX, pointY, width, height);
    canvas.style.cursor = "grabbing";
    renderDomainGraphCanvas();
    renderDomainGraphDetail();
    return;
  }
  if (state.graphPointer.isPanning) {
    const deltaX = event.clientX - state.graphPointer.startX;
    const deltaY = event.clientY - state.graphPointer.startY;
    const time = event.timeStamp || performance.now();
    const deltaTime = Math.max(16, time - state.graphPointer.lastTime);
    state.graphPointer.velocityX = (event.clientX - state.graphPointer.lastX) / deltaTime;
    state.graphPointer.velocityY = (event.clientY - state.graphPointer.lastY) / deltaTime;
    state.graphPointer.lastX = event.clientX;
    state.graphPointer.lastY = event.clientY;
    state.graphPointer.lastTime = time;
    if (Math.abs(deltaX) > 2 || Math.abs(deltaY) > 2) {
      state.graphPointer.didPan = true;
    }
    setGraphViewport(
      state.graphViewport.scale,
      state.graphPointer.originOffsetX + deltaX,
      state.graphPointer.originOffsetY + deltaY,
      true
    );
    canvas.style.cursor = "grabbing";
    renderDomainGraphCanvas();
    return;
  }
  if (!isWithinCanvas) {
    const hadHover = state.hoveredGraphClusterId || state.hoveredGraphNodeId || state.hoveredGraphLinkKey;
    state.hoveredGraphClusterId = null;
    state.hoveredGraphNodeId = null;
    state.hoveredGraphLinkKey = null;
    setGraphParallaxTargets(0, 0);
    canvas.style.cursor = "grab";
    if (hadHover) {
      renderDomainGraphCanvas();
      renderDomainGraphDetail();
    }
    return;
  }
  const normalizedX = ((pointX / Math.max(rect.width, 1)) - 0.5) * 2;
  const normalizedY = ((pointY / Math.max(rect.height, 1)) - 0.5) * 2;
  setGraphParallaxTargets(normalizedX * 28, normalizedY * 20);
  const hit = findDomainGraphHit(pointX, pointY);
  canvas.style.cursor = isGraphDragHit(hit) ? "grab" : hit ? "pointer" : "grab";
  const nextHoveredClusterId = hit && hit.type === "cluster" ? hit.item.id : null;
  const nextHoveredNodeId = hit && hit.type === "node" ? hit.item.id : null;
  const nextHoveredLinkKey = hit && hit.type === "link" ? graphLinkKey(hit.item) : null;
  if (
    state.hoveredGraphClusterId === nextHoveredClusterId
    && state.hoveredGraphNodeId === nextHoveredNodeId
    && state.hoveredGraphLinkKey === nextHoveredLinkKey
  ) {
    return;
  }
  state.hoveredGraphClusterId = nextHoveredClusterId;
  state.hoveredGraphNodeId = nextHoveredNodeId;
  state.hoveredGraphLinkKey = nextHoveredLinkKey;
  renderDomainGraphCanvas();
  renderDomainGraphDetail();
}

function handleDomainGraphCanvasClick(event) {
  if (state.graphPointer.didPan) {
    state.graphPointer.didPan = false;
    return;
  }
  const preparedCanvas = prepareDomainGraphCanvas();
  if (!preparedCanvas) {
    return;
  }
  const { canvas } = preparedCanvas;
  const rect = canvas.getBoundingClientRect();
  const pointX = event.clientX - rect.left;
  const pointY = event.clientY - rect.top;
  const hit = findDomainGraphHit(pointX, pointY);
  addGraphRipple(pointX, pointY, {
    color: graphRippleColorForHit(hit),
    size: hit?.type === "cluster" ? 136 : hit ? 92 : 68,
    duration: 720,
    width: hit ? 2.8 : 2.2,
  });
  if (state.graphSceneMode === "overview") {
    if (hit && hit.type === "cluster") {
      state.selectedGraphClusterId = hit.item.id;
      state.selectedGraphNodeId = null;
      state.selectedGraphLinkKey = null;
      updateDomainGraph();
      return;
    }
    renderDomainGraphDetail();
    return;
  }

  if (!hit) {
    state.selectedGraphNodeId = null;
    state.selectedGraphLinkKey = null;
    if (state.currentDomainGraphLayout) {
      const preparedCanvas = prepareDomainGraphCanvas();
      if (preparedCanvas) {
        zoomGraphToBounds(getCurrentGraphBounds(state.currentDomainGraphLayout), preparedCanvas.width, preparedCanvas.height, 72);
      }
    }
    renderDomainGraphCanvas();
    renderDomainGraphDetail();
    return;
  }
  if (hit.type === "node") {
    state.selectedGraphNodeId = state.selectedGraphNodeId === hit.item.id ? null : hit.item.id;
    state.selectedGraphLinkKey = null;
  } else {
    const key = graphLinkKey(hit.item);
    state.selectedGraphLinkKey = state.selectedGraphLinkKey === key ? null : key;
    state.selectedGraphNodeId = null;
  }
  if (state.selectedGraphNodeId || state.selectedGraphLinkKey) {
    startGraphSceneTransition(
      state.selectedGraphNodeId ? "focus-node" : "focus-link",
      state.selectedGraphNodeId ? { nodeId: state.selectedGraphNodeId } : { clusterId: state.selectedGraphClusterId },
      620
    );
    boostGraphCamera(0.24);
    centerGraphOnSelection(hit.type === "node" ? 0.12 : 0.06);
  }
  renderDomainGraphCanvas();
  renderDomainGraphDetail();
}

function handleDomainGraphCanvasDoubleClick(event) {
  const preparedCanvas = prepareDomainGraphCanvas();
  if (!preparedCanvas) {
    return;
  }
  const { canvas } = preparedCanvas;
  const rect = canvas.getBoundingClientRect();
  const pointX = event.clientX - rect.left;
  const pointY = event.clientY - rect.top;
  const hit = findDomainGraphHit(pointX, pointY);
  addGraphRipple(pointX, pointY, {
    color: graphRippleColorForHit(hit),
    size: hit?.type === "cluster" ? 168 : hit ? 124 : 92,
    duration: 860,
    width: 3,
  });

  if (state.graphSceneMode === "overview") {
    if (hit && hit.type === "cluster") {
      state.selectedGraphClusterId = hit.item.id;
      state.selectedGraphNodeId = null;
      state.selectedGraphLinkKey = null;
      updateDomainGraph();
    }
    return;
  }

  if (!hit) {
    if (state.currentDomainGraphLayout) {
      zoomGraphToBounds(getCurrentGraphBounds(state.currentDomainGraphLayout), preparedCanvas.width, preparedCanvas.height, 60);
    }
    return;
  }

  if (hit.type === "node") {
    state.selectedGraphNodeId = hit.item.id;
    state.selectedGraphLinkKey = null;
    startGraphSceneTransition("focus-node", { nodeId: state.selectedGraphNodeId }, 720);
    boostGraphCamera(0.28);
    centerGraphOnSelection(0.28);
  } else if (hit.type === "link") {
    state.selectedGraphLinkKey = graphLinkKey(hit.item);
    state.selectedGraphNodeId = null;
    startGraphSceneTransition("focus-link", { clusterId: state.selectedGraphClusterId }, 680);
    boostGraphCamera(0.26);
    centerGraphOnSelection(0.18);
  }
  renderDomainGraphCanvas();
  renderDomainGraphDetail();
}

function focusGraphCluster(clusterId) {
  if (!clusterId) {
    return;
  }
  if (!canRenderGraphHere() || !state.domainGraph) {
    navigateToMapWithFocus({ cluster: clusterId });
    return;
  }
  const searchInput = document.getElementById("graph-search");
  if (searchInput) {
    searchInput.value = "";
  }
  state.selectedGraphClusterId = clusterId;
  state.selectedGraphNodeId = null;
  state.selectedGraphLinkKey = null;
  startGraphSceneTransition("enter-region", { clusterId }, 980);
  boostGraphCamera(0.26);
  updateDomainGraph();
  document.querySelector(".graph-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function focusGraphDomain(domainId) {
  if (!domainId) {
    return;
  }
  if (!canRenderGraphHere() || !state.domainGraph) {
    navigateToMapWithFocus({ focus: domainId });
    return;
  }
  const searchInput = document.getElementById("graph-search");
  if (!searchInput) {
    return;
  }
  searchInput.value = domainId;
  state.selectedGraphClusterId = null;
  state.selectedGraphNodeId = domainId;
  state.selectedGraphLinkKey = null;
  startGraphSceneTransition("focus-node", { nodeId: domainId }, 720);
  boostGraphCamera(0.24);
  updateDomainGraph();
  document.querySelector(".graph-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function initializeGlobalInteractions() {
  document.getElementById("theme-toggle")?.addEventListener("click", () => {
    state.isDarkMode = !state.isDarkMode;
    applyTheme();
  });

  document.body.addEventListener("click", (event) => {
    const frontierModeButton = event.target.closest(".frontier-mode-button");
    if (frontierModeButton) {
      setFrontierMoveMode(frontierModeButton.dataset.frontierMode || "regions");
      return;
    }

    const clusterButton = event.target.closest("[data-focus-cluster]");
    if (clusterButton) {
      focusGraphCluster(clusterButton.dataset.focusCluster);
      return;
    }

    const focusButton = event.target.closest("[data-focus-domain]");
    if (focusButton) {
      focusGraphDomain(focusButton.dataset.focusDomain);
    }
  });
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

  if (state.selectedEvidenceId != null) {
    await loadEvidenceDetail(state.selectedEvidenceId);
  }
  if (state.selectedOutcomePredictionId != null) {
    await loadOutcomeReviewDetail(state.selectedOutcomePredictionId);
  }
  if (state.selectedStrongRejectionId != null) {
    await loadStrongRejectionDetail(state.selectedStrongRejectionId);
  }
}

function applyMapFocusFromLocation() {
  if (!canRenderGraphHere() || !state.domainGraph) {
    return;
  }
  const params = new URLSearchParams(window.location.search);
  const focusDomain = (params.get("focus") || "").trim();
  const clusterId = (params.get("cluster") || "").trim();
  if (focusDomain) {
    focusGraphDomain(focusDomain);
    return;
  }
  if (clusterId) {
    focusGraphCluster(clusterId);
  }
}

function renderError(error) {
  const message = error instanceof Error ? error.message : String(error);
  document.body.insertAdjacentHTML("beforeend", `<section class="panel"><div class="error">${escapeHtml(message)}</div></section>`);
}

async function loadHomePage() {
  const [costs, operatorHome, domainGraph, frontier] = await Promise.all([
    fetchJson("/api/costs"),
    fetchJson("/api/operator-home"),
    fetchJson("/api/domain-graph"),
    fetchJson("/api/frontier"),
  ]);
  renderCosts(costs);
  renderOperatorHome(operatorHome);
  state.domainGraph = domainGraph;
  renderFrontier(frontier);
  renderDomainGraph(domainGraph);
}

async function loadMapPage() {
  const [costs, operatorHome, domainGraph, frontier] = await Promise.all([
    fetchJson("/api/costs"),
    fetchJson("/api/operator-home"),
    fetchJson("/api/domain-graph"),
    fetchJson("/api/frontier"),
  ]);
  renderCosts(costs);
  renderOperatorHome(operatorHome);
  state.domainGraph = domainGraph;
  renderFrontier(frontier);
  renderDomainGraph(domainGraph);
  applyMapFocusFromLocation();
}

async function loadFrontierPage() {
  const [stats, topKilled, domainGraph, frontier] = await Promise.all([
    fetchJson("/api/stats"),
    fetchJson("/api/top-killed"),
    fetchJson("/api/domain-graph"),
    fetchJson("/api/frontier"),
  ]);
  renderStats(stats);
  renderTopKilled(topKilled);
  state.domainGraph = domainGraph;
  renderFrontier(frontier);
}

async function loadReviewPage() {
  await loadReviewData();
}

async function loadArchivePage() {
  const [costs, timeline, transmissions] = await Promise.all([
    fetchJson("/api/costs"),
    fetchJson("/api/transmission-timeline"),
    fetchJson("/api/transmissions"),
  ]);
  renderCosts(costs);
  renderTransmissionTimeline(timeline);
  renderTransmissions(transmissions);
}

async function loadCurrentPage() {
  const pageKey = currentPageKey();
  if (pageKey === "map") {
    await loadMapPage();
    return;
  }
  if (pageKey === "frontier") {
    await loadFrontierPage();
    return;
  }
  if (pageKey === "review") {
    await loadReviewPage();
    return;
  }
  if (pageKey === "archive") {
    await loadArchivePage();
    return;
  }
  await loadHomePage();
}

document.addEventListener("DOMContentLoaded", () => {
  initializeGlobalInteractions();
  applyTheme();
  loadCurrentPage().catch(renderError);
});
