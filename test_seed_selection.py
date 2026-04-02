import pytest

import config
import seed
import store


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "seed_selection_test.db"
    monkeypatch.setattr(store, "DB_PATH", str(db_path))
    store.init_db()
    return db_path


def test_shrunk_expected_value_preserves_explicit_zero() -> None:
    baseline = {"raw_expected_value": 0.33245132575757574}

    result = seed._shrunk_expected_value(
        {"attempts": 4, "raw_expected_value": 0.0},
        baseline,
        prior_strength=6.0,
    )

    expected = baseline["raw_expected_value"] * (6.0 / 10.0)
    assert result == pytest.approx(expected)
    assert result < baseline["raw_expected_value"]


def test_expected_value_multiplier_downweights_confirmed_zero_ev_seed() -> None:
    metrics = {
        "global_metrics": {"raw_expected_value": 0.33245132575757574},
        "domain_metrics": {
            "Tattooing": {
                "attempts": 4,
                "raw_expected_value": 0.0,
                "transmission_rate": 0.0,
                "late_stage_survival_rate": 0.0,
                "strong_rejection_rate": 0.0,
                "weak_grounding_rate": 0.0,
            }
        },
        "category_metrics": {
            "Body Art": {
                "attempts": 4,
                "raw_expected_value": 0.0,
            }
        },
    }

    multiplier, reason = seed._expected_value_multiplier(
        "Tattooing",
        "Body Art",
        metrics,
    )

    assert multiplier == pytest.approx(0.8905760207792207)
    assert multiplier < 1.0
    assert "low EV" in reason


def test_expected_value_multiplier_reports_category_tailwind() -> None:
    metrics = {
        "global_metrics": {"raw_expected_value": 0.33245132575757574},
        "domain_metrics": {
            "Blacksmithing": {
                "attempts": 1,
                "raw_expected_value": 0.0,
                "transmission_rate": 0.0,
                "late_stage_survival_rate": 0.0,
                "strong_rejection_rate": 0.0,
                "weak_grounding_rate": 0.0,
            }
        },
        "category_metrics": {
            "Craft": {
                "attempts": 7,
                "raw_expected_value": 1.091654761904762,
            }
        },
    }

    multiplier, reason = seed._expected_value_multiplier(
        "Blacksmithing",
        "Craft",
        metrics,
    )

    assert multiplier > 1.0
    assert "category tailwind" in reason
    assert "Craft" in reason


def test_describe_seed_quality_prefers_operator_rich_domain() -> None:
    strong = seed.describe_seed_quality(
        {
            "name": "Distributed Systems",
            "category": "Technology",
            "seed_queries": [
                "distributed systems load balancing failover",
                "queue routing latency control",
            ],
        }
    )
    weak = seed.describe_seed_quality(
        {
            "name": "Storytelling",
            "category": "Communication",
            "seed_queries": [
                "narrative structure universal patterns",
                "hero journey monomyth story",
            ],
        }
    )

    assert strong["band"] == "high"
    assert weak["band"] == "weak"
    assert strong["score"] > weak["score"]
    assert any("concrete mechanisms" in item for item in strong["strengths"])
    assert any("aesthetic or interpretive framing" in item for item in weak["concerns"])


def test_quality_multiplier_prefers_high_quality_and_downranks_weak_quality() -> None:
    high_multiplier, high_reason = seed._quality_multiplier(
        {
            "score": 0.82,
            "band": "high",
            "strengths": ["concrete mechanisms via routing"],
            "concerns": [],
        }
    )
    weak_multiplier, weak_reason = seed._quality_multiplier(
        {
            "score": 0.24,
            "band": "weak",
            "strengths": [],
            "concerns": ["aesthetic or interpretive framing via narrative"],
        }
    )

    assert high_multiplier > 1.0
    assert weak_multiplier < 1.0
    assert high_multiplier > weak_multiplier
    assert "seed quality high" in high_reason
    assert "seed quality weak" in weak_reason


def test_quality_matching_does_not_penalize_history_for_story_substring() -> None:
    profile = seed.describe_seed_quality(
        {
            "name": "Ancient Navigation",
            "category": "History",
            "seed_queries": [
                "Polynesian wayfinding navigation",
                "ancient celestial navigation techniques",
            ],
        }
    )

    assert not any(
        "aesthetic or interpretive framing" in concern
        for concern in profile["concerns"]
    )


def test_domain_to_seed_problem_frames_runtime_queries() -> None:
    runtime_seed = seed._domain_to_seed(
        {
            "name": "Distributed Systems",
            "category": "Technology",
            "seed_queries": [
                "  consensus algorithms distributed systems  ",
                "Byzantine   fault tolerance",
                "queue routing latency control",
                "load balancing failover schedule",
                "redundancy isolation control",
                "backpressure admission scheduling",
            ],
        }
    )

    assert set(runtime_seed) == {"name", "category", "seed_queries"}
    assert runtime_seed["seed_queries"] == [
        "failure modes in Distributed Systems consensus algorithms distributed systems",
        "control strategies in Distributed Systems Byzantine fault tolerance",
        "mechanisms in Distributed Systems queue routing latency control",
        "operating constraints in Distributed Systems load balancing failover schedule",
        "workflow interventions in Distributed Systems redundancy isolation control",
        "process bottlenecks in Distributed Systems backpressure admission scheduling",
    ]


def test_problem_frame_seed_queries_cycles_query_families_deterministically() -> None:
    queries = seed._problem_frame_seed_queries(
        "Control Systems",
        [
            "threshold switching",
            "feedback saturation",
            "queue routing",
            "voltage instability",
            "redundancy isolation",
            "timing drift",
        ],
    )

    assert queries == [
        "failure modes in Control Systems threshold switching",
        "control strategies in Control Systems feedback saturation",
        "mechanisms in Control Systems queue routing",
        "operating constraints in Control Systems voltage instability",
        "workflow interventions in Control Systems redundancy isolation",
        "process bottlenecks in Control Systems timing drift",
    ]


def test_problem_frame_seed_queries_remains_bounded_and_skips_empty_inputs() -> None:
    queries = seed._problem_frame_seed_queries(
        "Distributed Systems",
        [
            "queue routing latency control",
            "   ",
            "",
            "load balancing failover schedule",
        ],
    )

    assert len(queries) == 2
    assert all(query.strip() for query in queries)
    assert queries[0].startswith("failure modes in Distributed Systems ")
    assert queries[1].startswith("control strategies in Distributed Systems ")


def test_pick_seed_returns_quality_diagnostics(monkeypatch) -> None:
    strong_domains = [
        {
            "name": f"Strong Seed {index}",
            "category": "Technology",
            "seed_queries": [
                f"queue routing latency control {index}",
                f"load balancing failover schedule {index}",
            ],
        }
        for index in range(12)
    ]
    monkeypatch.setattr(
        seed,
        "_load_domains",
        lambda: [
            *strong_domains,
            {
                "name": "Storytelling",
                "category": "Communication",
                "seed_queries": [
                    "narrative structure universal patterns",
                    "hero journey monomyth story",
                ],
            },
        ],
    )
    monkeypatch.setattr(config, "PERSONALIZATION", False, raising=False)
    monkeypatch.setattr(config, "SEED_EXCLUSION_WINDOW", 0, raising=False)
    monkeypatch.setattr(store, "get_recent_domains", lambda _n=0: [])
    monkeypatch.setattr(store, "get_recent_seed_selection_context", lambda _n=0: {})
    monkeypatch.setattr(store, "get_seed_outcome_metrics", lambda: {})

    captured = {}

    def _choose_best(population, weights, k):
        captured["weights"] = list(weights)
        assert k == 1
        best_idx = max(range(len(population)), key=lambda idx: weights[idx])
        return [population[best_idx]]

    monkeypatch.setattr(seed.random, "choices", _choose_best)

    selected = seed.pick_seed()

    assert selected["name"] == "Strong Seed 0"
    assert selected["quality_profile"]["band"] == "high"
    assert "selection_diagnostics" in selected
    assert selected["selection_diagnostics"]["mode"] == "weighted"
    assert "quality-screened pool" in selected["selection_reason"]
    assert "retained 1 diversity-supported weak seeds" in selected["selection_reason"]
    assert "seed quality high" in selected["selection_reason"]
    assert selected["selection_diagnostics"]["weak_seed_add_back_count"] == 1
    assert selected["selection_diagnostics"]["candidate_pool_size"] == 13
    assert len(captured["weights"]) == 13


def test_pick_seed_retains_diversity_supported_weak_add_backs(monkeypatch) -> None:
    strong_domains = [
        {
            "name": f"Strong Seed {index}",
            "category": "Technology",
            "seed_queries": [
                f"queue routing latency control {index}",
                f"load balancing failover schedule {index}",
            ],
        }
        for index in range(12)
    ]
    weak_domains = [
        {
            "name": "Weak Seed A",
            "category": "Communication",
            "seed_queries": [
                "narrative structure universal patterns",
                "hero journey monomyth story",
            ],
        },
        {
            "name": "Weak Seed B",
            "category": "Art",
            "seed_queries": [
                "beauty story phenomenology",
                "narrative typography experience",
            ],
        },
        {
            "name": "Weak Seed C",
            "category": "Philosophy",
            "seed_queries": [
                "consciousness possible worlds philosophy",
                "unsolved problems universal patterns",
            ],
        },
    ]
    monkeypatch.setattr(seed, "_load_domains", lambda: [*strong_domains, *weak_domains])
    monkeypatch.setattr(config, "PERSONALIZATION", False, raising=False)
    monkeypatch.setattr(config, "SEED_EXCLUSION_WINDOW", 0, raising=False)
    monkeypatch.setattr(store, "get_recent_domains", lambda _n=0: [])
    monkeypatch.setattr(
        store,
        "get_recent_seed_selection_context",
        lambda _n=0: {
            "recent_categories": ["Technology"] * 12,
            "category_recent_counts": {"Technology": 12},
            "domain_last_seen": {},
            "category_last_seen": {"Technology": 0},
            "domain_low_yield_counts": {"Weak Seed C": 2},
        },
    )
    monkeypatch.setattr(store, "get_seed_outcome_metrics", lambda: {})

    captured = {}

    def _capture_population(population, weights, k):
        captured["names"] = [item["name"] for item in population]
        captured["weights"] = dict(zip(captured["names"], weights))
        assert k == 1
        return [population[0]]

    monkeypatch.setattr(seed.random, "choices", _capture_population)

    selected = seed.pick_seed()

    assert selected["selection_diagnostics"]["weak_seed_add_back_count"] == 2
    assert selected["selection_diagnostics"]["candidate_pool_size"] == 14
    assert "retained 2 diversity-supported weak seeds" in selected["selection_reason"]
    assert "Weak Seed A" in captured["names"]
    assert "Weak Seed B" in captured["names"]
    assert "Weak Seed C" not in captured["names"]
    assert captured["weights"]["Weak Seed A"] > 0.0
    assert captured["weights"]["Weak Seed B"] > 0.0


def test_pick_seed_random_floor_preserves_exploration_reason(monkeypatch) -> None:
    domains = [
        {
            "name": "Strong Seed",
            "category": "Technology",
            "seed_queries": [
                "queue routing latency control",
                "load balancing failover schedule",
            ],
        },
        {
            "name": "Storytelling",
            "category": "Communication",
            "seed_queries": [
                "narrative structure universal patterns",
                "hero journey monomyth story",
            ],
        },
    ]
    monkeypatch.setattr(seed, "_load_domains", lambda: domains)
    monkeypatch.setattr(config, "PERSONALIZATION", True, raising=False)
    monkeypatch.setattr(config, "SEED_EXCLUSION_WINDOW", 0, raising=False)
    monkeypatch.setattr(store, "get_recent_domains", lambda _n=0: [])
    monkeypatch.setattr(store, "get_recent_seed_selection_context", lambda _n=0: {})
    monkeypatch.setattr(
        store,
        "get_seed_outcome_metrics",
        lambda: {
            "global_metrics": {"raw_expected_value": 0.33},
            "domain_metrics": {},
            "category_metrics": {},
        },
    )
    monkeypatch.setattr(seed.random, "random", lambda: 0.0)
    monkeypatch.setattr(seed.random, "choice", lambda population: population[1])

    selected = seed.pick_seed()

    assert selected["name"] == "Storytelling"
    assert selected["selection_diagnostics"]["mode"] == "random_floor"
    assert "random exploration pick (20% diversity floor)" in selected["selection_reason"]
    assert "weak-quality seed" in selected["selection_reason"]


def test_seed_selection_metadata_round_trips_into_review_items(temp_db) -> None:
    store.save_exploration(
        seed_domain="Distributed Systems",
        seed_category="Technology",
        seed_selection={
            "mode": "weighted",
            "reason": "seed quality high (0.81): concrete mechanisms via queue, routing",
            "quality_profile": {
                "score": 0.81,
                "band": "high",
                "strengths": [
                    "concrete mechanisms via queue, routing",
                    "operator workflows via control, monitoring",
                ],
                "concerns": [],
            },
        },
        patterns_found=None,
        transmitted=False,
    )

    rows = store.list_recent_review_items(limit=1)

    assert len(rows) == 1
    assert rows[0]["seed_quality_band"] == "high"
    assert rows[0]["seed_selection"]["mode"] == "weighted"
    assert rows[0]["seed_selection"]["quality_profile"]["score"] == pytest.approx(0.81)
    assert rows[0]["seed_quality"]["strengths"][0].startswith("concrete mechanisms")
