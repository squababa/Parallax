import sqlite3

import pytest

import dashboard


def _create_dashboard_test_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE explorations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            seed_domain TEXT,
            jump_target_domain TEXT,
            total_score REAL,
            transmitted INTEGER NOT NULL DEFAULT 0,
            connection_description TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE strong_rejections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            seed_domain TEXT,
            target_domain TEXT,
            total_score REAL,
            salvage_reason TEXT,
            status TEXT
        )"""
    )
    conn.executemany(
        """INSERT INTO explorations (
            timestamp,
            seed_domain,
            jump_target_domain,
            total_score,
            transmitted,
            connection_description
        ) VALUES (?, ?, ?, ?, ?, ?)""",
        [
            ("2026-04-01T00:00:00+00:00", "Architecture", "Power grids", 0.91, 1, "Shared cascading behavior"),
            ("2026-04-02T00:00:00+00:00", "Architecture", "Wireless networks", 0.82, 0, "Load balancing"),
            ("2026-04-03T00:00:00+00:00", "Architecture", "Power grids", 0.74, 0, "Repeated bridge"),
            ("2026-04-04T00:00:00+00:00", "Locksmithing", "Working memory gating", 0.95, 1, "Selective gating"),
            ("2026-04-05T00:00:00+00:00", "Pottery", "Kiln control", None, 0, "Null-score test"),
        ],
    )
    conn.executemany(
        """INSERT INTO strong_rejections (
            timestamp,
            seed_domain,
            target_domain,
            total_score,
            salvage_reason,
            status
        ) VALUES (?, ?, ?, ?, ?, ?)""",
        [
            ("2026-04-05T12:00:00+00:00", "Juggling", "Transit scheduling", 0.94, "Packaging failed", "open"),
            ("2026-04-06T12:00:00+00:00", "Glassblowing", "Residual stress", 0.89, "Weak grounding", "open"),
            ("2026-04-07T12:00:00+00:00", "Storytelling", "Vocal physiology", 0.87, "Dismissed", "dismissed"),
        ],
    )
    conn.commit()
    conn.close()


def _create_empty_dashboard_test_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE explorations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            seed_domain TEXT,
            jump_target_domain TEXT,
            total_score REAL,
            transmitted INTEGER NOT NULL DEFAULT 0,
            connection_description TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE strong_rejections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            seed_domain TEXT,
            target_domain TEXT,
            total_score REAL,
            salvage_reason TEXT,
            status TEXT
        )"""
    )
    conn.commit()
    conn.close()


def test_get_domain_graph_aggregates_nodes_and_links(tmp_path, monkeypatch):
    db_path = tmp_path / "dashboard-test.db"
    _create_dashboard_test_db(db_path)
    monkeypatch.setattr(dashboard, "DB_PATH", str(db_path))

    payload = dashboard._get_domain_graph()

    assert payload["stats"]["unique_domains"] == 7
    assert payload["stats"]["unique_connections"] == 4
    assert payload["stats"]["transmitted_connections"] == 2
    assert payload["stats"]["successful_domains"] == 4
    assert payload["stats"]["total_explorations"] == 5
    assert payload["stats"]["total_transmissions"] == 2
    assert payload["stats"]["cluster_count"] == 3
    assert payload["stats"]["largest_cluster_size"] == 3

    architecture = next(node for node in payload["nodes"] if node["id"] == "Architecture")
    assert architecture["role"] == "seed"
    assert architecture["appearance_count"] == 3
    assert architecture["connection_count"] == 2
    assert architecture["transmitted_count"] == 1
    assert architecture["avg_score"] == pytest.approx(0.8233333333333334)
    assert architecture["cluster_id"] == "cluster-82b70d0d7045"

    power_grids = next(node for node in payload["nodes"] if node["id"] == "Power grids")
    assert power_grids["role"] == "target"
    assert power_grids["appearance_count"] == 2
    assert power_grids["connection_count"] == 1
    assert power_grids["cluster_id"] == "cluster-82b70d0d7045"

    repeated_bridge = next(
        link
        for link in payload["links"]
        if link["source"] == "Architecture" and link["target"] == "Power grids"
    )
    assert repeated_bridge["count"] == 2
    assert repeated_bridge["transmitted_count"] == 1
    assert repeated_bridge["max_score"] == 0.91
    assert repeated_bridge["avg_score"] == 0.825

    assert payload["clusters"] == [
        {
            "id": "cluster-82b70d0d7045",
            "label": "Architecture / Power grids",
            "node_count": 3,
            "link_count": 2,
            "transmitted_link_count": 1,
            "avg_score": pytest.approx(0.8233333333333334),
            "latest_timestamp": "2026-04-03T00:00:00+00:00",
            "top_domains": ["Architecture", "Power grids", "Wireless networks"],
            "seed_count": 1,
            "target_count": 2,
            "bridge_count": 0,
            "exploration_count": 3,
            "transmitted_exploration_count": 1,
            "open_salvage_count": 0,
        },
        {
            "id": "cluster-da7a7010c536",
            "label": "Locksmithing / Working memory gating",
            "node_count": 2,
            "link_count": 1,
            "transmitted_link_count": 1,
            "avg_score": 0.95,
            "latest_timestamp": "2026-04-04T00:00:00+00:00",
            "top_domains": ["Locksmithing", "Working memory gating"],
            "seed_count": 1,
            "target_count": 1,
            "bridge_count": 0,
            "exploration_count": 1,
            "transmitted_exploration_count": 1,
            "open_salvage_count": 0,
        },
        {
            "id": "cluster-37fa6da1138b",
            "label": "Kiln control / Pottery",
            "node_count": 2,
            "link_count": 1,
            "transmitted_link_count": 0,
            "avg_score": None,
            "latest_timestamp": "2026-04-05T00:00:00+00:00",
            "top_domains": ["Kiln control", "Pottery"],
            "seed_count": 1,
            "target_count": 1,
            "bridge_count": 0,
            "exploration_count": 1,
            "transmitted_exploration_count": 0,
            "open_salvage_count": 0,
        },
    ]


def test_get_frontier_snapshot_sorts_and_counts_signal(tmp_path, monkeypatch):
    db_path = tmp_path / "dashboard-test.db"
    _create_dashboard_test_db(db_path)
    monkeypatch.setattr(dashboard, "DB_PATH", str(db_path))

    payload = dashboard._get_frontier_snapshot()

    assert payload["stats"] == {
        "high_signal_seeds": 2,
        "high_signal_bridges": 2,
        "open_salvage_candidates": 2,
    }

    assert payload["launchpads"][0]["domain"] == "Locksmithing"
    assert payload["launchpads"][0]["exploration_count"] == 1
    assert payload["launchpads"][0]["transmitted_count"] == 1
    assert payload["launchpads"][0]["avg_score"] == 0.95
    assert payload["launchpads"][0]["cluster_id"] == "cluster-da7a7010c536"
    assert payload["launchpads"][0]["frontier_score"] == 63.5
    assert payload["launchpads"][1]["domain"] == "Architecture"
    assert payload["launchpads"][1]["avg_score"] == pytest.approx(0.8233333333333334)
    assert payload["launchpads"][1]["cluster_id"] == "cluster-82b70d0d7045"
    assert payload["launchpads"][1]["frontier_score"] == pytest.approx(48.03, abs=0.05)
    assert payload["launchpads"][-1]["domain"] == "Pottery"
    assert payload["launchpads"][-1]["avg_score"] is None

    assert payload["strong_bridges"][0]["source"] == "Locksmithing"
    assert payload["strong_bridges"][0]["target"] == "Working memory gating"
    assert payload["strong_bridges"][0]["transmitted_count"] == 1

    assert payload["salvage_candidates"][0]["source"] == "Juggling"
    assert payload["salvage_candidates"][0]["reason"] == "Packaging failed"

    assert [row["cluster_id"] for row in payload["next_regions"]] == [
        "cluster-da7a7010c536",
        "cluster-82b70d0d7045",
        "cluster-37fa6da1138b",
    ]
    assert payload["next_regions"][0]["label"] == "Locksmithing / Working memory gating"
    assert payload["next_regions"][0]["frontier_score"] == pytest.approx(58.5, abs=0.05)
    assert payload["next_regions"][0]["why"] == ["1 transmitting bridge", "avg score 0.950", "fresh activity"]
    assert payload["next_regions"][1]["label"] == "Architecture / Power grids"
    assert payload["next_regions"][1]["frontier_score"] == pytest.approx(53.03, abs=0.05)
    assert payload["next_regions"][1]["avg_score"] == pytest.approx(0.8233333333333334)
    assert payload["next_regions"][1]["why"] == ["1 transmitting bridge", "1 near miss", "avg score 0.823"]
    assert payload["next_regions"][2]["label"] == "Kiln control / Pottery"
    assert payload["next_regions"][2]["frontier_score"] == pytest.approx(10.0, abs=0.05)
    assert payload["next_regions"][2]["why"] == ["fresh activity"]

    assert [row["domain"] for row in payload["next_seeds"]] == [
        "Locksmithing",
        "Architecture",
        "Pottery",
    ]
    assert payload["next_seeds"][0]["cluster_id"] == "cluster-da7a7010c536"
    assert payload["next_seeds"][0]["frontier_score"] == pytest.approx(63.5, abs=0.05)
    assert payload["next_seeds"][0]["why"] == ["1 transmission", "1 transmitting neighbor", "avg score 0.950"]
    assert payload["next_seeds"][1]["cluster_id"] == "cluster-82b70d0d7045"
    assert payload["next_seeds"][1]["frontier_score"] == pytest.approx(48.03, abs=0.05)
    assert payload["next_seeds"][1]["avg_score"] == pytest.approx(0.8233333333333334)
    assert payload["next_seeds"][1]["why"] == ["1 transmission", "1 near miss", "1 transmitting neighbor"]
    assert payload["next_seeds"][2]["cluster_id"] == "cluster-37fa6da1138b"
    assert payload["next_seeds"][2]["frontier_score"] == pytest.approx(5.0, abs=0.05)
    assert payload["next_seeds"][2]["why"] == ["fresh activity"]


def test_dashboard_routes_use_multi_page_templates(tmp_path, monkeypatch):
    db_path = tmp_path / "dashboard-test.db"
    _create_dashboard_test_db(db_path)
    monkeypatch.setattr(dashboard, "DB_PATH", str(db_path))

    client = dashboard.app.test_client()
    home = client.get("/")
    map_page = client.get("/map")
    frontier = client.get("/frontier")
    review = client.get("/review")
    archive = client.get("/archive")

    home_text = home.get_data(as_text=True)
    map_text = map_page.get_data(as_text=True)
    frontier_text = frontier.get_data(as_text=True)
    review_text = review.get_data(as_text=True)
    archive_text = archive.get_data(as_text=True)

    assert home.status_code == 200
    assert map_page.status_code == 200
    assert frontier.status_code == 200
    assert review.status_code == 200
    assert archive.status_code == 200

    assert 'data-page="home"' in home_text
    assert 'data-page="map"' in map_text
    assert 'data-page="frontier"' in frontier_text
    assert 'data-page="review"' in review_text
    assert 'data-page="archive"' in archive_text

    assert "Open Map" in home_text
    assert "Review Pressure" in home_text
    assert "Frontier Preview" in home_text

    assert "Constellations And Internal Webs" in map_text
    assert "Zoom To Fit" in map_text

    assert "Frontier Navigator" in frontier_text
    assert "Failure Signals" in frontier_text
    assert "Regions" in frontier_text
    assert "Seeds" in frontier_text

    assert "Handle evidence, outcomes, and salvage work." in review_text
    assert "Strong Rejections Review" in review_text

    assert "Operational Spend" in archive_text
    assert "Transmission Archive" in archive_text

    for text in (home_text, map_text, frontier_text, review_text, archive_text):
        assert "/static/dashboard.css" in text
        assert "/static/dashboard.js" in text
        assert "BlackClaw" in text


def test_observatory_helpers_handle_empty_state(tmp_path, monkeypatch):
    db_path = tmp_path / "dashboard-empty.db"
    _create_empty_dashboard_test_db(db_path)
    monkeypatch.setattr(dashboard, "DB_PATH", str(db_path))

    graph_payload = dashboard._get_domain_graph()
    frontier_payload = dashboard._get_frontier_snapshot()

    assert graph_payload["stats"] == {
        "unique_domains": 0,
        "unique_connections": 0,
        "transmitted_connections": 0,
        "successful_domains": 0,
        "total_explorations": 0,
        "total_transmissions": 0,
        "avg_score": None,
        "cluster_count": 0,
        "largest_cluster_size": 0,
    }
    assert graph_payload["nodes"] == []
    assert graph_payload["links"] == []
    assert graph_payload["clusters"] == []

    assert frontier_payload["stats"] == {
        "high_signal_seeds": 0,
        "high_signal_bridges": 0,
        "open_salvage_candidates": 0,
    }
    assert frontier_payload["launchpads"] == []
    assert frontier_payload["strong_bridges"] == []
    assert frontier_payload["salvage_candidates"] == []
    assert frontier_payload["next_regions"] == []
    assert frontier_payload["next_seeds"] == []
