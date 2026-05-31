# PROMPT:
# Generate pytest tests for /stores/{id}/anomalies and /stores/{id}/heatmap.
# Cover: no anomalies when data is normal, QUEUE_SPIKE fires at correct threshold,
# CONVERSION_DROP respects min-visitor guard, DEAD_ZONE detects unvisited zones,
# HIGH_ABANDON_RATE fires at 50%, heatmap empty, heatmap normalization, scores.
#
# CHANGES MADE:
# - Fixture moved to conftest.py. Seeds via client.db_session.
# - All threshold boundary tests preserved with exact counts.

import uuid
import pytest
from datetime import datetime, timezone
from app.database import EventORM


def _now():
    return datetime.now(timezone.utc)


def _row(store_id, visitor_id, event_type, zone_id=None, dwell_ms=None, is_staff=False):
    return EventORM(
        event_id   = str(uuid.uuid4()),
        store_id   = store_id,
        camera_id  = "cam_01",
        visitor_id = visitor_id,
        event_type = event_type,
        timestamp  = _now(),
        zone_id    = zone_id,
        dwell_ms   = dwell_ms,
        is_staff   = is_staff,
        confidence = 0.9,
    )


def seed(client, rows):
    db = client.db_session
    for r in rows:
        db.add(r)
    db.commit()


def anomaly_types(client, store_id) -> set:
    data = client.get(f"/stores/{store_id}/anomalies").json()
    return {a["type"] for a in data.get("anomalies", [])}


class TestAnomalies:

    def test_no_anomalies_on_empty_store(self, client):
        resp = client.get("/stores/store_empty_anomaly/anomalies")
        assert resp.status_code == 200
        assert resp.json()["anomalies"] == []

    def test_no_anomalies_when_metrics_normal(self, client):
        sid = "store_healthy"
        rows = [_row(sid, f"v{i}", "ENTRY") for i in range(20)]
        rows += [_row(sid, f"v{i}", "ZONE_ENTER", zone_id="skincare") for i in range(20)]
        rows += [_row(sid, f"v{i}", "ZONE_EXIT",  zone_id="skincare") for i in range(20)]
        rows += [_row(sid, f"v{i}", "BILLING_QUEUE_JOIN") for i in range(15)]
        rows += [_row(sid, f"v{i}", "ZONE_ENTER", zone_id="billing") for i in range(15)]
        rows += [_row(sid, f"v{i}", "ZONE_EXIT",  zone_id="billing") for i in range(15)]
        seed(client, rows)
        types = anomaly_types(client, sid)
        assert "QUEUE_SPIKE"    not in types
        assert "CONVERSION_DROP" not in types

    def test_queue_spike_fires_above_threshold(self, client):
        sid = "store_queue_spike"
        rows = [_row(sid, f"v{i}", "ZONE_ENTER", zone_id="billing") for i in range(6)]
        seed(client, rows)
        resp = client.get(f"/stores/{sid}/anomalies")
        anomalies = resp.json()["anomalies"]
        types = {a["type"] for a in anomalies}
        assert "QUEUE_SPIKE" in types
        spike = next(a for a in anomalies if a["type"] == "QUEUE_SPIKE")
        assert spike["severity"] == "CRITICAL"
        assert spike["value"] == 6

    def test_queue_spike_does_not_fire_at_exactly_threshold(self, client):
        sid = "store_no_spike"
        seed(client, [_row(sid, f"v{i}", "ZONE_ENTER", zone_id="billing") for i in range(5)])
        assert "QUEUE_SPIKE" not in anomaly_types(client, sid)

    def test_conversion_drop_requires_minimum_visitors(self, client):
        sid = "store_too_few"
        seed(client, [_row(sid, f"v{i}", "ENTRY") for i in range(9)])
        assert "CONVERSION_DROP" not in anomaly_types(client, sid)

    def test_conversion_drop_fires_with_enough_visitors(self, client):
        sid = "store_conv_drop"
        seed(client, [_row(sid, f"v{i}", "ENTRY") for i in range(11)])
        resp = client.get(f"/stores/{sid}/anomalies")
        anomalies = resp.json()["anomalies"]
        types = {a["type"] for a in anomalies}
        assert "CONVERSION_DROP" in types
        alert = next(a for a in anomalies if a["type"] == "CONVERSION_DROP")
        assert alert["severity"] == "WARN"
        assert alert["value"] == 0.0

    def test_dead_zone_detected(self, client):
        sid = "store_dead_zone"
        seed(client, [
            _row(sid, "v1", "ENTRY"),
            _row(sid, "v1", "ZONE_ENTER", zone_id="skincare"),
            _row(sid, "v1", "ZONE_EXIT",  zone_id="perfume"),   # appeared but never entered
        ])
        resp = client.get(f"/stores/{sid}/anomalies")
        anomalies = resp.json()["anomalies"]
        types = {a["type"] for a in anomalies}
        assert "DEAD_ZONE" in types
        dz = next(a for a in anomalies if a["type"] == "DEAD_ZONE")
        assert dz["severity"] == "INFO"
        assert "perfume" in dz["message"]

    def test_high_abandon_rate_fires_above_50_pct(self, client):
        sid = "store_abandon"
        seed(client, [
            _row(sid, "v1", "BILLING_QUEUE_JOIN"),
            _row(sid, "v2", "BILLING_QUEUE_JOIN"),
            _row(sid, "v3", "BILLING_QUEUE_JOIN"),
            _row(sid, "v1", "BILLING_QUEUE_ABANDON"),
            _row(sid, "v2", "BILLING_QUEUE_ABANDON"),
        ])
        resp = client.get(f"/stores/{sid}/anomalies")
        anomalies = resp.json()["anomalies"]
        types = {a["type"] for a in anomalies}
        assert "HIGH_ABANDON_RATE" in types
        alert = next(a for a in anomalies if a["type"] == "HIGH_ABANDON_RATE")
        assert alert["severity"] == "WARN"
        assert alert["value"] > 50.0

    def test_high_abandon_does_not_fire_at_exactly_50_pct(self, client):
        sid = "store_boundary_abandon"
        seed(client, [
            _row(sid, "v1", "BILLING_QUEUE_JOIN"),
            _row(sid, "v2", "BILLING_QUEUE_JOIN"),
            _row(sid, "v1", "BILLING_QUEUE_ABANDON"),  # 1/2 = 50.0% exactly
        ])
        assert "HIGH_ABANDON_RATE" not in anomaly_types(client, sid)

    def test_all_severities_are_valid_strings(self, client):
        sid = "store_severity_check"
        seed(client, [
            *[_row(sid, f"v{i}", "ZONE_ENTER", zone_id="billing") for i in range(7)],
            *[_row(sid, f"v{i}", "ENTRY") for i in range(12)],
        ])
        anomalies = client.get(f"/stores/{sid}/anomalies").json()["anomalies"]
        for a in anomalies:
            assert a["severity"] in {"CRITICAL", "WARN", "INFO"}


class TestHeatmap:

    def test_empty_store_returns_empty_list(self, client):
        resp = client.get("/stores/store_no_zones/heatmap")
        assert resp.status_code == 200
        assert resp.json()["heatmap"] == []

    def test_highest_traffic_zone_has_score_1_0(self, client):
        sid = "store_heatmap_norm"
        seed(client, [
            _row(sid, "v1", "ZONE_ENTER", zone_id="skincare"),
            _row(sid, "v2", "ZONE_ENTER", zone_id="skincare"),
            _row(sid, "v3", "ZONE_ENTER", zone_id="skincare"),
            _row(sid, "v1", "ZONE_ENTER", zone_id="makeup"),
        ])
        heatmap = client.get(f"/stores/{sid}/heatmap").json()["heatmap"]
        assert heatmap[0]["zone_id"] == "skincare"
        assert heatmap[0]["normalized_score"] == 1.0
        assert heatmap[1]["zone_id"] == "makeup"
        assert heatmap[1]["normalized_score"] == round(1 / 3, 4)

    def test_heatmap_sorted_highest_first(self, client):
        sid = "store_heatmap_sort"
        seed(client, [
            _row(sid, "v1", "ZONE_ENTER", zone_id="makeup"),
            _row(sid, "v1", "ZONE_ENTER", zone_id="skincare"),
            _row(sid, "v2", "ZONE_ENTER", zone_id="skincare"),
        ])
        heatmap = client.get(f"/stores/{sid}/heatmap").json()["heatmap"]
        visits = [z["visit_frequency"] for z in heatmap]
        assert visits == sorted(visits, reverse=True)

    def test_heatmap_includes_avg_dwell(self, client):
        sid = "store_heatmap_dwell"
        seed(client, [
            _row(sid, "v1", "ZONE_ENTER", zone_id="skincare"),
            _row(sid, "v1", "ZONE_DWELL", zone_id="skincare", dwell_ms=90000),
        ])
        heatmap = client.get(f"/stores/{sid}/heatmap").json()["heatmap"]
        skincare = next(z for z in heatmap if z["zone_id"] == "skincare")
        assert skincare["avg_dwell_seconds"] == 90.0

    def test_staff_excluded_from_heatmap(self, client):
        sid = "store_staff_heatmap"
        seed(client, [
            _row(sid, "v1",     "ZONE_ENTER", zone_id="skincare", is_staff=False),
            _row(sid, "staff1", "ZONE_ENTER", zone_id="skincare", is_staff=True),
            _row(sid, "staff1", "ZONE_ENTER", zone_id="skincare", is_staff=True),
        ])
        heatmap = client.get(f"/stores/{sid}/heatmap").json()["heatmap"]
        skincare = next(z for z in heatmap if z["zone_id"] == "skincare")
        assert skincare["visit_frequency"] == 1
