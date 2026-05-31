# tests/test_edge_cases.py
# ─────────────────────────────────────────────────────────────────
# Robust edge-case verification for the Store Intelligence system.
#
# Covers:
#   - POS correlation boundary cases (missing files, malformed timestamps, zero/negative amounts)
#   - Funnel integrity on out-of-order events (monotonic capping validation)
#   - Division-by-zero guards on empty-store metrics
#   - Staff data exclusion from all routers
# ─────────────────────────────────────────────────────────────────

import csv
import uuid
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from app.conversion import load_pos_csv, POSTransaction, attribute_purchases
from app.services.analytics import AnalyticsService


# ── Helpers ────────────────────────────────────────────────────────────────────

def _seed_event(client, **overrides) -> dict:
    """Helper to insert a single event directly into the test database."""
    event = {
        "event_id":   str(uuid.uuid4()),
        "store_id":   "store_test",
        "camera_id":  "cam_01",
        "visitor_id": f"visitor_{uuid.uuid4().hex[:6]}",
        "event_type": "ENTRY",
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "is_staff":   False,
        "confidence": 0.90,
    }
    event.update(overrides)

    resp = client.post("/events/ingest", json=[event])
    assert resp.status_code == 200
    return event


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestPOSEdgeCases:

    def test_missing_pos_csv_returns_empty_list(self):
        """Should fail gracefully and return [] if file doesn't exist."""
        result = load_pos_csv("nonexistent_file.csv")
        assert result == []

    def test_malformed_columns_returns_empty_list(self, tmp_path):
        """Should return [] if required columns like transaction_id are missing."""
        csv_file = tmp_path / "bad_pos.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["payment_method", "amount"]) # missing transaction_id and timestamp
            writer.writerow(["UPI", "499.00"])

        result = load_pos_csv(str(csv_file))
        assert result == []

    def test_duplicate_transaction_ids_are_deduplicated(self, tmp_path):
        """Should load only the first transaction when duplicates exist."""
        csv_file = tmp_path / "dup_pos.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["transaction_id", "timestamp", "amount", "store_id"])
            writer.writerow(["TXN-001", "2024-01-15T10:00:00Z", "500", "store_test"])
            writer.writerow(["TXN-001", "2024-01-15T10:00:00Z", "500", "store_test"]) # duplicate ID

        result = load_pos_csv(str(csv_file))
        assert len(result) == 1
        assert result[0].transaction_id == "TXN-001"

    def test_negative_pos_amounts_are_clamped_to_zero(self, tmp_path):
        """Should clamp negative POS transaction amounts (e.g. refunds) to 0.0."""
        csv_file = tmp_path / "neg_pos.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["transaction_id", "timestamp", "amount", "store_id"])
            writer.writerow(["TXN-REFUND", "2024-01-15T10:00:00Z", "-150.00", "store_test"])

        result = load_pos_csv(str(csv_file))
        assert len(result) == 1
        assert result[0].amount == 0.0

    def test_unparseable_timestamp_skipped_cleanly(self, tmp_path):
        """Should skip rows with corrupted, unparseable timestamp formats."""
        csv_file = tmp_path / "bad_time.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["transaction_id", "timestamp", "amount", "store_id"])
            writer.writerow(["TXN-001", "this-is-not-a-date", "300", "store_test"])

        result = load_pos_csv(str(csv_file))
        assert len(result) == 0

    def test_greedy_attribution_matches_correctly(self):
        """Verify greedy closest-in-time matching works for POS transactions."""
        txns = [
            POSTransaction("T1", datetime(2024, 1, 15, 10, 0, 5, tzinfo=timezone.utc), 100.0, "store_test", {}),
            POSTransaction("T2", datetime(2024, 1, 15, 10, 0, 25, tzinfo=timezone.utc), 200.0, "store_test", {}),
        ]
        exits = [
            {"visitor_id": "v1", "store_id": "store_test", "timestamp": "2024-01-15T10:00:00Z"}, # 5 seconds diff from T1
            {"visitor_id": "v2", "store_id": "store_test", "timestamp": "2024-01-15T10:00:30Z"}, # 5 seconds diff from T2
        ]
        
        matches = attribute_purchases(txns, exits, window_seconds=10.0)
        assert len(matches) == 2
        assert matches[0]["visitor_id"] == "v1"
        assert matches[0]["transaction_id"] == "T1"
        assert matches[1]["visitor_id"] == "v2"
        assert matches[1]["transaction_id"] == "T2"


class TestEmptyAndZeroStates:

    def test_division_by_zero_guarded_in_metrics_when_empty(self, client):
        """Should return zeroes cleanly without crashes when no events exist in store."""
        resp = client.get("/stores/empty_store/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["unique_visitors"] == 0
        assert body["conversion_rate_pct"] == 0.0
        assert body["total_revenue"] == 0.0
        assert body["revenue_per_visitor"] == 0.0
        assert body["average_basket_value"] == 0.0

    def test_division_by_zero_guarded_in_heatmap_when_empty(self, client):
        """Should return empty heatmap array cleanly without crashes on empty store."""
        resp = client.get("/stores/empty_store/heatmap")
        assert resp.status_code == 200
        body = resp.json()
        assert body["heatmap"] == []

    def test_out_of_order_funnel_monotonicity(self, client):
        """
        Verify that out-of-order event ingestion is safely guarded.
        Even if billing queue events are processed but entries are missing,
        capping ensures counts do not increase down the stages.
        """
        store = "store_capping"
        # Seed only a BILLING_QUEUE_JOIN event (no ENTRY)
        _seed_event(client, store_id=store, visitor_id="v_cap", event_type="BILLING_QUEUE_JOIN")

        resp = client.get(f"/stores/{store}/funnel")
        assert resp.status_code == 200
        funnel = resp.json()["funnel"]
        
        # Entries = 0, therefore all subsequent stages must be capped at 0
        assert funnel[0]["count"] == 0 # Entry
        assert funnel[1]["count"] == 0 # Zone Visit
        assert funnel[2]["count"] == 0 # Billing Queue
        assert funnel[3]["count"] == 0 # Purchase


class TestStaffExclusionEdgeCases:

    def test_staff_completely_excluded_from_all_metrics(self, client):
        """Verify that staff members (is_staff=True) are entirely ignored in live analytics."""
        store = "store_staff_exclusion"
        
        # Seed staff events
        _seed_event(client, store_id=store, visitor_id="staff_01", event_type="ENTRY", is_staff=True)
        _seed_event(client, store_id=store, visitor_id="staff_01", event_type="ZONE_ENTER", zone_id="makeup", is_staff=True)
        _seed_event(client, store_id=store, visitor_id="staff_01", event_type="ZONE_EXIT", zone_id="makeup", is_staff=True)
        _seed_event(client, store_id=store, visitor_id="staff_01", event_type="ZONE_DWELL", zone_id="makeup", dwell_ms=10000, is_staff=True)
        _seed_event(client, store_id=store, visitor_id="staff_01", event_type="BILLING_QUEUE_JOIN", is_staff=True)

        # Seed one customer event
        _seed_event(client, store_id=store, visitor_id="cust_01", event_type="ENTRY", is_staff=False)
    
        from sqlalchemy import text
        all_rows = client.db_session.execute(text("select event_id, store_id, visitor_id, event_type, is_staff from events")).fetchall()
        print("TEMPORARY DB ROWS:", all_rows)

        # 1. Check Metrics
        metrics = client.get(f"/stores/{store}/metrics").json()
        print("METRICS RESPONSE:", metrics)
        assert metrics["unique_visitors"] == 1 # only customer
        assert metrics["purchases"] == 0       # staff billing queue ignored
        
        # 2. Check Heatmap
        heatmap = client.get(f"/stores/{store}/heatmap").json()["heatmap"]
        assert len(heatmap) == 0               # staff zone enters completely ignored

        # 3. Check Funnel
        funnel = client.get(f"/stores/{store}/funnel").json()["funnel"]
        assert funnel[0]["count"] == 1         # only customer
        assert funnel[2]["count"] == 0         # staff billing queue ignored
