# PROMPT:
# Generate pytest tests for /stores/{id}/metrics and /stores/{id}/funnel.
# Cover: empty store, zero purchases, staff exclusion, correct math, funnel
# monotonicity, re-entry visitor counted only once per funnel stage.
#
# CHANGES MADE:
# - Fixture moved to conftest.py. Tests seed via client.db_session.
# - All test methods accept only `client` — db_session is accessed via client.db_session.

import uuid
import pytest
from datetime import datetime, timezone
from app.database import EventORM


def _now():
    return datetime.now(timezone.utc)


def _row(store_id, visitor_id, event_type, zone_id=None, dwell_ms=None,
         is_staff=False, confidence=0.9):
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
        confidence = confidence,
    )


def seed(client, rows):
    db = client.db_session
    for r in rows:
        db.add(r)
    db.commit()


class TestMetrics:

    def test_empty_store_returns_zeros(self, client):
        resp = client.get("/stores/store_empty/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["unique_visitors"]      == 0
        assert body["conversion_rate_pct"]  == 0.0
        assert body["abandonment_rate_pct"] == 0.0
        assert body["queue_depth"]          == 0

    def test_unique_visitor_count_is_correct(self, client):
        sid = "store_visitors"
        seed(client, [
            _row(sid, "v1", "ENTRY"),
            _row(sid, "v2", "ENTRY"),
            _row(sid, "v3", "ENTRY"),
        ])
        resp = client.get(f"/stores/{sid}/metrics")
        assert resp.json()["unique_visitors"] == 3

    def test_staff_excluded_from_visitor_count(self, client):
        sid = "store_staff_test"
        seed(client, [
            _row(sid, "v1",     "ENTRY", is_staff=False),
            _row(sid, "v2",     "ENTRY", is_staff=False),
            _row(sid, "staff1", "ENTRY", is_staff=True),
            _row(sid, "staff2", "ENTRY", is_staff=True),
        ])
        assert client.get(f"/stores/{sid}/metrics").json()["unique_visitors"] == 2

    def test_zero_purchases_gives_zero_conversion(self, client):
        sid = "store_zero_purchase"
        seed(client, [_row(sid, f"v{i}", "ENTRY") for i in range(3)])
        body = client.get(f"/stores/{sid}/metrics").json()
        assert body["conversion_rate_pct"]  == 0.0
        assert body["abandonment_rate_pct"] == 0.0
        assert body["purchases"]            == 0

    def test_conversion_rate_math_is_correct(self, client):
        sid = "store_conv_math"
        seed(client, [
            _row(sid, "v1", "ENTRY"),
            _row(sid, "v2", "ENTRY"),
            _row(sid, "v1", "BILLING_QUEUE_JOIN"),
            _row(sid, "v2", "BILLING_QUEUE_JOIN"),
            _row(sid, "v2", "BILLING_QUEUE_ABANDON"),
        ])
        body = client.get(f"/stores/{sid}/metrics").json()
        assert body["purchases"]            == 1
        assert body["conversion_rate_pct"]  == 50.0
        assert body["abandonment_rate_pct"] == 50.0

    def test_abandonment_rate_100_percent(self, client):
        sid = "store_abandon_all"
        seed(client, [
            _row(sid, "v1", "ENTRY"),
            _row(sid, "v2", "ENTRY"),
            _row(sid, "v1", "BILLING_QUEUE_JOIN"),
            _row(sid, "v2", "BILLING_QUEUE_JOIN"),
            _row(sid, "v1", "BILLING_QUEUE_ABANDON"),
            _row(sid, "v2", "BILLING_QUEUE_ABANDON"),
        ])
        body = client.get(f"/stores/{sid}/metrics").json()
        assert body["purchases"]            == 0
        assert body["abandonment_rate_pct"] == 100.0
        assert body["conversion_rate_pct"]  == 0.0

    def test_avg_dwell_calculation(self, client):
        sid = "store_dwell"
        seed(client, [
            _row(sid, "v1", "ZONE_DWELL", zone_id="skincare", dwell_ms=30000),
            _row(sid, "v1", "ZONE_DWELL", zone_id="makeup",   dwell_ms=60000),
        ])
        body = client.get(f"/stores/{sid}/metrics").json()
        assert body["avg_dwell_seconds"] == 45.0


class TestFunnel:

    def test_empty_store_funnel_returns_zeros(self, client):
        resp = client.get("/stores/store_empty_funnel/funnel")
        assert resp.status_code == 200
        funnel = resp.json()["funnel"]
        assert len(funnel) == 4
        assert all(s["count"] == 0 for s in funnel)

    def test_funnel_stage_order_and_names(self, client):
        funnel = client.get("/stores/store_order/funnel").json()["funnel"]
        assert funnel[0]["name"] == "Store Entry"
        assert funnel[1]["name"] == "Zone Visit"
        assert funnel[2]["name"] == "Billing Queue"
        assert funnel[3]["name"] == "Purchase"

    def test_funnel_is_monotonically_non_increasing(self, client):
        sid = "store_monotonic"
        seed(client, [
            _row(sid, "v1", "ENTRY"),
            _row(sid, "v2", "ENTRY"),
            _row(sid, "v3", "ENTRY"),
            _row(sid, "v1", "ZONE_ENTER", zone_id="skincare"),
            _row(sid, "v2", "ZONE_ENTER", zone_id="makeup"),
            _row(sid, "v1", "BILLING_QUEUE_JOIN"),
            _row(sid, "v2", "BILLING_QUEUE_JOIN"),
            _row(sid, "v2", "BILLING_QUEUE_ABANDON"),
        ])
        funnel = client.get(f"/stores/{sid}/funnel").json()["funnel"]
        counts = [s["count"] for s in funnel]
        for i in range(1, len(counts)):
            assert counts[i] <= counts[i - 1], \
                f"Step {i+1} ({counts[i]}) > Step {i} ({counts[i-1]})"

    def test_reentry_visitor_counted_once_in_funnel(self, client):
        sid = "store_reentry_funnel"
        seed(client, [
            _row(sid, "visitor_A", "ENTRY"),
            _row(sid, "visitor_A", "EXIT"),
            _row(sid, "visitor_A", "REENTRY"),
            _row(sid, "visitor_B", "ENTRY"),
        ])
        funnel = client.get(f"/stores/{sid}/funnel").json()["funnel"]
        assert funnel[0]["count"] == 2   # 2 unique visitor_ids

    def test_staff_excluded_from_funnel(self, client):
        sid = "store_staff_funnel"
        seed(client, [
            _row(sid, "v1",     "ENTRY", is_staff=False),
            _row(sid, "staff1", "ENTRY", is_staff=True),
        ])
        funnel = client.get(f"/stores/{sid}/funnel").json()["funnel"]
        assert funnel[0]["count"] == 1

    def test_drop_off_pct_correct(self, client):
        sid = "store_dropoff"
        visitors = [f"v{i}" for i in range(10)]
        rows = [_row(sid, v, "ENTRY") for v in visitors]
        rows += [_row(sid, v, "ZONE_ENTER", zone_id="skincare") for v in visitors[:5]]
        seed(client, rows)
        funnel = client.get(f"/stores/{sid}/funnel").json()["funnel"]
        assert funnel[0]["count"]       == 10
        assert funnel[1]["count"]       == 5
        assert funnel[1]["drop_off_pct"] == 50.0
