# PROMPT:
# Generate pytest tests for the /events/ingest endpoint.
# Cover: valid batch, duplicate event_id, malformed JSON, missing required
# fields, empty array, batch too large, ZONE_DWELL without dwell_ms.
#
# CHANGES MADE:
# - Fixture moved to conftest.py (shared) — uses file-based SQLite per test
#   so FastAPI DI and the test both see the same database tables and rows.
# - TestClient is injected via the shared `client` fixture from conftest.py.

import uuid
import pytest
from fastapi.testclient import TestClient


# ── Helpers ────────────────────────────────────────────────────────────────────

def _event(**overrides) -> dict:
    """Build a minimal valid ENTRY event, with optional field overrides."""
    base = {
        "event_id":   str(uuid.uuid4()),
        "store_id":   "store_test",
        "camera_id":  "cam_01",
        "visitor_id": f"visitor_{uuid.uuid4().hex[:6]}",
        "event_type": "ENTRY",
        "confidence": 0.92,
    }
    base.update(overrides)
    return base


def post(client, events):
    return client.post("/events/ingest", json=events)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestIngestHappyPath:

    def test_single_valid_event(self, client):
        resp = post(client, [_event()])
        assert resp.status_code == 200
        body = resp.json()
        assert body["inserted_count"] == 1
        assert body["duplicate_count"] == 0
        assert body["invalid_count"] == 0
        assert body["status"] == "ok"

    def test_batch_of_multiple_events(self, client):
        events = [_event() for _ in range(5)]
        resp = post(client, events)
        assert resp.status_code == 200
        body = resp.json()
        assert body["received_count"] == 5
        assert body["inserted_count"] == 5
        assert body["duplicate_count"] == 0

    def test_all_event_types_accepted(self, client):
        types = [
            _event(event_type="ENTRY"),
            _event(event_type="EXIT"),
            _event(event_type="REENTRY"),
            _event(event_type="ZONE_ENTER",  zone_id="skincare"),
            _event(event_type="ZONE_EXIT",   zone_id="skincare"),
            _event(event_type="ZONE_DWELL",  zone_id="skincare", dwell_ms=45000),
            _event(event_type="BILLING_QUEUE_JOIN"),
            _event(event_type="BILLING_QUEUE_ABANDON"),
        ]
        resp = post(client, types)
        assert resp.status_code == 200
        body = resp.json()
        assert body["inserted_count"] == 8
        assert body["invalid_count"] == 0


class TestIngestIdempotency:

    def test_duplicate_event_id_is_ignored(self, client):
        ev = _event()
        resp1 = post(client, [ev])
        assert resp1.json()["inserted_count"] == 1

        resp2 = post(client, [ev])
        body2 = resp2.json()
        assert resp2.status_code == 200
        assert body2["inserted_count"] == 0
        assert body2["duplicate_count"] == 1

    def test_batch_with_mixed_new_and_duplicate(self, client):
        ev1 = _event()
        ev2 = _event()
        post(client, [ev1, ev2])

        ev3 = _event()
        ev4 = _event()
        resp = post(client, [ev1, ev2, ev3, ev4])
        body = resp.json()
        assert body["received_count"] == 4
        assert body["inserted_count"] == 2
        assert body["duplicate_count"] == 2

    def test_reentry_visitor_id_accepted(self, client):
        visitor = f"visitor_{uuid.uuid4().hex[:6]}"
        resp = post(client, [
            _event(event_type="ENTRY",   visitor_id=visitor),
            _event(event_type="EXIT",    visitor_id=visitor),
            _event(event_type="REENTRY", visitor_id=visitor),
        ])
        body = resp.json()
        assert body["inserted_count"] == 3
        assert body["duplicate_count"] == 0


class TestIngestValidation:

    def test_empty_array_rejected(self, client):
        resp = post(client, [])
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_batch_too_large_rejected(self, client):
        resp = post(client, [_event() for _ in range(501)])
        assert resp.status_code == 400
        assert "500" in resp.json()["detail"]

    def test_malformed_json_rejected(self, client):
        resp = client.post(
            "/events/ingest",
            content=b"this is not json at all",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_missing_required_field_store_id(self, client):
        bad = {"event_id": str(uuid.uuid4()), "event_type": "ENTRY",
               "camera_id": "cam_01", "visitor_id": "v1"}
        resp = post(client, [bad])
        assert resp.status_code == 200
        body = resp.json()
        assert body["invalid_count"] == 1
        assert body["inserted_count"] == 0

    def test_invalid_event_type_rejected(self, client):
        bad = _event(event_type="TELEPORT")
        resp = post(client, [bad])
        body = resp.json()
        assert body["invalid_count"] == 1
        assert body["inserted_count"] == 0

    def test_zone_dwell_without_dwell_ms_is_invalid(self, client):
        bad = _event(event_type="ZONE_DWELL", zone_id="skincare")
        resp = post(client, [bad])
        assert resp.json()["invalid_count"] == 1

    def test_zone_event_without_zone_id_is_invalid(self, client):
        bad = _event(event_type="ZONE_ENTER")
        resp = post(client, [bad])
        assert resp.json()["invalid_count"] == 1

    def test_confidence_out_of_range_is_invalid(self, client):
        bad = _event(confidence=1.5)
        resp = post(client, [bad])
        assert resp.json()["invalid_count"] == 1

    def test_batch_with_mix_of_valid_and_invalid(self, client):
        events = [
            _event(),
            _event(event_type="TELEPORT"),
            _event(event_type="ZONE_DWELL", zone_id="s"),
            _event(),
        ]
        resp = post(client, events)
        body = resp.json()
        assert body["inserted_count"] == 2
        assert body["invalid_count"] == 2
        assert body["status"] == "partial"


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["service_status"] == "ok"
        assert body["database_status"] == "connected"
        assert "last_updated" in body

    def test_root_returns_running(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"
