"""
app/routers/anomalies.py
────────────────────────
GET /stores/{store_id}/anomalies

Rule-based anomaly detection. Each rule runs a live SQL query against
the events table and emits a structured alert if the threshold is crossed.
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.analytics import AnalyticsService

log = logging.getLogger("store_intel.anomalies")

router = APIRouter(prefix="/stores", tags=["Anomalies"])

# ── Thresholds (easy to tune without touching SQL logic) ─────────────────────
QUEUE_SPIKE_THRESHOLD      = 5     # people simultaneously in billing queue
CONVERSION_DROP_THRESHOLD  = 10.0  # percent
CONVERSION_MIN_VISITORS    = 10    # only flag if enough data exists
ABANDON_RATE_THRESHOLD     = 50.0  # percent


@router.get("/{store_id}/anomalies")
def get_store_anomalies(store_id: str, db: Session = Depends(get_db)):
    """
    Runs all anomaly rules against the events table and returns a
    structured list of alerts. Returns an empty list if everything looks normal.
    Staff events are excluded from all calculations.
    """
    log.info("Running anomaly checks for store_id=%s", store_id)

    payload = AnalyticsService.get_anomalies(db, store_id)
    log.info("Anomaly check complete store_id=%s alerts=%d", store_id, len(payload["anomalies"]))
    return payload
