"""
app/routers/heatmap.py
──────────────────────
GET /stores/{store_id}/heatmap

Returns per-zone traffic intensity and dwell data suitable for overlaying
on a store floor plan image in the dashboard.

Business logic
──────────────
- visit_frequency   : raw count of ZONE_ENTER events for each zone (non-staff)
- avg_dwell_seconds : average milliseconds in ZONE_DWELL events / 1000
- normalized_score  : this zone's visits / max visits across all zones (0.0–1.0)
  This is what the dashboard uses to pick a heatmap colour (blue→red).

Safe empty handling
───────────────────
If no ZONE_ENTER events exist yet, returns an empty heatmap list rather
than crashing or returning a division-by-zero error.
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.analytics import AnalyticsService

log = logging.getLogger("store_intel.heatmap")

router = APIRouter(prefix="/stores", tags=["Heatmap"])


@router.get("/{store_id}/heatmap")
def get_store_heatmap(store_id: str, db: Session = Depends(get_db)):
    """
    Returns per-zone visit frequency, average dwell time, and a
    normalized traffic intensity score between 0.0 and 1.0.
    Staff events are excluded.
    """
    log.info("Computing heatmap for store_id=%s", store_id)

    payload = AnalyticsService.get_heatmap(db, store_id)
    log.info("Heatmap OK store_id=%s zones=%d", store_id, len(payload["heatmap"]))
    return payload

