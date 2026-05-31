"""
app/routers/metrics.py
──────────────────────
GET /stores/{store_id}/metrics

Returns high-level KPIs for one store, computed live from the events
table. All staff events are excluded from every calculation.

Business logic
──────────────
- unique_visitors    : DISTINCT visitor_ids that had an ENTRY event,
                       excluding anyone flagged is_staff=True.
- conversion_rate    : visitors who completed a purchase (had BILLING_QUEUE_JOIN
                       but NOT BILLING_QUEUE_ABANDON) / unique_visitors.
- avg_dwell_time     : average of all ZONE_DWELL dwell_ms values, in seconds.
- queue_depth        : visitors currently estimated to be inside the billing zone
                       (joined queue but not yet exited billing or abandoned).
- abandonment_rate   : visitors who abandoned / visitors who joined the queue.
- revenue_per_visitor: total POS revenue / unique visitors.
- average_basket_value: total POS revenue / number of purchases.

Empty-dataset safety
────────────────────
Every division is guarded. Missing transactions return 0.0 cleanly.
"""

import logging
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.analytics import AnalyticsService

log = logging.getLogger("store_intel.metrics")

router = APIRouter(prefix="/stores", tags=["Metrics"])


@router.get("/{store_id}/metrics")
def get_store_metrics(store_id: str, db: Session = Depends(get_db)):
    """
    Returns live store KPIs for the given store_id.
    Staff events (is_staff = 1) are excluded from every metric.
    """
    log.info("Computing metrics for store_id=%s", store_id)

    payload = AnalyticsService.get_metrics(db, store_id)

    if payload["unique_visitors"] == 0:
        log.warning("store_id=%s has no visitor data yet.", store_id)
        return JSONResponse(status_code=200, content={"warning": "No visitor data found.", **payload})

    log.info("Metrics OK store_id=%s visitors=%d conversion=%.1f%%",
             store_id, payload["unique_visitors"], payload["conversion_rate_pct"])
    return payload

