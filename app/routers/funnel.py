"""
app/routers/funnel.py
─────────────────────
GET /stores/{store_id}/funnel

Session-aware conversion funnel. Each visitor is counted at most once
per stage, even if they re-enter the store or visit a zone multiple times.

Funnel stages
─────────────
1. ENTRY       — visitor crossed the store entrance (ENTRY event, not staff)
2. ZONE_VISIT  — visitor entered at least one product zone (any zone != billing)
3. BILLING_QUEUE — visitor entered the billing queue (BILLING_QUEUE_JOIN)
4. PURCHASE    — visitor completed a purchase (BILLING_QUEUE_JOIN - BILLING_QUEUE_ABANDON)

Design decisions
────────────────
- "Session-based" here means: one visitor, one count per stage, regardless
  of how many REENTRY or ZONE_ENTER events they produced.
- We use COUNT(DISTINCT visitor_id) at every step, which is the standard
  industry approach for top-of-funnel to bottom-of-funnel conversion.
- Drop-off % at each step = how many people from the PREVIOUS step didn't
  continue. This helps the store manager pinpoint where customers are lost.
- Re-entry: a REENTRY visitor has already been counted at Stage 1 via their
  initial ENTRY, so their REENTRY events are safely ignored in funnel math.
  The DISTINCT ensures they aren't double-counted at later stages.
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.analytics import AnalyticsService

log = logging.getLogger("store_intel.funnel")

router = APIRouter(prefix="/stores", tags=["Funnel"])


@router.get("/{store_id}/funnel")
def get_store_funnel(store_id: str, db: Session = Depends(get_db)):
    """
    Returns the 4-stage conversion funnel with drop-off rates.
    Each visitor is counted exactly once per stage (no double-counting).
    Staff are excluded.
    """
    log.info("Computing funnel for store_id=%s", store_id)

    payload = AnalyticsService.get_funnel(db, store_id)

    stages = payload["funnel"]
    log.info("Funnel OK store_id=%s entry=%d zone=%d queue=%d purchase=%d",
             store_id, stages[0]["count"], stages[1]["count"], stages[2]["count"], stages[3]["count"])

    return payload

