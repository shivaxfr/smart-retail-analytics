"""
app/conversion.py
─────────────────
Real POS-based conversion analytics engine using database ORM.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy import text, func
from sqlalchemy.orm import Session

from app.database import get_db, POSTransactionORM

log = logging.getLogger("store_intel.conversion")

router = APIRouter(prefix="/stores", tags=["Conversion"])


# ── Attribution engine ────────────────────────────────────────────────────────

def attribute_purchases(
    db: Session,
    store_id: str,
    billing_exits: list[dict],
    window_seconds: float = 60.0,
) -> list[dict]:
    """
    Match POS transactions from DB to visitor billing-zone exits using a time window.
    """
    if not billing_exits:
        return []

    # Get all POS transactions for the store
    transactions = db.query(POSTransactionORM).filter(
        POSTransactionORM.store_id == store_id
    ).all()
    
    if not transactions:
        return []

    available = list(transactions)
    attributions = []

    for exit_event in billing_exits:
        exit_ts_raw = exit_event.get("timestamp")
        visitor_id  = exit_event.get("visitor_id")

        if not exit_ts_raw or not visitor_id:
            continue

        try:
            exit_ts = datetime.fromisoformat(str(exit_ts_raw))
            if exit_ts.tzinfo is None:
                exit_ts = exit_ts.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        best_txn   = None
        best_delta = float("inf")

        for txn in available:
            # Ensure txn timestamp is timezone aware for comparison
            txn_ts = txn.timestamp
            if txn_ts.tzinfo is None:
                txn_ts = txn_ts.replace(tzinfo=timezone.utc)
                
            delta = abs((txn_ts - exit_ts).total_seconds())
            if delta <= window_seconds and delta < best_delta:
                best_txn   = txn
                best_delta = delta

        if best_txn:
            available.remove(best_txn)
            attributions.append({
                "visitor_id":      visitor_id,
                "transaction_id":  best_txn.order_id,
                "amount":          best_txn.total_amount,
                "matched_at":      best_txn.timestamp.isoformat(),
                "delta_seconds":   round(best_delta, 1),
            })

    return attributions


# ── Conversion metrics builder ────────────────────────────────────────────────

def compute_conversion_metrics(
    store_id: str,
    db: Session,
    window_seconds: float = 60.0,
) -> dict:
    
    # ── Unique non-staff visitors ────────────────────────────────────────────
    res = db.execute(text("""
        SELECT COUNT(DISTINCT visitor_id)
        FROM events
        WHERE store_id  = :sid
          AND event_type = 'ENTRY'
          AND is_staff   = 0
    """), {"sid": store_id})
    unique_visitors: int = res.scalar() or 0

    # ── Billing zone exits ──────────────────────
    res = db.execute(text("""
        SELECT visitor_id, store_id, timestamp
        FROM events
        WHERE store_id  = :sid
          AND event_type = 'ZONE_EXIT'
          AND zone_id    = 'billing'
          AND is_staff   = 0
        ORDER BY timestamp
    """), {"sid": store_id})
    billing_exits = [
        {"visitor_id": row[0], "store_id": row[1], "timestamp": row[2]}
        for row in res
    ]

    # ── Run attribution ──────────────────────────────────────────────────────
    attributions = attribute_purchases(
        db=db,
        store_id=store_id,
        billing_exits=billing_exits,
        window_seconds=window_seconds,
    )

    pos_count = db.query(POSTransactionORM).filter(POSTransactionORM.store_id == store_id).count()

    # Compute revenue figures
    if pos_count == 0:
        res = db.execute(text("""
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id  = :sid AND event_type = 'BILLING_QUEUE_JOIN' AND is_staff = 0
        """), {"sid": store_id})
        q_joins = res.scalar() or 0

        res = db.execute(text("""
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id  = :sid AND event_type = 'BILLING_QUEUE_ABANDON' AND is_staff = 0
        """), {"sid": store_id})
        q_abandons = res.scalar() or 0

        purchases        = max(0, q_joins - q_abandons)
        total_revenue    = 0.0
        avg_basket       = 0.0
        source           = "camera_events_proxy"
    else:
        purchases        = len(attributions)
        total_revenue    = sum(a["amount"] for a in attributions)
        avg_basket       = total_revenue / purchases if purchases else 0.0
        source           = "pos_attributed"

    unique_purchasing = len({a["visitor_id"] for a in attributions})
    conversion_rate   = round((purchases / unique_visitors * 100), 2) if unique_visitors else 0.0
    revenue_per_visitor = round(total_revenue / unique_visitors, 2) if unique_visitors else 0.0

    return {
        "unique_visitors":       unique_visitors,
        "purchases":             purchases,
        "unique_purchasing_visitors": unique_purchasing,
        "conversion_rate_pct":   conversion_rate,
        "total_revenue":         round(total_revenue, 2),
        "revenue_per_visitor":   revenue_per_visitor,
        "average_basket_value":  round(avg_basket, 2),
        "attribution_window_s":  window_seconds,
        "data_source":           source,
        "attributions":          attributions,
    }


# ── POS Analytics builder ─────────────────────────────────────────────────────

def compute_pos_analytics(store_id: str, db: Session) -> dict:
    total_txns = db.query(func.count(func.distinct(POSTransactionORM.order_id))).filter(POSTransactionORM.store_id == store_id).scalar() or 0
    total_revenue = db.query(func.sum(POSTransactionORM.total_amount)).filter(POSTransactionORM.store_id == store_id).scalar() or 0.0
    
    top_brands = db.query(
        POSTransactionORM.brand_name, 
        func.sum(POSTransactionORM.qty).label('total_qty')
    ).filter(
        POSTransactionORM.store_id == store_id, 
        POSTransactionORM.brand_name != None
    ).group_by(POSTransactionORM.brand_name).order_by(text('total_qty DESC')).limit(5).all()
    
    top_products = db.query(
        POSTransactionORM.product_name, 
        func.sum(POSTransactionORM.qty).label('total_qty')
    ).filter(
        POSTransactionORM.store_id == store_id, 
        POSTransactionORM.product_name != None
    ).group_by(POSTransactionORM.product_name).order_by(text('total_qty DESC')).limit(5).all()

    return {
        "total_transactions": total_txns,
        "total_revenue": total_revenue,
        "top_brands": [{"brand": b[0], "qty": b[1]} for b in top_brands],
        "top_products": [{"product": p[0], "qty": p[1]} for p in top_products]
    }


# ── FastAPI endpoints ─────────────────────────────────────────────────────────

@router.get("/{store_id}/conversion")
def get_store_conversion(
    store_id: str,
    window: float = Query(default=60.0, description="POS attribution window in seconds"),
    db: Session = Depends(get_db),
):
    metrics = compute_conversion_metrics(store_id, db, window)
    return {"store_id": store_id, **metrics}

@router.get("/{store_id}/pos")
def get_store_pos(
    store_id: str,
    db: Session = Depends(get_db),
):
    analytics = compute_pos_analytics(store_id, db)
    return {"store_id": store_id, **analytics}
