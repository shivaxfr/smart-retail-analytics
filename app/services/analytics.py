"""
app/services/analytics.py
─────────────────────────
Decoupled analytics service layer for computing store metrics, funnel,
heatmap, and anomalies from the database. Excludes staff.
"""

import logging
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import Dict, List, Any

log = logging.getLogger("store_intel.analytics_service")

class AnalyticsService:

    @staticmethod
    def get_metrics(db: Session, store_id: str) -> Dict[str, Any]:
        """
        Returns live store KPIs for the given store_id.
        Staff events (is_staff = 1) are excluded from every metric.
        """
        log.info("Computing service metrics for store_id=%s", store_id)

        # ── 1. Unique non-staff visitors ────────────────────────────────────────
        res = db.execute(text("""
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id  = :sid
              AND event_type = 'ENTRY'
              AND is_staff   = 0
        """), {"sid": store_id})
        unique_visitors: int = res.scalar() or 0

        # ── 2. Billing queue metrics ─────────────────────────────────────────────
        res = db.execute(text("""
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id  = :sid
              AND event_type = 'BILLING_QUEUE_JOIN'
              AND is_staff   = 0
        """), {"sid": store_id})
        queue_joins: int = res.scalar() or 0

        res = db.execute(text("""
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id  = :sid
              AND event_type = 'BILLING_QUEUE_ABANDON'
              AND is_staff   = 0
        """), {"sid": store_id})
        abandons: int = res.scalar() or 0

        # A "purchase" is someone who joined the queue and did NOT abandon.
        purchases: int = max(0, queue_joins - abandons)

        # ── 3. Average dwell time (seconds) ─────────────────────────────────────
        res = db.execute(text("""
            SELECT AVG(dwell_ms)
            FROM events
            WHERE store_id  = :sid
              AND event_type = 'ZONE_DWELL'
              AND is_staff   = 0
              AND dwell_ms   IS NOT NULL
        """), {"sid": store_id})
        avg_dwell_ms: float = res.scalar() or 0.0
        avg_dwell_seconds: float = round(avg_dwell_ms / 1000.0, 2)

        # ── 4. Active queue depth (real-time estimate) ───────────────────────────
        res = db.execute(text("""
            SELECT COUNT(DISTINCT v.visitor_id)
            FROM (
                SELECT DISTINCT visitor_id
                FROM events
                WHERE store_id  = :sid
                  AND zone_id    = 'billing'
                  AND event_type = 'ZONE_ENTER'
                  AND is_staff   = 0
            ) AS v
            WHERE v.visitor_id NOT IN (
                SELECT DISTINCT visitor_id
                FROM events
                WHERE store_id  = :sid
                  AND zone_id    = 'billing'
                  AND event_type = 'ZONE_EXIT'
                  AND is_staff   = 0
            )
        """), {"sid": store_id})
        queue_depth: int = res.scalar() or 0

        # ── 5. Revenue figures (from POS metadata if present) ───────────────────
        res = db.execute(text("""
            SELECT COALESCE(SUM(CAST(json_extract(metadata, '$.pos_amount') AS REAL)), 0.0)
            FROM events
            WHERE store_id  = :sid
              AND event_type = 'BILLING_QUEUE_JOIN'
              AND is_staff   = 0
              AND json_extract(metadata, '$.pos_amount') IS NOT NULL
        """), {"sid": store_id})
        total_revenue: float = res.scalar() or 0.0

        revenue_per_visitor: float = round(total_revenue / unique_visitors, 2) if unique_visitors else 0.0
        average_basket_value: float = round(total_revenue / purchases, 2) if purchases else 0.0

        def safe_rate(num: float, den: float) -> float:
            if not den:
                return 0.0
            return round((num / den) * 100, 2)

        conversion_rate = safe_rate(purchases, unique_visitors)
        abandonment_rate = safe_rate(abandons, queue_joins)

        return {
            "store_id":            store_id,
            "unique_visitors":     unique_visitors,
            "purchases":           purchases,
            "conversion_rate_pct": conversion_rate,
            "abandonment_rate_pct": abandonment_rate,
            "avg_dwell_seconds":   avg_dwell_seconds,
            "queue_depth":         queue_depth,
            "total_revenue":       round(total_revenue, 2),
            "revenue_per_visitor": revenue_per_visitor,
            "average_basket_value": average_basket_value,
        }

    @staticmethod
    def get_funnel(db: Session, store_id: str) -> Dict[str, Any]:
        """
        Returns the 4-stage conversion funnel with drop-off rates.
        Excludes staff.
        """
        log.info("Computing service funnel for store_id=%s", store_id)

        # ── Stage 1: ENTRY ───────────────────────────────────────────────────────
        res = db.execute(text("""
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id  = :sid
              AND event_type = 'ENTRY'
              AND is_staff   = 0
        """), {"sid": store_id})
        entries: int = res.scalar() or 0

        # ── Stage 2: ZONE_VISIT (at least 1 product zone) ────────────────────────
        res = db.execute(text("""
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id   = :sid
              AND event_type  = 'ZONE_ENTER'
              AND is_staff    = 0
              AND zone_id    != 'billing'
              AND zone_id     IS NOT NULL
              AND visitor_id IN (
                  SELECT DISTINCT visitor_id
                  FROM events
                  WHERE store_id  = :sid
                    AND event_type = 'ENTRY'
                    AND is_staff   = 0
              )
        """), {"sid": store_id})
        zone_visits: int = res.scalar() or 0

        # ── Stage 3: BILLING_QUEUE ───────────────────────────────────────────────
        res = db.execute(text("""
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id  = :sid
              AND event_type = 'BILLING_QUEUE_JOIN'
              AND is_staff   = 0
        """), {"sid": store_id})
        queue_joins: int = res.scalar() or 0

        # ── Stage 4: PURCHASE ────────────────────────────────────────────────────
        res = db.execute(text("""
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id  = :sid
              AND event_type = 'BILLING_QUEUE_ABANDON'
              AND is_staff   = 0
        """), {"sid": store_id})
        abandons: int = res.scalar() or 0
        purchases: int = max(0, queue_joins - abandons)

        # Monotonic cap
        zone_visits = min(zone_visits, entries)
        queue_joins = min(queue_joins, zone_visits)
        purchases   = min(purchases,   queue_joins)

        def drop_off_pct(prev: int, curr: int) -> float:
            if not prev:
                return 0.0
            return round((1 - curr / prev) * 100, 2)

        return {
            "store_id": store_id,
            "funnel": [
                {
                    "step":            1,
                    "name":            "Store Entry",
                    "count":           entries,
                    "drop_off_pct":    0.0,
                },
                {
                    "step":            2,
                    "name":            "Zone Visit",
                    "count":           zone_visits,
                    "drop_off_pct":    drop_off_pct(entries, zone_visits),
                },
                {
                    "step":            3,
                    "name":            "Billing Queue",
                    "count":           queue_joins,
                    "drop_off_pct":    drop_off_pct(zone_visits, queue_joins),
                },
                {
                    "step":            4,
                    "name":            "Purchase",
                    "count":           purchases,
                    "drop_off_pct":    drop_off_pct(queue_joins, purchases),
                },
            ]
        }

    @staticmethod
    def get_heatmap(db: Session, store_id: str) -> Dict[str, Any]:
        """
        Returns per-zone visit frequency, average dwell time, and a
        normalized traffic intensity score between 0.0 and 1.0.
        Excludes staff.
        """
        log.info("Computing service heatmap for store_id=%s", store_id)

        # ── 1. Visit frequency per zone ──────────────────────────────────────────
        res = db.execute(text("""
            SELECT zone_id, COUNT(*) AS visits
            FROM events
            WHERE store_id  = :sid
              AND event_type = 'ZONE_ENTER'
              AND is_staff   = 0
              AND zone_id    IS NOT NULL
            GROUP BY zone_id
            ORDER BY visits DESC
        """), {"sid": store_id})
        zone_visits: dict[str, int] = {row[0]: row[1] for row in res}

        if not zone_visits:
            return {"store_id": store_id, "heatmap": []}

        # ── 2. Average dwell per zone ────────────────────────────────────────────
        res = db.execute(text("""
            SELECT zone_id, AVG(dwell_ms) AS avg_dwell_ms
            FROM events
            WHERE store_id  = :sid
              AND event_type = 'ZONE_DWELL'
              AND is_staff   = 0
              AND zone_id    IS NOT NULL
              AND dwell_ms   IS NOT NULL
            GROUP BY zone_id
        """), {"sid": store_id})
        zone_dwells: dict[str, float] = {row[0]: row[1] for row in res}

        max_visits: int = max(zone_visits.values())

        heatmap = []
        for zone_id, visits in zone_visits.items():
            score = round(visits / max_visits, 4) if max_visits > 0 else 0.0
            avg_dwell_ms = zone_dwells.get(zone_id, 0.0)
            avg_dwell_secs = round(avg_dwell_ms / 1000.0, 2)

            heatmap.append({
                "zone_id":           zone_id,
                "visit_frequency":   visits,
                "avg_dwell_seconds": avg_dwell_secs,
                "normalized_score":  score,
            })

        return {"store_id": store_id, "heatmap": heatmap}

    @staticmethod
    def get_anomalies(db: Session, store_id: str) -> Dict[str, Any]:
        """
        Runs all anomaly rules against the events table and returns a
        structured list of alerts. Excludes staff.
        """
        log.info("Computing service anomalies for store_id=%s", store_id)

        # Import thresholds directly from the router to maintain a single source of truth
        from app.routers.anomalies import (
            QUEUE_SPIKE_THRESHOLD,
            CONVERSION_DROP_THRESHOLD,
            CONVERSION_MIN_VISITORS,
            ABANDON_RATE_THRESHOLD,
        )

        alerts = []

        # ── Rule 1: QUEUE_SPIKE ──────────────────────────────────────────────────
        res = db.execute(text("""
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id  = :sid
              AND zone_id    = 'billing'
              AND event_type = 'ZONE_ENTER'
              AND is_staff   = 0
              AND visitor_id NOT IN (
                  SELECT visitor_id
                  FROM events
                  WHERE store_id  = :sid
                    AND zone_id    = 'billing'
                    AND event_type = 'ZONE_EXIT'
                    AND is_staff   = 0
              )
        """), {"sid": store_id})
        queue_depth: int = res.scalar() or 0

        if queue_depth > QUEUE_SPIKE_THRESHOLD:
            alerts.append({
                "type":     "QUEUE_SPIKE",
                "severity": "CRITICAL",
                "value":    queue_depth,
                "threshold": QUEUE_SPIKE_THRESHOLD,
                "message":  (f"Billing queue has {queue_depth} active visitors, "
                             f"exceeding the threshold of {QUEUE_SPIKE_THRESHOLD}. "
                             f"Open an additional checkout lane."),
            })

        # ── Rule 2: CONVERSION_DROP ──────────────────────────────────────────────
        res = db.execute(text("""
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id  = :sid
              AND event_type = 'ENTRY'
              AND is_staff   = 0
        """), {"sid": store_id})
        unique_visitors: int = res.scalar() or 0

        if unique_visitors >= CONVERSION_MIN_VISITORS:
            res = db.execute(text("""
                SELECT COUNT(DISTINCT visitor_id)
                FROM events
                WHERE store_id  = :sid
                  AND event_type = 'BILLING_QUEUE_JOIN'
                  AND is_staff   = 0
            """), {"sid": store_id})
            queue_joins: int = res.scalar() or 0

            res = db.execute(text("""
                SELECT COUNT(DISTINCT visitor_id)
                FROM events
                WHERE store_id  = :sid
                  AND event_type = 'BILLING_QUEUE_ABANDON'
                  AND is_staff   = 0
            """), {"sid": store_id})
            abandons: int = res.scalar() or 0

            purchases = max(0, queue_joins - abandons)
            conversion = (purchases / unique_visitors) * 100

            if conversion < CONVERSION_DROP_THRESHOLD:
                alerts.append({
                    "type":      "CONVERSION_DROP",
                    "severity":  "WARN",
                    "value":     round(conversion, 2),
                    "threshold": CONVERSION_DROP_THRESHOLD,
                    "message":   (f"Conversion rate is {round(conversion,1)}%, "
                                  f"below the {CONVERSION_DROP_THRESHOLD}% threshold. "
                                  f"Review product placement and pricing."),
                })

        # ── Rule 3: DEAD_ZONE ────────────────────────────────────────────────────
        res = db.execute(text("""
            SELECT DISTINCT zone_id
            FROM events
            WHERE store_id = :sid
              AND zone_id  IS NOT NULL
        """), {"sid": store_id})
        all_zones = {row[0] for row in res}

        res = db.execute(text("""
            SELECT DISTINCT zone_id
            FROM events
            WHERE store_id  = :sid
              AND event_type = 'ZONE_ENTER'
              AND is_staff   = 0
              AND zone_id    IS NOT NULL
        """), {"sid": store_id})
        visited_zones = {row[0] for row in res}

        dead_zones = all_zones - visited_zones - {"billing"}
        for zone in sorted(dead_zones):
            alerts.append({
                "type":     "DEAD_ZONE",
                "severity": "INFO",
                "value":    0,
                "threshold": 1,
                "message":  (f"Zone '{zone}' has received 0 customer visits. "
                             f"Consider reviewing its product mix or signage."),
            })

        # ── Rule 4: HIGH_ABANDON_RATE ────────────────────────────────────────────
        res = db.execute(text("""
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id  = :sid
              AND event_type = 'BILLING_QUEUE_JOIN'
              AND is_staff   = 0
        """), {"sid": store_id})
        qj: int = res.scalar() or 0

        if qj > 0:
            res = db.execute(text("""
                SELECT COUNT(DISTINCT visitor_id)
                FROM events
                WHERE store_id  = :sid
                  AND event_type = 'BILLING_QUEUE_ABANDON'
                  AND is_staff   = 0
            """), {"sid": store_id})
            ab: int = res.scalar() or 0
            abandon_pct = (ab / qj) * 100

            if abandon_pct > ABANDON_RATE_THRESHOLD:
                alerts.append({
                    "type":      "HIGH_ABANDON_RATE",
                    "severity":  "WARN",
                    "value":     round(abandon_pct, 2),
                    "threshold": ABANDON_RATE_THRESHOLD,
                    "message":   (f"{round(abandon_pct,1)}% of billing queue visitors abandoned. "
                                  f"Queue wait time may be too long. Assign more staff to billing."),
                })

        return {"store_id": store_id, "anomalies": alerts}
