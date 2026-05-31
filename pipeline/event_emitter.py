"""
pipeline/event_emitter.py
─────────────────────────
State machine that converts tracker output + zone checks into
structured events compatible with the FastAPI ingest endpoint.

PART 2 additions:
  REENTRY               : visitor who exited comes back
  BILLING_QUEUE_JOIN     : visitor enters the billing zone
  BILLING_QUEUE_ABANDON  : visitor leaves billing without a POS match
  Staff exclusion        : events from staff get is_staff=True
  POS correlation        : match billing zone exits to POS transactions
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pipeline.tracker import PersonTracker, VisitorState
from pipeline.zone_logic import ZoneManager


class EventEmitter:
    """
    Processes tracked visitors frame-by-frame and emits structured events.

    All 8 event types are now supported:
        ENTRY, EXIT, ZONE_ENTER, ZONE_EXIT, ZONE_DWELL,
        REENTRY, BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON
    """

    def __init__(
        self,
        zone_manager: ZoneManager,
        store_id: str = "store_01",
        camera_id: str = "cam_01",
        pos_transactions: list[dict] | None = None,
        pos_window_seconds: float = 30.0,
    ):
        """
        Args:
            zone_manager        : loaded ZoneManager
            store_id / camera_id: identifiers written into every event
            pos_transactions    : list of {"timestamp": ISO, ...} dicts
            pos_window_seconds  : how close (in seconds) a POS txn must
                                  be to a billing-zone exit to count as
                                  a purchase (default ±30 s)
        """
        self.zm         = zone_manager
        self.store_id   = store_id
        self.camera_id  = camera_id
        self.pos_window = pos_window_seconds

        # Parse POS timestamps into epoch floats for fast comparison
        self._pos_epochs: list[float] = []
        if pos_transactions:
            for txn in pos_transactions:
                ts = txn.get("timestamp", "")
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                self._pos_epochs.append(dt.timestamp())
            self._pos_epochs.sort()

        # Per-visitor sequence counter for metadata.session_seq
        self._seq: dict[int, int] = {}

        # Track which visitors have had BILLING_QUEUE_JOIN emitted
        # so we don't emit it twice if they re-enter billing zone
        self._billing_joined: set[int] = set()

    # ──────────────────────────────────────────────────────────────────────

    def process_frame(
        self,
        tracker: PersonTracker,
        tracked: list[VisitorState],
        new_ids: set[int],
        lost_ids: set[int],
        reentry_ids: set[int],
        timestamp: float,
    ) -> list[dict]:
        """
        Given one frame's results, emit all relevant events.

        Returns:
            List of event dicts ready for the ingest API.
        """
        events: list[dict] = []
        ts_iso = self._to_iso(timestamp)

        # ── 1. ENTRY events ──────────────────────────────────────────────
        for v in tracked:
            if v.track_id in new_ids:
                events.append(self._make_event(
                    visitor=v,
                    event_type="ENTRY",
                    timestamp=ts_iso,
                ))
                tracker.mark_entry_emitted(v.track_id)
                self._seq[v.track_id] = 1

        # ── 2. REENTRY events ────────────────────────────────────────────
        for v in tracked:
            if v.track_id in reentry_ids:
                events.append(self._make_event(
                    visitor=v,
                    event_type="REENTRY",
                    timestamp=ts_iso,
                ))
                tracker.mark_entry_emitted(v.track_id)
                self._seq[v.track_id] = self._seq.get(v.track_id, 0)
                # Reset billing state for the new visit
                self._billing_joined.discard(v.track_id)

        # ── 3. Zone transitions for active visitors ──────────────────────
        for v in tracked:
            cx, cy = v.last_center
            current_zones = set(self.zm.get_zones_for_point(cx, cy))
            previous_zones = v.current_zones

            entered_zones = current_zones - previous_zones
            exited_zones  = previous_zones - current_zones

            # ── Zones entered ─────────────────────────────────────────
            for zone_id in entered_zones:
                events.append(self._make_event(
                    visitor=v,
                    event_type="ZONE_ENTER",
                    timestamp=ts_iso,
                    zone_id=zone_id,
                ))
                v.zone_enter_ts[zone_id] = timestamp

                # Billing zone → BILLING_QUEUE_JOIN
                if (self.zm.is_billing_zone(zone_id)
                        and v.track_id not in self._billing_joined):
                    events.append(self._make_event(
                        visitor=v,
                        event_type="BILLING_QUEUE_JOIN",
                        timestamp=ts_iso,
                        zone_id=zone_id,
                    ))
                    self._billing_joined.add(v.track_id)

            # ── Zones exited ──────────────────────────────────────────
            for zone_id in exited_zones:
                enter_ts = v.zone_enter_ts.pop(zone_id, timestamp)
                dwell_ms = int((timestamp - enter_ts) * 1000)

                events.append(self._make_event(
                    visitor=v,
                    event_type="ZONE_EXIT",
                    timestamp=ts_iso,
                    zone_id=zone_id,
                ))

                if dwell_ms > 0:
                    events.append(self._make_event(
                        visitor=v,
                        event_type="ZONE_DWELL",
                        timestamp=ts_iso,
                        zone_id=zone_id,
                        dwell_ms=dwell_ms,
                    ))

                # Billing zone exit → check POS, maybe ABANDON
                if self.zm.is_billing_zone(zone_id):
                    if not self._has_pos_match(timestamp):
                        events.append(self._make_event(
                            visitor=v,
                            event_type="BILLING_QUEUE_ABANDON",
                            timestamp=ts_iso,
                            zone_id=zone_id,
                        ))
                    # Reset so they can join queue again if they return
                    self._billing_joined.discard(v.track_id)

            v.current_zones = current_zones

        # ── 4. EXIT events for lost visitors ─────────────────────────────
        for tid in lost_ids:
            v = tracker.visitors.get(tid)
            if v is None:
                continue

            # Flush remaining zones
            for zone_id in list(v.current_zones):
                enter_ts = v.zone_enter_ts.pop(zone_id, v.last_seen)
                dwell_ms = int((v.last_seen - enter_ts) * 1000)
                exit_iso = self._to_iso(v.last_seen)

                events.append(self._make_event(
                    visitor=v, event_type="ZONE_EXIT",
                    timestamp=exit_iso, zone_id=zone_id,
                ))
                if dwell_ms > 0:
                    events.append(self._make_event(
                        visitor=v, event_type="ZONE_DWELL",
                        timestamp=exit_iso, zone_id=zone_id,
                        dwell_ms=dwell_ms,
                    ))

                # If they were in billing zone, check abandon
                if self.zm.is_billing_zone(zone_id):
                    if not self._has_pos_match(v.last_seen):
                        events.append(self._make_event(
                            visitor=v, event_type="BILLING_QUEUE_ABANDON",
                            timestamp=exit_iso, zone_id=zone_id,
                        ))
                    self._billing_joined.discard(tid)

            v.current_zones = set()

            events.append(self._make_event(
                visitor=v,
                event_type="EXIT",
                timestamp=self._to_iso(v.last_seen),
            ))
            tracker.mark_exit_emitted(tid)

        return events

    # ── POS correlation ───────────────────────────────────────────────────

    def _has_pos_match(self, exit_epoch: float) -> bool:
        """
        Check if any POS transaction occurred within ± pos_window
        of the given exit time.

        Simple linear scan — fine for the small POS datasets in this challenge.
        """
        for pos_ts in self._pos_epochs:
            if abs(pos_ts - exit_epoch) <= self.pos_window:
                return True
        return False

    # ── Event builder ─────────────────────────────────────────────────────

    def _make_event(
        self,
        visitor: VisitorState,
        event_type: str,
        timestamp: str,
        zone_id: str | None = None,
        dwell_ms: int | None = None,
    ) -> dict:
        """Build an event dict matching the ingest API schema."""
        seq = self._seq.get(visitor.track_id, 0) + 1
        self._seq[visitor.track_id] = seq

        return {
            "event_id":   str(uuid.uuid4()),
            "store_id":   self.store_id,
            "camera_id":  self.camera_id,
            "visitor_id": visitor.visitor_id,
            "event_type": event_type,
            "timestamp":  timestamp,

            "zone_id":    zone_id,
            "dwell_ms":   dwell_ms,
            "is_staff":   visitor.is_staff,
            "confidence": round(visitor.confidence, 3),
            "metadata": {
                "session_seq":  seq,
                "visit_count":  visitor.visit_count,
                "bbox":         visitor.last_bbox,
            },
        }

    @staticmethod
    def _to_iso(epoch_seconds: float) -> str:
        return datetime.fromtimestamp(
            epoch_seconds, tz=timezone.utc
        ).isoformat()


# ── Convenience: load POS data from file ──────────────────────────────────────

def load_pos_transactions(path: str) -> list[dict]:
    """Load POS transactions from a JSON file. Returns [] if file is missing."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r") as f:
        return json.load(f)
