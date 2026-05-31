"""
pipeline/emit.py
────────────────
Converts raw tracker output into structured events that exactly match
the FastAPI ingest schema (app/models.py).

Separation of concerns
───────────────────────
  tracker.py  → "Who is this person and what is their state?"
  emit.py     → "What events does that state change produce?"

This file contains zero detection logic and zero tracking logic.
It only reads VisitorState objects and produces event dicts.

Events produced
───────────────
  ENTRY               first time a visitor is tracked
  EXIT                visitor has gone missing beyond max_lost_seconds
  REENTRY             visitor who previously exited has returned
  ZONE_ENTER          visitor's foot position entered a zone polygon
  ZONE_EXIT           visitor's foot position left a zone polygon
  ZONE_DWELL          emitted alongside ZONE_EXIT, carries dwell_ms

Challenge schema (every event must contain exactly these fields)
────────────────────────────────────────────────────────────────
  event_id   : UUID string
  store_id   : from config
  camera_id  : from config
  visitor_id : "visitor_0042" — stable across REENTRY
  event_type : one of the 8 event types
  timestamp  : ISO-8601 UTC
  zone_id    : null for ENTRY/EXIT, zone name for ZONE_* events
  dwell_ms   : null unless ZONE_DWELL
  is_staff   : bool, from staff detection rules
  confidence : float 0.0–1.0
  metadata   : dict with session_seq, visit_count, bbox
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from pipeline.tracker import PersonTracker, VisitorState
from pipeline.zone_logic import ZoneManager

log = logging.getLogger("pipeline.emit")


def _iso(epoch: float) -> str:
    """Convert epoch float to ISO-8601 UTC string."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


class EventEmitter:
    """
    Frame-by-frame event state machine.

    Call process_frame() once per video frame with the tracker's output.
    It returns a flat list of event dicts ready to write to JSONL or POST
    to the ingest API.
    """

    def __init__(
        self,
        zone_manager: ZoneManager,
        store_id: str  = "store_01",
        camera_id: str = "cam_01",
    ):
        self.zm        = zone_manager
        self.store_id  = store_id
        self.camera_id = camera_id

    # ── Main entry point ───────────────────────────────────────────────────────

    def process_frame(
        self,
        tracker: PersonTracker,
        tracked:     list[VisitorState],
        new_ids:     set[int],
        lost_ids:    set[int],
        reentry_ids: set[int],
        timestamp:   float,
    ) -> list[dict]:
        """
        Given one frame's tracking results, emit all events that occurred.

        Parameters
        ──────────
        tracker     : PersonTracker instance (needed to call mark_* helpers)
        tracked     : all VisitorState objects visible this frame
        new_ids     : track_ids of visitors seeing their very first frame
        lost_ids    : track_ids that have exceeded the lost-time threshold
        reentry_ids : track_ids that are re-appearances, not fresh arrivals
        timestamp   : current frame time (epoch float)

        Returns
        ───────
        List of event dicts — may be empty if nothing notable happened.
        """
        events: list[dict] = []

        # ── 1. ENTRY ──────────────────────────────────────────────────────────
        for v in tracked:
            if v.track_id in new_ids:
                events.append(self._build(v, "ENTRY", timestamp, tracker))
                tracker.mark_entry_emitted(v.track_id)

        # ── 2. REENTRY ────────────────────────────────────────────────────────
        for v in tracked:
            if v.track_id in reentry_ids:
                events.append(self._build(v, "REENTRY", timestamp, tracker))
                tracker.mark_entry_emitted(v.track_id)

        # ── 3. Zone transitions for all active visible visitors ───────────────
        for v in tracked:
            cx, cy         = v.last_center
            current_zones  = set(self.zm.get_zones_for_point(cx, cy))
            previous_zones = v.current_zones

            entered = current_zones - previous_zones
            exited  = previous_zones - current_zones

            for zone_id in entered:
                events.append(
                    self._build(v, "ZONE_ENTER", timestamp, tracker, zone_id=zone_id)
                )
                v.zone_enter_ts[zone_id] = timestamp   # start the dwell clock

            for zone_id in exited:
                enter_ts = v.zone_enter_ts.pop(zone_id, timestamp)
                dwell_ms = int((timestamp - enter_ts) * 1000)

                events.append(
                    self._build(v, "ZONE_EXIT", timestamp, tracker, zone_id=zone_id)
                )
                if dwell_ms > 0:
                    events.append(
                        self._build(
                            v, "ZONE_DWELL", timestamp, tracker,
                            zone_id=zone_id, dwell_ms=dwell_ms,
                        )
                    )

            v.current_zones = current_zones   # advance zone state

        # ── 4. EXIT for lost visitors ─────────────────────────────────────────
        for tid in lost_ids:
            v = tracker.visitors.get(tid)
            if v is None:
                continue

            # Flush any zones the person was still inside
            for zone_id in list(v.current_zones):
                enter_ts = v.zone_enter_ts.pop(zone_id, v.last_seen)
                dwell_ms = int((v.last_seen - enter_ts) * 1000)

                events.append(
                    self._build(v, "ZONE_EXIT", v.last_seen, tracker, zone_id=zone_id)
                )
                if dwell_ms > 0:
                    events.append(
                        self._build(
                            v, "ZONE_DWELL", v.last_seen, tracker,
                            zone_id=zone_id, dwell_ms=dwell_ms,
                        )
                    )
            v.current_zones = set()

            events.append(self._build(v, "EXIT", v.last_seen, tracker))
            tracker.mark_exit_emitted(tid)

        return events

    # ── Event builder ──────────────────────────────────────────────────────────

    def _build(
        self,
        v:          VisitorState,
        event_type: str,
        timestamp:  float,
        tracker:    PersonTracker,
        zone_id:    str | None = None,
        dwell_ms:   int | None = None,
    ) -> dict:
        """
        Build one event dict matching the challenge schema exactly.

        Note on session_seq
        ───────────────────
        session_seq is a monotonically increasing counter per visitor per
        visit session. It allows the API to sort events for a visitor even
        if they arrive out of order due to network batching.
        """
        seq = tracker.next_seq(v.track_id)

        return {
            # ── Required identity fields ──────────────────────────────────────
            "event_id":   str(uuid.uuid4()),
            "store_id":   self.store_id,
            "camera_id":  self.camera_id,
            "visitor_id": v.visitor_id,     # stable, survives REENTRY
            "event_type": event_type,
            "timestamp":  _iso(timestamp) if isinstance(timestamp, float) else timestamp,

            # ── Optional context fields ───────────────────────────────────────
            "zone_id":    zone_id,          # null for ENTRY / EXIT / REENTRY
            "dwell_ms":   dwell_ms,         # null unless ZONE_DWELL

            # ── Flags ─────────────────────────────────────────────────────────
            "is_staff":   v.is_staff,       # set by staff detection rules

            # ── Detection quality ─────────────────────────────────────────────
            "confidence": round(v.confidence, 3),

            # ── Rich metadata ─────────────────────────────────────────────────
            "metadata": {
                "session_seq": seq,          # ordering key within this visit
                "visit_count": v.visit_count,# 1 = first visit, 2+ = REENTRY
                "bbox":        v.last_bbox,  # [x1,y1,x2,y2] for spatial debug
                "on_screen_s": round(v.total_visible_seconds, 1),
            },
        }
