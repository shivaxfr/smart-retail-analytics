"""
pipeline/tracker.py
───────────────────
Manages person identity across video frames using YOLOv8 + ByteTrack.

The core problem this file solves
───────────────────────────────────
ByteTrack assigns a numeric `track_id` per continuous detection segment.
When a person walks behind a shelf (occlusion) or exits and re-enters,
ByteTrack may assign a NEW track_id even though it is the same person.

This class sits on top of ByteTrack and adds two layers of intelligence:

  1. REENTRY RESOLUTION
     If a person with a known appearance exits and a new track_id appears
     within a configurable time window, we assume it is the same person
     and reuse their original visitor_id. This emits REENTRY instead of
     a fresh ENTRY.

     Implementation strategy (rule-based, no ReID network needed):
     - We keep a "recently exited" table mapping (exit_position, exit_time)
     - For each new track, we check if there is a recently exited person
       whose last known position is spatially close AND exit time is recent
     - If yes → reuse visitor_id and flag as REENTRY
     - If no  → new visitor

  2. STAFF DETECTION
     Two independent rules, both set is_staff=True:
     A. Duration rule   : anyone visible for > staff_duration_seconds
                          (default 30 minutes). A customer does not browse
                          for 30 minutes without ever leaving.
     B. Frequency rule  : anyone who appears > staff_visit_threshold times
                          in the same session day. Staff open/close stores
                          and will appear many times vs a customer who visits
                          once or twice.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ultralytics import YOLO

log = logging.getLogger("pipeline.tracker")


# ── Per-visitor state ──────────────────────────────────────────────────────────

@dataclass
class VisitorState:
    """Everything the system knows about one tracked person."""

    # Identity
    visitor_id:     str           # stable "visitor_0042" string
    track_id:       int           # current ByteTrack integer ID

    # Timing
    first_seen:     float         # epoch seconds (absolute time)
    last_seen:      float         # updated every frame this person is visible
    last_bbox:      list[float]   # [x1, y1, x2, y2] — last known position
    last_center:    list[float]   # [cx, cy] — foot of bounding box

    # Detection quality
    confidence:     float = 0.0

    # Activity flags
    is_active:      bool  = True   # currently visible in the frame
    is_staff:       bool  = False  # flagged by staff detection rules
    entry_emitted:  bool  = False  # has the ENTRY event been emitted?
    exit_emitted:   bool  = False  # has the EXIT event been emitted?

    # Zone tracking (for zone_logic.py integration)
    current_zones:  set   = field(default_factory=set)
    zone_enter_ts:  dict  = field(default_factory=dict)  # {zone_id: epoch}

    # REENTRY / visit counting
    visit_count:    int   = 0      # incremented on each ENTRY + REENTRY
    session_seq:    int   = 0      # monotonically increasing event counter

    # Staff detection counters
    total_visible_seconds: float = 0.0   # accumulated on-screen time
    appearance_count:      int   = 0     # how many times they were "re-seen"


# ── Recently-exited record (for REENTRY matching) ─────────────────────────────

@dataclass
class ExitedRecord:
    """Snapshot of a visitor at their exit moment, used for REENTRY matching."""
    visitor_id:     str
    exit_time:      float          # epoch seconds
    exit_center:    list[float]    # [cx, cy] at moment of exit
    visit_count:    int
    total_visible_seconds: float
    appearance_count:      int


# ── Tracker ────────────────────────────────────────────────────────────────────

class PersonTracker:
    """
    Wraps YOLOv8 + ByteTrack and adds stable visitor IDs, REENTRY detection,
    and staff flagging.

    Public interface
    ────────────────
    tracker.update(frame, timestamp)
        → (tracked, new_ids, lost_ids, reentry_ids)

    tracked     : list[VisitorState] — all active visitors this frame
    new_ids     : set[int]  — visitor IDs first seen for the first time
    lost_ids    : set[int]  — visitor IDs that just went over max_lost threshold
    reentry_ids : set[int]  — track_ids that are actually returning visitors
    """

    PERSON_CLASS = 0  # COCO class 0 = person

    def __init__(
        self,
        model_path: str              = "yolov8n.pt",
        confidence: float            = 0.35,
        # REENTRY: how many seconds after exit a person can return and be
        # matched as the same person (not a new visitor)
        reentry_window_seconds: float = 120.0,
        # REENTRY: how close (pixels) the new detection must be to the last
        # known exit position to be considered the same person
        reentry_distance_px: float   = 200.0,
        # EXIT: how many seconds without detection before we declare EXIT
        max_lost_seconds: float      = 3.0,
        # STAFF rule A: continuous visible duration threshold (seconds)
        staff_duration_seconds: float = 1800.0,   # 30 minutes
        # STAFF rule B: how many appearances before we flag as staff
        staff_visit_threshold: int    = 5,
    ):
        self.model              = YOLO(model_path)
        self.confidence         = confidence
        self.reentry_window     = reentry_window_seconds
        self.reentry_dist_px    = reentry_distance_px
        self.max_lost_seconds   = max_lost_seconds
        self.staff_duration     = staff_duration_seconds
        self.staff_visits       = staff_visit_threshold

        # Active visitors: {track_id: VisitorState}
        self.visitors: dict[int, VisitorState] = {}

        # Visitors who exited recently (for REENTRY matching)
        self._exited: list[ExitedRecord] = []

        # Visitor ID counter (never reused)
        self._id_counter = 0

    # ── Public: update one frame ───────────────────────────────────────────────

    def update(
        self,
        frame,
        timestamp: float,
    ) -> tuple[list[VisitorState], set[int], set[int], set[int]]:
        """
        Process one video frame.

        Returns
        ───────
        tracked     : list[VisitorState]  all currently visible visitors
        new_ids     : set[int]  track_ids seeing their first frame → ENTRY
        lost_ids    : set[int]  track_ids gone too long → EXIT
        reentry_ids : set[int]  track_ids that are REENTRY, not fresh ENTRY
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=[self.PERSON_CLASS],
            conf=self.confidence,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        seen_track_ids: set[int] = set()
        new_ids:        set[int] = set()
        reentry_ids:    set[int] = set()
        tracked:  list[VisitorState] = []

        # ── Parse detections ──────────────────────────────────────────────────
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            for box, tid, conf in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.id.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy(),
            ):
                tid   = int(tid)
                conf  = float(conf)
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                cx    = (x1 + x2) / 2.0
                cy    = y2          # foot position (bottom center)

                seen_track_ids.add(tid)

                if tid in self.visitors:
                    # Known visitor — update their state
                    v = self.visitors[tid]
                    dt = timestamp - v.last_seen
                    v.total_visible_seconds += max(0.0, dt)
                    v.last_seen   = timestamp
                    v.last_bbox   = [x1, y1, x2, y2]
                    v.last_center = [cx, cy]
                    v.confidence  = conf
                    v.is_active   = True

                    # REENTRY check: was this person recently marked as exited?
                    if v.exit_emitted:
                        v.exit_emitted  = False
                        v.visit_count  += 1
                        v.appearance_count += 1
                        reentry_ids.add(tid)

                else:
                    # New ByteTrack ID — check if this is a REENTRY
                    matched_exit = self._match_reentry(cx, cy, timestamp)

                    if matched_exit:
                        # Reuse the old visitor's identity
                        v = VisitorState(
                            visitor_id    = matched_exit.visitor_id,
                            track_id      = tid,
                            first_seen    = timestamp,
                            last_seen     = timestamp,
                            last_bbox     = [x1, y1, x2, y2],
                            last_center   = [cx, cy],
                            confidence    = conf,
                            is_active     = True,
                            entry_emitted = True,   # they had an ENTRY before
                            visit_count   = matched_exit.visit_count + 1,
                            total_visible_seconds = matched_exit.total_visible_seconds,
                            appearance_count      = matched_exit.appearance_count + 1,
                        )
                        self.visitors[tid] = v
                        reentry_ids.add(tid)
                        log.info(
                            "REENTRY matched: %s (new track_id=%d)",
                            matched_exit.visitor_id, tid,
                        )
                    else:
                        # Genuinely new visitor
                        self._id_counter += 1
                        v = VisitorState(
                            visitor_id    = f"visitor_{self._id_counter:04d}",
                            track_id      = tid,
                            first_seen    = timestamp,
                            last_seen     = timestamp,
                            last_bbox     = [x1, y1, x2, y2],
                            last_center   = [cx, cy],
                            confidence    = conf,
                            is_active     = True,
                            visit_count   = 1,
                            appearance_count = 1,
                        )
                        self.visitors[tid] = v
                        new_ids.add(tid)

                # ── Staff detection ───────────────────────────────────────────
                if not v.is_staff:
                    # Rule A: duration
                    if v.total_visible_seconds > self.staff_duration:
                        v.is_staff = True
                        log.info(
                            "Staff flagged (duration): %s — %.0fs on screen",
                            v.visitor_id, v.total_visible_seconds,
                        )
                    # Rule B: frequency
                    elif v.appearance_count > self.staff_visits:
                        v.is_staff = True
                        log.info(
                            "Staff flagged (frequency): %s — %d appearances",
                            v.visitor_id, v.appearance_count,
                        )

                tracked.append(v)

        # ── Detect lost visitors → EXIT ────────────────────────────────────────
        lost_ids: set[int] = set()
        for tid, v in self.visitors.items():
            if tid in seen_track_ids:
                continue
            if v.exit_emitted:
                continue
            if (timestamp - v.last_seen) >= self.max_lost_seconds:
                v.is_active = False
                lost_ids.add(tid)

        return tracked, new_ids, lost_ids, reentry_ids

    # ── REENTRY matching ───────────────────────────────────────────────────────

    def _match_reentry(
        self, cx: float, cy: float, now: float
    ) -> Optional[ExitedRecord]:
        """
        Check if a new detection at (cx, cy) at time `now` matches any
        recently exited visitor within the time and distance windows.

        Returns the matching ExitedRecord, or None.
        """
        best: Optional[ExitedRecord] = None
        best_dist = float("inf")

        for rec in self._exited:
            time_since_exit = now - rec.exit_time
            if time_since_exit > self.reentry_window:
                continue   # too long ago

            dx   = cx - rec.exit_center[0]
            dy   = cy - rec.exit_center[1]
            dist = (dx**2 + dy**2) ** 0.5
            if dist < self.reentry_dist_px and dist < best_dist:
                best      = rec
                best_dist = dist

        if best:
            self._exited.remove(best)   # consume the match
        return best

    # ── Mark helpers (called by emit.py) ──────────────────────────────────────

    def mark_entry_emitted(self, track_id: int) -> None:
        if track_id in self.visitors:
            self.visitors[track_id].entry_emitted = True

    def mark_exit_emitted(self, track_id: int) -> None:
        v = self.visitors.get(track_id)
        if v:
            v.exit_emitted = True
            # Save to exited list for REENTRY matching
            self._exited.append(ExitedRecord(
                visitor_id    = v.visitor_id,
                exit_time     = v.last_seen,
                exit_center   = list(v.last_center),
                visit_count   = v.visit_count,
                total_visible_seconds = v.total_visible_seconds,
                appearance_count      = v.appearance_count,
            ))

    def next_seq(self, track_id: int) -> int:
        """Increment and return the session sequence counter for a visitor."""
        if track_id in self.visitors:
            self.visitors[track_id].session_seq += 1
            return self.visitors[track_id].session_seq
        return 0

    def cleanup_old(self, now: float, max_age: float = 300.0) -> None:
        """
        Remove exited visitors older than max_age seconds from memory.
        Also prune the reentry candidates that are past the reentry window.
        """
        # Prune old reentry candidates
        self._exited = [
            r for r in self._exited
            if (now - r.exit_time) <= self.reentry_window
        ]
        # Remove fully exited, aged-out visitors from main dict
        stale = [
            tid for tid, v in self.visitors.items()
            if v.exit_emitted and (now - v.last_seen) > max_age
        ]
        for tid in stale:
            del self.visitors[tid]
