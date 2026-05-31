"""
pipeline/run_pipeline.py
────────────────────────
CLI entry point: video -> detection -> tracking -> events -> JSONL file.

PART 2 updates:
  - Accepts --pos flag for POS transaction correlation
  - Passes reentry_ids to event emitter
  - Prints richer summary at the end

Usage:
    python -m pipeline.run_pipeline --video data/videos/sample.mp4
    python -m pipeline.run_pipeline --video clip.mp4 --pos data/pos_transactions.json
    python -m pipeline.run_pipeline --video clip.mp4 --skip 2 --conf 0.4
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

from pipeline.tracker import PersonTracker
from pipeline.zone_logic import ZoneManager
from pipeline.event_emitter import EventEmitter, load_pos_transactions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    stream=sys.stdout,
)
log = logging.getLogger("pipeline")


def run(
    video_path: str,
    layout_path: str = "data/store_layout.json",
    output_path: str = "data/events.jsonl",
    pos_path: str | None = None,
    model_path: str = "yolov8n.pt",
    confidence: float = 0.30,
    max_lost_seconds: float = 2.0,
    skip_frames: int = 0,
    camera_id: str = "cam_01",
    staff_threshold: float = 1800.0,
) -> list[dict]:
    """Process a video end-to-end and write events to JSONL."""

    # ── Validate ──────────────────────────────────────────────────────
    if not Path(video_path).exists():
        log.error("Video not found: %s", video_path)
        sys.exit(1)
    if not Path(layout_path).exists():
        log.error("Layout not found: %s", layout_path)
        sys.exit(1)

    # ── Load components ───────────────────────────────────────────────
    zone_manager = ZoneManager(layout_path)
    log.info("Zones loaded: %s", [z["zone_id"] for z in zone_manager.zones])

    pos_txns = []
    if pos_path:
        pos_txns = load_pos_transactions(pos_path)
        log.info("POS transactions loaded: %d", len(pos_txns))

    tracker = PersonTracker(
        model_path=model_path,
        confidence=confidence,
        max_lost_seconds=max_lost_seconds,
        staff_duration_seconds=staff_threshold,
    )

    emitter = EventEmitter(
        zone_manager=zone_manager,
        store_id=zone_manager.store_id,
        camera_id=camera_id,
        pos_transactions=pos_txns,
    )

    # ── Open video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0
    log.info("Video: %d frames @ %.1f FPS (%.1fs)", total_frames, fps, duration)

    base_time = time.time()
    all_events: list[dict] = []
    frame_idx = 0
    processed = 0

    log.info("Processing started ...")
    t_start = time.perf_counter()

    # ── Frame loop ────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if skip_frames > 0 and (frame_idx % (skip_frames + 1)) != 0:
            continue

        timestamp = base_time + (frame_idx / fps)

        # Tracker returns 4 values (updated in PART 2)
        tracked, new_ids, lost_ids, reentry_ids = tracker.update(frame, timestamp)

        events = emitter.process_frame(
            tracker, tracked, new_ids, lost_ids, reentry_ids, timestamp,
        )
        all_events.extend(events)

        tracker.cleanup_old(timestamp)
        processed += 1

        if processed % 100 == 0:
            active = sum(1 for v in tracker.visitors.values() if v.is_active)
            staff  = sum(1 for v in tracker.visitors.values() if v.is_staff)
            log.info(
                "Frame %d/%d | events: %d | active: %d | staff: %d",
                frame_idx, total_frames, len(all_events), active, staff,
            )

    cap.release()

    # ── Flush remaining visitors as EXIT ──────────────────────────────
    final_ts = base_time + (frame_idx / fps)
    for tid, v in list(tracker.visitors.items()):
        if v.entry_emitted and not v.exit_emitted:
            v.last_seen = final_ts
            flush_events = emitter.process_frame(
                tracker, [], set(), {tid}, set(), final_ts,
            )
            all_events.extend(flush_events)

    elapsed = time.perf_counter() - t_start

    # ── Write JSONL ───────────────────────────────────────────────────
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for ev in all_events:
            f.write(json.dumps(ev) + "\n")

    # ── Summary ───────────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("Pipeline complete in %.1fs", elapsed)
    log.info("  Frames processed : %d / %d", processed, total_frames)
    log.info("  Total events     : %d", len(all_events))
    log.info("  Output file      : %s", output)

    counts = Counter(ev["event_type"] for ev in all_events)
    for etype in [
        "ENTRY", "EXIT", "REENTRY",
        "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL",
        "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON",
    ]:
        log.info("    %-25s : %d", etype, counts.get(etype, 0))

    staff_visitors = {
        ev["visitor_id"] for ev in all_events if ev.get("is_staff")
    }
    if staff_visitors:
        log.info("  Staff detected   : %s", staff_visitors)

    log.info("=" * 50)
    return all_events


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Store Intelligence CCTV pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",   required=True,              help="Input video file")
    p.add_argument("--layout",  default="data/store_layout.json", help="Zone layout JSON")
    p.add_argument("--output",  default="data/events.jsonl",      help="Output JSONL path")
    p.add_argument("--pos",     default=None,               help="POS transactions JSON (optional)")
    p.add_argument("--model",   default="yolov8n.pt",       help="YOLO model weights")
    p.add_argument("--conf",    type=float, default=0.30,   help="Detection confidence")
    p.add_argument("--lost",    type=float, default=2.0,    help="Seconds before EXIT")
    p.add_argument("--skip",    type=int,   default=0,      help="Skip N frames between detections")
    p.add_argument("--camera",  default="cam_01",           help="Camera ID")
    p.add_argument("--staff-threshold", type=float, default=1800.0,
                   help="Seconds of visibility before marking as staff")

    args = p.parse_args()

    run(
        video_path=args.video,
        layout_path=args.layout,
        output_path=args.output,
        pos_path=args.pos,
        model_path=args.model,
        confidence=args.conf,
        max_lost_seconds=args.lost,
        skip_frames=args.skip,
        camera_id=args.camera,
        staff_threshold=args.staff_threshold,
    )


if __name__ == "__main__":
    main()
