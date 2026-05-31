"""
pipeline/detect.py
──────────────────
Universal Computer Vision pipeline that operates in three distinct camera modes:
  1. entrance : ENTRY / EXIT events via tripwire line crossings
  2. billing  : BILLING_QUEUE_JOIN / ABANDON events via spatial bounding checks
  3. browsing  : ZONE_ENTER / EXIT / DWELL events via product aisle polygon checks

This script loads layout definitions from data/camera_config.json, with
built-in safe coordinates fallbacks if configuration or layout data is empty.
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    stream=sys.stdout,
)
log = logging.getLogger("pipeline.detect")

# ── Constants ─────────────────────────────────────────────────────────────────

PERSON_CLASS_ID = 0      # COCO class index for "person"
SIDE_ABOVE      = "ABOVE"
SIDE_BELOW      = "BELOW"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _frame_to_iso(base_time: float, frame_idx: int, fps: float) -> str:
    """Convert a frame index to an ISO timestamp anchored to the video start."""
    epoch = base_time + (frame_idx / fps)
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _foot_y(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the Y coordinate of a person's foot (bottom edge of bounding box)."""
    return y2


def _side_of_line(foot_y: float, line_y: float) -> str:
    """Return which side of the virtual line this person is on."""
    return SIDE_ABOVE if foot_y < line_y else SIDE_BELOW


def _is_point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """Check if foot point (x, y) is inside a zone polygon using OpenCV."""
    pts = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (x, y), False) >= 0


def _make_event(
    event_type: str,
    track_id: int,
    confidence: float,
    frame_iso: str,
    store_id: str,
    camera_id: str,
    zone_id: str | None = None,
    dwell_ms: int | None = None,
) -> dict:
    """Build a structured event dict matching the FastAPI ingest schema."""
    return {
        "event_id":   str(uuid.uuid4()),
        "store_id":   store_id,
        "camera_id":  camera_id,
        "visitor_id": f"visitor_{track_id:04d}",
        "event_type": event_type,
        "timestamp":  frame_iso,
        "zone_id":    zone_id,
        "dwell_ms":   dwell_ms,
        "is_staff":   False,
        "confidence": round(confidence, 3),
        "metadata": {
            "track_id": track_id,
            "source":   "tripwire" if event_type in ["ENTRY", "EXIT"] else "spatial_polygon",
        },
    }


# ── Camera Config Loader ──────────────────────────────────────────────────────

def load_camera_config(config_path: str, camera_id: str) -> dict:
    """
    Load camera configuration strictly from JSON file.
    No fallbacks or hardcoded overrides.
    """
    if not os.path.exists(config_path):
        log.error("Camera config file not found: %s", config_path)
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # Print startup logs showing loaded config mappings
            log.info("Loaded camera configuration:")
            for cam_key, cam_data in data.items():
                m = cam_data.get("mode", "unknown")
                log.info("%s -> %s", cam_key, m.upper())
                
            if camera_id in data:
                log.info("Loaded config for requested camera %s from %s", camera_id, config_path)
                return data[camera_id]
            else:
                log.error("Camera ID %s not found in configuration file %s", camera_id, config_path)
                sys.exit(1)
    except Exception as exc:
        log.error("Failed to load camera config from %s: %s", config_path, exc)
        sys.exit(1)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(
    video_path: str,
    output_path: str     = "data/events.jsonl",
    output_video_path: str = "auto",
    config_path: str     = "data/camera_config.json",
    model_path: str      = "yolov8n.pt",
    store_id: str        = "store_01",
    camera_id: str       = "cam_entrance_1",
    line_pct: float      = 0.40,
    confidence: float    = 0.35,
    skip_frames: int     = 1,
) -> list[dict]:
    """Universal pipeline that processes standard CCTV videos based on modes."""

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not Path(video_path).exists():
        log.error("Video file not found: %s", video_path)
        sys.exit(1)

    # ── Load camera config ────────────────────────────────────────────────────
    cam_config = load_camera_config(config_path, camera_id)
    mode = cam_config.get("mode", "entrance").lower()
    tripwire_enabled = cam_config.get("tripwire", False)
    log.info("Camera mode resolved: %s | Tripwire Enabled: %s", mode.upper(), tripwire_enabled)

    # Resolve output video path
    if output_video_path == "auto":
        output_video_path = f"outputs/{camera_id.lower()}_annotated.mp4"

    # ── Load YOLOv8 model ─────────────────────────────────────────────────────
    log.info("Loading model: %s", model_path)
    try:
        model = YOLO(model_path)
    except Exception as exc:
        log.error("Failed to load YOLO model: %s", exc)
        sys.exit(1)

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Overwrite line_pct if custom value resolved in config
    if "line_pct" in cam_config:
        line_pct = float(cam_config["line_pct"])
    line_y       = int(height * line_pct)

    log.info(
        "Video size: %dx%d @ %.1f FPS | Total frames: %d",
        width, height, fps, total_frames
    )
    log.info("Confidence threshold: %.2f | Skipping: every %d", confidence, skip_frames)

    # ── State tracking ────────────────────────────────────────────────────────
    # 1. Entrance Mode States
    last_side: dict[int, str]   = {}
    # 2. Billing Mode States
    in_billing_zone: dict[int, bool] = {}
    # 3. Browsing Mode States
    visitor_zones: dict[int, set[str]] = {}
    zone_enter_ts: dict[tuple[int, str], float] = {}

    last_conf: dict[int, float] = {}
    all_events: list[dict] = []
    base_time  = time.time()
    frame_idx  = 0
    processed  = 0

    log.info("Starting universal pipeline processing...")
    t_start = time.perf_counter()

    # ── Video Writer Setup ────────────────────────────────────────────────────
    out_video = None
    if output_video_path:
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        # Using 'avc1' (H.264) because it is universally supported by web browsers (Streamlit st.video)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps / skip_frames, (width, height))

    # ── Frame loop ────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames for speed
        if frame_idx % skip_frames != 0:
            continue

        processed += 1
        frame_iso = _frame_to_iso(base_time, frame_idx, fps)
        timestamp_seconds = base_time + (frame_idx / fps)

        # ── Run YOLO + ByteTrack ──────────────────────────────────────────────
        try:
            results = model.track(
                frame,
                persist=True,
                classes=[PERSON_CLASS_ID],
                conf=confidence,
                verbose=False,
                tracker="bytetrack.yaml",
            )
        except Exception as exc:
            log.warning("Frame %d: detection failed (%s), skipping.", frame_idx, exc)
            continue

        # ── Visualizing camera modes ──────────────────────────────────────────
        mode_labels = {
            "browsing": "Customer Browsing Analytics",
            "entrance": "Entrance Analytics",
            "billing": "Billing & Queue Analytics",
            "backoffice": "Backoffice Operations"
        }
        display_label = mode_labels.get(mode, mode.upper())
        
        # Draw camera mode badge at top left
        cv2.rectangle(frame, (10, 10), (450, 55), (30, 30, 30), -1)
        cv2.putText(
            frame,
            f"CAM: {camera_id} | {display_label}",
            (25, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        # Draw spatial polygons / tripwire lines on frame
        if mode == "entrance" and tripwire_enabled:
            cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
            cv2.putText(
                frame,
                "TRIPWIRE LINE",
                (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        elif mode == "billing":
            polygon = cam_config.get("polygon", [[1200, 700], [1920, 700], [1920, 1080], [1200, 1080]])
            pts = np.array(polygon, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 165, 255), 2)
            cv2.putText(
                frame,
                "BILLING QUEUE ZONE",
                (polygon[0][0], polygon[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2,
            )
        elif mode == "browsing":
            zones = cam_config.get("zones", [
                {"zone_id": "skincare", "polygon": [[0, 200], [900, 200], [900, 650], [0, 650]]},
                {"zone_id": "makeup", "polygon": [[920, 200], [1920, 200], [1920, 650], [920, 650]]}
            ])
            for zone in zones:
                pts = np.array(zone["polygon"], dtype=np.int32)
                cv2.polylines(frame, [pts], True, (255, 255, 0), 2)
                cv2.putText(
                    frame,
                    zone["zone_id"].upper(),
                    (zone["polygon"][0][0] + 10, zone["polygon"][0][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

        # ── Parse Detections ──────────────────────────────────────────────────
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            for box, tid, conf_score in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.id.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy(),
            ):
                tid  = int(tid)
                conf = float(conf_score)
                x1, y1, x2, y2 = box.tolist()
                cx = (x1 + x2) / 2.0
                fy = _foot_y(x1, y1, x2, y2)

                # Draw green bounding box around tracked shopper
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID: {tid:04d}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                last_conf[tid] = conf

                # ── Mode 1: Entrance Tripwire Crossing check ──────────────────
                if mode == "entrance" and tripwire_enabled:
                    side = _side_of_line(fy, line_y)

                    if tid not in last_side:
                        last_side[tid] = side
                        continue

                    prev_side = last_side[tid]
                    if prev_side != side:
                        if prev_side == SIDE_ABOVE and side == SIDE_BELOW:
                            event = _make_event(
                                event_type="ENTRY",
                                track_id=tid,
                                confidence=conf,
                                frame_iso=frame_iso,
                                store_id=store_id,
                                camera_id=camera_id,
                            )
                            all_events.append(event)
                            log.info("ENTRY  visitor_%04d  frame=%d", tid, frame_idx)

                        elif prev_side == SIDE_BELOW and side == SIDE_ABOVE:
                            event = _make_event(
                                event_type="EXIT",
                                track_id=tid,
                                confidence=conf,
                                frame_iso=frame_iso,
                                store_id=store_id,
                                camera_id=camera_id,
                            )
                            all_events.append(event)
                            log.info("EXIT   visitor_%04d  frame=%d", tid, frame_idx)

                    last_side[tid] = side

                # ── Mode 2: Billing Queue check ────────────────────────────────
                elif mode == "billing":
                    polygon = cam_config.get("polygon", [[1200, 700], [1920, 700], [1920, 1080], [1200, 1080]])
                    in_zone = _is_point_in_polygon(cx, fy, polygon)

                    if tid not in in_billing_zone:
                        in_billing_zone[tid] = False

                    prev_in = in_billing_zone[tid]
                    if in_zone != prev_in:
                        if in_zone:
                            event = _make_event(
                                event_type="BILLING_QUEUE_JOIN",
                                track_id=tid,
                                confidence=conf,
                                frame_iso=frame_iso,
                                store_id=store_id,
                                camera_id=camera_id,
                                zone_id="billing",
                            )
                            all_events.append(event)
                            log.info("BILLING_QUEUE_JOIN  visitor_%04d  frame=%d", tid, frame_idx)
                        else:
                            event = _make_event(
                                event_type="BILLING_QUEUE_ABANDON",
                                track_id=tid,
                                confidence=conf,
                                frame_iso=frame_iso,
                                store_id=store_id,
                                camera_id=camera_id,
                                zone_id="billing",
                            )
                            all_events.append(event)
                            log.info("BILLING_QUEUE_ABANDON  visitor_%04d  frame=%d", tid, frame_idx)
                        in_billing_zone[tid] = in_zone

                # ── Mode 3: Shelf / Browsing transition check ──────────────────
                elif mode == "browsing":
                    zones = cam_config.get("zones", [
                        {"zone_id": "skincare", "polygon": [[0, 200], [900, 200], [900, 650], [0, 650]]},
                        {"zone_id": "makeup", "polygon": [[920, 200], [1920, 200], [1920, 650], [920, 650]]}
                    ])
                    if tid not in visitor_zones:
                        visitor_zones[tid] = set()

                    current_inside = set()
                    for zone in zones:
                        if _is_point_in_polygon(cx, fy, zone["polygon"]):
                            current_inside.add(zone["zone_id"])

                    prev_inside = visitor_zones[tid]
                    entered = current_inside - prev_inside
                    exited  = prev_inside - current_inside

                    for zone_id in entered:
                        event = _make_event(
                            event_type="ZONE_ENTER",
                            track_id=tid,
                            confidence=conf,
                            frame_iso=frame_iso,
                            store_id=store_id,
                            camera_id=camera_id,
                            zone_id=zone_id,
                        )
                        all_events.append(event)
                        zone_enter_ts[(tid, zone_id)] = timestamp_seconds
                        log.info("ZONE_ENTER  visitor_%04d  zone=%s", tid, zone_id)

                    for zone_id in exited:
                        event = _make_event(
                            event_type="ZONE_EXIT",
                            track_id=tid,
                            confidence=conf,
                            frame_iso=frame_iso,
                            store_id=store_id,
                            camera_id=camera_id,
                            zone_id=zone_id,
                        )
                        all_events.append(event)
                        log.info("ZONE_EXIT   visitor_%04d  zone=%s", tid, zone_id)

                        # Emit ZONE_DWELL
                        enter_t = zone_enter_ts.pop((tid, zone_id), timestamp_seconds)
                        dwell_ms = int((timestamp_seconds - enter_t) * 1000)
                        if dwell_ms > 0:
                            dwell_event = _make_event(
                                event_type="ZONE_DWELL",
                                track_id=tid,
                                confidence=conf,
                                frame_iso=frame_iso,
                                store_id=store_id,
                                camera_id=camera_id,
                                zone_id=zone_id,
                                dwell_ms=dwell_ms,
                            )
                            all_events.append(dwell_event)
                            log.info("ZONE_DWELL  visitor_%04d  zone=%s  dwell=%dms", tid, zone_id, dwell_ms)

                    visitor_zones[tid] = current_inside
                
                # ── Mode 4: Backoffice check ───────────────────────────────────
                elif mode == "backoffice":
                    # Backoffice tracks only, emits no customer events
                    pass

        # ── Headless execution (No GUI) ───────────────────────────────────────────
        if out_video is not None:
            out_video.write(frame)

        # ── Progress log ──────────────────────────────────────────────────────
        if processed % 100 == 0:
            pct = (frame_idx / total_frames * 100) if total_frames else 0
            log.info(
                "Progress: frame %d/%d (%.0f%%) | events so far: %d",
                frame_idx, total_frames, pct, len(all_events),
            )

    cap.release()
    if out_video is not None:
        out_video.release()
    cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t_start

    # ── Write output JSONL ────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for ev in all_events:
            f.write(json.dumps(ev) + "\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    counts = {}
    for ev in all_events:
        counts[ev["event_type"]] = counts.get(ev["event_type"], 0) + 1

    log.info("=" * 55)
    log.info("Detection complete in %.1fs", elapsed)
    log.info("  Frames read      : %d", frame_idx)
    log.info("  Frames processed : %d (every %d)", processed, skip_frames)
    log.info("  Unique tracks    : %d", len(last_conf))
    log.info("  Emitted events   : %d", len(all_events))
    for etype, count in counts.items():
        log.info("    %-20s : %d", etype, count)
    log.info("  Output           : %s", out)
    log.info("=" * 55)

    return all_events


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Universal CCTV detector — supports entrance, billing, and browsing modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",     required=True,                    help="Input video file")
    p.add_argument("--output",    default="data/events.jsonl",      help="Output JSONL path")
    p.add_argument("--out-video", default="auto",                   help="Output annotated video path (defaults to outputs/cam_X_annotated.mp4)")
    p.add_argument("--config",    default="data/camera_config.json",help="Camera configuration JSON path")
    p.add_argument("--model",     default="yolov8n.pt",             help="YOLOv8 weights")
    p.add_argument("--store-id",  default="store_01",               help="Store identifier")
    p.add_argument("--camera-id", default="CAM_1",                  help="Camera identifier")
    p.add_argument("--line-pct",  type=float, default=0.40,         help="Tripwire Y position (0–1)")
    p.add_argument("--conf",      type=float, default=0.35,         help="Detection confidence threshold")
    p.add_argument("--skip",      type=int,   default=1,            help="Process every Nth frame")

    args = p.parse_args()

    run(
        video_path  = args.video,
        output_path = args.output,
        output_video_path = args.out_video,
        config_path = args.config,
        model_path  = args.model,
        store_id    = args.store_id,
        camera_id   = args.camera_id,
        line_pct    = args.line_pct,
        confidence  = args.conf,
        skip_frames = args.skip,
    )


if __name__ == "__main__":
    main()
