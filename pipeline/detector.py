"""
pipeline/detector.py
────────────────────
Loads YOLOv8n and detects people in a single video frame.

This module ONLY does detection (bounding boxes + confidence).
Tracking (assigning persistent IDs) is handled by tracker.py.

YOLOv8n auto-downloads its weights (~6 MB) on first run.
"""

from ultralytics import YOLO


class PersonDetector:
    """
    Thin wrapper around YOLOv8n that filters for the 'person' class.

    Usage:
        detector = PersonDetector()
        boxes = detector.detect(frame)
        # boxes = [{"bbox": [x1,y1,x2,y2], "confidence": 0.87}, ...]
    """

    # In the COCO dataset YOLOv8 was trained on, class 0 = "person"
    PERSON_CLASS = 0

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.30):
        """
        Args:
            model_path : YOLO weights file (auto-downloads if missing)
            confidence : minimum detection score to keep (0.0 – 1.0)
        """
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        """
        Run person detection on one BGR frame.

        Args:
            frame: numpy array (H, W, 3) from OpenCV

        Returns:
            List of dicts, one per detected person:
            {
                "bbox":       [x1, y1, x2, y2],   # pixel coordinates
                "center":     [cx, cy],            # center of the box
                "confidence": float                # 0.0 – 1.0
            }
        """
        results = self.model.predict(
            frame,
            classes=[self.PERSON_CLASS],
            conf=self.confidence,
            verbose=False,          # suppress per-frame log spam
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box, conf in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = box.tolist()
                detections.append({
                    "bbox":       [x1, y1, x2, y2],
                    "center":     [(x1 + x2) / 2, (y1 + y2) / 2],
                    "confidence": float(conf),
                })

        return detections
