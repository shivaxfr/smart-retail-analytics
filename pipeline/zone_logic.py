"""
pipeline/zone_logic.py
──────────────────────
Rule-based zone assignment using point-in-polygon checks.

Each zone in the store layout is defined as a polygon (list of [x, y] points).
For every tracked person, we check which zone(s) their center point falls
inside, then detect transitions (entering / exiting a zone).

No ML here — just geometry.
"""

import json
from pathlib import Path


class ZoneManager:
    """
    Loads zones from a JSON layout file and checks person positions
    against those zones.

    Usage:
        zm = ZoneManager("data/store_layout.json")
        zones = zm.get_zones_for_point(cx, cy)
        # zones = ["entrance", "skincare"]
    """

    def __init__(self, layout_path: str):
        """
        Args:
            layout_path: path to the store_layout.json file
        """
        with open(layout_path, "r") as f:
            config = json.load(f)

        self.store_id     = config.get("store_id", "unknown_store")
        self.frame_width  = config.get("frame_width", 1920)
        self.frame_height = config.get("frame_height", 1080)
        self.zones        = config.get("zones", [])

    # ── public API ────────────────────────────────────────────────────────

    def get_zones_for_point(self, cx: float, cy: float) -> list[str]:
        """
        Return all zone_ids whose polygon contains the point (cx, cy).

        A person can be in multiple overlapping zones simultaneously.
        """
        hits = []
        for zone in self.zones:
            if self._point_in_polygon(cx, cy, zone["polygon"]):
                hits.append(zone["zone_id"])
        return hits

    def get_zone_type(self, zone_id: str) -> str:
        """Return the zone_type ('entrance', 'product', 'billing') for a zone."""
        for zone in self.zones:
            if zone["zone_id"] == zone_id:
                return zone.get("zone_type", "unknown")
        return "unknown"

    def is_entrance_zone(self, zone_id: str) -> bool:
        return self.get_zone_type(zone_id) == "entrance"

    def is_billing_zone(self, zone_id: str) -> bool:
        return self.get_zone_type(zone_id) == "billing"

    # ── geometry ──────────────────────────────────────────────────────────

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
        """
        Ray-casting algorithm for point-in-polygon.

        Shoots a horizontal ray from (x, y) to the right and counts how
        many polygon edges it crosses.  Odd count = inside, even = outside.

        This is the standard algorithm used in GIS — no external libraries.
        """
        n = len(polygon)
        inside = False
        j = n - 1

        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            # Does the edge cross the horizontal line at height y?
            if (yi > y) != (yj > y):
                # X-coordinate where the edge crosses that line
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_cross:
                    inside = not inside

            j = i

        return inside
