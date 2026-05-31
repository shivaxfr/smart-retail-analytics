"""
app/models.py
─────────────
Pydantic v2 schemas for the Store Intelligence event system.

These models define:
  - EventType   : all supported event kinds
  - EventMetadata : optional extra context per event
  - Event       : the canonical event that travels through the whole system

Pydantic validates every field automatically when you construct these objects,
so bad data is rejected before it ever touches the database.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── 1. Event type enum ────────────────────────────────────────────────────────

class EventType(str, Enum):
    """
    All event kinds the system recognises.
    Using `str` as a mixin means the values serialise to plain strings
    (e.g. "ENTRY") instead of numbers, which makes JSON much easier to read.
    """
    ENTRY                  = "ENTRY"
    EXIT                   = "EXIT"
    ZONE_ENTER             = "ZONE_ENTER"
    ZONE_EXIT              = "ZONE_EXIT"
    ZONE_DWELL             = "ZONE_DWELL"
    BILLING_QUEUE_JOIN     = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON  = "BILLING_QUEUE_ABANDON"
    REENTRY                = "REENTRY"


# ── 2. Optional metadata payload ─────────────────────────────────────────────

class EventMetadata(BaseModel):
    """
    Optional extra context attached to an event.

    - queue_depth : how many people are in the billing queue right now
    - sku_zone    : which product zone the visitor is in (e.g. "skincare")
    - session_seq : sequence number of this event within a single visit session
    """
    queue_depth: Optional[int]   = Field(None, ge=0, description="People in billing queue")
    sku_zone:    Optional[str]   = Field(None, max_length=64, description="Product zone label")
    session_seq: Optional[int]   = Field(None, ge=0, description="Event sequence within a visit")

    model_config = {"extra": "allow"}   # allow unknown keys so the schema stays forward-compatible


# ── 3. Core event model ───────────────────────────────────────────────────────

class Event(BaseModel):
    """
    The canonical event schema — every event in the system must match this.

    Required fields
    ───────────────
    event_id   : globally-unique identifier; used for idempotent ingestion
    store_id   : which store this event came from
    camera_id  : which camera captured it
    visitor_id : anonymised ID assigned by the tracker
    event_type : one of EventType
    timestamp  : when the event occurred (ISO-8601, UTC preferred)

    Optional fields
    ───────────────
    zone_id    : zone label (required for ZONE_* events, optional otherwise)
    dwell_ms   : milliseconds spent in a zone (required for ZONE_DWELL)
    is_staff   : True if the visitor is a staff member (default False)
    confidence : detector confidence score 0.0–1.0
    metadata   : EventMetadata bag for extra context
    """

    event_id:   str       = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique event ID")
    store_id:   str       = Field(..., min_length=1, max_length=64)
    camera_id:  str       = Field(..., min_length=1, max_length=64)
    visitor_id: str       = Field(..., min_length=1, max_length=128)
    event_type: EventType
    timestamp:  datetime  = Field(default_factory=lambda: datetime.now(timezone.utc))

    zone_id:    Optional[str]   = Field(None, max_length=64)
    dwell_ms:   Optional[int]   = Field(None, ge=0, description="Dwell time in milliseconds")
    is_staff:   bool            = Field(False)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata:   Optional[EventMetadata] = None

    # ── validators ────────────────────────────────────────────────────────────

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        """Accept ISO-8601 strings, attach UTC if naive."""
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        return v

    @model_validator(mode="after")
    def zone_events_need_zone_id(self) -> "Event":
        """ZONE_* events must carry a zone_id — catch this before DB write."""
        zone_events = {
            EventType.ZONE_ENTER,
            EventType.ZONE_EXIT,
            EventType.ZONE_DWELL,
        }
        if self.event_type in zone_events and not self.zone_id:
            raise ValueError(f"zone_id is required for event_type={self.event_type.value}")
        if self.event_type == EventType.ZONE_DWELL and self.dwell_ms is None:
            raise ValueError("dwell_ms is required for event_type=ZONE_DWELL")
        return self


# ── 4. Ingestion Models ───────────────────────────────────────────────────────

class EventBatch(BaseModel):
    events: list[Event]


class ErrorDetail(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class IngestResponse(BaseModel):
    accepted: int
    duplicates: int
    errors: int
    event_ids: list[str]
    # Compatibility with older ingest responses
    status: str
    received_count: int
    inserted_count: int
    duplicate_count: int
    invalid_count: int

