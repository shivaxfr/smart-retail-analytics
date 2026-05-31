"""
app/routers/ingest.py
─────────────────────
POST /events/ingest — the single entry point for all store events.

Key behaviours
──────────────
1. Accepts a JSON body that is EITHER:
     - a single Event object, OR
     - {"events": [ ... ]}  (an EventBatch of up to 500 events)

2. Validates every event with Pydantic BEFORE touching the database.
   Bad events are counted as errors and skipped — the rest still ingest.

3. Idempotency: if an event_id already exists in the DB the row is NOT
   updated; it is silently counted as a duplicate and skipped.
   This means callers can safely retry without creating ghost data.

4. Returns a structured IngestResponse so callers always know
   exactly what happened to each event they sent.
"""

import logging
from typing import Union

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import EventORM, get_db
from app.models import Event, EventBatch, ErrorDetail, IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["Ingest"])


# ── helper: convert one validated Event → EventORM row ───────────────────────

def _event_to_orm(event: Event) -> EventORM:
    """Map a validated Pydantic Event to a SQLAlchemy ORM row."""
    row = EventORM(
        event_id=event.event_id,
        store_id=event.store_id,
        camera_id=event.camera_id,
        visitor_id=event.visitor_id,
        event_type=event.event_type.value,   # store the string, not the enum
        timestamp=event.timestamp,
        zone_id=event.zone_id,
        dwell_ms=event.dwell_ms,
        is_staff=event.is_staff,
        confidence=event.confidence,
    )
    # metadata is a dict → JSON string
    row.set_metadata(
        event.metadata.model_dump(exclude_none=True) if event.metadata else None
    )
    return row


# ── main ingest endpoint ──────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=200,
    summary="Ingest one or many store events",
    responses={
        422: {"model": ErrorDetail, "description": "Validation error"},
        500: {"model": ErrorDetail, "description": "Server error"},
    },
)
async def ingest_events(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Ingest store events into the system.

    **Accepts two body shapes:**

    *Single event:*
    ```json
    { "store_id": "s1", "camera_id": "c1", "visitor_id": "v1", "event_type": "ENTRY" }
    ```

    *Batch of events:*
    ```json
    { "events": [ {...}, {...} ] }
    ```

    **Idempotency:** sending the same `event_id` twice is safe — the second
    call is counted as a duplicate and ignored.

    **Partial success:** if some events fail validation, the valid ones are
    still ingested. Check `errors` in the response.
    """
    # ── 1. Parse raw body (we do this manually to support both shapes) ────────
    try:
        raw = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content=ErrorDetail(
                error="Invalid JSON",
                detail="Request body is not valid JSON",
                code="PARSE_ERROR",
            ).model_dump(),
        )

    # ── 2. Normalise to a list of raw dicts ───────────────────────────────────
    if isinstance(raw, dict) and "events" in raw:
        # Batch shape: {"events": [...]}
        raw_events = raw["events"]
    elif isinstance(raw, dict):
        # Single event shape: {...}
        raw_events = [raw]
    elif isinstance(raw, list):
        # Plain list shape: [{...}, {...}]
        raw_events = raw
    else:
        return JSONResponse(
            status_code=400,
            content=ErrorDetail(
                error="Unexpected body format",
                detail="Send a single event object, a list of events, or {\"events\": [...]}",
                code="FORMAT_ERROR",
            ).model_dump(),
        )

    if not raw_events:
        return JSONResponse(
            status_code=400,
            content=ErrorDetail(
                error="Empty list",
                detail="The events array must not be empty.",
                code="EMPTY_BATCH",
            ).model_dump(),
        )

    if len(raw_events) > 500:
        return JSONResponse(
            status_code=400,
            content=ErrorDetail(
                error="Batch too large",
                detail="Maximum 500 events per request",
                code="BATCH_TOO_LARGE",
            ).model_dump(),
        )


    # ── 3. Validate each event with Pydantic ──────────────────────────────────
    valid_events: list[Event] = []
    error_count = 0

    for idx, raw_event in enumerate(raw_events):
        try:
            valid_events.append(Event.model_validate(raw_event))
        except ValidationError as exc:
            error_count += 1
            logger.warning(
                "Validation failed for event at index %d: %s",
                idx,
                exc.errors(include_url=False),
            )

    # ── 4. Write valid events to DB (handle duplicates per row) ───────────────
    accepted = 0
    duplicates = 0
    accepted_ids: list[str] = []

    for event in valid_events:
        row = _event_to_orm(event)
        try:
            db.add(row)
            db.flush()          # flush so IntegrityError fires NOW, not at commit
            accepted += 1
            accepted_ids.append(event.event_id)
        except IntegrityError:
            # PRIMARY KEY clash → duplicate event_id
            db.rollback()
            duplicates += 1
            logger.debug("Duplicate event_id skipped: %s", event.event_id)
        except Exception as exc:
            db.rollback()
            error_count += 1
            logger.error("Unexpected DB error for event %s: %s", event.event_id, exc)

    # Commit all accepted events in one transaction
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("Commit failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content=ErrorDetail(
                error="Database commit failed",
                detail=str(exc),
                code="DB_ERROR",
            ).model_dump(),
        )

    logger.info(
        "Ingest complete — accepted=%d duplicates=%d errors=%d",
        accepted, duplicates, error_count,
    )

    received_count = len(raw_events)
    if accepted == received_count:
        status = "ok"
    elif accepted == 0 and duplicates == 0:
        status = "rejected"
    else:
        status = "partial"

    return IngestResponse(
        accepted=accepted,
        duplicates=duplicates,
        errors=error_count,
        event_ids=accepted_ids,
        status=status,
        received_count=received_count,
        inserted_count=accepted,
        duplicate_count=duplicates,
        invalid_count=error_count,
    )

