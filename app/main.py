"""
app/main.py
───────────
Store Intelligence API  —  FastAPI application entry point.

Endpoints implemented here:
    GET  /                  → API alive check
    GET  /health            → service status + DB check + timestamp
    POST /events/ingest     → validate, store, deduplicate events

Run locally:
    uvicorn app.main:app --reload --port 8000

Swagger UI:
    http://localhost:8000/docs
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import EventORM, SessionLocal, create_tables
from app.models import Event
from app.routers import metrics, funnel, heatmap, anomalies, ingest
from app import conversion


# ── 1. LOGGING ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    stream=sys.stdout,
)
log = logging.getLogger("store_intel")


# ── 2. STARTUP / SHUTDOWN ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up Store Intelligence API …")
    create_tables()
    log.info("Database tables are ready.")
    yield
    log.info("Shutting down Store Intelligence API.")


# ── 3. FASTAPI APP ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Store Intelligence API",
    description="Ingest CCTV-derived store events and query real-time analytics.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 4. MIDDLEWARE ─────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info(
        "%s %s -> %s  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ── 5. HELPERS ────────────────────────────────────────────────────────────────

def _error(status: int, message: str, detail: str | None = None) -> JSONResponse:
    body: dict[str, Any] = {"error": message}
    if detail:
        body["detail"] = detail
    return JSONResponse(status_code=status, content=body)


def _get_db() -> Session:
    return SessionLocal()


def _event_to_orm(ev: Event) -> EventORM:
    row = EventORM(
        event_id   = ev.event_id,
        store_id   = ev.store_id,
        camera_id  = ev.camera_id,
        visitor_id = ev.visitor_id,
        event_type = ev.event_type.value,
        timestamp  = ev.timestamp,
        zone_id    = ev.zone_id,
        dwell_ms   = ev.dwell_ms,
        is_staff   = ev.is_staff,
        confidence = ev.confidence,
    )
    row.set_metadata(
        ev.metadata.model_dump(exclude_none=True) if ev.metadata else None
    )
    return row


# ── 5b. REGISTER ANALYTICS ROUTERS ──────────────────────────────────────────
app.include_router(metrics.router)
app.include_router(funnel.router)
app.include_router(heatmap.router)
app.include_router(anomalies.router)
app.include_router(conversion.router)
app.include_router(ingest.router)

# ── 6. ENDPOINTS ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Store Intelligence API",
        "version": "1.0.0",
        "status":  "running",
    }


@app.get("/health", tags=["Health"])
def health_check():
    db_status = "connected"
    db = _get_db()
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"error: {exc}"
        log.error("Health check DB error: %s", exc)
    finally:
        db.close()

    is_healthy = db_status == "connected"
    return JSONResponse(
        status_code=200 if is_healthy else 503,
        content={
            "service_status": "ok" if is_healthy else "degraded",
            "database_status": db_status,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        },
    )


# Inline /events/ingest removed in favor of app.routers.ingest.router

# ── 7. ERROR HANDLERS ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.error("Unhandled error: %s", exc)
    return _error(500, "Internal server error", str(exc))
