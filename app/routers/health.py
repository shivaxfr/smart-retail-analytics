"""
app/routers/health.py
─────────────────────
GET /health — liveness + readiness probe.

Returns 200 when the app is running AND can talk to the database.
Returns 503 when the DB is unreachable (so Docker / k8s can restart it).

This is the first endpoint checked by load balancers and monitoring tools.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Liveness + readiness check",
    response_description="Service status and DB connectivity",
)
def health_check(db: Session = Depends(get_db)):
    """
    Returns the overall health of the service.

    - **status**: `ok` or `degraded`
    - **db**: `connected` or an error message
    - **timestamp**: current UTC time
    """
    db_status = "connected"
    http_status = 200

    try:
        # Cheapest possible DB round-trip — just ask SQLite for the time
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"error: {exc}"
        http_status = 503

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=http_status,
        content={
            "status": "ok" if http_status == 200 else "degraded",
            "db": db_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
