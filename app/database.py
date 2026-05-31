"""
app/database.py
───────────────
SQLAlchemy database layer for the Store Intelligence system.

This file sets up:
  1. The SQLAlchemy engine (SQLite)
  2. A session factory used by FastAPI dependency injection
  3. The ORM model (EventORM) that maps Python objects to DB rows
  4. A create_tables() function called at startup
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ── DB file location ──────────────────────────────────────────────────────────

_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "store_intel.db"
DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{_DEFAULT_DB_PATH}")


# ── Engine ────────────────────────────────────────────────────────────────────

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

# Enable WAL mode for better concurrent read performance with SQLite
@event.listens_for(engine, "connect")
def _set_wal_mode(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.close()


# ── Session factory ───────────────────────────────────────────────────────────

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# ── ORM base ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Event ORM model ───────────────────────────────────────────────────────────

class EventORM(Base):
    """
    Maps one row in the `events` table to a Python object.
    """

    __tablename__ = "events"

    # Identity / dedup key — UNIQUE constraint drives idempotent ingest
    event_id   = Column(String(128), primary_key=True, index=True)

    # Where / who
    store_id   = Column(String(64),  nullable=False, index=True)
    camera_id  = Column(String(64),  nullable=False)
    visitor_id = Column(String(128), nullable=False, index=True)

    # What happened
    event_type = Column(String(32),  nullable=False, index=True)
    timestamp  = Column(DateTime,    nullable=False, index=True)

    # Zone context
    zone_id    = Column(String(64),  nullable=True,  index=True)
    dwell_ms   = Column(Integer,     nullable=True)

    # Flags / scores
    is_staff   = Column(Boolean,     nullable=False, default=False)
    confidence = Column(Float,       nullable=True)

    # Extra payload stored as JSON text
    metadata_  = Column("metadata", Text, nullable=True)

    ingested_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def set_metadata(self, obj: dict | None) -> None:
        self.metadata_ = json.dumps(obj) if obj else None

    def get_metadata(self) -> dict | None:
        return json.loads(self.metadata_) if self.metadata_ else None


# ── POS Transaction ORM model ──────────────────────────────────────────────────

class POSTransactionORM(Base):
    """
    Maps one row in the `pos_transactions` table to a Python object.
    """

    __tablename__ = "pos_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(128), index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    store_id = Column(String(64), nullable=False, index=True)
    customer_number = Column(String(128), nullable=True)
    product_name = Column(String(255), nullable=True)
    brand_name = Column(String(128), nullable=True)
    qty = Column(Integer, nullable=True, default=1)
    gmv = Column(Float, nullable=True, default=0.0)
    nmv = Column(Float, nullable=True, default=0.0)
    total_amount = Column(Float, nullable=True, default=0.0)


# ── Public helpers ────────────────────────────────────────────────────────────

def create_tables() -> None:
    """Create all tables defined above if they don't already exist."""
    if DATABASE_URL.startswith("sqlite:///"):
        db_path = Path(DATABASE_URL.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a database session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
