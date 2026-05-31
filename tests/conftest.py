# tests/conftest.py
# ─────────────────────────────────────────────────────────────────
# Shared pytest fixtures for the Store Intelligence test suite.
#
# Root problem solved here:
#   SQLite in-memory databases are connection-scoped. When FastAPI's
#   dependency injection creates a new Session, it opens a NEW connection,
#   which sees a completely empty (tableless) database — not the one the
#   test seeded.
#
# Fix:
#   Use SQLite file-based database with a unique name per test (via tmp_path
#   or a fixed test path), so all connections share the same physical file.
#   We patch `app.database.engine` and `app.database.SessionLocal` before
#   the test runs, and restore them after.
# ─────────────────────────────────────────────────────────────────

import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.database as db_module
from app.database import Base, get_db
from app.main import app


@pytest.fixture()
def client(tmp_path):
    """
    Provides a TestClient wired to a fresh, isolated SQLite test database.

    Uses a file-based SQLite DB (in pytest's tmp_path) so that all
    connections — from the test and from FastAPI's dependency injection —
    share the same physical file and see the same tables and rows.
    """
    db_file = tmp_path / f"test_{uuid.uuid4().hex}.db"
    db_url  = f"sqlite:///{db_file}"

    test_engine = create_engine(
        db_url, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)

    # Patch the module-level engine and SessionLocal used by FastAPI
    original_engine       = db_module.engine
    original_session_local = db_module.SessionLocal
    db_module.engine       = test_engine
    db_module.SessionLocal = TestingSession

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        # Also give tests direct DB access via c.db_session attribute
        c.db_session = TestingSession()
        yield c
        c.db_session.close()

    # Restore everything
    app.dependency_overrides.clear()
    db_module.engine       = original_engine
    db_module.SessionLocal = original_session_local
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()
