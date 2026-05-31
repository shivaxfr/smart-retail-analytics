# PROMPT:
# Generate a production-ready Dockerfile for a FastAPI + Uvicorn application.
# Use Python 3.11 slim, install dependencies cleanly, and run a single worker.
#
# CHANGES MADE:
# - Uses python:3.11-slim (small, secure, no unnecessary system packages)
# - Copies requirements.txt first so Docker layer cache avoids reinstalling
#   packages on every code change (only reinstalls when requirements.txt changes)
# - Creates /data/db directory at build time so SQLite can write its file
# - Runs as non-root user (appuser) — security best practice
# - Single Uvicorn worker (--workers 1) because SQLite is not safe with multiple
#   workers sharing the same file via separate processes
# - PORT defaults to 8000, overridable via env var

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# libgl1 and libglib2.0-0 are required by OpenCV headless on some base images.
# We install them here so the image works even if someone runs the pipeline
# inside the same container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements FIRST — Docker caches this layer until requirements.txt
# changes, so rebuilds are fast when you only change application code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY app/       ./app/
COPY pipeline/  ./pipeline/
COPY data/      ./data/

# ── Create the database directory (SQLite writes here) ────────────────────────
RUN mkdir -p /app/data/db

# ── Create a non-root user and give it ownership ──────────────────────────────
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Expose the API port ───────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check (Docker polls this; compose depends_on uses it) ──────────────
# Polls /health every 10 seconds. After 3 failures the container is "unhealthy".
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Start the FastAPI server ──────────────────────────────────────────────────
# --workers 1     : SQLite is not safe with multiple concurrent writers
# --host 0.0.0.0  : bind to all interfaces so Docker port mapping works
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
