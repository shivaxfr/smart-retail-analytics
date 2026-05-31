"""
pipeline/ingest_events.py
─────────────────────────
Reads events.jsonl produced by the pipeline and POSTs them to the
FastAPI ingest endpoint in batches.

Usage:
    # Make sure the API is running first:
    #   python -m uvicorn app.main:app --port 8000

    python -m pipeline.ingest_events
    python -m pipeline.ingest_events --file data/events.jsonl --url http://localhost:8000
    python -m pipeline.ingest_events --batch-size 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    stream=sys.stdout,
)
log = logging.getLogger("ingest")


def post_batch(events: list[dict], url: str) -> dict:
    """
    POST a batch of events to the ingest endpoint using only stdlib.
    No extra dependencies needed (no requests, no httpx).
    """
    body = json.dumps(events).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        log.error("HTTP %d: %s", e.code, error_body)
        return {"error": error_body, "inserted_count": 0, "duplicate_count": 0}
    except URLError as e:
        log.error("Connection failed: %s", e.reason)
        return {"error": str(e.reason), "inserted_count": 0, "duplicate_count": 0}


def ingest(
    jsonl_path: str = "data/events.jsonl",
    api_url: str = "http://localhost:8000",
    batch_size: int = 200,
):
    """Read a JSONL file and POST events to the API in batches."""
    path = Path(jsonl_path)
    if not path.exists():
        log.error("File not found: %s", path)
        sys.exit(1)

    # Read all events
    events = []
    for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as e:
            log.warning("Skipping line %d: %s", line_num, e)

    log.info("Loaded %d events from %s", len(events), path)

    if not events:
        log.warning("No events to ingest.")
        return

    # POST in batches
    ingest_url = f"{api_url.rstrip('/')}/events/ingest"
    total_inserted = 0
    total_duplicates = 0
    total_invalid = 0
    batch_num = 0

    for i in range(0, len(events), batch_size):
        batch = events[i : i + batch_size]
        batch_num += 1

        log.info("Sending batch %d (%d events) ...", batch_num, len(batch))
        result = post_batch(batch, ingest_url)

        inserted  = result.get("inserted_count", 0)
        dupes     = result.get("duplicate_count", 0)
        invalid   = result.get("invalid_count", 0)
        status    = result.get("status", "unknown")

        total_inserted   += inserted
        total_duplicates += dupes
        total_invalid    += invalid

        log.info(
            "  -> inserted=%d  duplicates=%d  invalid=%d  status=%s",
            inserted, dupes, invalid, status,
        )

    log.info("=" * 50)
    log.info("Ingestion complete.")
    log.info("  Total events    : %d", len(events))
    log.info("  Inserted        : %d", total_inserted)
    log.info("  Duplicates      : %d", total_duplicates)
    log.info("  Invalid         : %d", total_invalid)
    log.info("=" * 50)


def main():
    p = argparse.ArgumentParser(
        description="Ingest events.jsonl into the Store Intelligence API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file",       default="data/events.jsonl",        help="JSONL file to ingest")
    p.add_argument("--url",        default="http://localhost:8000",     help="API base URL")
    p.add_argument("--batch-size", type=int, default=200,              help="Events per batch")
    args = p.parse_args()

    ingest(jsonl_path=args.file, api_url=args.url, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
