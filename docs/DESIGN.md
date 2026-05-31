# Design Notes

This document explains the architecture of the Store Intelligence system —
how the components fit together, why they're structured this way, and
where the seams are.

---

## The Core Constraint

CCTV processing is blocking, CPU-heavy work. OpenCV's `cap.read()` blocks
until a frame is available. YOLO inference takes 20–80ms per frame on CPU.
If you try to run this inside a FastAPI endpoint, the event loop stalls,
every API request queues behind the inference, and the dashboard becomes
unusable.

The architecture exists to solve this: **keep OpenCV and YOLO completely
out of the API process.**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  OFFLINE: Detection Pipeline  (pipeline/)                           │
│                                                                     │
│   Video File                                                        │
│      │                                                              │
│      ▼                                                              │
│   detect.py ──── OpenCV frame reader                               │
│      │            + YOLOv8n inference                              │
│      │            + ByteTrack person IDs                           │
│      ▼                                                              │
│   tracker.py ─── VisitorState per track_id                         │
│      │            REENTRY matching (spatial + temporal)            │
│      │            Staff detection (duration + frequency)           │
│      ▼                                                              │
│   emit.py ────── VisitorState changes → event dicts                │
│      │            ENTRY / EXIT / REENTRY                           │
│      │            ZONE_ENTER / ZONE_EXIT / ZONE_DWELL              │
│      ▼                                                              │
│   events.jsonl   (one event per line, UTF-8)                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │  POST /events/ingest
                              │  (batch, idempotent)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ONLINE: FastAPI Backend  (app/)                                    │
│                                                                     │
│   POST /events/ingest                                               │
│      │  Pydantic validation                                         │
│      │  IntegrityError dedup (event_id PRIMARY KEY)                │
│      │  SQLAlchemy → SQLite                                         │
│      ▼                                                              │
│   events table (SQLite, WAL mode)                                   │
│      │                                                              │
│      ├── GET /stores/{id}/metrics     ─── SQL aggregations         │
│      ├── GET /stores/{id}/funnel      ─── COUNT(DISTINCT)          │
│      ├── GET /stores/{id}/heatmap     ─── GROUP BY zone_id         │
│      ├── GET /stores/{id}/anomalies   ─── threshold checks         │
│      └── GET /stores/{id}/conversion  ─── POS file attribution     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │  HTTP polling (5s interval)
                              ▼
┌──────────────────────────────────────────┐
│  Streamlit Dashboard  (dashboard/)       │
│  Visitors / Conversion / Queue / Revenue │
│  Funnel bars / Zone heatmap / Anomalies  │
└──────────────────────────────────────────┘
```

---

## Data Flow

### 1. Video → Raw Detections

`detect.py` opens the video with OpenCV and calls `model.track()` on each
frame. YOLOv8 runs detection (what is in this frame?) and ByteTrack runs
tracking (which object from last frame is this?). The output is a list of
`[x1, y1, x2, y2, track_id, confidence]` tuples per frame.

### 2. Raw Detections → VisitorState

`tracker.py` maintains a `VisitorState` object per `track_id`. Each
`update()` call:
- Updates `last_seen`, `last_center`, `total_visible_seconds`
- Detects new track_ids (fresh `ENTRY`) vs returning ones (`REENTRY`)
- Flags staff based on duration or visit frequency
- Pushes recently-exited visitors to the REENTRY candidate pool

### 3. VisitorState → Events

`emit.py` takes the tracker output and translates state changes into event
dicts. The emitter owns zone transition logic: it compares the visitor's
current zone membership against the previous frame's membership, emits
`ZONE_ENTER` for new zones and `ZONE_EXIT` + `ZONE_DWELL` for departed zones.

This separation is deliberate. The tracker does not know what an "event"
looks like. The emitter does not know how tracking works.

### 4. Events → JSONL File

All events are accumulated in memory during the run and written to a
`.jsonl` file at the end. One JSON object per line. UTF-8 encoding.

### 5. JSONL → API

`ingest_events.py` reads the file and POSTs batches of up to 500 events
to `POST /events/ingest`. The API validates each event with Pydantic,
writes valid ones to SQLite, and silently skips duplicates via
`IntegrityError` on the `event_id` PRIMARY KEY.

### 6. SQLite → Analytics

Each analytics endpoint runs parameterized SQL queries against the
`events` table. All queries filter `is_staff = 0`. Results are computed
live on each request — no materialized views, no caching.

---

## Detection Pipeline Detail

### Entrance Camera: Tripwire

For cameras positioned at store entrances, `detect.py` uses a virtual
horizontal line at a configurable height (default: 40% from the top).

```
  y=0  ──────────────────────────────────────  (top of frame)
       │                                     │
       │         (outside / street)          │
       │                                     │
  y=40%│ ─────────────────────────────────── │  ← tripwire line
       │                                     │
       │         (inside / store)            │
       │                                     │
  y=100%─────────────────────────────────────  (bottom of frame)
```

Each tracked person's foot position (bottom edge of bounding box) is
checked against the line each frame. When the side changes:
- ABOVE → BELOW: `ENTRY`
- BELOW → ABOVE: `EXIT`

The foot position is used instead of the box center because the head
bobs during walking and can cause false crossings.

### Interior Camera: Zone Polygons

For interior cameras, `zone_logic.py` loads polygon definitions from
`data/store_layout.json` and uses ray-casting to test whether a person's
foot point falls inside each polygon. Zone transitions generate
`ZONE_ENTER`, `ZONE_EXIT`, and `ZONE_DWELL` events.

### REENTRY Detection

When a `track_id` goes unseen for `max_lost_seconds` (default: 3s),
we emit `EXIT` and store an `ExitedRecord` containing the visitor's ID,
exit time, and last known position.

When a new `track_id` appears, before assigning a fresh `visitor_id`,
we check the exit pool: if the new detection is within 200px of an exited
visitor's last position AND the exit was less than 120 seconds ago, we
reuse the original `visitor_id` and emit `REENTRY` instead of `ENTRY`.

This handles the case where a person steps outside briefly (takes a call,
holds the door) and re-enters. ByteTrack will assign a new `track_id`
because the person was gone for more than a few frames. Our layer above
ByteTrack recovers the original identity.

### Staff Detection

Two independent rules, either one can flag `is_staff = True`:

**Rule A — Duration**: Total on-screen time exceeds 30 minutes. A shopper
doesn't spend 30 continuous minutes in a single store without leaving.

**Rule B — Frequency**: Appearance count exceeds 5 visits. Staff open the
store, restock shelves, close out the register — they appear many times
per day through the same entrance. A customer visits once, maybe twice.

Once flagged, `is_staff = True` is written into every subsequent event
for that visitor. The API's `AND is_staff = 0` filter then excludes them
from every metric automatically.

---

## Event Lifecycle

```
Detection  →  emit.py builds dict  →  JSONL file  →  POST to API
               │
               ├── event_id    : UUID (idempotency key)
               ├── visitor_id  : "visitor_0042" (stable across REENTRY)
               ├── event_type  : one of 8 types
               ├── timestamp   : ISO-8601 UTC from frame position
               ├── zone_id     : null for ENTRY/EXIT, zone name for ZONE_*
               ├── dwell_ms    : null unless ZONE_DWELL
               ├── is_staff    : set by staff detection, inherited by later events
               ├── confidence  : YOLO detection confidence score
               └── metadata
                    ├── session_seq : ordering key per visitor per visit
                    ├── visit_count : 1=first, 2+=REENTRY
                    └── bbox        : [x1,y1,x2,y2] for spatial debugging
```

`session_seq` exists specifically because events are batched and may arrive
at the API out of wall-clock order. Any consumer can reconstruct the exact
visitor path with `ORDER BY visitor_id, session_seq`.

---

## API Architecture

### Why synchronous endpoints

All endpoints are synchronous (`def`, not `async def`) because they run
SQL queries. SQLAlchemy's synchronous ORM would require wrapping in
`run_in_executor` inside an async function, which adds complexity for no
throughput benefit at this event volume. With a single SQLite database and
a single Uvicorn worker, synchronous is the right choice.

### Idempotent ingestion

The `event_id` column is the PRIMARY KEY of the events table. SQLite raises
`IntegrityError` on duplicate inserts. The ingest endpoint catches this per
row, increments `duplicate_count`, and continues processing the rest of the
batch. The client can re-POST the same JSONL file safely at any time.

### Staff exclusion

`AND is_staff = 0` appears in every SQL query in every analytics endpoint.
This is intentional redundancy — it means a staff detection bug in the
pipeline cannot silently corrupt the metrics. If staff events somehow reach
the DB, the API filters them out without any code change.

---

## AI-Assisted Decisions

The following design decisions were informed by testing multiple approaches
with AI assistance:

**Fixture architecture in tests**: The original fixtures used in-memory
SQLite databases (`sqlite:///:memory:`). The tests failed because each
`Session()` opened a separate connection, and SQLite in-memory databases
are connection-scoped — the API's DI session could not see tables created
by the test's session. The fix (patching `app.database.engine` at the
module level and using file-based databases per test) was identified by
tracing the exact error path: `no such table: events`.

**Funnel monotonic cap**: The funnel query returned raw counts which,
in the presence of out-of-order events or partial ingestion, could
produce a stage with a higher count than the previous stage. The fix
`zone_visits = min(zone_visits, entries)` at each step ensures the funnel
is always visually sensible regardless of data quality.

**POS attribution strategy**: A one-to-one greedy match (nearest neighbour
by time) was chosen over a many-to-one approach. The reasoning: in a
physical store, one billing event maps to one POS transaction. A greedy
match that consumes each transaction once models this correctly. A more
complex assignment algorithm (e.g. Hungarian algorithm) would be harder
to explain and adds no accuracy benefit for a single-store scenario.
