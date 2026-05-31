# Technical Choices

For each major decision, here is what I considered, what I tried, and
why I landed where I did. I've tried to be honest about the tradeoffs
rather than pretending every choice was obviously correct.

---

## 1. Detection Model — YOLOv8n

**Alternatives considered**: YOLOv5s, YOLOv8s, YOLOv8m, Detectron2 Faster-RCNN

**The question**: Fast vs accurate. A challenge CCTV clip is not a research
benchmark. Missing 5% of detections is acceptable. Running at 3 FPS because
the model is too heavy is not.

YOLOv8n runs at ~30ms/frame on a modern CPU for 640×480 video. YOLOv8s is
about 2× slower with maybe 5–10% better mAP on COCO. For people counting
at a store entrance — large, upright objects in relatively consistent
lighting — the nano model is sufficient. If I were deploying this on an
NVIDIA Jetson at the edge, I'd re-evaluate with TensorRT quantization.

Detectron2/Faster-RCNN was out immediately. It's not designed for
real-time inference and requires significantly more infrastructure to run.

**The `ultralytics` package specifically**: YOLOv8 ships with ByteTrack
built in via `model.track(..., tracker="bytetrack.yaml")`. This eliminates
a separate dependency and integration layer. That convenience matters when
the goal is a complete, runnable submission.

**Honest tradeoff**: YOLOv8n will miss detections in crowded scenes or when
people are partially occluded. The tracker's `persist=True` helps bridge
short gaps, but a busy entrance during peak hours will have more false
negatives than YOLOv8m would.

---

## 2. Tracking — ByteTrack

**Alternatives considered**: DeepSORT, BoT-SORT, StrongSORT, SORT (plain Kalman)

**The question**: How do you keep the same ID for a person across frames,
especially when they're occluded or briefly leave the frame?

DeepSORT is the most commonly cited answer. It uses a ReID (Re-Identification)
neural network to compare the appearance of detections across frames. The
problem: ReID models are large (typically 25–60 MB), require a separate
inference call per detection, and need to be matched to the camera's lighting
conditions to work well. For a challenge submission, shipping a second model
and managing its inference pipeline is significant overhead.

ByteTrack uses only motion: Kalman filter state estimation + IoU matching.
No appearance model. It's faster, lighter, and — for the specific case of
people walking in relatively straight paths in a retail environment —
comparably accurate to DeepSORT in practice. The key insight from the
ByteTrack paper is that low-confidence detections (which DeepSORT would
discard) still carry useful motion information and should be retained for
association before being dropped.

**What I gave up**: Long-term ReID. If a person leaves for 10 minutes and
returns, ByteTrack will assign a new track_id. This is exactly why
`tracker.py` implements its own spatial+temporal REENTRY matching on top of
ByteTrack — it's a deliberate layer added to compensate for ByteTrack's
short-term nature.

BoT-SORT and StrongSORT are improvements over ByteTrack, but they either
require additional dependencies or a camera motion model (homography
estimation). Not worth the complexity here.

---

## 3. Database — SQLite

**Alternatives considered**: PostgreSQL, DuckDB, ClickHouse

**The honest answer**: SQLite is the right choice for a single-store,
single-process deployment where the database is on the same machine as the
API.

A store generates at most a few thousand events per day. SQLite with WAL
mode handles this with microsecond latency. Setting up PostgreSQL adds a
second Docker service, connection pooling configuration, schema migrations,
and the cognitive overhead of managing a separate process. None of that
is justified at this scale.

DuckDB is interesting for analytics queries (it's column-oriented and fast
on aggregations), but it's not designed as a transactional write database.
Mixing event ingestion writes with analytical reads in DuckDB would require
careful transaction handling. Keeping ingestion in SQLite is simpler.

**The WAL mode decision**: SQLite's default journal mode (DELETE) acquires
an exclusive write lock that blocks all readers. WAL (Write-Ahead Logging)
allows concurrent reads during writes. Since the dashboard polls every 5
seconds while events might be ingesting, WAL is essential to avoid the
dashboard seeing "database is locked" errors.

**The `--workers 1` constraint**: SQLite's WAL mode supports concurrent
reads but still serializes writes. Multiple Uvicorn workers would race on
writes and produce occasional corruption. Single worker eliminates this at
the cost of parallelism — which is fine for this event volume.

**Scale-out path**: Change `DATABASE_URL` to a PostgreSQL connection string.
The SQLAlchemy ORM layer means zero application code changes. The only work
is removing `check_same_thread=False` (SQLite-specific) and tuning the
connection pool.

---

## 4. Anomaly Detection — Rule-Based

**Alternatives considered**: Isolation Forest, LSTM-based time series,
statistical Z-score on rolling windows

**The question**: How do you detect "something is wrong" without historical
data to train on?

The fundamental problem with ML-based anomaly detection for this challenge
is the cold-start problem. Isolation Forest and LSTM models need to learn
what "normal" looks like before they can identify "abnormal". On day 1 of
deployment, you have no training data. Rule-based detection has zero
cold-start cost.

The other issue is explainability. If an ML model flags an anomaly, the
response from a store manager will be "why?". With rule-based detection,
the answer is always precise: "The billing queue currently has 8 people,
which exceeds the threshold of 5." That's actionable. "The anomaly score
is 0.73" is not.

**What I gave up**: Adaptive thresholds. A fixed threshold of 5 people in
the billing queue might be correct for a small store and too sensitive for
a large one. A rolling-window Z-score approach would adapt to each store's
baseline automatically. This is the obvious next step if this were going to
production.

The thresholds are all named constants at the top of `anomalies.py`
specifically so they're easy to find and tune without touching SQL logic.

---

## 5. Funnel Design — Session-Based with DISTINCT

**Alternatives considered**:
- Event-count based (count all ENTRY events, including repeat visits)
- Visit-session segmentation (group events into sessions by time gap)
- Raw transition matrix (track which events follow which)

**The question**: If a visitor enters the store, browses skincare, leaves,
and comes back an hour later and buys something — do they count once or
twice in the funnel?

The answer depends on what the funnel is for. If it's for measuring
marketing effectiveness (did this person convert at all today?), they count
once. If it's for measuring each visit's conversion rate, they count twice.

I chose visitor-level (once): `COUNT(DISTINCT visitor_id)`. This gives the
answer to "of all the unique people who walked in today, what fraction
bought something?" — which is the standard retail conversion metric.

**The REENTRY handling**: Because `tracker.py` reuses the original
`visitor_id` for returning visitors, a REENTRY visitor has the same
`visitor_id` as their initial ENTRY. `COUNT(DISTINCT visitor_id)` therefore
counts them exactly once at stage 1 regardless of how many times they
entered and re-entered. This is the correct behavior without any special
REENTRY logic in the funnel query.

**The monotonic cap**: The funnel applies `zone_visits = min(zone_visits, entries)`
at each step. This exists because ZONE_ENTER events can arrive at the API
before the corresponding ENTRY event if the pipeline starts mid-session or
events are ingested out of order. Without the cap, you'd see a funnel that
goes up (more zone visitors than store entries), which is visually wrong
even if mathematically explainable.

**What I gave up**: Per-visit session segmentation. Grouping events into
sessions (by time gap) would let you compute conversion per visit rather
than per person-per-day. The implementation would require either a session
ID on each event (which the current schema supports via `session_seq`) or
a gap-based clustering in SQL (window functions on SQLite, which are
available but complex). This is left as a future improvement.

---

## 6. POS Attribution — Greedy Nearest-Neighbour

**Alternatives considered**: Hungarian algorithm (optimal assignment),
probabilistic matching, manual POS terminal integration

**The approach**: For each billing zone exit, find the closest unmatched
POS transaction in time. "Consume" the match so it can't be used again.

This is O(exits × transactions) — fine for a store that does hundreds of
transactions per day, not suitable for thousands. It's also greedy, meaning
it can make suboptimal assignments in edge cases (two people leaving billing
at the same second, one receipt between them).

The Hungarian algorithm (scipy.optimize.linear_sum_assignment) would find
the globally optimal assignment but adds a dependency and significantly
more complex code. For a single-store scenario with well-separated checkout
events, the greedy approach is empirically equivalent.

**Why POS attribution at all, rather than just using BILLING_QUEUE_JOIN - BILLING_QUEUE_ABANDON?**

Camera-based "purchase" counting is a proxy. It can be wrong if the camera
misses a billing zone exit (occlusion, low confidence frame) or if someone
stands near the billing zone without intending to buy. POS transactions are
ground truth — the money actually changed hands. When both data sources are
available, POS data should always win.

The `"data_source"` field in the `/conversion` response (`"pos_attributed"`
vs `"camera_events_proxy"`) makes it explicit which method was used.
