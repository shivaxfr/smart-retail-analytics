"""
dashboard/app.py
────────────────
Streamlit live dashboard for the Store Intelligence system.
Redesigned as a Professional Retail Intelligence Center.

Run:
    streamlit run dashboard/app.py
"""

import os
import time
import urllib.request
import urllib.error
import json
from datetime import datetime, timezone
import pandas as pd

import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

API_URL  = os.getenv("API_URL",  "http://localhost:8000")
STORE_ID = os.getenv("STORE_ID", "store_mumbai_01")
REFRESH_SECONDS = 5

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Store Intelligence Center",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern UI / CSS Styling ───────────────────────────────────────────────────

st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Hide Streamlit Header & Footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tighten top padding */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Metric Card Customization */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700;
        color: #f8fafc;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Anomalies & Badges */
    .badge-critical { background:#ef4444; color:#fff; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:700; }
    .badge-warn     { background:#f59e0b; color:#fff; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:700; }
    .badge-info     { background:#3b82f6; color:#fff; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:700; }
    
    .anomaly-card   { 
        background: rgba(15, 23, 42, 0.6); 
        border-radius: 8px; 
        padding: 12px 16px;
        margin-bottom: 12px; 
        border-left: 4px solid #64748b; 
    }
    .anomaly-critical { border-left-color: #ef4444; }
    .anomaly-warn     { border-left-color: #f59e0b; }
    .anomaly-info     { border-left-color: #3b82f6; }
    
    /* Titles */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── API helpers ────────────────────────────────────────────────────────────────

def _fetch(path: str) -> dict | list | None:
    """GET a JSON endpoint from the API. Returns None on any error."""
    url = f"{API_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError:
        return None
    except Exception:
        return None

def fetch_metrics() -> dict:
    return _fetch(f"/stores/{STORE_ID}/metrics") or {}

def fetch_funnel() -> list:
    data = _fetch(f"/stores/{STORE_ID}/funnel") or {}
    return data.get("funnel", [])

def fetch_heatmap() -> list:
    data = _fetch(f"/stores/{STORE_ID}/heatmap") or {}
    return data.get("heatmap", [])

def fetch_anomalies() -> list:
    data = _fetch(f"/stores/{STORE_ID}/anomalies") or {}
    return data.get("anomalies", [])

def fetch_pos_analytics() -> dict:
    return _fetch(f"/stores/{STORE_ID}/pos") or {}

# ── Severity helpers ───────────────────────────────────────────────────────────

SEVERITY_CLASS  = {"CRITICAL": "anomaly-critical", "WARN": "anomaly-warn", "INFO": "anomaly-info"}
BADGE_CLASS     = {"CRITICAL": "badge-critical",   "WARN": "badge-warn",   "INFO": "badge-info"}

def _heat_colour(score: float) -> str:
    r = int(score * 220)
    b = int((1 - score) * 220)
    return f"rgb({r}, 60, {b})"

def _bar(score: float, width: int = 120) -> str:
    px = int(score * width)
    colour = _heat_colour(score)
    return (
        f'<div style="background:#0f172a;border-radius:4px;width:{width}px;height:14px;border:1px solid #334155;">'
        f'<div style="background:{colour};border-radius:4px;width:{px}px;height:12px;"></div>'
        f'</div>'
    )

# ── Sidebar / Camera Selection ────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown("## 🏪 StorePulse AI")
    st.sidebar.caption("Retail Intelligence Center")
    st.sidebar.divider()
    
    st.sidebar.markdown("### 📷 Camera Feeds")
    camera_map = {
        "CAM_1 - Browsing": "Browsing Analytics",
        "CAM_2 - Entrance": "Entrance Analytics",
        "CAM_3 - Entrance": "Entrance Analytics",
        "CAM_4 - Backoffice": "Backoffice Operations",
        "CAM_5 - Billing": "Billing & Queue Analytics",
    }
    
    selected_cam = st.sidebar.radio(
        "Select Active Camera",
        options=list(camera_map.keys()),
        index=1 # Default to CAM_2
    )
    
    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ System Status")
    st.sidebar.markdown(f"**Store:** `{STORE_ID}`")
    st.sidebar.markdown(f"**Backend:** `{API_URL}`")
    st.sidebar.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    
    return selected_cam, camera_map[selected_cam]


# ── Video Validation ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=2, show_spinner=False)
def is_valid_video(path: str) -> dict:
    import cv2
    from pathlib import Path
    
    if not Path(path).exists():
        return {"valid": False, "reason": "File does not exist", "frames": 0, "duration": 0, "codec": "N/A", "size": 0}
        
    try:
        size = Path(path).stat().st_size / (1024*1024)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {"valid": False, "reason": "OpenCV cannot open file (missing moov atom)", "frames": 0, "duration": 0, "codec": "N/A", "size": size}
            
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frames / fps if fps > 0 else 0
        
        codec = 'Unknown'
        try:
            codec_float = cap.get(cv2.CAP_PROP_FOURCC)
            if codec_float > 0:
                codec = int(codec_float).to_bytes(4, 'little').decode('utf-8', 'ignore')
        except Exception:
            pass
            
        cap.release()
        
        if frames <= 0 or duration <= 0:
            return {"valid": False, "reason": "No frames or duration detected", "frames": int(frames), "duration": duration, "codec": codec, "size": size}
            
        return {"valid": True, "reason": "OK", "frames": int(frames), "duration": duration, "codec": codec, "size": size}
    except Exception as e:
        return {"valid": False, "reason": f"Exception: {str(e)}", "frames": 0, "duration": 0, "codec": "N/A", "size": 0}

def check_generation_status() -> bool:
    import psutil
    try:
        for proc in psutil.process_iter(['cmdline']):
            try:
                cmd = proc.info.get('cmdline')
                if cmd:
                    cmd_str = ' '.join(str(c) for c in cmd)
                    if 'generate_parallel.py' in cmd_str:
                        return True
            except Exception:
                continue
    except Exception:
        pass
    return False

# ── Main dashboard ────────────────────────────────────────────────────────────

def render():
    selected_cam, cam_mode = render_sidebar()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Verification Status")
    
    is_generating = check_generation_status()
    
    annotated_statuses = {}
    all_pass = True
    for i in range(1, 6):
        c_id = f"cam_{i}"
        path = os.path.join("outputs", f"{c_id}_annotated.mp4")
        val = is_valid_video(path)
        if val["valid"]:
            annotated_statuses[f"CAM_{i}"] = "🟢 VALID"
        else:
            all_pass = False
            if is_generating:
                annotated_statuses[f"CAM_{i}"] = "⏳ ENCODING"
            else:
                annotated_statuses[f"CAM_{i}"] = "🔴 CORRUPT/MISSING"
    
    if is_generating:
        gen_status = "Generating"
    elif not all_pass:
        gen_status = "Finalizing"
    else:
        gen_status = "Ready"
        
    st.sidebar.markdown(f"**Generation Status:** `{gen_status}`")
            
    for cam, status in annotated_statuses.items():
        st.sidebar.markdown(f"**{cam}:** {status}")
        
    st.sidebar.markdown(f"**Annotated Ready:** `{'YES' if all_pass else 'NO'}`")

    # ── Fetch all data ────────────────────────────────────────────────────────
    metrics   = fetch_metrics()
    funnel    = fetch_funnel()
    heatmap   = fetch_heatmap()
    anomalies = fetch_anomalies()
    pos_data  = fetch_pos_analytics()

    api_online = bool(metrics)

    if not api_online:
        st.error(
            f"⚠️  Cannot reach the API at **{API_URL}**. "
            "Make sure the FastAPI server is running: "
            "`python -m uvicorn app.main:app --port 8000`"
        )
        return

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("<h2>Store Intelligence Dashboard</h2>", unsafe_allow_html=True)
    
    # ── KPI Cards Row ─────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("👥 Total Visitors", metrics.get("unique_visitors", 0))
    with k2:
        st.metric("📈 Conversion Rate", f"{metrics.get('conversion_rate_pct', 0):.1f}%")
    with k3:
        st.metric("🧾 Queue Depth", metrics.get("queue_depth", 0))
    with k4:
        st.metric("💰 Rev / Visitor", f"₹{metrics.get('revenue_per_visitor', 0):,.0f}")

    st.markdown("---")

    # ── LIVE CCTV FEED ────────────────────────────────────────────────────────
    st.markdown("<h2>LIVE CCTV FEED</h2>", unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    cam_id = selected_cam.split(' - ')[0]
    cam_id_lower = cam_id.lower()
    annotated_path = os.path.join("outputs", f"{cam_id_lower}_annotated.mp4")
    original_path = os.path.join("data", "videos", f"{cam_id.replace('_', ' ')}.mp4")
    
    ann_val = is_valid_video(annotated_path)
    
    if ann_val["valid"]:
        resolved_path = annotated_path
        play_status = "Playing Annotated"
        playback_source = "Annotated Output"
    elif is_generating:
        resolved_path = None
        play_status = "Annotated video still being generated"
        playback_source = "None (Generating)"
    else:
        resolved_path = original_path
        play_status = "Playing Raw Fallback"
        playback_source = "Raw CCTV Fallback"
    
    from pathlib import Path
    
    selected_video = resolved_path
    
    tab_player, tab_debug = st.tabs(["CCTV Player", "Video Diagnostics"])
    
    with tab_player:
        if play_status == "Annotated video still being generated":
            st.info("⏳ **Annotated video still being generated.** Please wait for the background process to call VideoWriter.release().")
        elif not selected_video or not Path(selected_video).exists():
            st.error(f"Video not found: {selected_video}")
        else:
            video_path = Path(selected_video)
            st.write(f"Playing: {video_path}")
            video_bytes = video_path.read_bytes()
            st.video(video_bytes, format="video/mp4", autoplay=True, loop=True, muted=True)
            
    with tab_debug:
        st.markdown("### Video Diagnostics")
        st.markdown(f"**Selected Camera:** `{selected_cam}`")
        st.markdown(f"**Playback Source:** `{playback_source}`")
        
        st.markdown("#### Annotated Validation Details")
        st.markdown(f"**Annotated Exists:** `{Path(annotated_path).exists()}`")
        st.markdown(f"**Annotated Valid:** `{ann_val['valid']}`")
        st.markdown(f"**Frame Count:** `{ann_val['frames']}`")
        st.markdown(f"**Duration:** `{ann_val['duration']:.2f}s`")
        st.markdown(f"**Status Reason:** `{ann_val['reason']}`")
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h2>Analytics</h2>", unsafe_allow_html=True)

    # ── Analytics Panels Row 1 ────────────────────────────────────────────────
    r1c1, r1c2 = st.columns(2)
    
    with r1c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🔽 Conversion Funnel")
        if not funnel:
            st.info("No funnel data yet.")
        else:
            top = funnel[0]["count"] if funnel[0]["count"] > 0 else 1
            for step in funnel:
                count    = step["count"]
                pct      = round(count / top * 100, 1)
                drop_off = step.get("drop_off_pct", 0.0)
                
                st.markdown(f"**{step['step']}. {step['name']}**", unsafe_allow_html=True)
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.progress(pct / 100)
                with col_b:
                    st.markdown(f"`{count:,}`")
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🗺️ Zone Heatmap")
        if not heatmap:
            st.info("No zone data yet.")
        else:
            h1, h2, h3, h4 = st.columns([2, 1, 1, 2])
            h1.markdown("**Zone**")
            h2.markdown("**Visits**")
            h3.markdown("**Dwell**")
            h4.markdown("**Intensity**")
            st.divider()
            for zone in heatmap:
                score = zone.get("normalized_score", 0.0)
                r1, r2, r3, r4 = st.columns([2, 1, 1, 2])
                r1.markdown(f"🏷️ `{zone['zone_id']}`")
                r2.markdown(f"{zone['visit_frequency']:,}")
                r3.markdown(f"{zone['avg_dwell_seconds']:.0f}s")
                r4.markdown(_bar(score), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Analytics Panels Row 2 (POS Integration & Anomalies) ──────────────────
    r2c1, r2c2 = st.columns(2)
    
    with r2c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🛍️ Product Analytics (POS)")
        if pos_data and (pos_data.get("top_products") or pos_data.get("top_brands")):
            top_products = pd.DataFrame(pos_data["top_products"]).set_index("product")
            top_brands = pd.DataFrame(pos_data["top_brands"]).set_index("brand")
            
            tab1, tab2 = st.tabs(["Top Products", "Top Brands"])
            with tab1:
                st.bar_chart(top_products, color="#3b82f6")
            with tab2:
                st.bar_chart(top_brands, color="#10b981")
                
            st.info("💡 Correlation: Matched queue exits to POS receipts based on time proximity.")
        else:
            st.info("No POS data found. Run the ingestion script.")
        st.markdown('</div>', unsafe_allow_html=True)

    with r2c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⚠️ Active System Anomalies")
        if not anomalies:
            st.success("✅ System Nominal. No anomalies detected.")
        else:
            for a in anomalies:
                severity  = a.get("severity", "INFO")
                atype     = a.get("type", "UNKNOWN")
                message   = a.get("message", "")
                badge_cls = BADGE_CLASS.get(severity, "badge-info")
                card_cls  = SEVERITY_CLASS.get(severity, "anomaly-info")

                st.markdown(
                    f'<div class="anomaly-card {card_cls}">'
                    f'  <span class="{badge_cls}">{severity}</span>'
                    f'  &nbsp; <strong>{atype}</strong><br/>'
                    f'  <span style="color:#94a3b8;font-size:0.9rem">{message}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)


# ── Auto-refresh loop ─────────────────────────────────────────────────────────

render()
time.sleep(REFRESH_SECONDS)
st.rerun()
