# 🏪 Store Intelligence - Retail Analytics Platform

## Project Overview
Store Intelligence is a scalable, end-to-end retail video analytics and business intelligence (BI) platform. It transforms raw CCTV feeds and Point of Sale (POS) transaction data into actionable insights for retail managers. Leveraging computer vision (YOLOv8) for real-time person tracking and a robust FastAPI backend backed by SQLite, it provides an interactive Streamlit dashboard for real-time monitoring of store metrics such as visitor count, conversion rates, queue depth, and anomaly alerts.

## Features
- **Headless Video Processing Pipeline**: Uses YOLOv8 for robust, multi-camera zone tracking, tripwire counting, and dwell-time measurement.
- **POS Data Integration**: Correlates CCTV footfall with transaction data to compute live retail conversion rates.
- **RESTful API Backend**: A FastAPI server that aggregates analytics, funnel, and anomaly data across multiple stores.
- **Real-Time BI Dashboard**: A modern, interactive Streamlit frontend that streams processed video and live KPIs.
- **Intelligent Fallback**: Gracefully handles partial encodings or missing video streams by seamlessly falling back to raw CCTV streams while displaying correct generation statuses.

## Architecture
- **Data Layer**: SQLite Database (`store_intelligence.db`) managed via SQLAlchemy ORM.
- **Vision Pipeline**: Python + OpenCV + YOLOv8 + Norfair (for Object Tracking) deployed completely headlessly.
- **API Services**: FastAPI framework supporting asynchronous anomaly detection, queue aggregation, and metrics derivation.
- **Frontend Layer**: Streamlit powered BI dashboard utilizing HTML/CSS/JS embedding for custom UI components and live video stream rendering.

## Project Structure
```
store_intelligence/
│
├── app/                      # FastAPI Backend & Analytics Services
│   ├── main.py               # Application entrypoint
│   ├── database.py           # SQLAlchemy ORM and schema definition
│   ├── conversion.py         # Advanced POS/CCTV metric cross-correlation
│   ├── routers/              # API endpoints (anomalies, funnel, heatmap, metrics)
│   └── services/             # Core service logic
│
├── dashboard/                # Frontend Application
│   ├── app.py                # Main Streamlit dashboard script
│   ├── index.html            # Custom UI layout
│   └── index.css             # Glassmorphism styling and themes
│
├── data/                     # Configuration and Sample Data
│   ├── camera_config.json    # Camera zones, ROIs, and tracking settings
│   └── transactions.csv      # Sample Point-of-Sale data for ingestion
│
├── pipeline/                 # Computer Vision Detection Pipeline
│   ├── detect.py             # Headless YOLOv8 object detection
│   └── ingest_pos.py         # Script to seed POS data into SQLite
│
├── tests/                    # PyTest Unit & Integration Tests
│
├── generate_parallel.py      # Background worker orchestrator for heavy video tasks
├── requirements.txt          # Python package dependencies
├── docker-compose.yml        # Docker orchestration file
└── Dockerfile                # Environment container definition
```

## Installation

Ensure you have Python 3.10+ installed.

1. **Clone or Extract the Package**
   Extract the source code to your local machine.
   
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Run Instructions

Store Intelligence runs optimally via two distinct processes (the Backend API and the Frontend Dashboard). 

**Step 1: Start the FastAPI Backend**
```bash
python -m uvicorn app.main:app --port 8000
```
*Wait for the server to spin up at http://localhost:8000. You can explore the API documentation at `http://localhost:8000/docs`.*

**Step 2: Start the Dashboard**
Open a new terminal window and run:
```bash
python -m streamlit run dashboard/app.py
```
*The dashboard will automatically open in your browser at `http://localhost:8501`.*

## Demo Information
To test the pipeline and ingest sample POS data into the architecture, run the following scripts:
```bash
# Ingest mock POS data for conversion metrics
python pipeline/ingest_pos.py

# To simulate the heavy headless video encoding (Requires full raw CCTV data usually excluded from this lightweight package)
python generate_parallel.py
```
*Note: Due to file-size constraints for the submission package (50MB limit), the raw multi-gigabyte `.mp4` video files and PyTorch `.pt` models are excluded. To fully experience the live CCTV playback features, place valid MP4 videos in the `data/videos/` directory.*
