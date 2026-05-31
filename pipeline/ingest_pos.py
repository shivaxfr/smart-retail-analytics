import os
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy.orm import Session

# Add the parent directory to the Python path to import app modules
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal, POSTransactionORM

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

POS_DIR = Path("data/pos")
PROCESSED_DIR = POS_DIR / "processed"

def ingest_pos_data():
    if not POS_DIR.exists():
        log.warning(f"POS directory does not exist: {POS_DIR}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(POS_DIR.glob("*.csv"))
    if not csv_files:
        log.info("No new POS CSV files to process.")
        return

    db: Session = SessionLocal()
    try:
        for file_path in csv_files:
            log.info(f"Processing POS file: {file_path}")
            records_inserted = 0
            
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse timestamp from order_date and order_time
                    date_str = row.get("order_date", "").strip()
                    time_str = row.get("order_time", "").strip()
                    
                    try:
                        timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    except ValueError:
                        # Fallback to current time if parsing fails
                        timestamp = datetime.now(timezone.utc)
                    
                    # Safely handle numeric fields
                    def safe_int(val, default=1):
                        try: return int(val)
                        except (ValueError, TypeError): return default
                        
                    def safe_float(val, default=0.0):
                        try: return float(val)
                        except (ValueError, TypeError): return default

                    transaction = POSTransactionORM(
                        order_id=row.get("order_id", "").strip(),
                        timestamp=timestamp,
                        store_id=row.get("store_id", "store_mumbai_01").strip(),
                        customer_number=row.get("customer_number", "").strip() or None,
                        product_name=row.get("product_name", "").strip() or None,
                        brand_name=row.get("brand_name", "").strip() or None,
                        qty=safe_int(row.get("qty"), 1),
                        gmv=safe_float(row.get("GMV"), 0.0),
                        nmv=safe_float(row.get("NMV"), 0.0),
                        total_amount=safe_float(row.get("total_amount"), 0.0)
                    )
                    db.add(transaction)
                    records_inserted += 1
            
            db.commit()
            log.info(f"Successfully inserted {records_inserted} records from {file_path.name}")
            
            # Move file to processed
            dest_path = PROCESSED_DIR / file_path.name
            # If file already exists in processed, remove it first
            if dest_path.exists():
                dest_path.unlink()
            file_path.rename(dest_path)
            log.info(f"Moved {file_path.name} to processed folder.")
            
    except Exception as e:
        log.error(f"Error during POS ingestion: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    ingest_pos_data()
