import os
import shutil
import zipfile

src_dir = '.'
dest_dir = 'submission_package'

# Explicitly KEEP these:
# - source code (app/, dashboard/, pipeline/)
# - config files (.streamlit/, data/camera_config.json, docker-compose.yml, Dockerfile, .env.example)
# - requirements.txt
# - README.md
# - sample CSV (data/transactions.csv)
# - tests (tests/)
# - generate_parallel.py
# - docs/

def should_keep(rel_path):
    parts = rel_path.replace('\\', '/').split('/')
    if not parts:
        return False
        
    # Ignore hidden dirs or specific dirs entirely
    if parts[0] in ['.venv', 'venv', 'outputs', '.pytest_cache', 'submission_package', '__pycache__', '.git']:
        return False
        
    if '__pycache__' in parts:
        return False
        
    # If in data/
    if parts[0] == 'data':
        # Only keep camera_config.json and transactions.csv
        if len(parts) == 2 and parts[1] in ['camera_config.json', 'transactions.csv']:
            return True
        return False
        
    # Ignore root level models/media/logs
    if len(parts) == 1:
        if parts[0].endswith('.pt') or parts[0].endswith('.dll') or parts[0].endswith('.bz2') or parts[0].endswith('.mp4') or parts[0].endswith('.webm') or parts[0].endswith('.log') or parts[0].endswith('.zip'):
            return False
            
    # Keep others by default (app, dashboard, pipeline, tests, docs, .streamlit, run scripts, docker, env, requirements)
    return True

if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)
os.makedirs(dest_dir)

files_copied = []

for root, dirs, files in os.walk(src_dir):
    for f in files:
        fp = os.path.join(root, f)
        rel_path = os.path.relpath(fp, src_dir)
        
        if should_keep(rel_path):
            dest_path = os.path.join(dest_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(fp, dest_path)
            files_copied.append(rel_path)

print(f"Copied {len(files_copied)} files to {dest_dir}.")

# Create a generic README.md replacement or use existing
# Wait, user wants a professional README.md generated.
