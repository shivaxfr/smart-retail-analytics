import subprocess
import concurrent.futures
import time

cameras = [
    ("CAM_1", "CAM 1.mp4"),
    ("CAM_2", "CAM 2.mp4"),
    ("CAM_3", "CAM 3.mp4"),
    ("CAM_4", "CAM 4.mp4"),
    ("CAM_5", "CAM 5.mp4"),
]

def run_pipeline(cam):
    cam_id, video_file = cam
    print(f"[{cam_id}] Starting parallel generation...")
    try:
        subprocess.run([
            "python", "pipeline/detect.py", 
            "--video", f"data/videos/{video_file}", 
            "--camera-id", cam_id,
            "--skip", "2"
        ], check=True, capture_output=True)
        print(f"[{cam_id}] SUCCESS: Generated annotated video!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{cam_id}] ERROR: {e.stderr.decode('utf-8', errors='ignore')}")
        return False

if __name__ == "__main__":
    t0 = time.time()
    print("Launching all 5 camera pipelines in parallel...\n")
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(run_pipeline, cameras))
    
    elapsed = time.time() - t0
    if all(results):
        print(f"\nAll 5 videos generated successfully in {elapsed:.1f} seconds!")
    else:
        print(f"\nSome videos failed to generate. Elapsed: {elapsed:.1f} seconds.")
