import subprocess
import sys

cameras = [
    ("CAM_1", "CAM 1.mp4"),
    ("CAM_2", "CAM 2.mp4"),
    ("CAM_3", "CAM 3.mp4"),
    ("CAM_4", "CAM 4.mp4"),
    ("CAM_5", "CAM 5.mp4"),
]

for cam_id, video_file in cameras:
    print(f"\n[{cam_id}] Starting generation...")
    try:
        subprocess.run([
            "python", "pipeline/detect.py", 
            "--video", f"data/videos/{video_file}", 
            "--camera-id", cam_id,
            "--skip", "2"
        ], check=True)
        print(f"[{cam_id}] Successfully generated!")
    except subprocess.CalledProcessError as e:
        print(f"[{cam_id}] Failed: {e}")
        sys.exit(1)

print("\nAll videos generated successfully!")
