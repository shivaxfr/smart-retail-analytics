import subprocess
import concurrent.futures
import time

cameras = [
    ('CAM_1', 'CAM 1.mp4'),
    ('CAM_2', 'CAM 2.mp4'),
    ('CAM_3', 'CAM 3.mp4'),
]

def run_pipeline(cam):
    cam_id, video_file = cam
    print(f'[{cam_id}] Starting repair generation...')
    try:
        subprocess.run([
            'python', 'pipeline/detect.py', 
            '--video', f'data/videos/{video_file}', 
            '--camera-id', cam_id,
            '--skip', '2'
        ], check=True)
        print(f'[{cam_id}] SUCCESS: Generated annotated video!')
        return True
    except subprocess.CalledProcessError as e:
        print(f'[{cam_id}] ERROR.')
        return False

if __name__ == '__main__':
    t0 = time.time()
    print('Launching repair pipeline for CAM_1, CAM_2, CAM_3...')
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(run_pipeline, cameras))
    elapsed = time.time() - t0
    if all(results):
        print(f'\nRepair generated successfully in {elapsed:.1f} seconds!')
    else:
        print(f'\nSome repair jobs failed. Elapsed: {elapsed:.1f} seconds.')
