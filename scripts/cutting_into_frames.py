import cv2
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

DIR_WITH_VIDEOS = Path("data/ouc-cge/videos")
OUTPUT_DIR = Path("data/ouc-cge/separate_frames")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORMAT = ".png"
QUALITY = 3

def get_frame(video_path: Path):

    video_stem = video_path.stem
    video_output_dir = OUTPUT_DIR / video_stem

    if video_output_dir.exists() and any(video_output_dir.iterdir()):
        return {"video" : video_stem, "status" : "skipped"}
    
    video_output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(f"{video_path}")
    if capture.isOpened():
        frame_idx = 0
        start_time = time.perf_counter()

        while True:
            retur, frame = capture.read()
            if not retur:
                break

            frame_name = video_output_dir / f"f_{frame_idx:04d}{FORMAT}"
            cv2.imwrite(str(frame_name), frame, [cv2.IMWRITE_PNG_COMPRESSION, QUALITY])
            frame_idx += 1

        capture.release()
        duration = time.perf_counter() - start_time

        return {
            "video" : video_stem,
            "status" : "success",
            "frames" : frame_idx,
            "time" : round(duration, 2)
        }
    else:
        return {
            "video" : video_stem,
            "status" : "error"
        }

def main():
    video_list = list(DIR_WITH_VIDEOS.rglob("*.mp4"))
    num_workers = os.cpu_count() 

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(get_frame, video_list), total=len(video_list)))
            

if __name__ == "__main__":
    main()       

