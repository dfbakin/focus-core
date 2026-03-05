import cv2
from pathlib import Path

way_to_data = Path("data/ways_to_data.txt")
dir_to_save = Path("data/separate_frames")
step = 30

with open(way_to_data, "r") as f:
    for line in f:
        way, label = line.strip().split()
        way_to_file = Path(way)
        