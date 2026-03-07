from pathlib import Path

project_root = Path(".")
data_dir = Path("data/ouc-cge/videos")
mapp = {"low" : 0, "mid" : 1, "high" : 2}

all_videos = list(data_dir.rglob("*.mp4"))
output_file = Path("data/ouc-cge/ways_to_data.txt")

with open(output_file, "w") as f:
    for video in all_videos:
        if video.parent.name in mapp:
            f.write(f"{video.relative_to(project_root)} {mapp[video.parent.name]}\n")
            

