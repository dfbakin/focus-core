"""OUC-CGE dataset: classroom group engagement recognition.

Dataset: https://osf.io/brd2c/
Paper: "A Video Dataset for Classroom Group Engagement Recognition" (Lu et al., 2025)

3 engagement levels: Low (0), Medium (1), High (2)
7,705 segments of 10-second clips, 1280x720, 30fps
"""

from src.data.video_dataset import VideoDataset
from pathlib import Path 
import pandas as pd
import cv2
import numpy as np
import torch


ENGAGEMENT_CLASSES = ["low", "medium", "high"]


class OUCCGEDataset(VideoDataset):
    """OUC-CGE video clip dataset."""

    def __init__(self, root: str, split: str = "train", config=None, transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.path_of_filenames_list = self.root / f"{split}.csv"
        self.sample = pd.read_csv(self.path_of_filenames_list, sep=' ', names=["path", "label"])

        self.config = config
        self.is_slowfast = getattr(config, "name", '').startswith("slowfast")

        if self.is_slowfast:
            self.slow_num_fr = getattr(config, "slow_num_frames", 8)
            self.slow_rate = getattr(config, "slow_sample_rate", 8)

            self.fast_num_fr = getattr(config, "fast_num_frames", 32)
            self.fast_rate = getattr(config, "fast_sample_rate", 2)
        else:
            self.slow_num_fr = getattr(config, "num_frames", 8)
            self.slow_rate = getattr(config, "sample_rate", 8)

            self.fast_num_fr = None
            self.fast_rate = None
            
    def __len__(self) -> int:
        return len(self.sample)

    def __getitem__(self, index: int) -> dict:
        video_path = self.root / Path(self.sample.iloc[index, 0])

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            window_size = self.slow_num_fr * self.slow_rate

            if total_frames > window_size:
                start_point = np.random.randint(0, total_frames - window_size)
            else:
                start_point = 0
            
            def _get_frame(num_frames, rate):
                list_frames = []
                for i in range(num_frames):
                    idx = start_point + i * rate

                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

                    retur, frame_bgr = cap.read()
                    if retur:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        list_frames.append(frame_rgb)
                    else:
                        print(f"Broken frame: {idx}, in video by {video_path}")
                        list_frames.append(list_frames[i - 1])

                np_array_frames = np.array(list_frames)
                tensor_cv = torch.tensor(np_array_frames)
                t_tensor = tensor_cv.permute(0, 3, 1, 2).float()
                norm_tensor = t_tensor / 255.0
                return norm_tensor
            
            self.s_flow_tensor = _get_frame(self.slow_num_fr, self.slow_rate)
            data = {
                "s_flow": self.s_flow_tensor,
                "label": torch.tensor(self.sample.iloc[index, 1], dtype=torch.int64),
                "path": str(video_path)
            }
            if self.is_slowfast:
                self.f_flow_tensor = _get_frame(self.fast_num_fr, self.fast_rate)
                data["f_flow"] = self.f_flow_tensor

            cap.release()

            if self.transform:
                data["s_flow"] = self.transform(data["s_flow"])
                if self.is_slowfast:
                    data["f_flow"] = self.transform(data["f_flow"])
            return data
        else:
            raise RuntimeError(f"Error: video by {video_path} cloud not open")

    @property
    def num_classes(self) -> int:
        return 3

    @property
    def class_names(self) -> list[str]:
        return ENGAGEMENT_CLASSES
