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
import logging


ENGAGEMENT_CLASSES = ["low", "medium", "high"]
ORIGINAL_FPS = 30
CLIP_DURATION = 10
BASIC_NUM_FRAMES = 8
logger = logging.getLogger(__name__)

class OUCCGEDataset(VideoDataset):
    """OUC-CGE video clip dataset."""

    def __init__(self, root: str | Path | None = None, split: str = "train", fps: list[int] = [ORIGINAL_FPS], num_frames: list[int] = [BASIC_NUM_FRAMES], transform=None):
        if root is None:
            root = Path(__file__).parent.parent.parent / "data" / "ouc-cge"
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.path_of_filenames_list = self.root / "data_labels" / f"{split}.csv"
        self.sample = pd.read_csv(self.path_of_filenames_list, sep=' ', names=["path", "label"])

        self.num_frames = num_frames
        self.fps = fps
        self.list_of_rates = []
        for fps_num in self.fps:
            self.list_of_rates.append(ORIGINAL_FPS // fps_num)
            
    def __len__(self) -> int:
        return len(self.sample)

    def _get_frame(self, cap, video_path, start_point, num_frames, rate):
        list_frames = []
        for i in range(num_frames):
            idx = start_point + i * rate

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            retur, frame_bgr = cap.read()
            if retur:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                list_frames.append(frame_rgb)
            else:
                logger.warning(f"Broken frame: {idx}, in video by {video_path}")
                list_frames.append(list_frames[i - 1])

        np_array_frames = np.array(list_frames)
        tensor_cv = torch.tensor(np_array_frames)
        t_tensor = tensor_cv.permute(0, 3, 1, 2).float()
        norm_tensor = t_tensor / 255.0
        return norm_tensor
    
    def __getitem__(self, index: int) -> dict:
        video_path = self.root / Path(self.sample.iloc[index, 0])

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            window_size = max(f * r for f, r in zip(self.num_frames, self.list_of_rates))

            if total_frames > window_size:
                start_point = np.random.randint(0, total_frames - window_size)
            else:
                start_point = 0

            data = {}
            for i, (n_frames, n_rate) in enumerate(zip(self.num_frames, self.list_of_rates)):
                current_tensor = self._get_frame(cap=cap, video_path=video_path, start_point=start_point, num_frames=n_frames, rate=n_rate)
                
                if self.transform:
                    data[f"flow_num_{i}"] = self.transform(current_tensor)
                else:
                    data[f"flow_num_{i}"] = current_tensor
            
            cap.release()
            data["label"] = torch.tensor(self.sample.iloc[index, 1], dtype=torch.int64)
            data["path"] = str(video_path)
            return data
        else:
            raise RuntimeError(f"Error: video by {video_path} cloud not open")

    @property
    def num_classes(self) -> int:
        return 3

    @property
    def class_names(self) -> list[str]:
        return ENGAGEMENT_CLASSES
    
if __name__ == "__main__":
    ouc_cge = OUCCGEDataset(split="train", fps=[3, 15], num_frames=[8, 32])
    sample_1 = ouc_cge[0]
    print(sample_1)
