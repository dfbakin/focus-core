"""OUC-CGE dataset: classroom group engagement recognition.

Dataset: https://osf.io/brd2c/
Paper: "A Video Dataset for Classroom Group Engagement Recognition" (Lu et al., 2025)

3 engagement levels: Low (0), Medium (1), High (2)
~7,705 segments, 30fps. Real clip lengths are heterogeneous (median ~5.4s),
some clips are short or corrupt, so the manifest is filtered on first use.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.data.video_dataset import VideoDataset

ENGAGEMENT_CLASSES = ["low", "medium", "high"]
NATIVE_FPS = 30
logger = logging.getLogger(__name__)


class OUCCGEDataset(VideoDataset):
    """OUC-CGE video clip dataset.

    Produces one entry per temporal pathway under keys ``flow_num_{i}`` with
    shape (C, T, H, W) after the transform, plus ``label`` and ``path``.

    A single pathway (default) matches the SLOW baseline. Pass two pathways,
    e.g. ``fps=[3.75, 15], num_frames=[8, 32]``, for SlowFast.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        fps: list[int | float] = [3.75],
        num_frames: list[int] = [8],
        transform=None,
        use_cache: bool = True,
    ):
        if root is None:
            raise ValueError("root is required, e.g. root='data/ouc-cge'")
        if len(fps) != len(num_frames):
            raise ValueError(
                f"fps and num_frames must have equal length, got {fps} / {num_frames}"
            )
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.num_frames = list(num_frames)
        self.fps = list(fps)
        # Integer sample stride per pathway (every Nth native-fps frame).
        self.list_of_rates = [max(1, int(round(NATIVE_FPS / f))) for f in self.fps]
        # The window each pathway needs; the dataset window is the max of these
        # so a clip that survives filtering is guaranteed long enough for ALL
        # pathways (this keeps cleaning and __getitem__ consistent).
        self.window_size = max(n * r for n, r in zip(self.num_frames, self.list_of_rates))

        self.path_of_filenames_list = self.root / "data_labels" / f"{split}.csv"
        self.sample = self._load_manifest(self.path_of_filenames_list)
        self._cleaning_dataset(window_size=self.window_size, use_cache=use_cache)

        logger.info(
            "OUC-CGE split=%s clips=%d fps=%s num_frames=%s rates=%s window=%d",
            self.split, len(self.sample), self.fps, self.num_frames,
            self.list_of_rates, self.window_size,
        )

    @staticmethod
    def _load_manifest(path: Path) -> pd.DataFrame:
        """Parse the space-separated manifest robustly.

        Tolerates CRLF line endings and drops rows whose label is missing or
        not an integer (e.g. the stray ``videos/anonymity/a.mp4`` row in the
        published test split).
        """
        df = pd.read_csv(path, sep=" ", names=["path", "label"])
        df["path"] = df["path"].astype(str).str.strip()
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        n_before = len(df)
        df = df.dropna(subset=["path", "label"])
        df = df[df["path"].str.len() > 0]
        df["label"] = df["label"].astype(int)
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.warning(
                "Dropped %d malformed manifest row(s) from %s", n_dropped, path.name
            )
        return df.reset_index(drop=True)

    def _cache_path(self) -> Path:
        return (
            self.root / "data_labels" / ".cache" / f"{self.split}_w{self.window_size}.csv"
        )

    def _cleaning_dataset(self, window_size: int, use_cache: bool = True) -> None:
        """Drop clips that are missing, unreadable, or shorter than the window.

        Result is cached per (split, window) so repeated runs skip the scan.
        """
        cache = self._cache_path()
        if use_cache and cache.exists():
            self.sample = self._load_manifest(cache)
            logger.info("Loaded cached filtered manifest: %s (%d clips)",
                        cache, len(self.sample))
            return

        dropped = []
        for video_idx in range(len(self.sample)):
            video_path = self.root / str(self.sample.iloc[video_idx, 0])
            if not video_path.exists():
                dropped.append(video_idx)
                continue
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                dropped.append(video_idx)
                cap.release()
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < window_size:
                dropped.append(video_idx)
            else:
                is_read, _ = cap.read()  # first-frame decode sanity check
                if not is_read:
                    dropped.append(video_idx)
            cap.release()

        if dropped:
            logger.warning(
                "split=%s: filtered out %d/%d clips (short/missing/corrupt)",
                self.split, len(dropped), len(self.sample),
            )
        self.sample = self.sample.drop(labels=dropped, axis=0).reset_index(drop=True)

        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            self.sample.to_csv(cache, sep=" ", header=False, index=False)

    def __len__(self) -> int:
        return len(self.sample)

    def _get_frames(self, cap, video_path, start_point, num_frames, rate):
        """Decode `num_frames` frames starting at `start_point`, stride `rate`.

        Uses a single seek followed by sequential grab()/retrieve(): we grab
        (no decode) the frames we skip and only decode the ones we keep. This
        avoids one keyframe seek per sampled frame, which is the dominant cost
        for compressed HD video.
        """
        needed = {i * rate for i in range(num_frames)}
        last_needed = max(needed)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_point)

        list_frames = []
        pos = 0
        while pos <= last_needed:
            grabbed = cap.grab()
            if pos in needed:
                if grabbed:
                    ret, frame_bgr = cap.retrieve()
                else:
                    ret, frame_bgr = False, None
                if ret:
                    list_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                elif list_frames:
                    logger.warning("Broken frame at offset %d in %s; reusing previous",
                                   start_point + pos, video_path)
                    list_frames.append(list_frames[-1])
                else:
                    raise RuntimeError(
                        f"first sampled frame is undecodable in {video_path}"
                    )
            pos += 1

        # Pad in case the stream ended early (truncated clip).
        while len(list_frames) < num_frames:
            list_frames.append(list_frames[-1])

        np_frames = np.ascontiguousarray(np.stack(list_frames[:num_frames]))
        tensor = torch.from_numpy(np_frames).permute(0, 3, 1, 2).float() / 255.0
        return tensor

    def __getitem__(self, index: int) -> dict:
        video_path = self.root / str(self.sample.iloc[index, 0])
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            label = torch.tensor(int(self.sample.iloc[index, 1]), dtype=torch.int64)

            if total_frames < self.window_size:
                raise RuntimeError(
                    f"{total_frames} frames < window {self.window_size} in {video_path}"
                )
            # +1 so a clip with exactly window_size frames yields start=0
            # instead of np.random.randint(0, 0), which raises.
            start_point = int(np.random.randint(0, total_frames - self.window_size + 1))

            data = {}
            for i, (n_frames, n_rate) in enumerate(zip(self.num_frames, self.list_of_rates)):
                clip = self._get_frames(cap, video_path, start_point, n_frames, n_rate)
                data[f"flow_num_{i}"] = self.transform(clip) if self.transform else clip
        finally:
            cap.release()

        data["label"] = label
        data["path"] = str(video_path)
        return data

    @property
    def num_classes(self) -> int:
        return 3

    @property
    def class_names(self) -> list[str]:
        return ENGAGEMENT_CLASSES
