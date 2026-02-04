#flow_frame_dataset.py
# Dataset for loading optical-flow clips stored as .npy files.
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

class FlowFrameDataset(Dataset):
    """
    Reads clips from a 'split' file with lines:
        REL_PATH  START  END  LABEL
    Loads optical flow from a parallel root where files are:
        flow_00001.npy, flow_00002.npy, ...
    Each npy has shape [2, H, W] with float32 (u, v).
    Returns a tensor of shape [T, 2, H, W] and an int label.
    """
    def __init__(
        self,
        root_path: str,
        annotationfile_path: str,
        num_segments: int,
        frames_per_segment: int,
        flowfile_template: str = 'flow_{:05d}.npy',
        transform=None,
        test_mode: bool = False
    ):
        self.root = Path(root_path)
        self.ann  = Path(annotationfile_path)
        self.nseg = num_segments
        self.fps  = frames_per_segment
        self.lenT = self.nseg * self.fps
        self.tfm  = transform
        self.test_mode = test_mode
        self.templ = flowfile_template

        # parse split file
        self.recs = []
        with self.ann.open() as f:
            for line in f:
                sp = line.strip().split()
                if len(sp) < 4:  # skip empty/bad lines
                    continue
                rel, s, e, lab = sp[0], int(sp[1]), int(sp[2]), int(sp[3])
                self.recs.append((rel, s, e, lab))

    def __len__(self) -> int:
        return len(self.recs)

    def _sample_indices(self, s: int, e: int) -> List[int]:
        """
        Sample self.lenT indices evenly from the inclusive interval [s, e].
        If the interval is shorter than T, pad by repeating the last index.
        """
        total = max(1, e - s + 1)
        if self.lenT <= total:
            idx = np.linspace(0, total - 1, self.lenT).round().astype(int)
            return [s + int(i) for i in idx]
        # pad
        xs = list(range(s, e + 1))
        while len(xs) < self.lenT:
            xs.append(xs[-1])
        return xs

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        rel, s, e, lab = self.recs[i]
        clip_dir = self.root / rel
        n_available = len(list(clip_dir.glob("flow_*.npy")))
        if n_available == 0:
            raise RuntimeError(f"No flow npy files found in {clip_dir}")

        # clamp requested range to available files (indexed 1..n_available)
        s = max(1, min(s, n_available))
        e = max(1, min(e, n_available))

        idxs = self._sample_indices(s, e)

        # load .npy per index → torch [2,H,W] → build list
        frames: List[torch.Tensor] = []
        last_ok = None
        for k in idxs:
            p = clip_dir / self.templ.format(k)
            if not p.exists():
                # fallback to last available file if index missing
                p = clip_dir / self.templ.format(n_available)
            arr = np.load(p)
            t   = torch.from_numpy(arr)
            frames.append(t)
            last_ok = t

        # stack to [T,2,H,W]
        x = torch.stack(frames, dim=0)

        if self.tfm is not None:
            x = self.tfm(x)

        return x, lab
