"""
VideoFrameDataset and ImglistToTensor

Adapted from:
  Raivo Eli Koot, "Video Dataset Loading in PyTorch"
  GitHub repository: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
  BSD-2-Clause license.

Original copyright: (c) Raivo Eli Koot.
This version contains minor adaptations for the bachelor thesis project
on football action recognition by Nikolay Nikolov (2025).
"""
import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from typing import List, Union, Tuple, Any


class VideoRecord(object):
    """
    Helper class for VideoFrameDataset. Represents one video sample.

    Row format expected from the annotation file:
        VIDEO_FOLDER_PATH  START_FRAME  END_FRAME  LABEL_INDEX [LABEL_INDEX_2 ...]
    """
    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])

    @property
    def path(self) -> str:
        return self._path

    @property
    def start_frame(self) -> int:
        return int(self._data[1])

    @property
    def end_frame(self) -> int:
        return int(self._data[2])

    @property
    def num_frames(self) -> int:
        # +1 because end frame is inclusive
        return self.end_frame - self.start_frame + 1

    @property
    def label(self) -> Union[int, List[int]]:
        if len(self._data) == 4:
            return int(self._data[3])
        else:
            return [int(label_id) for label_id in self._data[3:]]


class VideoFrameDataset(torch.utils.data.Dataset):
    r"""
    Efficient dataset that samples NUM_SEGMENTS * FRAMES_PER_SEGMENT frames
    from each video folder on disk and returns either a list of PIL Images
    or a Tensor [T, C, H, W] if ImglistToTensor is used.

    Folder structure under root_path (example):
        root_path/
            scoring/
                video_a/
                    img_00001.jpg ... img_00073.jpg
                video_b/
                    ...
            tackling/
            red_cards/

    Annotation file lines:
        VIDEO_FOLDER_PATH  START_FRAME  END_FRAME  LABEL_INDEX
    where VIDEO_FOLDER_PATH is relative to root_path, e.g. "scoring/video_a/"
    """
    def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 num_segments: int = 3,
                 frames_per_segment: int = 1,
                 imagefile_template: str = 'img_{:05d}.jpg',
                 transform=None,
                 test_mode: bool = False):
        super(VideoFrameDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.test_mode = test_mode

        self._parse_annotationfile()
        self._sanity_check_samples()

    # helpers
    def _load_image(self, directory: str, idx: int) -> Image.Image:
        return Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert('RGB')

    def _parse_annotationfile(self):
        self.video_list: List[VideoRecord] = []
        with open(self.annotationfile_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  
                parts = line.split()
                if len(parts) < 4:
                    continue
                self.video_list.append(VideoRecord(parts, self.root_path))

    def _sanity_check_samples(self):
        need = self.num_segments * self.frames_per_segment
        for record in self.video_list:
            if record.num_frames <= 0 or record.start_frame == record.end_frame:
                print(f"\nDataset Warning: {record.path} seems to have zero RGB frames on disk!\n")
            elif record.num_frames < need:
                print(f"\nDataset Info: {record.path} has {record.num_frames} frames, "
                      f"requested {need}. Short clips will be padded by repeating the last frame.\n")

    def _get_start_indices(self, record: VideoRecord) -> np.ndarray:
        """
        Choose start indices for each segment. If the video is too short,
        return zeros so we start at the first frame (padding is applied later).
        """
        need = self.num_segments * self.frames_per_segment
        total = record.num_frames

        if total < need:
            return np.zeros(self.num_segments, dtype=int)

        if self.test_mode:
            distance_between_indices = (total - self.frames_per_segment + 1) / float(self.num_segments)
            start_indices = np.array([
                int(distance_between_indices / 2.0 + distance_between_indices * x)
                for x in range(self.num_segments)
            ])
        else:
            max_valid_start_index = (total - self.frames_per_segment + 1) // self.num_segments
            if max_valid_start_index <= 0:
                start_indices = np.zeros(self.num_segments, dtype=int)
            else:
                start_indices = (
                    np.multiply(list(range(self.num_segments)), max_valid_start_index)
                    + np.random.randint(max_valid_start_index, size=self.num_segments)
                )
        return start_indices

    # dataset protocol
    def __getitem__(self, idx: int) -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple['torch.Tensor', Union[int, List[int]]],
        Tuple[Any, Union[int, List[int]]]]:
        record: VideoRecord = self.video_list[idx]
        frame_start_indices: np.ndarray = self._get_start_indices(record)
        return self._get(record, frame_start_indices)

    def _get(self, record: VideoRecord, frame_start_indices: np.ndarray) -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple['torch.Tensor', Union[int, List[int]]],
        Tuple[Any, Union[int, List[int]]]]:
        """
        Load frames starting at each chosen index; pad by repeating the last frame
        when we run out of frames in short videos.
        """
        frame_start_indices = frame_start_indices + record.start_frame
        images: List[Image.Image] = []

        for start_index in frame_start_indices:
            frame_index = int(start_index)

            last_img = None
            for _ in range(self.frames_per_segment):
                if frame_index <= record.end_frame:
                    image = self._load_image(record.path, frame_index)
                    last_img = image
                    images.append(image)
                    frame_index += 1
                else:
                    images.append(last_img if last_img is not None
                                  else self._load_image(record.path, record.end_frame))

        if self.transform is not None:
            images = self.transform(images)

        return images, record.label

    def __len__(self):
        return len(self.video_list)


class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images [0,255] to a FloatTensor
    of shape (T, C, H, W) in [0,1].
    """
    @staticmethod
    def forward(img_list: List[Image.Image]) -> 'torch.Tensor':
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])

