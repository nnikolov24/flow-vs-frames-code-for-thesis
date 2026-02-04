#extract_frames.py
#Convert raw .avi football clips into folders of JPEG frames.

import cv2
from pathlib import Path

SRC = Path("/home/u306092/data/football_actions_raw")   # my raw data: the avi. format videos from my dataset
DST = Path("/home/u306092/data/football_actions_frames")

CLASSES = ["scoring","tackling","red_cards"]
DST.mkdir(parents=True, exist_ok=True)

def save_video_to_frames(avi_path: Path, out_dir: Path):
    # Read an .avi file and save all frames as JPEG images into out_dir.
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {avi_path}")
    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok: break
        cv2.imwrite(str(out_dir / f"img_{idx:05d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        idx += 1
    cap.release()
    if idx == 1:
        print("WARNING: no frames for", avi_path)
# Process all videos for each class
for c in CLASSES:
    for avi in (SRC/c).glob("*.avi"):
        out_dir = DST/c/avi.stem  # e.g., ...\frames\scoring\goal_001
        if (out_dir/"img_00001.jpg").exists():    # Skip if frames already exist
            continue
        print("Extracting:", avi)
        save_video_to_frames(avi, out_dir)

print("DONE →", DST)
