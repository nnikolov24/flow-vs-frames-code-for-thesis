# scripts/extract_flow_farneback_npy.py
#Compute dense optical flow (Farnebäck) between consecutive RGB frames and save each flow field as a .npy file
# Implementation uses cv2.calcOpticalFlowFarneback (OpenCV).
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_sorted_images(folder: Path):
    return sorted(folder.glob("img_*.jpg"))

def load_gray(p: Path):
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def save_flow_npy(out_path: Path, flow_hw2: np.ndarray):
    # flow_hw2: [H, W, 2] float32 -> save as [2, H, W]
    np.save(out_path, np.moveaxis(flow_hw2.astype(np.float32), -1, 0))

def process_clip(frames_dir: Path, out_dir: Path,
                 resize_hw=None,
                 farneback_params=(0.5, 3, 15, 3, 5, 1.2, 0)):
    """
     Compute Farnebäck optical flow for one clip.

    frames_dir: path to clip folder, containing img_*.jpg frames
    out_dir:    path to output folder for flow_*.npy
    resize_hw:  (H, W) to resize frames, or None to keep original size
    farneback_params: (pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    """
    imgs = read_sorted_images(frames_dir)
    ensure_dir(out_dir)

    if len(imgs) == 0:
        return
    # load first frame
    prev_gray = load_gray(imgs[0])
    if resize_hw is not None:
        prev_gray = cv2.resize(prev_gray, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA)

    # pad first step with zero flow (keep indexing aligned: flow_00001.npy exists)
    H, W = prev_gray.shape[:2]
    save_flow_npy(out_dir / f"flow_{1:05d}.npy", np.zeros((H, W, 2), dtype=np.float32))

    # compute flow for the rest
    for i in range(1, len(imgs)):
        nxt_gray = load_gray(imgs[i])
        if resize_hw is not None:
            nxt_gray = cv2.resize(nxt_gray, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, nxt_gray, None,
            farneback_params[0], farneback_params[1], farneback_params[2],
            farneback_params[3], farneback_params[4], farneback_params[5],
            farneback_params[6]
        )  # [H, W, 2] float32

        save_flow_npy(out_dir / f"flow_{i+1:05d}.npy", flow)
        prev_gray = nxt_gray

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_root", required=True, help="root with class/clip/img_*.jpg")
    ap.add_argument("--flow_root", required=True, help="root to save flow .npy in mirrored tree")
    ap.add_argument("--height", type=int, default=224, help="optional resize height")
    ap.add_argument("--width",  type=int, default=224, help="optional resize width")
    args = ap.parse_args()

    rgb_root  = Path(args.rgb_root)
    flow_root = Path(args.flow_root)
    resize_hw = (args.height, args.width) if args.height > 0 and args.width > 0 else None

    classes = sorted([p for p in rgb_root.iterdir() if p.is_dir()])
    for cls_dir in classes:
        clips = sorted([p for p in cls_dir.iterdir() if p.is_dir()])
        for clip_dir in tqdm(clips, desc=f"{cls_dir.name}", unit="clip"):
            rel = clip_dir.relative_to(rgb_root)          # e.g., scoring/scoring_117
            out = flow_root / rel
            process_clip(clip_dir, out, resize_hw=resize_hw)

if __name__ == "__main__":
    main()
