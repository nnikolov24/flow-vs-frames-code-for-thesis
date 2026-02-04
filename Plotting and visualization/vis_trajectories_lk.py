"""
vis_trajectories_lk.py

This script visualizes motion trajectories in a video clip using the Lucas-Kanade method.
It tracks corner features across consecutive RGB frames and draws their trajectories on the last frame
of the clip for motion analysis.

Useful for highlighting differences in motion patterns across football events like scoring, tackling,
or red card incidents.

Inspired by public OpenCV Lucas-Kanade examples:
https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
https://github.com/opencv/opencv/blob/master/samples/python/lk_track.py

"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def load_rgb_frames(rgb_dir: Path):
    """Load sorted RGB frames img_*.jpg from a clip folder."""
    paths = sorted(rgb_dir.glob("img_*.jpg"))
    if not paths:
        raise RuntimeError(f"No img_*.jpg found in {rgb_dir}")
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Failed to read {p}")
        frames.append(img)
    return frames


def compute_trajectories(frames,
                         max_corners=200,
                         quality=0.01,
                         min_dist=10,
                         win_size=(15, 15),
                         max_level=3):
    """
    Track feature points over all frames using Lucas–Kanade optical flow.

    Returns:
        trajectories: list of arrays of shape [L, 2] (x, y positions)
        last_frame: last RGB frame (BGR)
    """
    # Convert all frames to gray once
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    # Detect initial points on first frame
    p0 = cv2.goodFeaturesToTrack(
        grays[0],
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_dist,
        blockSize=7,
    )
    if p0 is None:
        raise RuntimeError("No good features found in first frame")

    trajectories = [[(float(x), float(y))] for [[x, y]] in p0]

    lk_params = dict(
        winSize=win_size,
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    prev_gray = grays[0]
    prev_pts = p0

    for k in range(1, len(grays)):
        gray = grays[k]

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts, None, **lk_params
        )

        status = status.reshape(-1)
        new_trajectories = []
        good_next = []

        for traj, (pt, st) in zip(trajectories, zip(next_pts, status)):
            if st == 1:
                x, y = pt.ravel()
                traj.append((float(x), float(y)))
                new_trajectories.append(traj)
                good_next.append([[x, y]])

        if not good_next:
            break

        trajectories = new_trajectories
        prev_pts = np.array(good_next, dtype=np.float32)
        prev_gray = gray

    return trajectories, frames[-1]


def draw_trajectories(base_frame, trajectories,
                      min_length=5,
                      line_thickness=2):
    """
    Draw trajectories on base_frame (BGR) and return the image.

    min_length: only draw tracks with at least this many points.
    """
    out = base_frame.copy()
    h, w = out.shape[:2]

    # Random colors for different tracks
    rng = np.random.default_rng(1234)

    for traj in trajectories:
        if len(traj) < min_length:
            continue
        pts = np.array(traj, dtype=np.int32)

        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        color = rng.integers(50, 255, size=3).tolist()
        # Draw as polyline
        cv2.polylines(out, [pts], isClosed=False, color=color,
                      thickness=line_thickness, lineType=cv2.LINE_AA)

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Visualize multi-frame motion trajectories for one clip."
    )
    ap.add_argument("--rgb_dir", required=True,
                    help="Folder with img_*.jpg frames (one clip).")
    ap.add_argument("--out_image", required=True,
                    help="Output PNG path.")
    ap.add_argument("--max_corners", type=int, default=200,
                    help="Number of feature points to track.")
    ap.add_argument("--min_length", type=int, default=5,
                    help="Minimum trajectory length to draw.")
    args = ap.parse_args()

    rgb_dir = Path(args.rgb_dir)
    out_path = Path(args.out_image)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading frames from {rgb_dir}")
    frames = load_rgb_frames(rgb_dir)
    print(f"Loaded {len(frames)} frames")

    print("Computing trajectories with Lucas–Kanade...")
    trajectories, last_frame = compute_trajectories(frames)

    print(f"Got {len(trajectories)} trajectories "
          f"(before length filtering).")

    vis = draw_trajectories(
        last_frame,
        trajectories,
        min_length=args.min_length,
        line_thickness=2,
    )

    cv2.imwrite(str(out_path), vis)
    print(f"Saved trajectory visualization to {out_path}")


if __name__ == "__main__":
    main()
