# C:\data\make_splits.py
import random
from pathlib import Path

ROOT = Path("/home/u306092/data/football_actions_frames")
CLASSES = ["scoring","tackling","red_cards"]
label_map = {c:i for i,c in enumerate(CLASSES)}

samples = []
for c in CLASSES:
    for vid_dir in (ROOT/c).iterdir():
        if vid_dir.is_dir():
            n_frames = len(list(vid_dir.glob("img_*.jpg")))
            if n_frames >= 1:
                rel = f"{c}/{vid_dir.name}/"
                samples.append((rel, 1, n_frames, label_map[c]))

random.seed(42)
random.shuffle(samples)
n = len(samples)
n_train = int(0.7*n)
n_val   = int(0.15*n)

splits = {
    r"C:\data\splits_train.txt": samples[:n_train],
    r"C:\data\splits_val.txt":   samples[n_train:n_train+n_val],
    r"C:\data\splits_test.txt":  samples[n_train+n_val:]
}

for path, rows in splits.items():
    with open(path, "w", encoding="utf-8") as f:
        for rel, s, e, y in rows:
            f.write(f"{rel} {s} {e} {y}\n")

print("Wrote:")
for k in splits: print(k)
