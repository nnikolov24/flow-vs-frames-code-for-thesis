# eval_x3d_rgb_TK.py
"""
Clip-length ablation for the RGB (appearance-only) X3D-S model.

This script evaluates the trained appearance-only X3D-S model on the
validation split while varying the number of DISTINCT RGB frames per clip.

- K_FRAMES (2, 4, 8, 16) is read from the environment. For example, K_FRAMES=4 python scripts/eval_x3d_rgb_TK.py


- For each batch, I:
    1) keep only K_FRAMES evenly spaced frames,
    2) pad back to T=16 by repeating the last selected frame
       (so the model always sees 16 frames, but only K are unique),
    3) resize + convert frames exactly as in eval_x3d_rgb.py,
    4) apply the same mean/std normalization as in training,
    5) run the model in eval mode.

- The script prints validation accuracy/loss and a classification report,
  and appends the results to clip_length_ablation.txt
"""

from pathlib import Path
import sys
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

from torchvision import transforms
from torchvision.transforms import functional as TF

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.video_frame_dataset import VideoFrameDataset, ImglistToTensor
from models.x3d_rgb import X3D_RGB

# Paths
ROOT = Path("/home/u306092")

DATA_ROOT  = ROOT / "data" / "football_actions_frames"
SPLITS_DIR = ROOT / "data" / "football_actions_splits"
VAL_SPLIT  = SPLITS_DIR / "splits_val.txt"

RUNS_DIR   = ROOT / "thesis2025"/ "runs_rgb_single"
CKPT       = RUNS_DIR / "x3d_rgb_appearance_only.pt"

# Hyperparameters
NUM_CLASSES = 3
BATCH_SIZE  = 8
NUM_WORKERS = 4
TARGET_SIZE = (224, 224)

# Kinetics-ish normalization used in training
MEAN_1D = torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32)
STD_1D  = torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32)

# Number of DISTINCT frames to keep (2, 4, 8, 16)
K_FRAMES = int(os.getenv("K_FRAMES", "16"))
print(f"[INFO] Evaluating RGB model with K_FRAMES = {K_FRAMES}")

# Helper: subselect and pad frames
def subselect_and_pad(xb: torch.Tensor, k: int) -> torch.Tensor:
    """
    xb: [B, T, C, H, W] RGB clip
    k : number of DISTINCT frames to keep

    Returns: [B, T, C, H, W] where only k frames are unique and the
             rest are repetitions of the last selected frame.
    """
    B, T, C, H, W = xb.shape

    # Baseline: keep all frames if k >= T
    if k >= T:
        return xb

    # Choose k indices evenly spaced between 0 and T-1
    idx = torch.linspace(0, T - 1, steps=k).round().long().to(xb.device)
    xb_k = xb[:, idx, :, :, :]  # [B, k, C, H, W]

    # Pad back to T by repeating the last selected frame
    last = xb_k[:, -1:, :, :, :]              # [B, 1, C, H, W]
    repeat = T - k
    pad = last.repeat(1, repeat, 1, 1, 1)     # [B, T-k, C, H, W]

    xb_full = torch.cat([xb_k, pad], dim=1)   # [B, T, C, H, W]
    return xb_full

val_tf = transforms.Compose([
    # resize each PIL image to 224x224
    transforms.Lambda(lambda imgs: [TF.resize(im, TARGET_SIZE) for im in imgs]),
    # convert list of PIL images -> tensor [T, C, H, W] in [0,1]
    ImglistToTensor(),
])

# Dataset and DataLoader
val_ds = VideoFrameDataset(
    root_path=str(DATA_ROOT),
    annotationfile_path=str(VAL_SPLIT),
    num_segments=8,              # 8 × 2 = 16 frames total
    frames_per_segment=2,
    imagefile_template='img_{:05d}.jpg',
    transform=val_tf,
    test_mode=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = X3D_RGB(num_classes=NUM_CLASSES, pretrained=False).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss(reduction="sum")

MEAN = MEAN_1D.view(1, 3, 1, 1, 1).to(device)
STD  = STD_1D.view(1, 3, 1, 1, 1).to(device)

# Evaluation loop
all_preds = []
all_labels = []
total_loss = 0.0

with torch.no_grad():
    for xb, yb in val_loader:
        # xb: [B, T, C, H, W] from VideoFrameDataset + ImglistToTensor

        # 1) Apply K_FRAMES sub-selection + padding in [B, T, C, H, W] space
        xb = subselect_and_pad(xb, K_FRAMES)

        # 2) Reorder to [B, 3, T, H, W] as expected by X3D
        xb = xb.permute(0, 2, 1, 3, 4).contiguous()

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # 3) Normalize with same mean/std used in training
        xb = (xb - MEAN) / STD

        # 4) Forward pass
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item()

        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(yb.cpu().numpy())

# Metrics
y_true = np.concatenate(all_labels)
y_pred = np.concatenate(all_preds)

val_acc = accuracy_score(y_true, y_pred)
val_loss = total_loss / len(val_ds)

print(f"[RESULT] K_FRAMES={K_FRAMES}: "
      f"val_acc={val_acc:.4f} | val_loss={val_loss:.4f}")

report = classification_report(
    y_true,
    y_pred,
    target_names=["scoring", "tackling", "red_cards"],
)
print(report)

# Save
results_file = RUNS_DIR / "clip_length_ablation.txt"
results_file.parent.mkdir(parents=True, exist_ok=True)

with results_file.open("a") as f:
    f.write(f"K_FRAMES={K_FRAMES}  "
            f"val_acc={val_acc:.4f}  val_loss={val_loss:.4f}\n")
    f.write(report + "\n\n")

print(f"[INFO] Appended results to {results_file}")


