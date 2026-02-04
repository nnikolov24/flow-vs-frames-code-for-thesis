"""
eval_x3d_flow_TK.py

Clip-length ablation for the flow-based X3D-S model.

This script evaluates the trained motion-only X3D-S model on the validation
split while varying the number of distinct flow frames per clip.
- K_FRAMES (2, 4, 8, 16, ...) is read from the environment:
    K_FRAMES=4 python scripts/eval_x3d_flow_TK.py
- For each batch, it keeps only K_FRAMES evenly spaced flow frames, pads
  back to the original temporal length (T=16) by repeating the last frame( meaning we will only have K unique frames), applies the same normalization
  as in training, and runs the model in evaluation mode.
- Computes validation accuracy, loss, and a classification report
"""

from pathlib import Path
import sys
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.flow_frame_dataset import FlowFrameDataset
from models.x3d_flow import X3D_FLOW

ROOT = Path("/home/u306092")

DATA_ROOT  = ROOT / "data" / "football_actions_flow_farneback_npy"
SPLITS_DIR = ROOT / "data" / "football_actions_splits"
VAL_SPLIT  = SPLITS_DIR / "splits_val.txt"

RUNS_DIR   = ROOT / "thesis2025" / "runs_flow_farneback_npy"
CKPT       = RUNS_DIR / "x3d_flow_best.pt"

NUM_CLASSES = 3
BATCH_SIZE  = 8
NUM_WORKERS = 4

MEAN = torch.tensor([0.5, 0.5]).view(1, 2, 1, 1, 1)
STD  = torch.tensor([0.5, 0.5]).view(1, 2, 1, 1, 1)


# How many distinct frames are kept (2,4,8,16).
# The rest will be padded so the model still sees T=16.
K_FRAMES = int(os.getenv("K_FRAMES", "16"))
print(f"[INFO] Evaluating flow model with K_FRAMES = {K_FRAMES}")

FLOW_BOUND = 20.0  # clamp Farneback flow to [-20,20] then divide by 20 → [-1,1]

def subselect_and_pad(xb: torch.Tensor, k: int) -> torch.Tensor:
    """
    xb: [B, T, 2, H, W] flow clip
    k : number of *unique* frames to keep

    Returns: [B, T, 2, H, W] with only k informative frames.
    """
    B, T, C, H, W = xb.shape

    # Baseline: keep all frames
    if k >= T:
        return xb

    # choose k indices evenly spaced between 0 and T-1
    idx = torch.linspace(0, T - 1, steps=k).round().long().to(xb.device)  # [k]

    # take those k frames
    xb_k = xb[:, idx, :, :, :]  # [B, k, 2, H, W]

    # pad back to T by repeating the last chosen frame
    last = xb_k[:, -1:, :, :, :]                 # [B, 1, 2, H, W]
    repeat = T - k
    pad = last.repeat(1, repeat, 1, 1, 1)        # [B, T-k, 2, H, W]
    xb_full = torch.cat([xb_k, pad], dim=1)      # [B, T, 2, H, W]
    return xb_full

val_ds = FlowFrameDataset(
    root_path=str(DATA_ROOT),
    annotationfile_path=str(VAL_SPLIT),
    num_segments=8,          # original T=16  (8 segments × 2 frames)
    frames_per_segment=2,
    transform=None,
    test_mode=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = X3D_FLOW(num_classes=NUM_CLASSES, pretrained=False).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss(reduction="sum")

all_preds = []
all_labels = []
total_loss = 0.0

with torch.no_grad():
    for xb, yb in val_loader:
        # xb: [B, T, 2, H, W]  (T should be 16)
        xb = subselect_and_pad(xb, K_FRAMES)        # keep K, pad to T

        # model wants [B, 2, T, H, W]
        xb = xb.permute(0, 2, 1, 3, 4).contiguous()

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        
         # same mean/std normalization as in training
        mean = MEAN.to(xb.device)
        std  = STD.to(xb.device)
        xb = (xb - mean) / std

        # normalize flow to roughly [-1,1]
        xb = torch.clamp(xb, -FLOW_BOUND, FLOW_BOUND) / FLOW_BOUND

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

results_file = RUNS_DIR / "clip_length_ablation.txt"
with results_file.open("a") as f:
    f.write(f"K_FRAMES={K_FRAMES}  val_acc={val_acc:.4f}  val_loss={val_loss:.4f}\n")
    f.write(report + "\n\n")

print(f"[INFO] Appended results to {results_file}")
