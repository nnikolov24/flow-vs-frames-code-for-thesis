"""
eval_x3d_flow.py

Evaluate the trained flow-based X3D-S model on the validation split
(Farnebäck optical flow saved as .npy). Loads x3d_flow_best.pt, computes
loss/accuracy, saves classification report and confusion matrix.
"""

from pathlib import Path
import sys, os, time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# normalization for flow in [0,1] -> zero-center at 0.5
MEAN = torch.tensor([0.5, 0.5]).view(1, 2, 1, 1, 1)
STD  = torch.tensor([0.5, 0.5]).view(1, 2, 1, 1, 1)

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

val_ds = FlowFrameDataset(
    root_path=str(DATA_ROOT),
    annotationfile_path=str(VAL_SPLIT),
    num_segments=8, frames_per_segment=2,
    transform=None, test_mode=True
)

val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = X3D_FLOW(num_classes=NUM_CLASSES, pretrained=False).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss(reduction="sum")
all_preds, all_labels = [], []
total_loss = 0.0

with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.permute(0, 2, 1, 3, 4).contiguous()

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        
        # same mean/std normalization as in training
        mean = MEAN.to(xb.device)
        std  = STD.to(xb.device)
        xb = (xb - mean) / std

        # Flow scale to [-1,1]
        FLOW_BOUND = 20.0
        xb = torch.clamp(xb, -FLOW_BOUND, FLOW_BOUND) / FLOW_BOUND

        # Optional probes:
        #xb = xb.flip(dims=[2])       # reverse time

        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item()
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(yb.cpu().numpy())

y_true = np.concatenate(all_labels)
y_pred = np.concatenate(all_preds)
val_acc  = accuracy_score(y_true, y_pred)
val_loss = total_loss / len(val_ds)

print(f"Flow val acc={val_acc:.4f} | loss={val_loss:.4f}")
print(classification_report(y_true, y_pred, target_names=["scoring","tackling","red_cards"]))
# Save classification report
report = classification_report(y_true, y_pred, target_names=["scoring","tackling","red_cards"])
report_path = RUNS_DIR / "classification_report_flow.txt"
with open(report_path, "w") as f:
    f.write(report)
print(f"Saved classification report to {report_path}")


# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
classes = ["scoring", "tackling", "red_cards"]

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix — Flow model")
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()

# Save the confusion matrix image
out_path = RUNS_DIR / "confusion_matrix_flow.png"
plt.savefig(out_path, dpi=200)
print(f"[Confusion matrix saved to {out_path}")
plt.close()
