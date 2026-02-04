# eval_x3d_rgb.py
"""
Evaluate the trained appearance-only X3D-S model on the validation split.

- Loads the checkpoint x3d_rgb_appearance_only.pt, runs the model in
  evaluation mode, and computes overall validation loss and accuracy.
- Writes a classification report (precision,recall,F1 per class, accuracy, weighted and macro average)
  and saves a confusion matrix.
"""

from pathlib import Path
import sys, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import functional as TF

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.x3d_rgb import X3D_RGB
from scripts.video_frame_dataset import VideoFrameDataset, ImglistToTensor

ROOT = Path("/home/u306092")
DATA_ROOT  = ROOT / "data" / "football_actions_frames"
SPLITS_DIR = ROOT / "data" / "football_actions_splits"
VAL_SPLIT  = SPLITS_DIR / "splits_val.txt"
RUNS_DIR   = ROOT / "thesis2025" / "runs"
CKPT       = RUNS_DIR / "x3d_rgb_appearance_only.pt"

NUM_CLASSES = int(os.environ.get("NUM_CLASSES", 3))
BATCH_SIZE  = int(os.environ.get("BS", 8))
NUM_WORKERS = int(os.environ.get("NW", 4))
TARGET_SIZE = (224, 224)

# Kinetics-ish normalization used in training
MEAN = torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32)
STD  = torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32)

# per-sample transform
# 1) resize every PIL frame to 224x224
# 2) convert list of frames -> tensor [T, C, H, W] in [0,1]
val_tf = transforms.Compose([
    transforms.Lambda(lambda imgs: [TF.resize(im, TARGET_SIZE) for im in imgs]),
    ImglistToTensor(),
])

# dataset & loader
val_ds = VideoFrameDataset(
    root_path=str(DATA_ROOT),
    annotationfile_path=str(VAL_SPLIT),
    num_segments=8,              # 8 x 2 = 16 frames
    frames_per_segment=2,
    imagefile_template='img_{:05d}.jpg',
    transform=val_tf,
    test_mode=True
)

val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = X3D_RGB(num_classes=NUM_CLASSES, pretrained=False).to(device)

state = torch.load(CKPT, map_location=device)
model.load_state_dict(state)
model.eval()
# sanity checck print a few validation samples
class_names = ["scoring(0)", "tackling(1)", "red_cards(2)"]  # order must match your labels 0/1/2
printed, max_print = 0, 24

# reuse training-time normalization
mean = MEAN.to(device).view(1, 3, 1, 1, 1)
std  = STD.to(device).view(1, 3, 1, 1, 1)

with torch.no_grad():
    for xb, yb in val_loader:
        # xb: [B, T, C, H, W] -> model expects [B, 3, T, H, W]
        xb = xb.permute(0, 2, 1, 3, 4).contiguous()
        xb = xb.to(device, non_blocking=True)
        mean = mean.to(xb.device)
        std  = std.to(xb.device)
        xb = (xb - mean) / std

        yb = yb.to(device, non_blocking=True)

        logits = model(xb)                 # [B, 3]
        probs  = F.softmax(logits, dim=1)  # [B, 3]

        for i in range(xb.size(0)):
            gt_id   = int(yb[i].item())
            pred_id = int(torch.argmax(probs[i]).item())
            conf    = float(probs[i, pred_id].item())
            print(f"[QC {printed:03d}] GT={class_names[gt_id]:12s}  "
                  f"PRED={class_names[pred_id]:12s}  conf={conf:0.3f}")
            if printed < 5:  # only print first few
                print("     probs:", [round(p, 3) for p in probs[i].cpu().tolist()])
            printed += 1
            if printed >= max_print:
                break
        if printed >= max_print:
            break

print("---- end of quick sanity check ----") ## here the sanity check ends


# evaluation
criterion = nn.CrossEntropyLoss(reduction="sum")
all_preds, all_labels = [], []
total_loss = 0.0

with torch.no_grad():
    for xb, yb in val_loader:
        # xb: [B, T, C, 224, 224]  -> match training layout
        xb = xb.permute(0, 2, 1, 3, 4).contiguous()   # [B, 3, T, 224, 224]

        # normalize per-channel
        mean = MEAN.to(xb.device).view(1, 3, 1, 1, 1)
        std  = STD.to(xb.device).view(1, 3, 1, 1, 1)
        xb = (xb - mean) / std

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(yb.cpu().numpy())

y_true = np.concatenate(all_labels)
y_pred = np.concatenate(all_preds)

val_acc  = accuracy_score(y_true, y_pred)
val_loss = total_loss / len(val_ds)

# reports
timestamp = time.strftime("%Y%m%d_%H%M%S")
report_txt = RUNS_DIR / f"rgb_val_report_{timestamp}.txt"
cm_png     = RUNS_DIR / f"rgb_val_confusion_matrix_{timestamp}.png"

target_names = ["scoring(0)", "tackling(1)", "red_cards(2)"]
report = classification_report(y_true, y_pred, target_names=target_names, digits=4)

report_txt.write_text(
    f"Checkpoint: {CKPT.name}\n"
    f"Samples: {len(val_ds)}\n"
    f"Val Loss: {val_loss:.4f}\n"
    f"Val Acc:  {val_acc:.4f}\n\n"
    f"{report}\n"
)

cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
fig = plt.figure(figsize=(5,4), dpi=150)
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix (VAL)')
plt.xticks([0,1,2], target_names, rotation=30, ha='right')
plt.yticks([0,1,2], target_names)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(cm_png)
plt.close(fig)

print(f"Val acc={val_acc:.4f} | loss={val_loss:.4f}")
print(f"Report -> {report_txt}")
print(f"Confusion matrix -> {cm_png}")


