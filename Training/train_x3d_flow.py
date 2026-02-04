# train_x3d_flow.py
#Train the flow-based X3D-S model on optical flow
import os
from pathlib import Path
import argparse
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd, matplotlib.pyplot as plt

from flow_frame_dataset import FlowFrameDataset

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
from x3d_flow import X3D_FLOW

FLOW_ROOT = Path("/home/u306092/data/football_actions_flow_farneback_npy")

SPLIT_DIR  = Path("/home/u306092/data/football_actions_splits")
DEF_TR     = SPLIT_DIR / "splits_train.txt"
DEF_VA     = SPLIT_DIR / "splits_val.txt"

NUM_CLASSES = int(os.getenv("NUM_CLASSES", "3"))
EPOCHS      = int(os.getenv("EPOCHS", "10"))
BS          = int(os.getenv("BS", "8"))
LR          = float(os.getenv("LR", "1e-4"))
WORKERS     = int(os.getenv("WORKERS", "4"))

NUM_SEGMENTS   = int(os.getenv("SEGMENTS", "8"))
FRAMES_PER_SEG = int(os.getenv("FRAMES_PER_SEG", "2"))
H, W = 224, 224

# normalization for flow in [0,1] → zero-center at 0.5
MEAN = torch.tensor([0.5, 0.5]).view(1, 2, 1, 1, 1)
STD  = torch.tensor([0.5, 0.5]).view(1, 2, 1, 1, 1)
def make_flow_ds(txt, test_mode=False):
    return FlowFrameDataset(
        root_path=str(FLOW_ROOT),
        annotationfile_path=str(txt),
        num_segments=NUM_SEGMENTS,
        frames_per_segment=FRAMES_PER_SEG,
        transform=None,
        test_mode=test_mode
    )

def collate_to_bcthw(batch):
    xs, ys = zip(*batch)                    # xs: [B, T, 2, H, W]
    x = torch.stack(xs, dim=0)              # [B, T, 2, H, W]
    x = x.permute(0, 2, 1, 3, 4).contiguous()  # -> [B, 2, T, H, W]
    y = torch.tensor(ys, dtype=torch.long)
    return x, y

def run_epoch(model, loader, train, device, optimizer, criterion):
    model.train(train)
    loss_sum, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        xb = (xb - MEAN.to(xb.device)) / STD.to(xb.device)
        FLOW_BOUND = 20.0
        xb = torch.clamp(xb, -FLOW_BOUND, FLOW_BOUND) / FLOW_BOUND  # [-1,1]


        logits = model(xb)
        loss = criterion(logits, yb)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        loss_sum += loss.detach().item() * xb.size(0)
        correct  += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return loss_sum/max(1,n), correct/max(1,n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_split", type=str, default=str(DEF_TR))
    ap.add_argument("--val_split",   type=str, default=str(DEF_VA))
    ap.add_argument("--run_dir",     type=str, default=str(Path.home() / "thesis2025" / "runs_flow"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = make_flow_ds(args.train_split, test_mode=False)
    val_ds   = make_flow_ds(args.val_split,   test_mode=True)
    

    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True,
                          num_workers=WORKERS, pin_memory=True,
                          collate_fn=collate_to_bcthw)
    val_dl   = DataLoader(val_ds,   batch_size=BS, shuffle=False,
                          num_workers=WORKERS, pin_memory=True,
                          collate_fn=collate_to_bcthw)

    model = X3D_FLOW(num_classes=NUM_CLASSES, pretrained=True).to(device)
    for p in model.parameters(): p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    hist, best_acc = {"epoch":[], "train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}, 0.0
    outdir = Path(args.run_dir); outdir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc = run_epoch(model, train_dl, True,  device, optimizer, criterion)
        va_loss, va_acc = run_epoch(model, val_dl,   False, device, optimizer, criterion)
        hist["epoch"].append(ep); hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["train_acc"].append(tr_acc); hist["val_acc"].append(va_acc)
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), outdir / "x3d_flow_best.pt")
            print(f"Saved best: val_acc={best_acc:.3f} -> {outdir/'x3d_flow_best.pt'}")
        print(f"Epoch {ep}: train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}  val_loss={va_loss:.4f} val_acc={va_acc:.3f}")

    # curves
    df = pd.DataFrame(hist); df.to_csv(outdir/"x3d_flow_trainval_curves.csv", index=False)
    plt.figure(); plt.plot(df.epoch, df.train_loss, label="train"); plt.plot(df.epoch, df.val_loss, label="val")
    plt.legend(); plt.title("Flow — Loss"); plt.savefig(outdir/"x3d_flow_loss.png", dpi=200)
    plt.figure(); plt.plot(df.epoch, df.train_acc, label="train"); plt.plot(df.epoch, df.val_acc, label="val")
    plt.legend(); plt.title("Flow — Accuracy"); plt.savefig(outdir/"x3d_flow_acc.png", dpi=200)

if __name__ == "__main__":
    main()
