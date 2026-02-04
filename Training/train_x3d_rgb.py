# train_x3d_rgb.py
"""
Train the appearance-only X3D-S model on RGB frame clips.

- Uses VideoFrameDataset (adapted from Video-Dataset-Loading-Pytorch by Koot)
  to sample T = num_segments * frames_per_segment frames per clip.
- applies a TimeShuffle transform during training to randomize
  frame order and remove motion cues when APPEARANCE_SHUFFLE is 1.
- Trains the model with cross-entropy loss and AdamW, logs train/val loss
  and accuracy per epoch, saves the best checkpoint, and saves
  learning curves.
"""

import os
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
import random

#tiny transform to destroy temporal order (appearance-only)
class TimeShuffle:
    def __call__(self, imgs):
        # imgs: list of PIL images length T
        random.shuffle(imgs)  # in-place permutation
        return imgs


from video_frame_dataset import VideoFrameDataset, ImglistToTensor

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
from x3d_rgb import X3D_RGB


#paths
FRAMES_ROOT = Path("/home/u306092/data/football_actions_frames")
SPLIT_DIR   = Path("/home/u306092/data/football_actions_splits")
DEFAULT_TRAIN_TXT   = SPLIT_DIR / "splits_train.txt"
DEFAULT_VAL_TXT     = SPLIT_DIR / "splits_val.txt"

# hyperparams (can override via env)
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "3"))
EPOCHS      = int(os.getenv("EPOCHS", "5"))
BS          = int(os.getenv("BS", "8"))
LR          = float(os.getenv("LR", "1e-4"))
WORKERS     = int(os.getenv("WORKERS", "4"))
# randomize frame order to remove motion cues (appearance-only)
APPEARANCE_SHUFFLE = os.getenv("APPEARANCE_SHUFFLE", "0") == "1"

# temporal settings: NUM_SEGMENTS * FRAMES_PER_SEG = T
NUM_SEGMENTS     = int(os.getenv("SEGMENTS", "8"))
FRAMES_PER_SEG   = int(os.getenv("FRAMES_PER_SEG", "2"))  # T=16 default
H, W = 224, 224

# Kinetics-ish normalization
# channel dimension is dim=1 (B, 3, T, H, W)
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
STD  = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)


def make_dataset(txt, test_mode=False):
    tfms = [
        transforms.Lambda(lambda imgs: [transforms.functional.resize(im, [H, W]) for im in imgs]),
    ]
    # shuffle only for training when toggle is ON
    if not test_mode and APPEARANCE_SHUFFLE:
        tfms.append(TimeShuffle())  # destroys temporal order

    tfms.append(ImglistToTensor())  # -> [T, C, H, W] in [0,1]
    tfm = transforms.Compose(tfms)    
    return VideoFrameDataset(
        root_path=str(FRAMES_ROOT),
        annotationfile_path=str(txt),
        num_segments=NUM_SEGMENTS,
        frames_per_segment=FRAMES_PER_SEG,
        imagefile_template='img_{:05d}.jpg',
        transform=tfm,
        test_mode=test_mode
    )

def collate_to_bcthw(batch):
    xs, ys = zip(*batch)                   # xs: [B, T, C, H, W]
    x = torch.stack(xs, dim=0)             # [B, T, C, H, W]
    x = x.permute(0, 2, 1, 3, 4).contiguous()  # -> [B, C, T, H, W]
    y = torch.tensor(ys, dtype=torch.long)
    return x, y

def run_epoch(model, loader, train, device, optimizer, criterion):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)          # [B,3,T,H,W]
        yb = yb.to(device, non_blocking=True)
        xb = (xb - MEAN.to(xb.device)) / STD.to(xb.device)

        logits = model(xb)
        loss = criterion(logits, yb)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.detach().item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return total_loss / max(n,1), correct / max(n,1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_split", type=str, default=str(DEFAULT_TRAIN_TXT))
    parser.add_argument("--val_split",   type=str, default=str(DEFAULT_VAL_TXT))
    parser.add_argument("--run_dir",     type=str, default=str(Path.home() / "thesis2025" / "runs"))
    args = parser.parse_args()

    train_split_path = Path(args.train_split)
    val_split_path   = Path(args.val_split)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] APPEARANCE_SHUFFLE = {'ON' if APPEARANCE_SHUFFLE else 'OFF'}")

    train_ds = make_dataset(train_split_path, test_mode=False)
    val_ds   = make_dataset(val_split_path,   test_mode=True)


    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True,
                          num_workers=WORKERS, pin_memory=True,
                          collate_fn=collate_to_bcthw)
    val_dl   = DataLoader(val_ds,   batch_size=BS, shuffle=False,
                          num_workers=WORKERS, pin_memory=True,
                          collate_fn=collate_to_bcthw)

    model = X3D_RGB(num_classes=NUM_CLASSES, pretrained=True).to(device)

    # fine-tune ALL layers (unfrozen)
    for p in model.parameters():
        p.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    #tracking for plots and "save best"
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0

# saving outputs
    outdir = Path(args.run_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # real training
    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc = run_epoch(model, train_dl, True,  device, optimizer, criterion)
        va_loss, va_acc = run_epoch(model, val_dl,   False, device, optimizer, criterion)

    # log for curves
        hist["epoch"].append(ep)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(va_acc)

    # save "best" checkpoint by validation accuracy
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), outdir / "x3d_rgb_appearance_only_best.pt")
            print(f"Saved best: val_acc={best_acc:.3f} -> {outdir/'x3d_rgb_appearance_only_best.pt'}")

        print(f"Epoch {ep}: train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}  "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.3f}")
        # proof weights are changing: track head weight norm
        head = model.net.blocks[-1].proj      # Linear layer
        head_norm = head.weight.data.norm().item()
        if ep == 1:
            prev_head_norm = head_norm
        print(f"[ep {ep}] head |W|={head_norm:.6f}  Δ={head_norm - prev_head_norm:+.6f}")
        prev_head_norm = head_norm



    out = Path.home() / "thesis2025" / "runs" / "x3d_rgb_appearance_only.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print("Saved:", out)
    # Save learning curves
    df = pd.DataFrame(hist)
    df.to_csv(outdir / "x3d_rgb_trainval_curves.csv", index=False)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title("X3D RGB — Loss")
    plt.savefig(outdir / "x3d_rgb_loss_curve.png", dpi=200)

    plt.figure()
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["val_acc"], label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
    plt.title("X3D RGB — Accuracy")
    plt.savefig(outdir / "x3d_rgb_acc_curve.png", dpi=200)


if __name__ == "__main__":
    main()
