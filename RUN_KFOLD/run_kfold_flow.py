# scripts/run_kfold_flow.py
# Run K-fold cross-validation for the flow-based X3D-S model.
import os
import sys
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folds_dir",
        type=str,
        required=True,
        help="Folder with fold1, fold2, ... each containing train.txt and val.txt",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Root output dir where fold1, fold2, ... run dirs will be created",
    )
    args = parser.parse_args()

    folds_dir = Path(args.folds_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # Read hyperparams from environment (EPOCHS, LR, BS, WORKERS)
    env_epochs  = os.environ.get("EPOCHS", "10")
    env_lr      = os.environ.get("LR", "1e-4")
    env_bs      = os.environ.get("BS", "8")
    env_workers = os.environ.get("WORKERS", "4")

    # Loop over fold1, fold2, ...
    for fold_dir in sorted(p for p in folds_dir.iterdir() if p.is_dir()):
        fold_name = fold_dir.name
        train_split = fold_dir / "train.txt"
        val_split   = fold_dir / "val.txt"
        run_dir     = out_root / fold_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        cmd_str = (
            f"EPOCHS={env_epochs} LR={env_lr} BS={env_bs} WORKERS={env_workers} "
            f"{py} scripts/train_x3d_flow.py "
            f"--train_split {train_split} "
            f"--val_split {val_split} "
            f"--run_dir {run_dir}"
        )

        print("Running:", cmd_str)
        subprocess.run(["bash", "-lc", cmd_str], check=True)


if __name__ == "__main__":
    main()
