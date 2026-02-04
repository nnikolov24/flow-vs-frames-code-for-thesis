#Run K-fold cross-validation for the appearance-only X3D-S RGB model.
import argparse, os
from pathlib import Path
import subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds_dir", required=True, help="dir with fold*/train.txt,val.txt")
    ap.add_argument("--out_root",  required=True, help="root dir to store per-fold runs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr",     type=float, default=1e-4)
    ap.add_argument("--bs",     type=int, default=8)
    ap.add_argument("--workers",type=int, default=4)
    args = ap.parse_args()

    folds = sorted([p for p in Path(args.folds_dir).iterdir()
                    if p.is_dir() and p.name.startswith("fold")])

    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    for fold_dir in folds:
        run_dir = Path(args.out_root) / fold_dir.name
        run_dir.mkdir(parents=True, exist_ok=True)
        train_split = fold_dir / "train.txt"
        val_split   = fold_dir / "val.txt"

        cmd = [
            "python", "scripts/train_x3d_rgb.py",
            "--train_split", str(train_split),
            "--val_split",   str(val_split),
            "--run_dir",     str(run_dir)
        ]
        env = os.environ.copy()
        env["EPOCHS"]  = str(args.epochs)
        env["LR"]      = str(args.lr)
        env["BS"]      = str(args.bs)
        env["WORKERS"] = str(args.workers)
        # shuffle for training
        env["APPEARANCE_SHUFFLE"] = "1"

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    main()
