# scripts/aggregate_kfold_flow.py
#Summarise 5-fold cross-validation results for the flow-based X3D model.

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, required=True,
                    help="Root with fold1, fold2, ... from run_kfold_flow.py")
    args = ap.parse_args()

    root = Path(args.runs_root)
    accs = []

    for k in range(1, 6):
        fold_dir = root / f"fold{k}"
        csv_path = fold_dir / "x3d_flow_trainval_curves.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        best = df["val_acc"].max()
        accs.append(best)
        print(f"Fold {k}: best val_acc = {best:.4f}")

    if not accs:
        print("No folds found.")
        return

    accs = np.array(accs)
    print(f"\nk={len(accs)} folds")
    print(f"Val Acc (best per fold): mean={accs.mean():.4f}  std={accs.std():.4f}")
    print("Per fold:", ", ".join(f"{a:.4f}" for a in accs))

if __name__ == "__main__":
    main()
