#Summarise 5-fold cross-validation results for the appearance-only X3D model.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def extract_best_acc(csv_path: Path):
    df = pd.read_csv(csv_path)
    return float(df["val_acc"].max())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True, help="root with fold1..fold5 run dirs")
    args = ap.parse_args()

    accs = []
    for k in range(1, 6):
        csv_path = Path(args.runs_root) / f"fold{k}" / "x3d_rgb_trainval_curves.csv"
        if csv_path.exists():
            accs.append(extract_best_acc(csv_path))
        else:
            print(f"Missing: {csv_path}")

    accs = np.array(accs, dtype=float)
    print(f"K={len(accs)} folds")
    if len(accs) > 0:
        mean = accs.mean()
        std  = accs.std(ddof=1) if len(accs) > 1 else 0.0
        print(f"Val Acc (best per fold): mean={mean:.4f}  std={std:.4f}")
        print("Per fold:", ", ".join(f"{a:.4f}" for a in accs))

if __name__ == "__main__":
    main()
