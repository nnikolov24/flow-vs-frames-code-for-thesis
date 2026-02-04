# Uses StratifiedKFold from scikit-learn to create label-balanced folds.
import argparse
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

def read_annot(p: Path):
    lines, labels = [], []
    with p.open("r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            lines.append(ln)
            labels.append(int(parts[-1]))  # last token is the label id
    return lines, labels

def write_lines(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(r + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True, help="path to all_samples.txt")
    ap.add_argument("--out_dir", required=True, help="output dir for kfold splits")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    annot = Path(args.annot)
    out_dir = Path(args.out_dir)

    lines, labels = read_annot(annot)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for k, (train_idx, val_idx) in enumerate(skf.split(lines, labels), start=1):
        fold_dir = out_dir / f"fold{k}"
        train_rows = [lines[i] for i in train_idx]
        val_rows   = [lines[i] for i in val_idx]
        write_lines(fold_dir / "train.txt", train_rows)
        write_lines(fold_dir / "val.txt",   val_rows)
        print(f"Fold {k}: train={len(train_rows)} val={len(val_rows)} -> {fold_dir}")

if __name__ == "__main__":
    main()
