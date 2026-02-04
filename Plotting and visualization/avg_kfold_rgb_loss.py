#avg_kfold_rgb_loss.py

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root dir with fold1, fold2, ... (RGB k-fold runs)"
    )
    args = ap.parse_args()
    root = Path(args.root)

    csv_paths = sorted(root.glob("fold*/x3d_rgb_trainval_curves.csv"))
    if not csv_paths:
        raise RuntimeError(f"No CSVs found under {root}/fold*/x3d_rgb_trainval_curves.csv")

    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df["fold"] = p.parent.name
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Group by epoch, average across folds
    group = all_df.groupby("epoch")
    mean_df = group[["train_loss", "val_loss"]].mean()
    std_df  = group[["train_loss", "val_loss"]].std()

    epochs = mean_df.index.values

    # Plot average loss
    plt.figure(figsize=(10, 4))

    plt.plot(epochs, mean_df["train_loss"], label="train loss", color="tab:blue")
    plt.plot(epochs, mean_df["val_loss"],   label="val loss",   color="tab:orange")

    
    plt.fill_between(
        epochs,
        mean_df["train_loss"] - std_df["train_loss"],
        mean_df["train_loss"] + std_df["train_loss"],
        alpha=0.2,
        color="tab:blue"
    )
    plt.fill_between(
        epochs,
        mean_df["val_loss"] - std_df["val_loss"],
        mean_df["val_loss"] + std_df["val_loss"],
        alpha=0.2,
        color="tab:orange"
    )

    # axis labels bigger
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)

    # show every epoch on x-axis and make tick numbers bigger
    plt.xticks(epochs)
    plt.tick_params(axis='both', labelsize=12)

    # title + legend bigger
    plt.title("RGB – Average Loss Across 5 Folds", fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)

    out_png = root / "rgb_kfold_loss_avg.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved RGB average-loss figure to: {out_png}")

if __name__ == "__main__":
    main()
