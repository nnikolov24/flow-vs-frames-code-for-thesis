import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True,
                    help="Path containing fold1, fold2, ... fold5 folders")
    ap.add_argument("--out", default="kfold_avg_flow.png",
                    help="Output image file")
    args = ap.parse_args()

    root = Path(args.runs_root)

    csvs = sorted(root.glob("fold*/x3d_flow_trainval_curves.csv"))
    if len(csvs) == 0:
        raise RuntimeError("No CSV files found!")

    dfs = [pd.read_csv(c) for c in csvs]

    # Align by epoch count
    max_epochs = min(df["epoch"].max() for df in dfs)
    dfs = [df[df["epoch"] <= max_epochs] for df in dfs]

    train_loss = np.stack([df["train_loss"].values for df in dfs])
    val_loss   = np.stack([df["val_loss"].values for df in dfs])

    train_acc  = np.stack([df["train_acc"].values for df in dfs])
    val_acc    = np.stack([df["val_acc"].values for df in dfs])

    epochs = dfs[0]["epoch"].values

    # Compute average
    mean_train_loss = train_loss.mean(axis=0)
    mean_val_loss   = val_loss.mean(axis=0)
    std_train_loss  = train_loss.std(axis=0)
    std_val_loss    = val_loss.std(axis=0)

    mean_train_acc = train_acc.mean(axis=0)
    mean_val_acc   = val_acc.mean(axis=0)
    std_train_acc  = train_acc.std(axis=0)
    std_val_acc    = val_acc.std(axis=0)

    # Plot loss curve
    plt.figure(figsize=(10,5))
    plt.title("Flow – Average Loss Across 5 Folds")
    plt.fill_between(epochs, mean_train_loss-std_train_loss,
                     mean_train_loss+std_train_loss, alpha=0.2, color="blue")
    plt.fill_between(epochs, mean_val_loss-std_val_loss,
                     mean_val_loss+std_val_loss, alpha=0.2, color="orange")
    plt.plot(epochs, mean_train_loss, label="train loss")
    plt.plot(epochs, mean_val_loss, label="val loss")
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.xticks(epochs)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.savefig(args.out, dpi=300)
    pdf_out = Path(args.out).with_suffix(".pdf")
    plt.savefig(pdf_out, bbox_inches="tight")

    # Acc curve
    plt.figure(figsize=(10,5))
    plt.title("Flow – Average Accuracy Across 5 Folds")
    plt.fill_between(epochs, mean_train_acc-std_train_acc,
                     mean_train_acc+std_train_acc, alpha=0.2, color="blue")
    plt.fill_between(epochs, mean_val_acc-std_val_acc,
                     mean_val_acc+std_val_acc, alpha=0.2, color="orange")
    plt.plot(epochs, mean_train_acc, label="train acc")
    plt.plot(epochs, mean_val_acc, label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("kfold_avg_acc_flow.png", dpi=200)

    print(f"Saved averaged plots to:")
    print("   ", args.out)
    print("   kfold_avg_acc_flow.png")

if __name__ == "__main__":
    main()
