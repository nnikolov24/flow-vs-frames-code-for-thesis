import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_FILE = Path.home() / "thesis" / "repo" / "results_clip_length.txt"

# Lists to store parsed values
Ks = []
accs = []
losses = []

#Parse the text file
with open(RESULTS_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split(",")
        K = int(parts[0].split("=")[1])
        ACC = float(parts[1].split("=")[1])
        LOSS = float(parts[2].split("=")[1])

        Ks.append(K)
        accs.append(ACC)
        losses.append(LOSS)

# Convert to numpy arrays
Ks = np.array(Ks)
accs = np.array(accs)
losses = np.array(losses)

# Sort by K_FRAMES
order = np.argsort(Ks)
Ks = Ks[order]
accs = accs[order]
losses = losses[order]

# Output directory
OUT_DIR = Path.home() / "thesis2025" / "ablations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot accuracy
plt.figure(figsize=(6,4))
plt.plot(Ks, accs, marker="o", linewidth=2)
plt.title("Motion-Only Model — Accuracy vs Clip Length")
plt.xlabel("Number of frames used (K)")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "accuracy_vs_clip_length.png", dpi=200)
plt.close()

# Plot loss
plt.figure(figsize=(6,4))
plt.plot(Ks, losses, marker="o", color="red", linewidth=2)
plt.title("Motion-Only Model — Loss vs Clip Length")
plt.xlabel("Number of frames used (K)")
plt.ylabel("Validation Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "loss_vs_clip_length.png", dpi=200)
plt.close()

print("Saved:")
print("  -", OUT_DIR / "accuracy_vs_clip_length.png")
print("  -", OUT_DIR / "loss_vs_clip_length.png")
