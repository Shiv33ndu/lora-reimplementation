import json
import matplotlib.pyplot as plt

RANKS = [1, 2, 4, 8]
val_accs = []

for r in RANKS:
    with open(f"results/roberta_sst2_rank_ablation/r{r}/metrics.json") as f:
        metrics = json.load(f)

    val_accs.append(metrics[-1]["val_acc"])

# Plot
plt.figure(figsize=(6, 4))
plt.plot(RANKS, val_accs, marker="o")
plt.xlabel("LoRA Rank (r)")
plt.ylabel("Validation Accuracy")
plt.title("LoRA Rank Ablation on SST-2 (RoBERTa-base)")
plt.grid(True)

# Save
plt.savefig("results/roberta_sst2_rank_ablation/rank_ablation.png", dpi=150)
plt.show()
