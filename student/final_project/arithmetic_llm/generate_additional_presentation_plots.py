import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
ADV_CSV = "student/final_project/adversarial_results/adversarial_numeric_20260307_153810.csv"
BASELINE_CSV = "student/final_project/adversarial_results/baseline_table_20260306_180705.csv"
OUTPUT_DIR = "student/final_project/adversarial_results/plots_final_numeric_only"

PERTURBATION_ORDER = ["off_by_1", "off_by_2", "off_by_5", "off_by_10", "random_offset"]
PERTURBATION_MAG = {"off_by_1": 1, "off_by_2": 2, "off_by_5": 5, "off_by_10": 10, "random_offset": 12}
DIFFICULTY_ORDER = ["easy", "medium", "hard"]
COLORS = sns.color_palette("Set2", n_colors=5)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
adv = pd.read_csv(ADV_CSV)
baseline = pd.read_csv(BASELINE_CSV) if os.path.exists(BASELINE_CSV) else None

# 1. Difficulty × Perturbation Heatmap
heatmap_data = adv.groupby(["difficulty", "perturbation_level"]).flipped.mean().reset_index()
heatmap_data["flip_rate"] = heatmap_data["flipped"] * 100
pivot = heatmap_data.pivot(index="difficulty", columns="perturbation_level", values="flip_rate")
pivot = pivot.loc[DIFFICULTY_ORDER, PERTURBATION_ORDER]
plt.figure(figsize=(11, 5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", vmin=0, vmax=100, cbar_kws={"label": "Flip Rate (%)"})
plt.xlabel("Perturbation Level")
plt.ylabel("Difficulty")
plt.title("Flip Rate by Difficulty and Perturbation Level")
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "plot5_flip_rate_heatmap.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(plot_path)

# 2. Flip Rate vs Baseline Accuracy (Difficulty Comparison)
if baseline is not None:
    acc = baseline.groupby("difficulty").baseline_correct.mean() * 100
    flip = adv.groupby("difficulty").flipped.mean() * 100
    plt.figure(figsize=(7, 7))
    for i, d in enumerate(DIFFICULTY_ORDER):
        plt.scatter(acc[d], flip[d], color=COLORS[i], s=120, label=d)
        plt.text(acc[d]+1, flip[d], d, fontsize=14, fontweight="bold")
    plt.xlabel("Baseline Accuracy (%)")
    plt.ylabel("Adversarial Flip Rate (%)")
    plt.title("Adversarial Flip Rate vs Baseline Accuracy")
    plt.xlim(0, 105)
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "plot6_flip_vs_accuracy.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(plot_path)

# 3. Perturbation Magnitude vs Flip Probability Curve
flip_curve = adv.groupby("perturbation_level").flipped.mean().reindex(PERTURBATION_ORDER)
mag = [PERTURBATION_MAG[p] for p in PERTURBATION_ORDER]
plt.figure(figsize=(8, 5))
plt.plot(mag, flip_curve.values * 100, marker="o", color=COLORS[0], linewidth=2)
plt.xticks(mag, PERTURBATION_ORDER)
plt.xlabel("Perturbation Magnitude")
plt.ylabel("Flip Probability (%)")
plt.title("Flip Probability vs Perturbation Magnitude")
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "plot7_flip_probability_curve.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(plot_path)

# 4. Outcome Distribution by Perturbation
outcomes = []
for p in PERTURBATION_ORDER:
    subset = adv[adv.perturbation_level == p]
    total = len(subset)
    correct = ((subset.adversarial_still_correct == True) & (subset.adversarial_parse_success == True)).sum()
    flipped = (subset.flipped == True).sum()
    parse_fail = (subset.adversarial_parse_success == False).sum()
    outcomes.append([correct, flipped, parse_fail])
outcomes = np.array(outcomes)
plt.figure(figsize=(10, 6))
bottom = np.zeros(len(PERTURBATION_ORDER))
labels = ["Correct", "Flipped", "Parse Fail"]
for i, label in enumerate(labels):
    plt.bar(PERTURBATION_ORDER, outcomes[:, i], bottom=bottom, color=COLORS[i], label=label)
    # Removed bar-top count labels for cleaner presentation
    bottom += outcomes[:, i]
plt.xlabel("Perturbation Level")
plt.ylabel("Count")
plt.title("Prediction Outcomes by Perturbation Level")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "plot8_outcome_distribution.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(plot_path)

# 5. Flip Rate vs Parse Rate Diagnostic
parse_rate = adv.groupby("perturbation_level").adversarial_parse_success.mean().reindex(PERTURBATION_ORDER) * 100
flip_rate = adv.groupby("perturbation_level").flipped.mean().reindex(PERTURBATION_ORDER) * 100
plt.figure(figsize=(8, 6))
for i, p in enumerate(PERTURBATION_ORDER):
    plt.scatter(parse_rate[p], flip_rate[p], color=COLORS[i], s=120)
    plt.text(parse_rate[p]+1, flip_rate[p], p, fontsize=13, fontweight="bold")
plt.xlabel("Parse Success Rate (%)")
plt.ylabel("Flip Rate (%)")
plt.title("Flip Rate vs Parse Success")
plt.xlim(0, 105)
plt.ylim(0, 105)
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "plot9_flip_vs_parse.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(plot_path)
