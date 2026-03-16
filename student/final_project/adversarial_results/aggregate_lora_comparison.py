import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

# Model info and clean accuracy values
model_info = [
    ("LoRA-4", "student/final_project/adversarial_results/lora_r4_numeric_full/adversarial_numeric_summary_*.json", 0.40),
    ("LoRA-8", "student/final_project/adversarial_results/lora_numeric_full/adversarial_numeric_summary_*.json", 0.475),
    ("LoRA-16", "student/final_project/adversarial_results/lora_r16_numeric_full/adversarial_numeric_summary_*.json", 0.49),
]

rows = []

for model, path_pattern, clean_acc in model_info:
    files = glob.glob(path_pattern)
    if not files:
        print(f"No summary file found for {model}")
        continue

    latest_file = max(files, key=os.path.getmtime)

    with open(latest_file, "r") as f:
        summary = json.load(f)

    def flip_rate(key: str):
        stats = summary.get("stats_by_level", {})
        if key not in stats:
            print(f"WARNING: {model} | {latest_file} | missing key: {key}")
            return None
        entry = stats[key]
        total = entry.get("total")
        flipped = entry.get("flipped")
        if total is None or flipped is None:
            print(f"WARNING: {model} | {latest_file} | missing 'total' or 'flipped' in {key}")
            return None
        return flipped / total if total > 0 else None

    rows.append({
        "Model": model,
        "CleanAccuracy": clean_acc,
        "OffBy1FlipRate": flip_rate("off_by_1"),
        "OffBy2FlipRate": flip_rate("off_by_2"),
        "OffBy5FlipRate": flip_rate("off_by_5"),
        "OffBy10FlipRate": flip_rate("off_by_10"),
        "RandomOffsetFlipRate": flip_rate("random_offset")
    })

df = pd.DataFrame(rows)

# enforce model order
model_order = ["LoRA-4", "LoRA-8", "LoRA-16"]
df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
df = df.sort_values("Model").reset_index(drop=True)

print("\nComparison table:")
print(df.round(4))

# save wide CSV
output_dir = "student/final_project/adversarial_results"
os.makedirs(output_dir, exist_ok=True)

wide_csv = os.path.join(output_dir, "comparison_results.csv")
df.to_csv(wide_csv, index=False)

# save long CSV
long_df = df.melt(
    id_vars=["Model", "CleanAccuracy"],
    value_vars=["OffBy1FlipRate", "OffBy2FlipRate", "OffBy5FlipRate", "OffBy10FlipRate", "RandomOffsetFlipRate"],
    var_name="Perturbation",
    value_name="FlipRate"
)

# nicer perturbation labels
label_map = {
    "OffBy1FlipRate": "Off-by-1",
    "OffBy2FlipRate": "Off-by-2",
    "OffBy5FlipRate": "Off-by-5",
    "OffBy10FlipRate": "Off-by-10",
    "RandomOffsetFlipRate": "Random Offset"
}
long_df["Perturbation"] = long_df["Perturbation"].map(label_map)

long_csv = os.path.join(output_dir, "comparison_results_long.csv")
long_df.to_csv(long_csv, index=False)

# Plot 1: Clean Accuracy vs Model
plt.figure(figsize=(6, 4))
plt.bar(df["Model"], df["CleanAccuracy"])
plt.ylabel("Clean Accuracy")
plt.title("Clean Accuracy vs LoRA Rank")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_vs_rank.png"))
plt.close()

# Plot 2: Flip Rate vs Model for numeric perturbations
plt.figure(figsize=(8, 6))
plot_cols = [
    ("OffBy1FlipRate", "Off-by-1"),
    ("OffBy2FlipRate", "Off-by-2"),
    ("OffBy5FlipRate", "Off-by-5"),
    ("OffBy10FlipRate", "Off-by-10"),
]

for col, label in plot_cols:
    plt.plot(df["Model"], df[col], marker="o", label=label)

plt.ylabel("Flip Rate")
plt.title("Adversarial Flip Rate vs LoRA Rank")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fliprate_vs_rank.png"))
plt.close()

print(f"\nSaved wide CSV: {wide_csv}")
print(f"Saved long CSV: {long_csv}")
print(f"Saved plot: {os.path.join(output_dir, 'accuracy_vs_rank.png')}")
print(f"Saved plot: {os.path.join(output_dir, 'fliprate_vs_rank.png')}")
