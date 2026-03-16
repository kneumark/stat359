import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data extracted from CSVs
numeric_labels = ["±1", "±2", "±5", "±10", "Random"]
numeric_wrong_revision = [0.619, 0.611, 0.617, 0.646, 0.651]
language_labels = ["Might be wrong", "Pretty sure", "TA says", "Official key", "Autograder", "Do not argue"]
language_wrong_revision = [0.632, 0.622, 0.594, 0.615, 0.620, 0.646]
politeness_labels = ["Polite", "Firm", "Rude", "Threat"]
politeness_wrong_revision = [0.642, 0.609, 0.594, 0.600]

OUTPUT_DIR = "student/final_project/adversarial_results/plots_cross_axis_wrong_revision_heatmaps_strict"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Combined heatmap
combined_matrix = np.zeros((max(len(numeric_labels), len(language_labels), len(politeness_labels)), 3))
combined_matrix[:len(numeric_labels), 0] = numeric_wrong_revision
combined_matrix[:len(language_labels), 1] = language_wrong_revision
combined_matrix[:len(politeness_labels), 2] = politeness_wrong_revision

row_labels = [
    f"Numeric: {lbl}" if i < len(numeric_labels) else ""
    for i, lbl in enumerate(numeric_labels + [""] * (max(len(numeric_labels), len(language_labels), len(politeness_labels)) - len(numeric_labels)))
]
row_labels = [lbl if lbl else f"Language: {language_labels[i-len(numeric_labels)]}" if i < len(numeric_labels)+len(language_labels) else f"Politeness: {politeness_labels[i-len(numeric_labels)-len(language_labels)]}" for i, lbl in enumerate(row_labels)]
col_labels = ["Numeric", "Language", "Politeness"]

plt.figure(figsize=(8, 6))
sns.heatmap(combined_matrix, annot=True, fmt=".3f", cmap="YlOrRd", xticklabels=col_labels, yticklabels=row_labels)
plt.title("Wrong Revision Rate by Axis (Heatmap)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cross_axis_wrong_revision_combined_heatmap.png"))
plt.close()

# Individual heatmaps
for axis, labels, values in zip(["numeric", "language", "politeness"], [numeric_labels, language_labels, politeness_labels], [numeric_wrong_revision, language_wrong_revision, politeness_wrong_revision]):
    matrix = np.array(values).reshape(-1, 1)
    plt.figure(figsize=(2.5, len(labels)))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlOrRd", xticklabels=[axis.capitalize()], yticklabels=labels)
    plt.title(f"Wrong Revision Rate ({axis.capitalize()})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cross_axis_wrong_revision_{axis}_heatmap.png"))
    plt.close()
