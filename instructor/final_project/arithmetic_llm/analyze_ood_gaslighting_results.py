#!/usr/bin/env python3
"""Analyze and plot OOD gaslighting experiment results."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt

DIFFICULTY_ORDER = [
    "in_distribution",
    "slight_ood",
    "moderate_ood",
    "strong_ood",
    "very_strong_ood",
]


def load_metrics(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_line(
    x_labels: List[str],
    series: Dict[str, List[float]],
    title: str,
    y_label: str,
    out_path: str,
) -> None:
    plt.figure(figsize=(9, 5))
    x = list(range(len(x_labels)))

    for label, values in series.items():
        plt.plot(x, values, marker="o", label=label)

    plt.xticks(x, x_labels, rotation=20)
    plt.title(title)
    plt.xlabel("Difficulty")
    plt.ylabel(y_label)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze OOD gaslighting experiment metrics")
    parser.add_argument("--metrics-json", required=True, help="Path to *_metrics.json")
    parser.add_argument("--output-dir", default="proposal_results/plots", help="Output directory for plots")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    metrics = load_metrics(args.metrics_json)
    by_bucket = metrics["by_bucket"]

    available_buckets = [b for b in DIFFICULTY_ORDER if b in by_bucket]
    if not available_buckets:
        available_buckets = list(by_bucket.keys())

    labels = [bucket.replace("_", " ") for bucket in available_buckets]

    control_acc = [by_bucket[b]["control_accuracy"] for b in available_buckets]
    adv_acc = [by_bucket[b]["adversarial_accuracy"] for b in available_buckets]
    defense_acc = [by_bucket[b]["defense_accuracy"] for b in available_buckets]

    flip_adv = [by_bucket[b]["flip_rate_adversarial"] for b in available_buckets]
    flip_def = [by_bucket[b]["flip_rate_defense"] for b in available_buckets]

    wrong_flip_adv = [by_bucket[b]["wrong_flip_rate_adversarial"] for b in available_buckets]
    wrong_flip_def = [by_bucket[b]["wrong_flip_rate_defense"] for b in available_buckets]

    hall_control = [by_bucket[b]["hallucinated_justification_rate_control"] for b in available_buckets]
    hall_adv = [by_bucket[b]["hallucinated_justification_rate_adversarial"] for b in available_buckets]
    hall_def = [by_bucket[b]["hallucinated_justification_rate_defense"] for b in available_buckets]

    plot_line(
        labels,
        {
            "Control": control_acc,
            "Adversarial": adv_acc,
            "Defense": defense_acc,
        },
        "Accuracy vs Difficulty",
        "Accuracy",
        os.path.join(args.output_dir, "accuracy_vs_difficulty.png"),
    )

    plot_line(
        labels,
        {
            "Flip Rate (Adversarial)": flip_adv,
            "Flip Rate (Defense)": flip_def,
        },
        "Flip Rate vs Difficulty",
        "Flip Rate",
        os.path.join(args.output_dir, "flip_rate_vs_difficulty.png"),
    )

    plot_line(
        labels,
        {
            "Wrong-Flip (Adversarial)": wrong_flip_adv,
            "Wrong-Flip (Defense)": wrong_flip_def,
        },
        "Wrong-Flip Rate vs Difficulty",
        "Wrong-Flip Rate",
        os.path.join(args.output_dir, "wrong_flip_vs_difficulty.png"),
    )

    plot_line(
        labels,
        {
            "Hallucination Control": hall_control,
            "Hallucination Adversarial": hall_adv,
            "Hallucination Defense": hall_def,
        },
        "Hallucinated Justification Rate vs Difficulty",
        "Rate",
        os.path.join(args.output_dir, "hallucination_vs_difficulty.png"),
    )

    print("Plots written to:", args.output_dir)


if __name__ == "__main__":
    main()
