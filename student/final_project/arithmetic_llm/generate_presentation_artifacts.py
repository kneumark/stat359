#!/usr/bin/env python3
"""Generate final presentation plots and summary tables for adversarial results."""

import csv
import os
import argparse
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PERTURBATION_ORDER = [
    "off_by_1",
    "off_by_2",
    "off_by_5",
    "off_by_10",
    "random_offset",
]

PERTURBATION_LABELS = {
    "off_by_1": "off_by_1",
    "off_by_2": "off_by_2",
    "off_by_5": "off_by_5",
    "off_by_10": "off_by_10",
    "random_offset": "random_offset",
}

DIFFICULTY_ORDER = ["easy", "medium", "hard"]

COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "tertiary": "#2ca02c",
    "difficulty": {
        "easy": "#2ca02c",
        "medium": "#ff7f0e",
        "hard": "#d62728",
    },
    "perturbation": "#1f77b4",
}


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_bool(v: str) -> bool:
    return str(v).strip().lower() == "true"


def to_int(v: str) -> Optional[int]:
    s = str(v).strip()
    if s in {"", "PARSE_FAIL", "ERROR"}:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
        }
    )


def add_count_labels(ax, bars, labels: List[str]) -> None:
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.0,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def compute_baseline_metrics(baseline_rows: List[Dict[str, str]]) -> Dict:
    total = len(baseline_rows)
    correct = sum(1 for r in baseline_rows if to_bool(r["baseline_correct"]))

    by_diff = {}
    for d in DIFFICULTY_ORDER:
        subset = [r for r in baseline_rows if r["difficulty"] == d]
        d_total = len(subset)
        d_correct = sum(1 for r in subset if to_bool(r["baseline_correct"]))
        by_diff[d] = {
            "total": d_total,
            "correct": d_correct,
            "rate": (d_correct / d_total * 100.0) if d_total else 0.0,
        }

    return {
        "total": total,
        "correct": correct,
        "rate": (correct / total * 100.0) if total else 0.0,
        "by_difficulty": by_diff,
    }


def compute_adversarial_metrics(adv_rows: List[Dict[str, str]]) -> Dict:
    by_perturb = {}
    for p in PERTURBATION_ORDER:
        subset = [r for r in adv_rows if r["perturbation_level"] == p]
        total = len(subset)
        flipped = sum(1 for r in subset if to_bool(r["flipped"]))
        parsed = sum(1 for r in subset if to_bool(r["adversarial_parse_success"]))
        still = sum(1 for r in subset if to_bool(r["adversarial_still_correct"]))

        abs_errors = []
        for r in subset:
            if not to_bool(r["flipped"]):
                continue
            pred = to_int(r["adversarial_pred"])
            correct = to_int(r["correct_answer"])
            if pred is None or correct is None:
                continue
            abs_errors.append(abs(pred - correct))

        by_perturb[p] = {
            "total": total,
            "flipped": flipped,
            "flip_rate": (flipped / total * 100.0) if total else 0.0,
            "parsed": parsed,
            "parse_rate": (parsed / total * 100.0) if total else 0.0,
            "still_correct": still,
            "maintained_rate": (still / total * 100.0) if total else 0.0,
            "mean_abs_error_after_flip": (sum(abs_errors) / len(abs_errors)) if abs_errors else np.nan,
            "flip_numeric_count": len(abs_errors),
        }

    by_diff = {}
    for d in DIFFICULTY_ORDER:
        subset = [r for r in adv_rows if r["difficulty"] == d]
        total = len(subset)
        flipped = sum(1 for r in subset if to_bool(r["flipped"]))
        parsed = sum(1 for r in subset if to_bool(r["adversarial_parse_success"]))
        by_diff[d] = {
            "total": total,
            "flipped": flipped,
            "flip_rate": (flipped / total * 100.0) if total else 0.0,
            "parsed": parsed,
            "parse_rate": (parsed / total * 100.0) if total else 0.0,
        }

    return {
        "by_perturbation": by_perturb,
        "by_difficulty": by_diff,
    }


def plot_baseline_accuracy_by_difficulty(baseline_metrics: Dict, output_path: str) -> None:
    vals = [baseline_metrics["by_difficulty"][d]["rate"] for d in DIFFICULTY_ORDER]

    fig, ax = plt.subplots()
    bars = ax.bar(
        DIFFICULTY_ORDER,
        vals,
        color=[COLORS["difficulty"][d] for d in DIFFICULTY_ORDER],
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Difficulty")
    ax.set_title("Baseline Accuracy by Difficulty")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_flip_rate_vs_perturbation(adv_metrics: Dict, output_path: str) -> None:
    vals = [adv_metrics["by_perturbation"][p]["flip_rate"] for p in PERTURBATION_ORDER]
    labels = [
        f"{adv_metrics['by_perturbation'][p]['flipped']}/{adv_metrics['by_perturbation'][p]['total']}"
        for p in PERTURBATION_ORDER
    ]

    x_labels = [PERTURBATION_LABELS[p] for p in PERTURBATION_ORDER]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(
        x_labels,
        vals,
        color=COLORS["perturbation"],
        edgecolor="black",
        linewidth=0.8,
    )
    # Removed bar-top count labels for cleaner presentation
    ax.set_ylim(0, 105)
    ax.set_ylabel("Flip Rate (%)")
    ax.set_xlabel("Error Magnitude")
    ax.set_title("Flip Rate vs Error Magnitude")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_flip_rate_by_difficulty(adv_metrics: Dict, output_path: str) -> None:
    vals = [adv_metrics["by_difficulty"][d]["flip_rate"] for d in DIFFICULTY_ORDER]
    labels = [
        f"{adv_metrics['by_difficulty'][d]['flipped']}/{adv_metrics['by_difficulty'][d]['total']}"
        for d in DIFFICULTY_ORDER
    ]

    fig, ax = plt.subplots()
    bars = ax.bar(
        DIFFICULTY_ORDER,
        vals,
        color=[COLORS["difficulty"][d] for d in DIFFICULTY_ORDER],
        edgecolor="black",
        linewidth=0.8,
    )
    # Removed bar-top count labels for cleaner presentation
    ax.set_ylim(0, 105)
    ax.set_ylabel("Flip Rate (%)")
    ax.set_xlabel("Difficulty")
    ax.set_title("Flip Rate by Difficulty")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_parse_success_by_difficulty(adv_metrics: Dict, output_path: str) -> None:
    vals = [adv_metrics["by_difficulty"][d]["parse_rate"] for d in DIFFICULTY_ORDER]
    labels = [
        f"{adv_metrics['by_difficulty'][d]['parsed']}/{adv_metrics['by_difficulty'][d]['total']}"
        for d in DIFFICULTY_ORDER
    ]

    fig, ax = plt.subplots()
    bars = ax.bar(
        DIFFICULTY_ORDER,
        vals,
        color=[COLORS["difficulty"][d] for d in DIFFICULTY_ORDER],
        edgecolor="black",
        linewidth=0.8,
    )
    # Removed bar-top count labels for cleaner presentation
    ax.set_ylim(0, 105)
    ax.set_ylabel("Parse Success Rate (%)")
    ax.set_xlabel("Difficulty")
    ax.set_title("Parse Success Rate by Difficulty")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_flip_rate_heatmap(adv_rows: List[Dict[str, str]], output_path: str) -> None:
    """Generate a heatmap showing flip_rate by difficulty × perturbation_level."""
    # Compute flip_rate for each difficulty × perturbation_level combination
    data = []
    for d in DIFFICULTY_ORDER:
        for p in PERTURBATION_ORDER:
            subset = [
                r for r in adv_rows 
                if r["difficulty"] == d and r["perturbation_level"] == p
            ]
            if not subset:
                continue
            total = len(subset)
            flipped = sum(1 for r in subset if to_bool(r["flipped"]))
            flip_rate = (flipped / total * 100.0) if total > 0 else 0.0
            data.append({
                "difficulty": d,
                "perturbation_level": p,
                "flip_rate": flip_rate
            })
    
    # Create DataFrame and pivot table
    df = pd.DataFrame(data)
    pivot = df.pivot(
        index="difficulty", 
        columns="perturbation_level", 
        values="flip_rate"
    )
    
    # Order columns as specified
    pivot = pivot[PERTURBATION_ORDER]
    
    # Order rows by difficulty
    pivot = pivot.reindex(DIFFICULTY_ORDER)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Flip Rate (%)"},
        linewidths=0.5,
        linecolor="gray",
        vmin=0,
        vmax=100,
        ax=ax
    )
    
    # Format labels
    ax.set_xlabel("Error Magnitude", fontsize=15)
    ax.set_ylabel("Difficulty", fontsize=15)
    ax.set_title("Flip Rate by Difficulty and Error Magnitude", fontsize=18, fontweight="bold")
    
    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_results_summary_csv(path: str, baseline_metrics: Dict, adv_metrics: Dict) -> None:
    rows = []

    rows.append(
        {
            "section": "baseline_overall",
            "metric": "accuracy",
            "group": "all",
            "numerator": baseline_metrics["correct"],
            "denominator": baseline_metrics["total"],
            "rate_percent": f"{baseline_metrics['rate']:.2f}",
        }
    )

    for d in DIFFICULTY_ORDER:
        m = baseline_metrics["by_difficulty"][d]
        rows.append(
            {
                "section": "baseline_by_difficulty",
                "metric": "accuracy",
                "group": d,
                "numerator": m["correct"],
                "denominator": m["total"],
                "rate_percent": f"{m['rate']:.2f}",
            }
        )

    for p in PERTURBATION_ORDER:
        m = adv_metrics["by_perturbation"][p]
        rows.append(
            {
                "section": "adversarial_by_perturbation",
                "metric": "flip_rate",
                "group": p,
                "numerator": m["flipped"],
                "denominator": m["total"],
                "rate_percent": f"{m['flip_rate']:.2f}",
            }
        )

    for d in DIFFICULTY_ORDER:
        m = adv_metrics["by_difficulty"][d]
        rows.append(
            {
                "section": "adversarial_by_difficulty",
                "metric": "flip_rate",
                "group": d,
                "numerator": m["flipped"],
                "denominator": m["total"],
                "rate_percent": f"{m['flip_rate']:.2f}",
            }
        )

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "section",
                "metric",
                "group",
                "numerator",
                "denominator",
                "rate_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate presentation artifacts from baseline/adversarial CSVs")
    parser.add_argument(
        "--baseline-csv",
        type=str,
        default="student/final_project/adversarial_results/baseline_table_20260306_180705.csv",
        help="Path to baseline CSV",
    )
    parser.add_argument(
        "--adv-csv",
        type=str,
        default="student/final_project/adversarial_results/adversarial_numeric_20260306_190253.csv",
        help="Path to adversarial CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="student/final_project/adversarial_results/plots_final",
        help="Directory to save plots and summary table",
    )
    args = parser.parse_args()

    baseline_csv = args.baseline_csv
    adv_csv = args.adv_csv
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    setup_matplotlib()

    baseline_rows = read_csv(baseline_csv)
    adv_rows = read_csv(adv_csv)

    baseline_metrics = compute_baseline_metrics(baseline_rows)
    adv_metrics = compute_adversarial_metrics(adv_rows)

    plot_baseline_accuracy_by_difficulty(
        baseline_metrics,
        os.path.join(output_dir, "plot1_baseline_accuracy_by_difficulty.png"),
    )
    plot_flip_rate_vs_perturbation(
        adv_metrics,
        os.path.join(output_dir, "plot2_flip_rate_vs_numeric_perturbation.png"),
    )
    plot_flip_rate_by_difficulty(
        adv_metrics,
        os.path.join(output_dir, "plot3_flip_rate_by_difficulty.png"),
    )
    plot_parse_success_by_difficulty(
        adv_metrics,
        os.path.join(output_dir, "plot4_parse_success_by_difficulty.png"),
    )
    plot_flip_rate_heatmap(
        adv_rows,
        os.path.join(output_dir, "plot5_flip_rate_heatmap.png"),
    )

    summary_csv = os.path.join(output_dir, "results_summary.csv")
    write_results_summary_csv(summary_csv, baseline_metrics, adv_metrics)

    print("=" * 80)
    print("FINAL KEY METRICS")
    print("=" * 80)
    print(f"Baseline accuracy: {baseline_metrics['correct']}/{baseline_metrics['total']} = {baseline_metrics['rate']:.2f}%")
    print("\nBaseline accuracy by difficulty:")
    for d in DIFFICULTY_ORDER:
        m = baseline_metrics["by_difficulty"][d]
        print(f"  {d}: {m['correct']}/{m['total']} = {m['rate']:.2f}%")

    print("\nFlip rate by perturbation:")
    for p in PERTURBATION_ORDER:
        m = adv_metrics["by_perturbation"][p]
        print(f"  {p}: {m['flipped']}/{m['total']} = {m['flip_rate']:.2f}%")

    print("\nMaintained correctness by perturbation:")
    for p in PERTURBATION_ORDER:
        m = adv_metrics["by_perturbation"][p]
        print(f"  {p}: {m['still_correct']}/{m['total']} = {m['maintained_rate']:.2f}%")

    print("\nParse success by perturbation:")
    for p in PERTURBATION_ORDER:
        m = adv_metrics["by_perturbation"][p]
        print(f"  {p}: {m['parsed']}/{m['total']} = {m['parse_rate']:.2f}%")

    print("\nFlip rate by difficulty:")
    for d in DIFFICULTY_ORDER:
        m = adv_metrics["by_difficulty"][d]
        print(f"  {d}: {m['flipped']}/{m['total']} = {m['flip_rate']:.2f}%")

    print("\nParse success by difficulty:")
    for d in DIFFICULTY_ORDER:
        m = adv_metrics["by_difficulty"][d]
        print(f"  {d}: {m['parsed']}/{m['total']} = {m['parse_rate']:.2f}%")

    print("\nMean absolute error magnitude after flips by perturbation:")
    for p in PERTURBATION_ORDER:
        m = adv_metrics["by_perturbation"][p]
        mae = m["mean_abs_error_after_flip"]
        if np.isnan(mae):
            print(f"  {p}: N/A (no numeric flipped predictions)")
        else:
            print(f"  {p}: {mae:.3f} (n={m['flip_numeric_count']})")

    print("\nOutput artifacts:")
    print(f"  Plots dir: {output_dir}")
    print(f"  Summary table: {summary_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
