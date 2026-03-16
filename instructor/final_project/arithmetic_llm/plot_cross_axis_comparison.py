#!/usr/bin/env python3
"""Create digestible visualizations across numeric, language, and politeness axis sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_float(rows: List[Dict[str, str]], key: str) -> List[float]:
    return [float(r[key]) for r in rows]


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.3f}", offset: float = 0.001) -> None:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _dynamic_ylim(values: List[float], minimum_pad: float = 0.02) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    y_min = min(values)
    y_max = max(values)
    spread = y_max - y_min
    pad = max(minimum_pad, spread * 0.20)
    low = max(0.0, y_min - pad)
    high = min(1.0, y_max + pad)
    if high - low < 0.05:
        center = (high + low) / 2
        low = max(0.0, center - 0.03)
        high = min(1.0, center + 0.03)
    return low, high


def short_level_label(axis_name: str, level: str) -> str:
    if axis_name == "Numeric":
        return {
            "off_by_1": "±1",
            "off_by_2": "±2",
            "off_by_5": "±5",
            "off_by_10": "±10",
            "random_offset": "Random",
        }.get(level, level)

    if axis_name == "Language":
        return {
            "might_be_wrong": "Might be wrong",
            "pretty_sure": "Pretty sure",
            "ta_says": "TA says",
            "official_key": "Official key",
            "autograder": "Autograder",
            "do_not_argue": "Do not argue",
        }.get(level, level)

    if axis_name == "Politeness":
        return {
            "polite_request": "Polite",
            "firm_correction": "Firm",
            "rude_or_insulting": "Rude",
            "threatening_complaint": "Threat",
        }.get(level, level)

    return level


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cross-axis breakpoint comparisons")
    parser.add_argument("--numeric-csv", required=True, help="Path to numeric breakpoint_sweep_levels.csv")
    parser.add_argument("--language-csv", required=True, help="Path to language breakpoint_sweep_levels.csv")
    parser.add_argument("--politeness-csv", required=True, help="Path to politeness breakpoint_sweep_levels.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory for combined plots")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    numeric = load_csv_rows(args.numeric_csv)
    language = load_csv_rows(args.language_csv)
    politeness = load_csv_rows(args.politeness_csv)

    axes_data = {
        "Numeric": numeric,
        "Language": language,
        "Politeness": politeness,
    }

    axis_colors = {
        "Numeric": "#1f77b4",
        "Language": "#ff7f0e",
        "Politeness": "#2ca02c",
    }

    # 1) Wrong-revision bars (one panel per axis, readable level names)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    all_wrong = []
    for rows in axes_data.values():
        all_wrong.extend(to_float(rows, "wrong_revision_rate"))
    wrong_low, wrong_high = _dynamic_ylim(all_wrong, minimum_pad=0.015)
    wrong_offset = max(0.001, (wrong_high - wrong_low) * 0.02)

    for i, (axis_name, rows) in enumerate(axes_data.items()):
        x = list(range(len(rows)))
        y = to_float(rows, "wrong_revision_rate")
        labels = [short_level_label(axis_name, r["level"]) for r in rows]

        bars = axs[i].bar(x, y, width=0.6, color=axis_colors[axis_name])
        annotate_bars(axs[i], bars, fmt="{:.3f}", offset=wrong_offset)
        axs[i].set_title(axis_name)
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(labels, rotation=25, ha="right")
        axs[i].set_ylim(wrong_low, wrong_high)
        axs[i].grid(alpha=0.3)
        axs[i].set_xlabel("")
        axs[i].tick_params(axis="y", labelleft=True)
    for ax in axs:
        ax.set_ylabel("Wrong revision rate")
    fig.suptitle("Wrong Revision Rate by Axis", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(os.path.join(args.output_dir, "cross_axis_wrong_revision_faceted.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # 2) Delta-accuracy bars
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    all_delta = []
    for rows in axes_data.values():
        all_delta.extend(to_float(rows, "delta_accuracy"))
    delta_low, delta_high = _dynamic_ylim(all_delta, minimum_pad=0.02)
    delta_offset = max(0.001, (delta_high - delta_low) * 0.02)

    for i, (axis_name, rows) in enumerate(axes_data.items()):
        x = list(range(len(rows)))
        y = to_float(rows, "delta_accuracy")
        labels = [short_level_label(axis_name, r["level"]) for r in rows]

        bars = axs[i].bar(x, y, width=0.6, color=axis_colors[axis_name])
        annotate_bars(axs[i], bars, fmt="{:.3f}", offset=delta_offset)
        axs[i].set_title(axis_name)
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(labels, rotation=25, ha="right")
        axs[i].set_ylim(delta_low, delta_high)
        axs[i].grid(alpha=0.3)
        axs[i].set_xlabel("")
        axs[i].tick_params(axis="y", labelleft=True)
    for ax in axs:
        ax.set_ylabel("Delta accuracy")
    fig.suptitle("Accuracy Drop by Axis", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(os.path.join(args.output_dir, "cross_axis_delta_accuracy_faceted.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # 3) Grouped accuracy bars per axis/level (baseline vs adversarial vs defense)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    bar_w = 0.24
    for i, (axis_name, rows) in enumerate(axes_data.items()):
        x = list(range(len(rows)))
        baseline = to_float(rows, "baseline_accuracy")
        adversarial = to_float(rows, "adversarial_accuracy")
        defense = to_float(rows, "defense_accuracy")
        labels = [short_level_label(axis_name, r["level"]) for r in rows]

        bars_base = axs[i].bar([v - bar_w for v in x], baseline, width=bar_w, label="Baseline", color="#4daf4a")
        bars_adv = axs[i].bar(x, adversarial, width=bar_w, label="Adversarial", color="#e41a1c")
        bars_def = axs[i].bar([v + bar_w for v in x], defense, width=bar_w, label="Defense", color="#377eb8")
        annotate_bars(axs[i], bars_base, fmt="{:.3f}", offset=0.003)
        annotate_bars(axs[i], bars_adv, fmt="{:.3f}", offset=0.001)
        annotate_bars(axs[i], bars_def, fmt="{:.3f}", offset=0.001)
        axs[i].set_title(axis_name)
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(labels, rotation=25, ha="right")
        axs[i].set_ylim(0, 0.85)
        axs[i].grid(alpha=0.25, axis="y")
        axs[i].set_xlabel("")
        axs[i].tick_params(axis="y", labelleft=True)

    for ax in axs:
        ax.set_ylabel("Accuracy")
    handles, labels = axs[0].get_legend_handles_labels()
    legend_map = {}
    for handle, label in zip(handles, labels):
        if label not in legend_map:
            legend_map[label] = handle
    fig.suptitle("Baseline vs Adversarial vs Defense Accuracy", y=0.99)
    fig.legend(
        list(legend_map.values()),
        list(legend_map.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    plt.savefig(os.path.join(args.output_dir, "cross_axis_accuracy_grouped_bars.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # 3b) Grouped bars (adversarial vs defense only, dynamic y-range)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    bar_w = 0.32
    all_adv_def_vals: List[float] = []
    for rows in axes_data.values():
        all_adv_def_vals.extend(to_float(rows, "adversarial_accuracy"))
        all_adv_def_vals.extend(to_float(rows, "defense_accuracy"))

    max_adv_def = max(all_adv_def_vals) if all_adv_def_vals else 0.1
    if max_adv_def <= 0.10:
        ylim_top = 0.08
        panel_title = "Adversarial vs Defense Accuracy (Zoomed)"
    else:
        ylim_top = min(1.0, max_adv_def * 1.15)
        panel_title = "Adversarial vs Defense Accuracy"

    for i, (axis_name, rows) in enumerate(axes_data.items()):
        x = list(range(len(rows)))
        adversarial = to_float(rows, "adversarial_accuracy")
        defense = to_float(rows, "defense_accuracy")
        labels = [short_level_label(axis_name, r["level"]) for r in rows]

        bars_adv = axs[i].bar([v - bar_w / 2 for v in x], adversarial, width=bar_w, label="Adversarial", color="#e41a1c")
        bars_def = axs[i].bar([v + bar_w / 2 for v in x], defense, width=bar_w, label="Defense", color="#377eb8")
        annotate_bars(axs[i], bars_adv, fmt="{:.3f}", offset=max(0.001, ylim_top * 0.01))
        annotate_bars(axs[i], bars_def, fmt="{:.3f}", offset=max(0.001, ylim_top * 0.01))
        axs[i].set_title(axis_name)
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(labels, rotation=25, ha="right")
        axs[i].set_ylim(0, ylim_top)
        axs[i].grid(alpha=0.25, axis="y")
        axs[i].set_xlabel("")
        axs[i].tick_params(axis="y", labelleft=True)

    for ax in axs:
        ax.set_ylabel("Accuracy")
    handles, labels = axs[0].get_legend_handles_labels()
    legend_map = {}
    for handle, label in zip(handles, labels):
        if label not in legend_map:
            legend_map[label] = handle
    fig.suptitle(panel_title, y=0.985)
    fig.legend(
        list(legend_map.values()),
        list(legend_map.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    plt.savefig(os.path.join(args.output_dir, "cross_axis_accuracy_zoomed_adv_def.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # 4) Axis-level averages for quick takeaway
    axis_names = list(axes_data.keys())
    avg_wrong = []
    avg_delta = []
    avg_defense = []
    for axis_name in axis_names:
        rows = axes_data[axis_name]
        avg_wrong.append(sum(to_float(rows, "wrong_revision_rate")) / len(rows))
        avg_delta.append(sum(to_float(rows, "delta_accuracy")) / len(rows))
        avg_defense.append(sum(to_float(rows, "defense_accuracy")) / len(rows))

    x = list(range(len(axis_names)))
    bar_w = 0.24
    plt.figure(figsize=(10, 5))
    plt.bar([v - bar_w for v in x], avg_wrong, width=bar_w, label="Avg Wrong Revision")
    plt.bar(x, avg_delta, width=bar_w, label="Avg Delta Accuracy")
    plt.bar([v + bar_w for v in x], avg_defense, width=bar_w, label="Avg Defense Accuracy")
    plt.xticks(x, axis_names)
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.title("Axis Averages (Quick Summary)")
    plt.grid(alpha=0.25, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "cross_axis_axis_averages.png"), dpi=180)
    plt.close()

    # 5) Creative alternative (heatmap): condition-level risk/impact metrics
    fig, axs = plt.subplots(1, 3, figsize=(17, 6), sharey=False)
    metric_labels = ["Wrong rev", "Delta acc", "Def-Adv"]

    heatmap_image = None
    all_metric_values: List[float] = []
    for rows in axes_data.values():
        all_metric_values.extend(to_float(rows, "wrong_revision_rate"))
        all_metric_values.extend(to_float(rows, "delta_accuracy"))
        all_metric_values.extend(
            [float(r["defense_accuracy"]) - float(r["adversarial_accuracy"]) for r in rows]
        )
    m_vmin = min(all_metric_values) if all_metric_values else -0.1
    m_vmax = max(all_metric_values) if all_metric_values else 1.0

    for i, (axis_name, rows) in enumerate(axes_data.items()):
        matrix = [
            [
                float(r["wrong_revision_rate"]),
                float(r["delta_accuracy"]),
                float(r["defense_accuracy"]) - float(r["adversarial_accuracy"]),
            ]
            for r in rows
        ]

        heatmap_image = axs[i].imshow(matrix, aspect="auto", cmap="viridis", vmin=m_vmin, vmax=m_vmax)
        row_labels = [short_level_label(axis_name, r["level"]) for r in rows]

        axs[i].set_title(axis_name)
        axs[i].set_xticks([0, 1, 2])
        axs[i].set_xticklabels(metric_labels, rotation=20, ha="right")
        axs[i].set_yticks(list(range(len(row_labels))))
        axs[i].set_yticklabels(row_labels)

        for r_idx, row_vals in enumerate(matrix):
            for c_idx, value in enumerate(row_vals):
                text_color = "white" if value > (m_vmin + m_vmax) / 2 else "black"
                axs[i].text(c_idx, r_idx, f"{value:.3f}", ha="center", va="center", fontsize=8, color=text_color)

    fig.suptitle("Condition Risk/Impact Heatmap", y=0.98)
    if heatmap_image is not None:
        cax = fig.add_axes([0.92, 0.19, 0.015, 0.63])
        cbar = fig.colorbar(heatmap_image, cax=cax)
        cbar.set_label("Metric value")
    fig.subplots_adjust(left=0.08, right=0.90, top=0.88, bottom=0.16, wspace=0.24)
    plt.savefig(os.path.join(args.output_dir, "cross_axis_delta_accuracy_ranked.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # 6) Creative alternative: axis-specific heatmaps for baseline/adversarial/defense
    fig, axs = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
    stage_labels = ["Baseline", "Adversarial", "Defense"]

    all_stage_values: List[float] = []
    for rows in axes_data.values():
        all_stage_values.extend(to_float(rows, "baseline_accuracy"))
        all_stage_values.extend(to_float(rows, "adversarial_accuracy"))
        all_stage_values.extend(to_float(rows, "defense_accuracy"))
    vmin = min(all_stage_values) if all_stage_values else 0.0
    vmax = max(all_stage_values) if all_stage_values else 1.0

    heatmap_image = None
    for i, (axis_name, rows) in enumerate(axes_data.items()):
        matrix = [
            [float(r["baseline_accuracy"]), float(r["adversarial_accuracy"]), float(r["defense_accuracy"])]
            for r in rows
        ]
        heatmap_image = axs[i].imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)

        row_labels = [short_level_label(axis_name, r["level"]) for r in rows]
        axs[i].set_title(axis_name)
        axs[i].set_xticks([0, 1, 2])
        axs[i].set_xticklabels(stage_labels, rotation=20, ha="right")
        axs[i].set_yticks(list(range(len(row_labels))))
        axs[i].set_yticklabels(row_labels)
        axs[i].tick_params(axis="y", labelleft=True)

        for r_idx, row_vals in enumerate(matrix):
            for c_idx, value in enumerate(row_vals):
                axs[i].text(c_idx, r_idx, f"{value:.3f}", ha="center", va="center", fontsize=8, color="black")

    fig.suptitle("Accuracy Heatmap by Condition (Baseline → Adversarial → Defense)", y=0.98)
    if heatmap_image is not None:
        cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
        cbar = fig.colorbar(heatmap_image, cax=cax)
        cbar.set_label("Accuracy")
    fig.subplots_adjust(left=0.08, right=0.90, top=0.90, bottom=0.16, wspace=0.22)
    plt.savefig(os.path.join(args.output_dir, "cross_axis_accuracy_heatmap.png"), dpi=180, bbox_inches="tight")
    plt.close()

    summary = {
        "sources": {
            "numeric_csv": args.numeric_csv,
            "language_csv": args.language_csv,
            "politeness_csv": args.politeness_csv,
        },
        "num_levels": {
            "numeric": len(numeric),
            "language": len(language),
            "politeness": len(politeness),
        },
        "outputs": {
            "wrong_revision_faceted": os.path.join(args.output_dir, "cross_axis_wrong_revision_faceted.png"),
            "delta_accuracy_faceted": os.path.join(args.output_dir, "cross_axis_delta_accuracy_faceted.png"),
            "accuracy_grouped_bars": os.path.join(args.output_dir, "cross_axis_accuracy_grouped_bars.png"),
            "accuracy_zoomed_adv_def": os.path.join(args.output_dir, "cross_axis_accuracy_zoomed_adv_def.png"),
            "axis_averages": os.path.join(args.output_dir, "cross_axis_axis_averages.png"),
            "delta_accuracy_ranked": os.path.join(args.output_dir, "cross_axis_delta_accuracy_ranked.png"),
            "accuracy_heatmap": os.path.join(args.output_dir, "cross_axis_accuracy_heatmap.png"),
        },
    }

    summary_path = os.path.join(args.output_dir, "cross_axis_plot_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Combined plots written to:", args.output_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
