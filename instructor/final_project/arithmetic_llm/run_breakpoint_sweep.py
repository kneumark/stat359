#!/usr/bin/env python3
"""Run pressure-ladder sweep and estimate breakpoint shifts.

This script repeatedly calls `run_experiment` with increasing adversarial
prompt strength, then estimates where wrong-revision rate increases sharply.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from .run_ood_gaslighting_experiment import run_experiment


NUMERIC_LEVELS = ["off_by_1", "off_by_2", "off_by_5", "off_by_10", "random_offset"]
LANGUAGE_LEVELS = [
    "might_be_wrong",
    "pretty_sure",
    "ta_says",
    "official_key",
    "autograder",
    "do_not_argue",
]
POLITENESS_LEVELS = ["polite_request", "firm_correction", "rude_or_insulting", "threatening_complaint"]


def _threshold_crossing(level_values: List[float], threshold: float) -> Optional[float]:
    if not level_values:
        return None

    if level_values[0] >= threshold:
        return 0.0

    for i in range(1, len(level_values)):
        prev_val = level_values[i - 1]
        curr_val = level_values[i]
        if curr_val >= threshold:
            if curr_val == prev_val:
                return float(i)
            frac = (threshold - prev_val) / (curr_val - prev_val)
            return (i - 1) + max(0.0, min(1.0, frac))

    return None


def _steepest_slope(level_values: List[float]) -> Optional[Dict[str, Any]]:
    if len(level_values) < 2:
        return None

    best_idx = 0
    best_slope = level_values[1] - level_values[0]
    for i in range(1, len(level_values) - 1):
        slope = level_values[i + 1] - level_values[i]
        if slope > best_slope:
            best_slope = slope
            best_idx = i

    return {
        "from_index": best_idx,
        "to_index": best_idx + 1,
        "delta": best_slope,
        "midpoint_index": best_idx + 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adversarial pressure sweep and estimate breakpoint")
    parser.add_argument("--model-path", required=True, help="Path to instruction model checkpoint")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer directory")
    parser.add_argument("--output-dir", default="proposal_breakpoint_sweep", help="Output directory")
    parser.add_argument("--num-per-bucket", type=int, default=100, help="Samples per bucket per level")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--device", default="auto", help="cuda|mps|cpu|auto")
    parser.add_argument("--max-gen-length", type=int, default=128, help="Generation max length")
    parser.add_argument(
        "--difficulty-preset",
        type=str,
        default="high_control",
        choices=["original", "easier", "high_control"],
        help="Difficulty preset",
    )
    parser.add_argument(
        "--buckets",
        type=str,
        default="in_distribution,slight_ood,moderate_ood",
        help="Comma-separated buckets to include",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="numeric",
        choices=["numeric", "language", "politeness"],
        help="Which adversarial axis to sweep while holding others fixed",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="",
        help="Optional comma-separated levels for selected axis (default uses full axis ladder)",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default="legacy",
        choices=["legacy", "strict"],
        help="Use legacy tolerant parse or strict parse mode",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Wrong-revision threshold used for breakpoint crossing estimate",
    )

    args = parser.parse_args()

    if args.device == "auto":
        import torch

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = args.device

    os.makedirs(args.output_dir, exist_ok=True)
    selected_buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]

    if args.levels.strip():
        levels = [lvl.strip() for lvl in args.levels.split(",") if lvl.strip()]
    elif args.axis == "numeric":
        levels = NUMERIC_LEVELS
    elif args.axis == "language":
        levels = LANGUAGE_LEVELS
    else:
        levels = POLITENESS_LEVELS

    per_level_rows: List[Dict[str, Any]] = []

    for idx, level in enumerate(levels):
        print(f"\n=== Running level {idx + 1}/{len(levels)}: {level} ===")
        level_dir = os.path.join(args.output_dir, f"level_{idx + 1}_{level}")
        os.makedirs(level_dir, exist_ok=True)

        result = run_experiment(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            output_dir=level_dir,
            num_per_bucket=args.num_per_bucket,
            seed=args.seed + idx,
            device=device,
            max_gen_length=args.max_gen_length,
            prompt_style="medium",
            difficulty_preset=args.difficulty_preset,
            control_only=False,
            selected_buckets=selected_buckets if selected_buckets else None,
            output_mode=args.output_mode,
            active_axis=args.axis,
            numeric_offset_level=(level if args.axis == "numeric" else "off_by_2"),
            language_pressure_level=(level if args.axis == "language" else "ta_says"),
            politeness_level=(level if args.axis == "politeness" else "firm_correction"),
        )

        overall = result["metrics"]["overall"]
        baseline_accuracy = overall["control_accuracy"]
        adversarial_accuracy = overall["adversarial_accuracy"]
        wrong_revision_rate = overall["wrong_flip_rate_adversarial"]
        row = {
            "axis": args.axis,
            "level": level,
            "level_index": idx,
            "num_samples": overall["num_samples"],
            "eligible_count": overall["gaslight_eligible_count"],
            "baseline_accuracy": baseline_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
            "defense_accuracy": overall["defense_accuracy"],
            "wrong_revision_rate": wrong_revision_rate,
            "wrong_flip_rate_defense": overall["wrong_flip_rate_defense"],
            "delta_accuracy": baseline_accuracy - adversarial_accuracy,
            "flip_rate_adversarial": overall["flip_rate_adversarial"],
            "flip_rate_defense": overall["flip_rate_defense"],
            "summary_path": result["summary_path"],
            "metrics_json_path": result["metrics_json_path"],
            "raw_path": result["raw_path"],
        }
        per_level_rows.append(row)

    adv_curve = [r["wrong_revision_rate"] for r in per_level_rows]
    defense_curve = [r["wrong_flip_rate_defense"] for r in per_level_rows]

    adv_break_cross = _threshold_crossing(adv_curve, args.threshold)
    def_break_cross = _threshold_crossing(defense_curve, args.threshold)

    adv_steep = _steepest_slope(adv_curve)
    def_steep = _steepest_slope(defense_curve)

    analysis = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path,
            "output_dir": args.output_dir,
            "device": device,
            "axis": args.axis,
            "num_per_bucket": args.num_per_bucket,
            "difficulty_preset": args.difficulty_preset,
            "buckets": selected_buckets,
            "levels": levels,
            "output_mode": args.output_mode,
            "threshold": args.threshold,
        },
        "per_level": per_level_rows,
        "breakpoint": {
            "adversarial_threshold_crossing_index": adv_break_cross,
            "defense_threshold_crossing_index": def_break_cross,
            "threshold_crossing_shift_defense_minus_adversarial": (
                (def_break_cross - adv_break_cross)
                if (def_break_cross is not None and adv_break_cross is not None)
                else None
            ),
            "adversarial_steepest_slope": adv_steep,
            "defense_steepest_slope": def_steep,
            "steepest_midpoint_shift_defense_minus_adversarial": (
                (def_steep["midpoint_index"] - adv_steep["midpoint_index"])
                if (def_steep is not None and adv_steep is not None)
                else None
            ),
        },
    }

    breakpoint_estimate = {
        "adversarial_threshold_crossing_index": adv_break_cross,
        "defense_threshold_crossing_index": def_break_cross,
        "threshold_crossing_shift_defense_minus_adversarial": analysis["breakpoint"]["threshold_crossing_shift_defense_minus_adversarial"],
        "adversarial_steepest_midpoint_index": (adv_steep["midpoint_index"] if adv_steep else None),
        "defense_steepest_midpoint_index": (def_steep["midpoint_index"] if def_steep else None),
    }

    grouped_metrics = []
    for row in per_level_rows:
        grouped_metrics.append(
            {
                "axis": args.axis,
                "level": row["level"],
                "wrong_revision_rate": row["wrong_revision_rate"],
                "baseline_accuracy": row["baseline_accuracy"],
                "delta_accuracy": row["delta_accuracy"],
                "breakpoint_estimate": breakpoint_estimate,
                "eligible_count": row["eligible_count"],
            }
        )

    analysis["grouped_metrics"] = grouped_metrics

    json_path = os.path.join(args.output_dir, "breakpoint_sweep_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    csv_path = os.path.join(args.output_dir, "breakpoint_sweep_levels.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "axis",
                "level",
                "level_index",
                "num_samples",
                "eligible_count",
                "baseline_accuracy",
                "adversarial_accuracy",
                "defense_accuracy",
                "wrong_revision_rate",
                "wrong_flip_rate_defense",
                "delta_accuracy",
                "flip_rate_adversarial",
                "flip_rate_defense",
                "summary_path",
                "metrics_json_path",
                "raw_path",
            ],
        )
        writer.writeheader()
        for row in per_level_rows:
            writer.writerow(row)

    x = list(range(len(levels)))
    plt.figure(figsize=(9, 5))
    plt.plot(x, adv_curve, marker="o", label="Wrong-revision rate (Adversarial)")
    plt.plot(x, defense_curve, marker="o", label="Wrong-revision rate (Defense)")
    plt.xticks(x, levels, rotation=25)
    plt.ylim(0, 1)
    plt.xlabel("Attack pressure level")
    plt.ylabel("Wrong revision rate")
    plt.title("Breakpoint Sweep: Wrong Revision vs Pressure")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "breakpoint_sweep_curve.png")
    plt.savefig(plot_path, dpi=180)
    plt.close()

    txt_path = os.path.join(args.output_dir, "breakpoint_sweep_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("BREAKPOINT SWEEP SUMMARY\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Active axis: {args.axis}\n")
        f.write("Levels (in order): " + ", ".join(levels) + "\n")
        f.write(f"Threshold: {args.threshold}\n\n")
        f.write("Per-level grouped metrics\n")
        for r in per_level_rows:
            f.write(
                f"- {r['level']}: wrong_revision_rate={r['wrong_revision_rate']:.4f}, "
                f"baseline_accuracy={r['baseline_accuracy']:.4f}, "
                f"delta_accuracy={r['delta_accuracy']:.4f}, eligible={r['eligible_count']}\n"
            )
        f.write("\nBreakpoint estimate\n")
        f.write(f"- adversarial threshold crossing index: {adv_break_cross}\n")
        f.write(f"- defense threshold crossing index: {def_break_cross}\n")
        f.write(
            "- threshold crossing shift (defense - adversarial): "
            f"{analysis['breakpoint']['threshold_crossing_shift_defense_minus_adversarial']}\n"
        )
        f.write(f"- adversarial steepest slope: {adv_steep}\n")
        f.write(f"- defense steepest slope: {def_steep}\n")
        f.write(
            "- steepest midpoint shift (defense - adversarial): "
            f"{analysis['breakpoint']['steepest_midpoint_shift_defense_minus_adversarial']}\n"
        )

    print("\n" + "=" * 60)
    print("BREAKPOINT SWEEP COMPLETE")
    print("=" * 60)
    print("Summary JSON:", json_path)
    print("Summary CSV:", csv_path)
    print("Summary TXT:", txt_path)
    print("Curve PNG:", plot_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
