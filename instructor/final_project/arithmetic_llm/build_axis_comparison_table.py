#!/usr/bin/env python3
"""Build a consolidated comparison table across numeric/language/politeness sweeps."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fmt_pct(x: str) -> str:
    return f"{float(x) * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-axis comparison table")
    parser.add_argument("--numeric-csv", required=True)
    parser.add_argument("--language-csv", required=True)
    parser.add_argument("--politeness-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    data = {
        "Numeric": read_csv(args.numeric_csv),
        "Language": read_csv(args.language_csv),
        "Politeness": read_csv(args.politeness_csv),
    }

    consolidated_rows: List[Dict[str, str]] = []
    for axis_name, rows in data.items():
        for row in rows:
            consolidated_rows.append(
                {
                    "axis": axis_name,
                    "level": row["level"],
                    "num_samples": row["num_samples"],
                    "eligible_count": row["eligible_count"],
                    "baseline_accuracy": row["baseline_accuracy"],
                    "adversarial_accuracy": row["adversarial_accuracy"],
                    "defense_accuracy": row["defense_accuracy"],
                    "wrong_revision_rate": row["wrong_revision_rate"],
                    "wrong_flip_rate_defense": row["wrong_flip_rate_defense"],
                    "delta_accuracy": row["delta_accuracy"],
                    "flip_rate_adversarial": row["flip_rate_adversarial"],
                    "flip_rate_defense": row["flip_rate_defense"],
                }
            )

    out_csv = os.path.join(args.output_dir, "axis_comparison_table.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "axis",
                "level",
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
            ],
        )
        writer.writeheader()
        for row in consolidated_rows:
            writer.writerow(row)

    out_md = os.path.join(args.output_dir, "axis_comparison_table.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Consolidated Axis Comparison\n\n")
        for axis_name in ["Numeric", "Language", "Politeness"]:
            f.write(f"## {axis_name}\n\n")
            f.write("| Level | Baseline | Adversarial | Defense | Wrong Revision | Delta Accuracy | Eligible |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for row in [r for r in consolidated_rows if r["axis"] == axis_name]:
                f.write(
                    "| {level} | {baseline} | {adv} | {defense} | {wrong_rev} | {delta} | {eligible} |\n".format(
                        level=row["level"],
                        baseline=fmt_pct(row["baseline_accuracy"]),
                        adv=fmt_pct(row["adversarial_accuracy"]),
                        defense=fmt_pct(row["defense_accuracy"]),
                        wrong_rev=fmt_pct(row["wrong_revision_rate"]),
                        delta=fmt_pct(row["delta_accuracy"]),
                        eligible=row["eligible_count"],
                    )
                )
            f.write("\n")

    print("Wrote:", out_csv)
    print("Wrote:", out_md)


if __name__ == "__main__":
    main()
