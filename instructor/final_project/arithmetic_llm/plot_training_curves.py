#!/usr/bin/env python3
"""Plot training/validation curves from training logs.

Generates:
- train/val loss vs step
- train/val perplexity vs step (exp(loss), clipped for stability)
- learning rate vs step
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_training_log(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Training log is empty or malformed: {path}")
    return data


def maybe_load_training_step_log(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def to_perplexity(loss: float) -> float:
    # Clip for numerical stability in case of unusually large loss values.
    return math.exp(min(20.0, max(-20.0, float(loss))))


def smooth_series(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) <= 2:
        return values

    smoothed: List[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        current_window = min(window, index + 1)
        smoothed.append(running_sum / current_window)
    return smoothed


def ema_series(values: List[float], alpha: float) -> List[float]:
    if not values:
        return values
    smoothed: List[float] = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return smoothed


def downsample_xy(x: List[int], y: List[float], max_points: int = 1200) -> tuple[List[int], List[float]]:
    if len(x) <= max_points:
        return x, y
    stride = max(1, len(x) // max_points)
    x_ds = x[::stride]
    y_ds = y[::stride]
    if x_ds[-1] != x[-1]:
        x_ds.append(x[-1])
        y_ds.append(y[-1])
    return x_ds, y_ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training/validation curves from JSON logs")
    parser.add_argument("--training-log", required=True, help="Path to training_log.json")
    parser.add_argument(
        "--training-step-log",
        default="",
        help="Optional path to training_step_log.json (if omitted, auto-detected next to training_log)"
    )
    parser.add_argument(
        "--x-axis",
        choices=["epoch", "step"],
        default="epoch",
        help="X-axis mode for plots (default: epoch)"
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save plots")
    parser.add_argument("--title-prefix", default="", help="Optional title prefix")
    args = parser.parse_args()

    rows = load_training_log(args.training_log)
    step_log_path = args.training_step_log or os.path.join(
        os.path.dirname(args.training_log),
        "training_step_log.json"
    )
    step_rows = maybe_load_training_step_log(step_log_path)
    ensure_dir(args.output_dir)

    val_steps = [int(r["step"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    epochs = [int(r["epoch"]) for r in rows]
    epoch_train_loss = [float(r["train_loss"]) for r in rows]
    epoch_lr = [float(r["learning_rate"]) for r in rows]

    if step_rows:
        train_steps = [int(r["step"]) for r in step_rows]
        train_loss = [float(r["train_loss"]) for r in step_rows]
        lr_steps = [int(r["step"]) for r in step_rows]
        lr = [float(r["learning_rate"]) for r in step_rows]
    else:
        train_steps = [int(r["step"]) for r in rows]
        train_loss = [float(r["train_loss"]) for r in rows]
        lr_steps = [int(r["step"]) for r in rows]
        lr = [float(r["learning_rate"]) for r in rows]

    if args.x_axis == "step":
        smooth_window = max(30, len(train_loss) // 80)
        train_loss_smoothed = smooth_series(train_loss, smooth_window)
        ema_alpha = min(0.08, max(0.015, 180.0 / max(1, len(train_loss_smoothed))))
        train_loss_trend = ema_series(train_loss_smoothed, ema_alpha)
        train_steps_ds, train_loss_trend_ds = downsample_xy(train_steps, train_loss_trend)
        train_ppl_ds = [to_perplexity(v) for v in train_loss_trend_ds]
        val_ppl = [to_perplexity(v) for v in val_loss]
    else:
        smooth_window = 1
        train_steps_ds = epochs
        train_loss_trend_ds = epoch_train_loss
        lr_steps = epochs
        lr = epoch_lr
        val_steps = epochs
        train_ppl_ds = [to_perplexity(v) for v in train_loss_trend_ds]
        val_ppl = [to_perplexity(v) for v in val_loss]

    title_prefix = f"{args.title_prefix} - " if args.title_prefix else ""

    x_label = "Step" if args.x_axis == "step" else "Epoch"

    # 1) Loss curves (separate panels for scale correctness)
    fig, (ax_train, ax_val) = plt.subplots(
        2,
        1,
        figsize=(9, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]}
    )

    ax_train.plot(
        train_steps_ds,
        train_loss_trend_ds,
        linewidth=2.0,
        color="#1f77b4",
        label=(
            f"Train Loss Trend (MA+EMA, w={smooth_window})"
            if args.x_axis == "step"
            else "Train Loss"
        )
    )
    ax_train.set_ylabel("Train Loss")
    ax_train.set_title(f"{title_prefix}Loss by {x_label}")
    ax_train.grid(alpha=0.3)
    ax_train.legend(loc="upper right")

    ax_val.plot(
        val_steps,
        val_loss,
        marker="o",
        linewidth=2.0,
        color="#ff7f0e",
        label="Validation Loss"
    )
    ax_val.set_xlabel(x_label)
    ax_val.set_ylabel("Val Loss")
    ax_val.grid(alpha=0.3)
    ax_val.legend(loc="upper right")

    plt.tight_layout()
    loss_path = os.path.join(args.output_dir, "training_validation_loss_curve.png")
    fig.savefig(loss_path, dpi=180)
    plt.close(fig)

    # 2) Perplexity curves (separate panels for scale correctness)
    fig, (ax_train_ppl, ax_val_ppl) = plt.subplots(
        2,
        1,
        figsize=(9, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]}
    )

    ax_train_ppl.plot(
        train_steps_ds,
        train_ppl_ds,
        linewidth=2.0,
        color="#1f77b4",
        label="Train Perplexity"
    )
    ax_train_ppl.set_ylabel("Train PPL")
    ax_train_ppl.set_title(f"{title_prefix}Perplexity by {x_label}")
    ax_train_ppl.grid(alpha=0.3)
    ax_train_ppl.legend(loc="upper right")

    ax_val_ppl.plot(
        val_steps,
        val_ppl,
        marker="o",
        linewidth=2.0,
        color="#ff7f0e",
        label="Validation Perplexity"
    )
    ax_val_ppl.set_xlabel(x_label)
    ax_val_ppl.set_ylabel("Val PPL")
    ax_val_ppl.grid(alpha=0.3)
    ax_val_ppl.legend(loc="upper right")

    plt.tight_layout()
    ppl_path = os.path.join(args.output_dir, "training_validation_perplexity_curve.png")
    fig.savefig(ppl_path, dpi=180)
    plt.close(fig)

    # 3) Learning rate curve
    plt.figure(figsize=(8, 5))
    plt.plot(lr_steps, lr, color="#ff7f0e", label="Learning Rate")
    plt.xlabel(x_label)
    plt.ylabel("Learning Rate")
    plt.title(f"{title_prefix}Learning Rate by {x_label}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    lr_path = os.path.join(args.output_dir, "learning_rate_curve.png")
    plt.savefig(lr_path, dpi=180)
    plt.close()

    summary = {
        "training_log": args.training_log,
        "training_step_log": step_log_path if step_rows else "",
        "x_axis": args.x_axis,
        "num_epochs": len(rows),
        "num_train_points": len(train_steps),
        "num_val_points": len(val_steps),
        "plots": {
            "loss": loss_path,
            "perplexity": ppl_path,
            "learning_rate": lr_path,
        },
    }
    summary_path = os.path.join(args.output_dir, "training_curves_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:", loss_path)
    print("Wrote:", ppl_path)
    print("Wrote:", lr_path)
    print("Wrote:", summary_path)


if __name__ == "__main__":
    main()
