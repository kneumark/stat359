#!/usr/bin/env python3
"""Run a baseline vs post-finetune robustness comparison for Arithmetic LLM.

This utility standardizes a common project workflow:
1) Evaluate a baseline checkpoint on OOD gaslighting robustness.
2) Optionally run targeted instruction fine-tuning from a checkpoint.
3) Evaluate the post-finetune checkpoint with the exact same robustness settings.
4) Save a compact before/after comparison summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Optional

from .run_ood_gaslighting_experiment import (
    run_experiment,
    NUMERIC_OFFSET_LEVELS,
    LANGUAGE_PRESSURE_TEXT,
    POLITENESS_TEXT,
)
from .train_instruction import train_instruction_model
from .training_config import TrainingConfig
from .output_naming import create_numbered_output_dir


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_summary(
    summary_path: str,
    baseline_metrics: Dict[str, Any],
    post_metrics: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    baseline = baseline_metrics["overall"]
    post = post_metrics["overall"]

    keys = [
        "control_accuracy",
        "adversarial_accuracy",
        "defense_accuracy",
        "wrong_flip_rate_adversarial",
        "wrong_flip_rate_defense",
    ]

    deltas = {key: post.get(key, 0.0) - baseline.get(key, 0.0) for key in keys}

    summary = {
        "metadata": metadata,
        "baseline": baseline,
        "post": post,
        "deltas": deltas,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def write_comparison_csv(
    csv_path: str,
    baseline_metrics: Dict[str, Any],
    post_metrics: Dict[str, Any],
) -> None:
    baseline = baseline_metrics["overall"]
    post = post_metrics["overall"]

    fields = [
        "metric",
        "baseline",
        "post",
        "delta",
    ]
    rows = [
        "control_accuracy",
        "adversarial_accuracy",
        "defense_accuracy",
        "wrong_flip_rate_adversarial",
        "wrong_flip_rate_defense",
        "gaslight_eligible_count",
        "num_samples",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for metric in rows:
            base_val = baseline.get(metric, 0.0)
            post_val = post.get(metric, 0.0)
            delta = (
                (post_val - base_val)
                if isinstance(base_val, (int, float)) and isinstance(post_val, (int, float))
                else ""
            )
            writer.writerow(
                {
                    "metric": metric,
                    "baseline": base_val,
                    "post": post_val,
                    "delta": delta,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline vs post-finetune robustness comparison"
    )

    parser.add_argument("--baseline-model-path", type=str, required=True, help="Baseline checkpoint path")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Tokenizer directory")
    parser.add_argument("--output-dir", type=str, default="proposal_before_after", help="Output root directory")

    parser.add_argument(
        "--after-model-path",
        type=str,
        default="",
        help="Optional post model checkpoint path (if set, skip finetuning)",
    )
    parser.add_argument(
        "--instruction-corpus-path",
        type=str,
        default="",
        help="Instruction corpus for optional finetuning",
    )
    parser.add_argument(
        "--finetune-checkpoint",
        type=str,
        default="",
        help="Checkpoint to finetune from (default: baseline model)",
    )
    parser.add_argument("--finetune-learning-rate", type=float, default=2e-5)
    parser.add_argument("--finetune-batch-size", type=int, default=32)
    parser.add_argument("--finetune-num-epochs", type=int, default=2)
    parser.add_argument("--finetune-warmup-steps", type=int, default=300)
    parser.add_argument("--finetune-gradient-clip", type=float, default=1.0)
    parser.add_argument("--finetune-save-every", type=int, default=500)

    parser.add_argument("--device", type=str, default="auto", help="cuda|mps|cpu|auto")
    parser.add_argument("--max-gen-length", type=int, default=256)
    parser.add_argument("--num-per-bucket", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="medium",
        choices=["very_mild", "mild", "medium", "strong", "very_strong", "extreme"],
    )
    parser.add_argument(
        "--difficulty-preset",
        type=str,
        default="high_control",
        choices=["original", "easier", "high_control"],
    )
    parser.add_argument("--buckets", type=str, default="in_distribution,slight_ood,moderate_ood")
    parser.add_argument("--output-mode", type=str, default="strict", choices=["strict", "legacy"])

    parser.add_argument(
        "--active-axis",
        type=str,
        default="legacy",
        choices=["legacy", "numeric", "language", "politeness"],
    )
    parser.add_argument(
        "--numeric-offset-level",
        type=str,
        default="off_by_2",
        choices=NUMERIC_OFFSET_LEVELS,
    )
    parser.add_argument(
        "--language-pressure-level",
        type=str,
        default="ta_says",
        choices=list(LANGUAGE_PRESSURE_TEXT.keys()),
    )
    parser.add_argument(
        "--politeness-level",
        type=str,
        default="firm_correction",
        choices=list(POLITENESS_TEXT.keys()),
    )

    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B for finetune run")
    parser.add_argument("--wandb-project", type=str, default="stat359-arithmetic-llm")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="before-after")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )

    args = parser.parse_args()

    device = resolve_device(args.device)
    buckets = [item.strip() for item in args.buckets.split(",") if item.strip()]

    run_dir = create_numbered_output_dir(args.output_dir, "before_after")
    run_id = os.path.basename(run_dir)
    baseline_dir = os.path.join(run_dir, "baseline_eval")
    post_dir = os.path.join(run_dir, "post_eval")
    finetune_dir = os.path.join(run_dir, "finetune_models")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 72)
    print("STEP 1/3: BASELINE ROBUSTNESS EVALUATION")
    print("=" * 72)
    baseline_result = run_experiment(
        model_path=args.baseline_model_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=baseline_dir,
        num_per_bucket=args.num_per_bucket,
        seed=args.seed,
        device=device,
        max_gen_length=args.max_gen_length,
        prompt_style=args.prompt_style,
        difficulty_preset=args.difficulty_preset,
        control_only=False,
        selected_buckets=buckets,
        output_mode=args.output_mode,
        active_axis=args.active_axis,
        numeric_offset_level=args.numeric_offset_level,
        language_pressure_level=args.language_pressure_level,
        politeness_level=args.politeness_level,
    )

    post_model_path: Optional[str] = None

    if args.after_model_path:
        post_model_path = args.after_model_path
        print("\nUsing provided post model path; skipping finetuning.")
    elif args.instruction_corpus_path:
        print("\n" + "=" * 72)
        print("STEP 2/3: FINETUNE FROM CHECKPOINT")
        print("=" * 72)
        finetune_source = args.finetune_checkpoint or args.baseline_model_path
        finetune_config = TrainingConfig(
            learning_rate=args.finetune_learning_rate,
            batch_size=args.finetune_batch_size,
            num_epochs=args.finetune_num_epochs,
            warmup_steps=args.finetune_warmup_steps,
            gradient_clip=args.finetune_gradient_clip,
            save_every=args.finetune_save_every,
            device=device,
        )

        final_ckpt = train_instruction_model(
            instruction_corpus_path=args.instruction_corpus_path,
            tokenizer_path=args.tokenizer_path,
            foundational_checkpoint=finetune_source,
            output_dir=finetune_dir,
            config=finetune_config,
            model_config=None,
            wandb_enabled=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=(args.wandb_entity or None),
            wandb_run_name=(args.wandb_run_name or f"before-after-{run_id}"),
            wandb_tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()],
            wandb_mode=args.wandb_mode,
        )

        candidate_best = os.path.join(os.path.dirname(final_ckpt), "best_model.pt")
        post_model_path = candidate_best if os.path.exists(candidate_best) else final_ckpt
    else:
        raise ValueError(
            "Need either --after-model-path (compare two existing models) or "
            "--instruction-corpus-path (run finetune)."
        )

    print("\n" + "=" * 72)
    print("STEP 3/3: POST ROBUSTNESS EVALUATION")
    print("=" * 72)
    post_result = run_experiment(
        model_path=post_model_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=post_dir,
        num_per_bucket=args.num_per_bucket,
        seed=args.seed,
        device=device,
        max_gen_length=args.max_gen_length,
        prompt_style=args.prompt_style,
        difficulty_preset=args.difficulty_preset,
        control_only=False,
        selected_buckets=buckets,
        output_mode=args.output_mode,
        active_axis=args.active_axis,
        numeric_offset_level=args.numeric_offset_level,
        language_pressure_level=args.language_pressure_level,
        politeness_level=args.politeness_level,
    )

    summary_json = os.path.join(run_dir, "before_after_summary.json")
    summary_csv = os.path.join(run_dir, "before_after_summary.csv")

    write_summary(
        summary_path=summary_json,
        baseline_metrics=baseline_result["metrics"],
        post_metrics=post_result["metrics"],
        metadata={
            "run_id": run_id,
            "device": device,
            "baseline_model_path": args.baseline_model_path,
            "post_model_path": post_model_path,
            "tokenizer_path": args.tokenizer_path,
            "num_per_bucket": args.num_per_bucket,
            "difficulty_preset": args.difficulty_preset,
            "buckets": buckets,
            "prompt_style": args.prompt_style,
            "output_mode": args.output_mode,
            "active_axis": args.active_axis,
            "numeric_offset_level": args.numeric_offset_level,
            "language_pressure_level": args.language_pressure_level,
            "politeness_level": args.politeness_level,
        },
    )
    write_comparison_csv(
        csv_path=summary_csv,
        baseline_metrics=baseline_result["metrics"],
        post_metrics=post_result["metrics"],
    )

    print("\n" + "=" * 72)
    print("BEFORE/AFTER ROBUSTNESS RUN COMPLETE")
    print("=" * 72)
    print(f"Run directory: {run_dir}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
