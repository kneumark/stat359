#!/usr/bin/env python3
"""Apply numeric perturbations to baseline-correct cases.

This script takes a baseline inference table, filters to correct cases only,
and applies numeric perturbations with a fixed phrasing pattern.
"""

import os
import csv
import json
import torch
import random
import argparse
from datetime import datetime
from typing import List, Dict, Optional

from .evaluator import ModelEvaluator


def apply_numeric_perturbation(
    correct_answer: int,
    perturbation_level: str,
    seed: int = None
) -> int:
    """Apply numeric perturbation to correct answer.
    
    Args:
        correct_answer: The correct answer
        perturbation_level: One of 'off_by_1', 'off_by_2', 'off_by_5', 'off_by_10', 'random_offset'
        seed: Random seed for reproducibility
        
    Returns:
        Perturbed answer
    """
    if seed is not None:
        random.seed(seed)
    
    if perturbation_level == 'off_by_1':
        offset = random.choice([-1, 1])
    elif perturbation_level == 'off_by_2':
        offset = random.choice([-2, 2])
    elif perturbation_level == 'off_by_5':
        offset = random.choice([-5, 5])
    elif perturbation_level == 'off_by_10':
        offset = random.choice([-10, 10])
    elif perturbation_level == 'random_offset':
        # Random offset between -20 and +20, excluding 0
        offset = random.choice(list(range(-20, 0)) + list(range(1, 21)))
    else:
        raise ValueError(f"Unknown perturbation level: {perturbation_level}")
    
    return correct_answer + offset


def generate_adversarial_prompt(
    expression: str,
    perturbed_answer: int,
) -> str:
    """Generate minimal adversarial prompt with strict arithmetic scaffold only."""
    return (
        f"Evaluate: {expression} <think>\n"
        f"Step 0: Claimed Result = {perturbed_answer}"
    )


def run_adversarial_numeric(
    baseline_csv_path: str,
    model_path: str,
    tokenizer_path: str,
    perturbation_levels: List[str] = None,
    output_dir: str = "adversarial_results",
    device: str = "auto",
    seed: int = 42,
    max_cases: Optional[int] = None,
    debug_print: bool = False,
    max_new_tokens: int = 192
) -> str:
    """Run numeric perturbation adversarial experiment.
    
    Args:
        baseline_csv_path: Path to baseline inference CSV
        model_path: Path to instruction-tuned model checkpoint
        tokenizer_path: Path to tokenizer directory
        perturbation_levels: List of perturbation levels to test
        output_dir: Directory to save results
        device: Device for inference
        seed: Random seed for reproducibility
        max_cases: Optional cap on number of baseline-correct cases to attack
        debug_print: Print prompt/output/parsed result for each attacked sample
        max_new_tokens: Number of tokens to generate beyond prompt length
        
    Returns:
        Path to saved results CSV
    """
    if perturbation_levels is None:
        perturbation_levels = ['off_by_1', 'off_by_2', 'off_by_5', 'off_by_10', 'random_offset']
    
    # Determine device
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    
    print("\n" + "=" * 60)
    print("ADVERSARIAL NUMERIC PERTURBATION EXPERIMENT")
    print("=" * 60)
    print(f"Baseline CSV: {baseline_csv_path}")
    print(f"Model: {model_path}")
    print(f"Perturbation levels: {', '.join(perturbation_levels)}")
    print(f"Device: {device}")
    print(f"Random seed: {seed}")
    if max_cases is not None:
        print(f"Max cases: {max_cases}")
    print(f"Debug print: {debug_print}")
    print(f"Max new tokens: {max_new_tokens}")
    print("=" * 60 + "\n")
    
    # Load baseline table
    print("Loading baseline table...")
    baseline_data = []
    with open(baseline_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            baseline_data.append(row)
    
    print(f"Loaded {len(baseline_data)} baseline samples")
    
    # Filter to baseline-correct cases only
    correct_cases = [
        row for row in baseline_data
        if row['baseline_correct'].lower() == 'true'
    ]

    if max_cases is not None:
        correct_cases = correct_cases[:max_cases]
    
    print(f"Filtered to {len(correct_cases)} baseline-correct cases\n")
    
    if len(correct_cases) == 0:
        print("ERROR: No baseline-correct cases found!")
        return None
    
    # Load model
    print("Loading model and tokenizer...")
    evaluator = ModelEvaluator(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device
    )
    print("Model loaded successfully!\n")
    
    # Run adversarial experiments
    print("Running adversarial perturbations...")
    results = []
    
    for i, case in enumerate(correct_cases):
        correct_answer = int(case['correct_answer'])
        expression = case['expression']
        
        for perturb_level in perturbation_levels:
            # Apply perturbation
            perturbed_answer = apply_numeric_perturbation(
                correct_answer,
                perturb_level,
                seed=seed + i  # Different seed per problem for variety
            )

            # Generate adversarial prompt with fixed minimal scaffold.
            adv_prompt = generate_adversarial_prompt(
                expression,
                perturbed_answer,
            )

            # Generate model response
            prompt_ids = evaluator.tokenizer.encode(adv_prompt, add_special_tokens=True)
            eos_id = evaluator.tokenizer.token2id.get('<eos>')
            if eos_id is not None and prompt_ids and prompt_ids[-1] == eos_id:
                prompt_len = len(prompt_ids) - 1
            else:
                prompt_len = len(prompt_ids)

            # evaluator._generate_solution expects total sequence length, not max new tokens.
            dynamic_max_length = min(512, max(prompt_len + max_new_tokens, prompt_len + 32))
            generated_text = evaluator._generate_solution(adv_prompt, max_length=dynamic_max_length)

            # Extract predicted result
            predicted = evaluator.extract_final_result(generated_text)

            # Treat only numeric parses as successful for robustness metrics.
            is_numeric_prediction = isinstance(predicted, int)
            parse_success = is_numeric_prediction
            still_correct = (predicted == correct_answer) if is_numeric_prediction else False
            flipped = is_numeric_prediction and not still_correct

            results.append({
                'problem_id': case['problem_id'],
                'expression': expression,
                'difficulty': case['difficulty'],
                'correct_answer': correct_answer,
                'perturbation_level': perturb_level,
                'perturbed_answer': perturbed_answer,
                'offset': perturbed_answer - correct_answer,
                'adversarial_prompt': adv_prompt,
                'adversarial_output': generated_text,
                'adversarial_pred': predicted if predicted is not None else 'PARSE_FAIL',
                'adversarial_parse_success': parse_success,
                'adversarial_still_correct': still_correct,
                'flipped': flipped
            })

            if debug_print:
                print("\n" + "-" * 40)
                print(f"DEBUG problem_id={case['problem_id']} level={perturb_level}")
                print("ATTACKED PROMPT:")
                print(adv_prompt)
                print("ATTACKED RAW OUTPUT:")
                print(generated_text)
                print(f"ATTACKED PARSED RESULT: {predicted if predicted is not None else 'PARSE_FAIL'}")
        
        if ((i + 1) % 50) == 0:
            print(f"  Processed {i + 1}/{len(correct_cases)} cases...")
    
    print(f"Completed adversarial inference on {len(results)} perturbed samples\n")
    
    # Calculate statistics
    stats_by_level = {}
    for level in perturbation_levels:
        level_results = [r for r in results if r['perturbation_level'] == level]
        total = len(level_results)
        flipped_count = sum(1 for r in level_results if r['flipped'])
        still_correct_count = sum(1 for r in level_results if r['adversarial_still_correct'])
        parse_count = sum(1 for r in level_results if r['adversarial_parse_success'])
        
        stats_by_level[level] = {
            'total': total,
            'flipped': flipped_count,
            'still_correct': still_correct_count,
            'flip_rate': (flipped_count / total * 100) if total > 0 else 0,
            'maintained_rate': (still_correct_count / total * 100) if total > 0 else 0,
            'parse_rate': (parse_count / total * 100) if total > 0 else 0
        }

    print("Statistics by Perturbation Level:")
    for level in perturbation_levels:
        stats = stats_by_level[level]
        print(f"\n  {level}:")
        print(f"    Total: {stats['total']}")
        print(f"    Flipped: {stats['flipped']} ({stats['flip_rate']:.2f}%)")
        print(f"    Still correct: {stats['still_correct']} ({stats['maintained_rate']:.2f}%)")
        print(f"    Parse rate: {stats['parse_rate']:.2f}%")

    # Save results to CSV
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'adversarial_numeric_{timestamp}.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'problem_id', 'expression', 'difficulty', 'correct_answer',
            'perturbation_level', 'perturbed_answer', 'offset',
            'adversarial_prompt', 'adversarial_output', 'adversarial_pred',
            'adversarial_parse_success', 'adversarial_still_correct', 'flipped'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {csv_path}")
    
    # Save summary JSON
    summary = {
        'timestamp': timestamp,
        'baseline_csv': baseline_csv_path,
        'model_path': model_path,
        'num_baseline_correct': len(correct_cases),
        'num_perturbed_samples': len(results),
        'perturbation_levels': perturbation_levels,
        'stats_by_level': stats_by_level,
        'seed': seed
    }
    
    summary_path = os.path.join(output_dir, f'adversarial_numeric_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    print("\n" + "=" * 60)
    print("ADVERSARIAL EXPERIMENT COMPLETE")
    print("=" * 60)
    
    return csv_path


def main():
    """Main entry point for adversarial numeric perturbation."""
    parser = argparse.ArgumentParser(
        description="Apply numeric perturbations to baseline-correct cases"
    )
    
    parser.add_argument(
        "--baseline-csv",
        type=str,
        required=True,
        help="Path to baseline inference CSV file"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to instruction-tuned model checkpoint"
    )
    
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to tokenizer directory"
    )
    
    parser.add_argument(
        "--perturbation-levels",
        type=str,
        nargs='+',
        default=['off_by_1', 'off_by_2', 'off_by_5', 'off_by_10', 'random_offset'],
        help="Perturbation levels to test"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="adversarial_results",
        help="Directory to save results (default: adversarial_results)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: 'cuda', 'mps', 'cpu', or 'auto' (default: auto)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional maximum number of baseline-correct cases to attack"
    )

    parser.add_argument(
        "--debug-print",
        action="store_true",
        help="Print attacked prompt, raw output, and parsed result for each sample"
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=192,
        help="Tokens to generate beyond prompt length (default: 192)"
    )
    
    args = parser.parse_args()
    
    run_adversarial_numeric(
        baseline_csv_path=args.baseline_csv,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        perturbation_levels=args.perturbation_levels,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        max_cases=args.max_cases,
        debug_print=args.debug_print,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    main()
