#!/usr/bin/env python3
"""Generate baseline inference table for adversarial experiments.

This script evaluates the instruction-tuned model on a test set and creates
a baseline table with problem details, predictions, and correctness labels.
"""

import os
import json
import csv
import torch
import argparse
from datetime import datetime
from typing import List, Dict, Optional

from .evaluator import ModelEvaluator, eval_expression
from .generator import ExpressionGenerator


def calculate_difficulty(expression: str) -> str:
    """Calculate difficulty based on expression depth and operators.
    
    Args:
        expression: Arithmetic expression string
        
    Returns:
        Difficulty label: 'easy', 'medium', or 'hard'
    """
    # Count parentheses depth
    max_depth = 0
    current_depth = 0
    for char in expression:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1
    
    # Count number of operators
    num_operators = expression.count('+') + expression.count('-')
    
    # Classify difficulty
    if max_depth <= 1 and num_operators <= 2:
        return 'easy'
    elif max_depth <= 3 and num_operators <= 5:
        return 'medium'
    else:
        return 'hard'


def generate_baseline_table(
    model_path: str,
    tokenizer_path: str,
    num_samples: int = 500,
    max_depth: int = 4,
    num_range: tuple = (1, 20),
    output_dir: str = "adversarial_results",
    device: str = "auto"
) -> str:
    """Generate baseline inference table.
    
    Args:
        model_path: Path to instruction-tuned model checkpoint
        tokenizer_path: Path to tokenizer directory
        num_samples: Number of test expressions to generate
        max_depth: Maximum expression depth
        num_range: Range of numbers to use
        output_dir: Directory to save baseline table
        device: Device for inference
        
    Returns:
        Path to saved baseline CSV file
    """
    # Determine device
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    
    print("\n" + "=" * 60)
    print("GENERATING BASELINE INFERENCE TABLE")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Samples: {num_samples}")
    print(f"Device: {device}")
    print("=" * 60 + "\n")
    
    # Load model
    print("Loading model and tokenizer...")
    evaluator = ModelEvaluator(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device
    )
    print("Model loaded successfully!\n")
    
    # Generate test expressions
    print(f"Generating {num_samples} test expressions...")
    generator = ExpressionGenerator(
        max_depth=max_depth,
        num_range=num_range,
        invalid_rate=0.0  # Only valid expressions
    )
    
    test_data = []
    problem_id = 0
    
    while len(test_data) < num_samples:
        expression = generator.generate()
        
        # Get ground truth
        result = eval_expression(expression)
        if result['answer'] == 'ERROR':
            continue
        
        # Calculate difficulty
        difficulty = calculate_difficulty(expression)
        
        test_data.append({
            'problem_id': problem_id,
            'expression': expression,
            'difficulty': difficulty,
            'correct_answer': result['answer']
        })
        problem_id += 1
        
        if (problem_id % 100) == 0:
            print(f"  Generated {problem_id} valid expressions...")
    
    print(f"Generated {len(test_data)} test expressions\n")
    
    # Run baseline inference
    print("Running baseline inference...")
    baseline_results = []
    
    for i, item in enumerate(test_data):
        # Generate prediction
        prompt = f"Evaluate: {item['expression']} <think>"
        generated_text = evaluator._generate_solution(prompt, max_length=256)
        
        # Extract predicted result
        predicted = evaluator.extract_final_result(generated_text)
        
        # Determine parse success and correctness
        parse_success = predicted is not None
        correct = (predicted == item['correct_answer']) if parse_success else False
        
        baseline_results.append({
            'problem_id': item['problem_id'],
            'expression': item['expression'],
            'difficulty': item['difficulty'],
            'correct_answer': item['correct_answer'],
            'baseline_output': generated_text,
            'baseline_pred': predicted if predicted is not None else 'PARSE_FAIL',
            'baseline_parse_success': parse_success,
            'baseline_correct': correct
        })
        
        if ((i + 1) % 50) == 0:
            print(f"  Processed {i + 1}/{len(test_data)} samples...")
    
    print(f"Completed baseline inference on {len(baseline_results)} samples\n")
    
    # Calculate statistics
    total = len(baseline_results)
    correct_count = sum(1 for r in baseline_results if r['baseline_correct'])
    parse_count = sum(1 for r in baseline_results if r['baseline_parse_success'])
    
    accuracy = (correct_count / total * 100) if total > 0 else 0
    parse_rate = (parse_count / total * 100) if total > 0 else 0
    
    # Stats by difficulty
    difficulty_stats = {}
    for difficulty in ['easy', 'medium', 'hard']:
        subset = [r for r in baseline_results if r['difficulty'] == difficulty]
        if subset:
            correct_in_subset = sum(1 for r in subset if r['baseline_correct'])
            difficulty_stats[difficulty] = {
                'count': len(subset),
                'correct': correct_in_subset,
                'accuracy': (correct_in_subset / len(subset) * 100)
            }
    
    print("Baseline Statistics:")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct_count}")
    print(f"  Parse success: {parse_count}")
    print(f"  Overall accuracy: {accuracy:.2f}%")
    print(f"  Parse rate: {parse_rate:.2f}%")
    print("\nAccuracy by difficulty:")
    for difficulty, stats in difficulty_stats.items():
        print(f"  {difficulty.capitalize()}: {stats['correct']}/{stats['count']} ({stats['accuracy']:.2f}%)")
    
    # Save baseline table to CSV
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'baseline_table_{timestamp}.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'problem_id', 'expression', 'difficulty', 'correct_answer',
            'baseline_output', 'baseline_pred', 'baseline_parse_success',
            'baseline_correct'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(baseline_results)
    
    print(f"\nBaseline table saved to: {csv_path}")
    
    # Save summary JSON
    summary = {
        'timestamp': timestamp,
        'model_path': model_path,
        'num_samples': len(baseline_results),
        'overall_accuracy': accuracy,
        'parse_rate': parse_rate,
        'correct_count': correct_count,
        'difficulty_stats': difficulty_stats
    }
    
    summary_path = os.path.join(output_dir, f'baseline_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    print("\n" + "=" * 60)
    print("BASELINE TABLE GENERATION COMPLETE")
    print("=" * 60)
    
    return csv_path


def main():
    """Main entry point for baseline table generation."""
    parser = argparse.ArgumentParser(
        description="Generate baseline inference table for adversarial experiments"
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
        "--num-samples",
        type=int,
        default=500,
        help="Number of test expressions to generate (default: 500)"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum expression depth (default: 4)"
    )
    
    parser.add_argument(
        "--num-range",
        type=int,
        nargs=2,
        default=[1, 20],
        help="Range of numbers in expressions (default: 1 20)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="adversarial_results",
        help="Directory to save baseline table (default: adversarial_results)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: 'cuda', 'mps', 'cpu', or 'auto' (default: auto)"
    )
    
    args = parser.parse_args()
    
    generate_baseline_table(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        num_samples=args.num_samples,
        max_depth=args.max_depth,
        num_range=tuple(args.num_range),
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
