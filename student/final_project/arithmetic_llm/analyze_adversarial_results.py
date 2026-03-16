#!/usr/bin/env python3
"""Analyze and plot adversarial experiment results.

This script loads baseline and adversarial results CSVs and generates plots for:
- Baseline accuracy vs difficulty
- Flip rate vs numeric offset
"""

import os
import csv
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime


def load_baseline_csv(csv_path: str) -> list:
    """Load baseline inference results from CSV.
    
    Args:
        csv_path: Path to baseline CSV file
        
    Returns:
        List of baseline result dictionaries
    """
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string bools to actual bools
            row['baseline_correct'] = row['baseline_correct'].lower() == 'true'
            row['baseline_parse_success'] = row['baseline_parse_success'].lower() == 'true'
            data.append(row)
    return data


def load_adversarial_csv(csv_path: str) -> list:
    """Load adversarial experiment results from CSV.
    
    Args:
        csv_path: Path to adversarial CSV file
        
    Returns:
        List of adversarial result dictionaries
    """
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string bools to actual bools
            row['adversarial_still_correct'] = row['adversarial_still_correct'].lower() == 'true'
            row['adversarial_parse_success'] = row['adversarial_parse_success'].lower() == 'true'
            row['flipped'] = row['flipped'].lower() == 'true'
            row['offset'] = int(row['offset'])
            data.append(row)
    return data


def plot_baseline_accuracy_by_difficulty(baseline_data: list, output_path: str):
    """Plot baseline accuracy by difficulty level.
    
    Args:
        baseline_data: List of baseline result dicts
        output_path: Path to save plot
    """
    # Calculate accuracy by difficulty
    difficulty_stats = {}
    for difficulty in ['easy', 'medium', 'hard']:
        subset = [r for r in baseline_data if r['difficulty'] == difficulty]
        if subset:
            correct = sum(1 for r in subset if r['baseline_correct'])
            total = len(subset)
            difficulty_stats[difficulty] = {
                'total': total,
                'correct': correct,
                'accuracy': (correct / total * 100) if total > 0 else 0
            }
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    difficulties = ['easy', 'medium', 'hard']
    accuracies = [difficulty_stats.get(d, {'accuracy': 0})['accuracy'] for d in difficulties]
    totals = [difficulty_stats.get(d, {'total': 0})['total'] for d in difficulties]
    
    bars = ax.bar(difficulties, accuracies, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
    
    # Add count labels on bars
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'n={total}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Difficulty Level', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Baseline Accuracy by Difficulty', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved baseline accuracy plot to: {output_path}")
    plt.close()


def plot_flip_rate_by_offset(adversarial_data: list, output_path: str):
    """Plot flip rate vs numeric offset.
    
    Args:
        adversarial_data: List of adversarial result dicts
        output_path: Path to save plot
    """
    # Group by perturbation level
    level_order = ['off_by_1', 'off_by_2', 'off_by_5', 'off_by_10', 'random_offset']
    level_labels = ['±1', '±2', '±5', '±10', 'Random\n(±1-20)']
    
    flip_rates = []
    maintained_rates = []
    level_names = []
    
    for level in level_order:
        subset = [r for r in adversarial_data if r['perturbation_level'] == level]
        if subset:
            total = len(subset)
            flipped = sum(1 for r in subset if r['flipped'])
            still_correct = sum(1 for r in subset if r['adversarial_still_correct'])
            
            flip_rates.append((flipped / total * 100) if total > 0 else 0)
            maintained_rates.append((still_correct / total * 100) if total > 0 else 0)
            level_names.append(level)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(level_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, flip_rates, width, label='Flipped (Incorrect)', 
                    color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, maintained_rates, width, label='Maintained (Correct)', 
                    color='#2ecc71', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Numeric Offset Level', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Model Response to Numeric Perturbations', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved flip rate plot to: {output_path}")
    plt.close()


def plot_flip_rate_by_difficulty_and_offset(adversarial_data: list, output_path: str):
    """Plot flip rate by difficulty level and perturbation offset.
    
    Args:
        adversarial_data: List of adversarial result dicts
        output_path: Path to save plot
    """
    # Group by difficulty and perturbation level
    level_order = ['off_by_1', 'off_by_2', 'off_by_5', 'off_by_10', 'random_offset']
    level_labels = ['±1', '±2', '±5', '±10', 'Random']
    difficulties = ['easy', 'medium', 'hard']
    
    difficulty_data = {d: [] for d in difficulties}
    
    for difficulty in difficulties:
        for level in level_order:
            subset = [
                r for r in adversarial_data
                if r['difficulty'] == difficulty and r['perturbation_level'] == level
            ]
            if subset:
                total = len(subset)
                flipped = sum(1 for r in subset if r['flipped'])
                flip_rate = (flipped / total * 100) if total > 0 else 0
            else:
                flip_rate = 0
            difficulty_data[difficulty].append(flip_rate)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(level_labels))
    width = 0.25
    
    colors = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'}
    
    for i, difficulty in enumerate(difficulties):
        offset = (i - 1) * width
        ax.bar(x + offset, difficulty_data[difficulty], width, 
               label=difficulty.capitalize(), color=colors[difficulty], alpha=0.8)
    
    ax.set_xlabel('Numeric Offset Level', fontsize=12)
    ax.set_ylabel('Flip Rate (%)', fontsize=12)
    ax.set_title('Flip Rate by Difficulty and Perturbation Level', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved difficulty-offset interaction plot to: {output_path}")
    plt.close()


def plot_parse_success_by_difficulty(adversarial_data: list, output_path: str):
    """Plot parse success rate by difficulty level.
    
    Args:
        adversarial_data: List of adversarial result dicts
        output_path: Path to save plot
    """
    # Calculate parse success by difficulty
    difficulty_stats = {}
    for difficulty in ['easy', 'medium', 'hard']:
        subset = [r for r in adversarial_data if r['difficulty'] == difficulty]
        if subset:
            total = len(subset)
            parsed = sum(1 for r in subset if r['adversarial_parse_success'])
            difficulty_stats[difficulty] = {
                'total': total,
                'parsed': parsed,
                'parse_rate': (parsed / total * 100) if total > 0 else 0
            }
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    difficulties = ['easy', 'medium', 'hard']
    parse_rates = [difficulty_stats.get(d, {'parse_rate': 0})['parse_rate'] for d in difficulties]
    totals = [difficulty_stats.get(d, {'total': 0})['total'] for d in difficulties]
    
    bars = ax.bar(difficulties, parse_rates, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
    
    # Add count labels on bars
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'n={total}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Difficulty Level', fontsize=12)
    ax.set_ylabel('Parse Success Rate (%)', fontsize=12)
    ax.set_title('Adversarial Parse Success by Difficulty', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved parse success plot to: {output_path}")
    plt.close()


def analyze_results(
    baseline_csv_path: str,
    adversarial_csv_path: str,
    output_dir: str = "adversarial_results/plots"
):
    """Analyze and plot adversarial experiment results.
    
    Args:
        baseline_csv_path: Path to baseline CSV
        adversarial_csv_path: Path to adversarial CSV
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 60)
    print("ANALYZING ADVERSARIAL EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Baseline CSV: {baseline_csv_path}")
    print(f"Adversarial CSV: {adversarial_csv_path}")
    print("=" * 60 + "\n")
    
    # Load data
    print("Loading data...")
    baseline_data = load_baseline_csv(baseline_csv_path)
    adversarial_data = load_adversarial_csv(adversarial_csv_path)
    
    print(f"Loaded {len(baseline_data)} baseline samples")
    print(f"Loaded {len(adversarial_data)} adversarial samples\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    print("Generating plots...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Baseline accuracy by difficulty
    plot_path_1 = os.path.join(output_dir, f'baseline_accuracy_by_difficulty_{timestamp}.png')
    plot_baseline_accuracy_by_difficulty(baseline_data, plot_path_1)
    
    # Plot 2: Flip rate by offset
    plot_path_2 = os.path.join(output_dir, f'flip_rate_by_offset_{timestamp}.png')
    plot_flip_rate_by_offset(adversarial_data, plot_path_2)
    
    # Plot 3: Flip rate by difficulty and offset
    plot_path_3 = os.path.join(output_dir, f'flip_rate_by_difficulty_and_offset_{timestamp}.png')
    plot_flip_rate_by_difficulty_and_offset(adversarial_data, plot_path_3)
    
    # Plot 4: Parse success by difficulty
    plot_path_4 = os.path.join(output_dir, f'parse_success_by_difficulty_{timestamp}.png')
    plot_parse_success_by_difficulty(adversarial_data, plot_path_4)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


def main():
    """Main entry point for analysis and plotting."""
    parser = argparse.ArgumentParser(
        description="Analyze and plot adversarial experiment results"
    )
    
    parser.add_argument(
        "--baseline-csv",
        type=str,
        required=True,
        help="Path to baseline inference CSV file"
    )
    
    parser.add_argument(
        "--adversarial-csv",
        type=str,
        required=True,
        help="Path to adversarial results CSV file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="adversarial_results/plots",
        help="Directory to save plots (default: adversarial_results/plots)"
    )
    
    args = parser.parse_args()
    
    analyze_results(
        baseline_csv_path=args.baseline_csv,
        adversarial_csv_path=args.adversarial_csv,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
