# 018 vs 025 Strict Rerun (Quick Comparison)

This note compares:
- `018_proposal_cross_axis_plots` (legacy-mode full cross-axis outputs)
- `025_proposal_cross_axis_plots_strict` (strict-mode full cross-axis rerun after strict parser/prompt fixes)

## Axis-level averages

| Run | Axis | Baseline Acc | Adversarial Acc | Defense Acc | Wrong Revision | Delta Accuracy |
|---|---|---:|---:|---:|---:|---:|
| 018 (legacy) | Numeric | 0.7810 | 0.0090 | 0.0367 | 0.9910 | 0.7720 |
| 018 (legacy) | Language | 0.7767 | 0.0208 | 0.0350 | 0.9792 | 0.7559 |
| 018 (legacy) | Politeness | 0.7767 | 0.0183 | 0.0381 | 0.9817 | 0.7584 |
| 025 (strict) | Numeric | 0.7807 | 0.3713 | 0.3441 | 0.6287 | 0.4094 |
| 025 (strict) | Language | 0.7772 | 0.3783 | 0.3149 | 0.6217 | 0.3989 |
| 025 (strict) | Politeness | 0.7788 | 0.3888 | 0.3204 | 0.6112 | 0.3899 |

## Interpretation

- Baseline accuracy is effectively unchanged between runs (~0.78).
- After strict-mode fixes, adversarial/defense parse paths are no longer collapsing to near-zero useful outputs.
- Compared with legacy results, strict rerun shows substantially less catastrophic collapse (lower wrong-revision, higher adversarial/defense accuracy).
- Core robustness issue still remains: adversarial pressure still causes large degradation from baseline across all axes.

## Files

- Legacy combined table: `018_proposal_cross_axis_plots/axis_comparison_table.csv`
- Strict combined table: `025_proposal_cross_axis_plots_strict/axis_comparison_table.csv`
- Strict plots summary: `025_proposal_cross_axis_plots_strict/cross_axis_plot_summary.json`
