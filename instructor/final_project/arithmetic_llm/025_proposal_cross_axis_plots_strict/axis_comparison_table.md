# Consolidated Axis Comparison

## Numeric

| Level | Baseline | Adversarial | Defense | Wrong Revision | Delta Accuracy | Eligible |
|---|---:|---:|---:|---:|---:|---:|
| off_by_1 | 77.00% | 38.10% | 29.44% | 61.90% | 38.90% | 462 |
| off_by_2 | 75.83% | 38.90% | 32.75% | 61.10% | 36.93% | 455 |
| off_by_5 | 78.83% | 38.27% | 35.94% | 61.73% | 40.57% | 473 |
| off_by_10 | 79.50% | 35.43% | 38.16% | 64.57% | 44.07% | 477 |
| random_offset | 79.17% | 34.95% | 35.79% | 65.05% | 44.22% | 475 |

## Language

| Level | Baseline | Adversarial | Defense | Wrong Revision | Delta Accuracy | Eligible |
|---|---:|---:|---:|---:|---:|---:|
| might_be_wrong | 77.00% | 36.80% | 30.95% | 63.20% | 40.20% | 462 |
| pretty_sure | 75.83% | 37.80% | 33.63% | 62.20% | 38.03% | 455 |
| ta_says | 78.83% | 40.59% | 32.35% | 59.41% | 38.24% | 473 |
| official_key | 79.67% | 38.49% | 30.54% | 61.51% | 41.17% | 478 |
| autograder | 78.17% | 37.95% | 31.13% | 62.05% | 40.21% | 469 |
| do_not_argue | 76.83% | 35.36% | 30.37% | 64.64% | 41.48% | 461 |

## Politeness

| Level | Baseline | Adversarial | Defense | Wrong Revision | Delta Accuracy | Eligible |
|---|---:|---:|---:|---:|---:|---:|
| polite_request | 77.33% | 35.78% | 31.03% | 64.22% | 41.56% | 464 |
| firm_correction | 75.83% | 39.12% | 31.65% | 60.88% | 36.71% | 455 |
| rude_or_insulting | 78.83% | 40.59% | 32.14% | 59.41% | 38.24% | 473 |
| threatening_complaint | 79.50% | 40.04% | 33.33% | 59.96% | 39.46% | 477 |

