# 2026-09-08 strict-extractor robustness result

## Purpose

Post-submission measurement diagnostic before the 2026-09-09 Andrea Zanette meeting.

Question: could the reported correctness movement, especially the low-nonzero-p0 pre-own-exposure gain, be an artifact of the canonical GSM8K scorer's generic last-number fallback combined with large changes in termination/output behavior?

This check does **not** retrain any policy. It keeps the frozen p0 bins, exposure timing, cross-fit directions, covariates, termination labels, and training rewards fixed, and replaces only correctness C with a stricter extractor that preserves the canonical boxed/final-answer patterns while disabling the generic last-number fallback.

Implementation:
- `probe/scoring.py::extract_numeric_answer_strict`
- `analyses/strict_extractor_robustness.py`
- `tests/test_strict_extractor_robustness.py`

Local unit test reported by Shanglin:
- `python -m pytest -q tests/test_strict_extractor_robustness.py`
- 4 passed

## Reported canonical result

Whole-panel correctness:

| policy | canonical C | strict C |
| --- | ---: | ---: |
| pi0 | 0.5045 | 0.4886 |
| GRPO endpoint | 0.5774 | 0.5706 |

Therefore:
- canonical GRPO endpoint delta C = +7.29 pp
- strict GRPO endpoint delta C = +8.20 pp

The p0 scorer-label disagreement induced by disabling the generic last-number fallback is 1.59%.

Primary low-nonzero-p0 adjusted correctness:

| snapshot | U canonical | U strict | E canonical | E strict | strict U-E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25% | +10.02 pp | +10.57 pp | +6.25 pp | +6.74 pp | +3.83 pp |
| 45% | +12.80 pp | +13.53 pp | +7.42 pp | +8.56 pp | +4.97 pp |
| 65% | +12.80 pp | +14.14 pp | +10.90 pp | +12.43 pp | +1.71 pp |

## Interpretation

The generic last-number fallback is **not** driving the primary pre-own-exposure correctness result. Removing it leaves the effect intact and slightly increases the estimated unexposed gains at all three frozen checkpoints.

This specifically rules against the narrow explanation that the main correctness movement is produced by the canonical scorer increasingly finding a convenient trailing number as policies learn to terminate differently.

It does **not** establish that unconditional extracted-answer correctness is a pure reasoning-ability measure. Broader output-format and stopping behavior remain possible components of the measured behavior, and conditioning only on terminated responses would introduce post-treatment selection.

For discussion with Andrea, the safe concise statement is:

> I worried that the pre-exposure correctness gain might be an artifact of termination changes interacting with our last-number fallback. I re-scored the frozen response banks with that fallback disabled; the low-nonzero pre-exposure gains remain about 10.6, 13.5, and 14.1 pp, so that specific extraction artifact does not explain the result.

