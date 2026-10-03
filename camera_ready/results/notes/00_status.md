Status: results computed 2026-10-03 ~16:40 UTC by the pre-registered pipeline (`camera_ready/analysis/analyze.py`, `bridge.py`) on the GPU box with code commit `45d6130`; independent fact-check: see `FACTCHECK.md`. All numbers below come from `camera_ready/results/analysis.json` (SHA-256 `2f1fa340…4419`) and `bridge.json` (`e816c7f0…b082`). Units: percentage points (pp) for contrasts, ratios for advantage mass.

**Sample.** Main inference uses the three box-trained pairs (seeds 43, 44, 45; 2 × RTX 4080 SUPER per run). The A40 seed-42 pair (the submitted discovery run) is shown in its own row and never pooled into the main sample. Endpoint K for the primary cells is 16 (the 64-response version requires all 64 endpoint responses in every pair; seed 45's third extra batch could not be scheduled before the pre-registered cut-off, so K = 64 was not available for all pairs).

**What can be said, using only the pre-registered sentences.**

- Primary cells (step 3736; bins 0 and (0,.25]; C and R; d and q): all eight are **"unresolved at this number of runs"** — every two-sided 95 % t-interval across the three pairs contains 0 and reaches beyond ±3 pp. No interval lies entirely above or entirely below 0, so the paper's Test-2 sentence does not have to change on the rule's terms. The eight intervals carry no multiplicity adjustment.
- Reweighting is **realized** in every new pair: the MaxRL/GRPO advantage-mass ratio is above 1 in bins 0 and (0,.25] at step 3736, and also at all four EVAL_STEPS.
- The pre-exposure gain is **replicated** under the pre-registered rule: the adjusted not-yet-exposed correctness gain in (0,.25] is positive at all three cutoffs in every new seed.
- The random-schedule exposure contrast (U − E) is **uninformative** at three seeds: the per-seed values differ in sign and the across-seed intervals are wide.
- Bridge: **PASS** — box evaluations of the same frozen weights are compatible with the stored A40 evaluations within the pre-registered margins. This says nothing about whether training on the box behaves like training on A40.
- Supplementary, mixing hardware (A40 discovery pair pooled with the three box pairs, K = 16, n = 4, shown only because the bridge passed): reported in §2 next to the primary table. It is not the pre-registered main inference.
