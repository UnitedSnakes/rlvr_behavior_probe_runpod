"""Absolute versus whole-panel-centered correctness contrasts.

This post-outcome diagnostic uses existing accepted analysis exports. Centering
is an algebraic comparison, not a causal adjustment for overall model quality.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
GRPO_AGGREGATE = REPO_ROOT / "analyses" / "canonical_snapshot_crossfit" / "aggregate_sanity.csv"
MAXRL_AGGREGATE = REPO_ROOT / "analyses" / "canonical_maxrl_snapshot_crossfit" / "aggregate_sanity.csv"
OBJECTIVE_COMPARISON = (
    REPO_ROOT
    / "analyses"
    / "canonical_maxrl_grpo_objective_comparison"
    / "objective_comparison.csv"
)
BINS = ("0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)")
SNAPSHOTS = tuple(range(5, 101, 5))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_analysis(*, output_dir: Path = OUTPUT_DIR) -> list[dict]:
    grpo = {int(row["snapshot_pct"]): float(row["C"]) for row in _read_csv(GRPO_AGGREGATE)}
    maxrl = {int(row["snapshot_pct"]): float(row["C"]) for row in _read_csv(MAXRL_AGGREGATE)}
    if grpo[0] != maxrl[0]:
        raise AssertionError("GRPO and MaxRL must share the same pi0 aggregate baseline")
    if set(SNAPSHOTS) - set(grpo) or set(SNAPSHOTS) - set(maxrl):
        raise AssertionError("aggregate exports are missing frozen 5%-through-100% snapshots")

    objective_rows = _read_csv(OBJECTIVE_COMPARISON)
    output_rows = []
    for row in objective_rows:
        cutoff = int(row["snapshot_pct"])
        bin_name = row["bin"]
        if cutoff not in SNAPSHOTS or bin_name not in BINS:
            raise AssertionError(f"unexpected objective-comparison row: {cutoff}, {bin_name}")
        absolute = float(row["maxrl_minus_grpo_delta_C"])
        panel_difference = maxrl[cutoff] - grpo[cutoff]
        output_rows.append(
            {
                "snapshot_pct": cutoff,
                "bin": bin_name,
                "bin_maxrl_minus_grpo_pp": 100 * absolute,
                "panel_maxrl_minus_grpo_pp": 100 * panel_difference,
                "bin_minus_panel_contrast_pp": 100 * (absolute - panel_difference),
            }
        )

    if len(output_rows) != len(SNAPSHOTS) * len(BINS):
        raise AssertionError("expected exactly 20 snapshots x 5 frozen bins")
    for bin_name in BINS:
        seen = sorted(
            row["snapshot_pct"] for row in output_rows if row["bin"] == bin_name
        )
        if seen != list(SNAPSHOTS):
            raise AssertionError(f"incomplete snapshot trajectory for {bin_name}: {seen}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "centered_correctness.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = []
    for bin_name in BINS:
        rows = sorted(
            [row for row in output_rows if row["bin"] == bin_name],
            key=lambda row: row["snapshot_pct"],
        )
        summary.append(
            {
                "bin": bin_name,
                "endpoint_absolute_pp": rows[-1]["bin_maxrl_minus_grpo_pp"],
                "endpoint_centered_pp": rows[-1]["bin_minus_panel_contrast_pp"],
                "mean_absolute_pp": mean(row["bin_maxrl_minus_grpo_pp"] for row in rows),
                "mean_centered_pp": mean(row["bin_minus_panel_contrast_pp"] for row in rows),
                "positive_centered_snapshots": sum(
                    row["bin_minus_panel_contrast_pp"] > 0 for row in rows
                ),
            }
        )
    (output_dir / "centered_correctness_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    print(json.dumps(run_analysis(), indent=2))


if __name__ == "__main__":
    main()
