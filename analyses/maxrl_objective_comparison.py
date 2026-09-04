from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_FIELDS = {
    "snapshot_pct",
    "direction",
    "bin",
    "cumulative_abs_advantage_per_panel_question",
    "exploratory_dapo_is_abs_mass_per_panel_question",
    "actual_is_ess_fraction",
    "delta_R",
    "delta_T",
    "delta_C",
}


def _read_symmetric(path: Path) -> dict[tuple[int, str], dict]:
    rows: dict[tuple[int, str], dict] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_FIELDS.issubset(reader.fieldnames):
            raise ValueError(
                f"signal/movement CSV missing required fields: {sorted(REQUIRED_FIELDS)}"
            )
        for raw in reader:
            if raw["direction"] != "symmetric":
                continue
            key = (int(raw["snapshot_pct"]), raw["bin"])
            if key in rows:
                raise ValueError(f"duplicate symmetric objective row {key}")
            row = dict(raw)
            for field in (
                "cumulative_abs_advantage_per_panel_question",
                "exploratory_dapo_is_abs_mass_per_panel_question",
                "actual_is_ess_fraction",
                "delta_R",
                "delta_T",
                "delta_C",
            ):
                value = raw[field]
                row[field] = None if value == "" else float(value)
            rows[key] = row
    if not rows:
        raise ValueError(f"no symmetric rows found in {path}")
    return rows


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def build_objective_comparison(
    grpo_rows: dict[tuple[int, str], dict],
    maxrl_rows: dict[tuple[int, str], dict],
) -> list[dict]:
    if set(grpo_rows) != set(maxrl_rows):
        missing = sorted(set(grpo_rows) - set(maxrl_rows))
        extra = sorted(set(maxrl_rows) - set(grpo_rows))
        raise ValueError(
            f"GRPO/MaxRL comparison key mismatch; missing_maxrl={missing}, "
            f"extra_maxrl={extra}"
        )

    out: list[dict] = []
    for key in sorted(grpo_rows):
        pct, bin_label = key
        g = grpo_rows[key]
        m = maxrl_rows[key]
        g_signal = g["cumulative_abs_advantage_per_panel_question"]
        m_signal = m["cumulative_abs_advantage_per_panel_question"]
        g_proxy = g["exploratory_dapo_is_abs_mass_per_panel_question"]
        m_proxy = m["exploratory_dapo_is_abs_mass_per_panel_question"]

        row = {
            "snapshot_pct": pct,
            "bin": bin_label,
            "grpo_signal": g_signal,
            "maxrl_signal": m_signal,
            "maxrl_over_grpo_signal": _ratio(m_signal, g_signal),
            "maxrl_minus_grpo_signal": (
                None if g_signal is None or m_signal is None else m_signal - g_signal
            ),
            "grpo_is_proxy": g_proxy,
            "maxrl_is_proxy": m_proxy,
            "maxrl_over_grpo_is_proxy": _ratio(m_proxy, g_proxy),
            "grpo_actual_is_ess_fraction": g["actual_is_ess_fraction"],
            "maxrl_actual_is_ess_fraction": m["actual_is_ess_fraction"],
        }
        for metric in ("R", "T", "C"):
            gv = g[f"delta_{metric}"]
            mv = m[f"delta_{metric}"]
            row[f"grpo_delta_{metric}"] = gv
            row[f"maxrl_delta_{metric}"] = mv
            row[f"maxrl_minus_grpo_delta_{metric}"] = (
                None if gv is None or mv is None else mv - gv
            )
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write empty objective comparison")
    fields = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(
    *,
    grpo_csv: Path,
    maxrl_csv: Path,
    output_dir: Path,
) -> dict:
    grpo = _read_symmetric(grpo_csv)
    maxrl = _read_symmetric(maxrl_csv)
    rows = build_objective_comparison(grpo, maxrl)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "objective_comparison.csv", rows)

    summary = {
        "status": "COMPLETE_NO_AUTOMATIC_H2_H3_VERDICT",
        "rows": len(rows),
        "snapshots": sorted({int(row["snapshot_pct"]) for row in rows}),
        "bins": sorted({row["bin"] for row in rows}),
        "interpretation": (
            "The predeclared H2/H3 decision remains qualitative. Compare the "
            "magnitude/shape of realized signal reallocation with the matched "
            "DeltaC contrast; DeltaT and DeltaR remain separate supporting outcomes."
        ),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**summary, "comparison_rows": rows, "output_dir": str(destination)}


def _fmt(value, scale: float = 1.0, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{scale * float(value):.{digits}f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare canonical GRPO and MaxRL signal-vs-behavior allocation."
    )
    parser.add_argument(
        "--grpo-csv",
        type=Path,
        default=Path(
            "analyses/canonical_ledger_crossfit_signal/signal_movement_join.csv"
        ),
    )
    parser.add_argument("--maxrl-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("controlled_run_outputs/maxrl_grpo_objective_comparison"),
    )
    args = parser.parse_args(argv)

    result = run_comparison(
        grpo_csv=args.grpo_csv,
        maxrl_csv=args.maxrl_csv,
        output_dir=args.output_dir,
    )

    final_pct = max(result["snapshots"])
    final_rows = [
        row for row in result["comparison_rows"]
        if int(row["snapshot_pct"]) == final_pct
    ]
    print(f"CANONICAL GRPO VS MAXRL @ {final_pct}%")
    print(
        f"{'bin':<10} {'sig M/G':>9} {'dC G':>9} {'dC M':>9} "
        f"{'M-G dC':>9} {'dT G':>9} {'dT M':>9} {'dR G':>9} {'dR M':>9}"
    )
    for row in final_rows:
        print(
            f"{row['bin']:<10} "
            f"{_fmt(row['maxrl_over_grpo_signal']):>9} "
            f"{_fmt(row['grpo_delta_C'], scale=100.0):>9} "
            f"{_fmt(row['maxrl_delta_C'], scale=100.0):>9} "
            f"{_fmt(row['maxrl_minus_grpo_delta_C'], scale=100.0):>9} "
            f"{_fmt(row['grpo_delta_T'], scale=100.0):>9} "
            f"{_fmt(row['maxrl_delta_T'], scale=100.0):>9} "
            f"{_fmt(row['grpo_delta_R'], scale=100.0):>9} "
            f"{_fmt(row['maxrl_delta_R'], scale=100.0):>9}"
        )

    print()
    print(f"outputs: {result['output_dir']}")
    print(result["status"])


if __name__ == "__main__":
    main()
