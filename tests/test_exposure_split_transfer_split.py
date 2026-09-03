from __future__ import annotations

import math

from analyses import exposure_split_transfer as est


def _movement(
    *,
    pct: int,
    direction: str,
    index: int,
    bin_label: str = "(0,.25]",
    d_r: float,
    d_t: float,
    d_c: float,
) -> dict:
    return {
        "snapshot_pct": pct,
        "direction": direction,
        "dataset_index": index,
        "bin": bin_label,
        "delta_R": d_r,
        "delta_T": d_t,
        "delta_C": d_c,
    }


def test_directional_split_uses_strict_snapshot_boundary() -> None:
    rows = [
        _movement(
            pct=25,
            direction="A-bin/B-base",
            index=i,
            d_r=0.01 * (i + 1),
            d_t=0.02 * (i + 1),
            d_c=0.03 * (i + 1),
        )
        for i in range(4)
    ]
    # Snapshot S=2 sees ledger steps 0 and 1. A question first exposed at
    # ledger step 2 is NOT exposed to the saved policy at snapshot step 2.
    exposure_steps = {0: 0, 1: 1, 2: 2, 3: 3}

    split = est.build_exposure_split_directional(
        rows,
        exposure_steps=exposure_steps,
        snapshot_schedule={25: 2},
        target_pcts=(25,),
    )
    by_status = {row["exposure_status"]: row for row in split}

    assert by_status["exposed"]["n_questions"] == 2
    assert by_status["unexposed"]["n_questions"] == 2
    assert math.isclose(by_status["exposed"]["delta_C"], (0.03 + 0.06) / 2)
    assert math.isclose(by_status["unexposed"]["delta_C"], (0.09 + 0.12) / 2)


def test_symmetric_split_weights_direction_means_equally_not_question_counts() -> None:
    directional = [
        {
            "snapshot_pct": 25,
            "snapshot_step": 2,
            "direction": "A-bin/B-base",
            "bin": "(0,.25]",
            "exposure_status": "exposed",
            "n_questions": 20,
            "delta_R": 0.10,
            "delta_T": 0.20,
            "delta_C": 0.10,
        },
        {
            "snapshot_pct": 25,
            "snapshot_step": 2,
            "direction": "B-bin/A-base",
            "bin": "(0,.25]",
            "exposure_status": "exposed",
            "n_questions": 2,
            "delta_R": 0.30,
            "delta_T": 0.40,
            "delta_C": 0.30,
        },
        {
            "snapshot_pct": 25,
            "snapshot_step": 2,
            "direction": "A-bin/B-base",
            "bin": "(0,.25]",
            "exposure_status": "unexposed",
            "n_questions": 5,
            "delta_R": 0.08,
            "delta_T": 0.18,
            "delta_C": 0.08,
        },
        {
            "snapshot_pct": 25,
            "snapshot_step": 2,
            "direction": "B-bin/A-base",
            "bin": "(0,.25]",
            "exposure_status": "unexposed",
            "n_questions": 50,
            "delta_R": 0.12,
            "delta_T": 0.22,
            "delta_C": 0.12,
        },
    ]

    sym = est.symmetrize_exposure_split(directional)
    by_status = {row["exposure_status"]: row for row in sym}
    assert math.isclose(by_status["exposed"]["delta_C"], 0.20)
    assert math.isclose(by_status["unexposed"]["delta_C"], 0.10)
    assert by_status["exposed"]["n_questions_A"] == 20
    assert by_status["exposed"]["n_questions_B"] == 2

    contrast = est.build_transfer_contrasts(sym)
    assert len(contrast) == 1
    row = contrast[0]
    assert math.isclose(row["delta_C_exposed"], 0.20)
    assert math.isclose(row["delta_C_unexposed"], 0.10)
    assert math.isclose(row["transfer_ratio_C"], 0.5)
    assert row["transfer_classification"] == "mixed"


def test_default_split_outputs_only_predeclared_25_45_65_snapshots() -> None:
    schedule = {5: 1, 25: 2, 45: 3, 65: 4, 100: 5}
    rows = []
    for pct in schedule:
        for direction in est.DIRECTIONS:
            rows.extend(
                [
                    _movement(
                        pct=pct,
                        direction=direction,
                        index=0,
                        d_r=0.1,
                        d_t=0.2,
                        d_c=0.1,
                    ),
                    _movement(
                        pct=pct,
                        direction=direction,
                        index=1,
                        d_r=0.1,
                        d_t=0.2,
                        d_c=0.1,
                    ),
                ]
            )

    directional = est.build_exposure_split_directional(
        rows,
        exposure_steps={0: 0, 1: 4},
        snapshot_schedule=schedule,
    )
    assert {row["snapshot_pct"] for row in directional} == {25, 45, 65}
