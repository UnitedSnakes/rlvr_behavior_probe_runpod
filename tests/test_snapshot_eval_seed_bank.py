from __future__ import annotations


def test_snapshot_question_seed_uses_independent_c_bank():
    import controlled_run.eval_snapshot as eval_snapshot
    from diagnose_p0_signal_budget import half_seed

    base_seed = 42
    dataset_index = 17

    snapshot_seed = eval_snapshot.snapshot_question_seed(base_seed, dataset_index)
    half_a = half_seed(base_seed, dataset_index, "A")
    half_b = half_seed(base_seed, dataset_index, "B")

    assert snapshot_seed == base_seed * 100_000 + dataset_index + 75_000
    assert snapshot_seed != half_a
    assert snapshot_seed != half_b


def test_snapshot_question_seed_is_stable_across_policy_snapshots():
    import controlled_run.eval_snapshot as eval_snapshot

    # Snapshot percentage is deliberately not an input: the same prompt reuses
    # common random numbers across pi_005 ... pi_100 while staying independent
    # of the K32 A/B baseline banks.
    assert eval_snapshot.snapshot_question_seed(42, 123) == eval_snapshot.snapshot_question_seed(42, 123)
    assert eval_snapshot.snapshot_question_seed(42, 124) != eval_snapshot.snapshot_question_seed(42, 123)
