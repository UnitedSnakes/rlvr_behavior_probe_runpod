from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from probe.scoring import extract_numeric_answer, numeric_equal, _to_number
from probe.utils import read_jsonl


def rescore_file(path: Path, in_place: bool) -> Path:
    rows = read_jsonl(path)
    changed_labels = 0
    changed_predictions = 0
    total = 0

    for row in rows:
        gold = _to_number(str(row["gold"]))
        if gold is None:
            raise ValueError(f"Could not parse gold for qid={row['qid']}: {row['gold']!r}")

        n_correct = 0
        for rollout in row["rollouts"]:
            total += 1
            old_pred = rollout.get("pred_value")
            old_correct = bool(rollout.get("correct", False))

            pred, token, method = extract_numeric_answer(rollout["text"])
            correct = bool(numeric_equal(pred, gold))

            if pred != old_pred:
                changed_predictions += 1
            if correct != old_correct:
                changed_labels += 1

            rollout["pred_value"] = pred
            rollout["pred_token"] = token
            rollout["extract_method"] = method
            rollout["correct"] = correct
            n_correct += int(correct)

        row["gold_value"] = gold
        row["n_correct"] = n_correct

    if in_place:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        out = path
    else:
        out = path.with_name(path.stem + "_rescored" + path.suffix)

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"{path}: rescored {total} rollouts; "
        f"prediction changed={changed_predictions}, correctness changed={changed_labels} -> {out}"
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", required=True)
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite *_raw.jsonl after creating a .bak copy.",
    )
    args = p.parse_args()

    rd = Path(args.result_dir)
    for name in ("sft_raw.jsonl", "rl_raw.jsonl"):
        path = rd / name
        if path.exists():
            rescore_file(path, args.in_place)


if __name__ == "__main__":
    main()
