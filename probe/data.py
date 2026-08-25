from __future__ import annotations

import json
import random
import re
from pathlib import Path

from datasets import load_dataset


_GOLD_RE = re.compile(r"####\s*([^\n]+)")


def extract_gsm8k_gold(answer_text: str) -> str:
    match = _GOLD_RE.search(answer_text)
    if match is None:
        raise ValueError(
            f"Could not find GSM8K gold answer in: {answer_text!r}"
        )

    return match.group(1).strip()


def prepare_questions(path, n_questions=30, seed=42):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if len(rows) != n_questions:
            raise ValueError(
                f"{path} has {len(rows)} questions but "
                f"--questions={n_questions}. Delete it to resample."
            )

        return rows

    dataset = list(load_dataset("openai/gsm8k", "main", split="test"))

    rng = random.Random(seed)
    rng.shuffle(dataset)
    dataset = dataset[:n_questions]

    rows = []
    for qid, item in enumerate(dataset):
        rows.append(
            {
                "qid": qid,
                "question": item["question"],
                "gold": extract_gsm8k_gold(item["answer"]),
            }
        )

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return rows
