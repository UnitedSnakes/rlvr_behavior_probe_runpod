from __future__ import annotations
import json, random, re
from pathlib import Path
from datasets import load_dataset

_GOLD_RE = re.compile(r"####\s*([^\n]+)")

def extract_gsm8k_gold(answer_text: str) -> str:
    m = _GOLD_RE.search(answer_text)
    if not m: raise ValueError(f"Could not find gold answer in: {answer_text!r}")
    return m.group(1).strip()

def prepare_questions(path, n_questions=30, seed=42):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(rows) != n_questions:
            raise ValueError(f"{path} has {len(rows)} questions but --questions={n_questions}. Delete it to resample.")
        return rows
    ds = list(load_dataset("openai/gsm8k", "main", split="test"))
    rng = random.Random(seed); rng.shuffle(ds); ds = ds[:n_questions]
    rows=[]
    for i,item in enumerate(ds):
        rows.append({"qid":i,"question":item["question"],"gold":extract_gsm8k_gold(item["answer"])})
    with path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+"\n")
    return rows
