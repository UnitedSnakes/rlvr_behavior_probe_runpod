from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def resolve_device(requested: str) -> str:
    if requested != "auto": return requested
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def resolve_dtype(name: str):
    return {"float32":torch.float32,"float16":torch.float16,"bfloat16":torch.bfloat16}[name]

def empty_device_cache() -> None:
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available(): torch.mps.empty_cache()

def append_jsonl(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_jsonl(path):
    path = Path(path)
    if not path.exists(): return []
    out=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out
