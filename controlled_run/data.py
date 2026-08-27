from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict

import numpy as np

from controlled_run.constants import CONTROLLED_SYSTEM_PROMPT
from probe.data import extract_gsm8k_gold


MAX_SFT_TOKENS = 2048
DEFAULT_SHINGLE_SIZE = 5
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.80


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_verified_trace(row: dict) -> tuple[int, str] | None:
    uuid = str(row.get("uuid", "<missing uuid>"))
    generations = row.get("generations")
    complete = row.get("is_reasoning_complete")
    verified = row.get("correctness_math_verify")

    if not all(isinstance(values, (list, tuple)) for values in (generations, complete, verified)):
        raise ValueError(
            f"{uuid}: expected sequence fields generations, "
            "is_reasoning_complete, and correctness_math_verify"
        )

    lengths = (len(generations), len(complete), len(verified))
    if len(set(lengths)) != 1:
        raise ValueError(
            f"{uuid}: sequence lengths do not match: "
            f"generations={lengths[0]}, complete={lengths[1]}, verified={lengths[2]}"
        )

    for index, (text, is_complete, is_verified) in enumerate(
        zip(generations, complete, verified)
    ):
        if bool(is_complete) and bool(is_verified) and isinstance(text, str) and text.strip():
            return index, text

    return None


def _nfkc_strip(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).strip()


def normalize_problem(text: str, aggressive: bool = False) -> str:
    normalized = _nfkc_strip(text).lower()

    if aggressive:
        normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
        normalized = "".join(
            " " if unicodedata.category(char).startswith("P") else char
            for char in normalized
        )

    return " ".join(normalized.split())


def word_shingles(text: str, n: int = DEFAULT_SHINGLE_SIZE) -> set[str]:
    if n <= 0:
        raise ValueError("n must be positive")

    words = normalize_problem(text, aggressive=True).split()
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}

    return {
        " ".join(words[start : start + n])
        for start in range(len(words) - n + 1)
    }


class NearDuplicateIndex:
    def __init__(
        self,
        reference_texts: list[str],
        n: int = DEFAULT_SHINGLE_SIZE,
        threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    ):
        if n <= 0:
            raise ValueError("n must be positive")
        if not 0 < threshold <= 1:
            raise ValueError("threshold must lie in (0, 1]")

        self.n = n
        self.threshold = float(threshold)
        self.reference_shingles = [word_shingles(text, n=n) for text in reference_texts]
        inverted: dict[str, set[int]] = defaultdict(set)
        for index, shingles in enumerate(self.reference_shingles):
            for shingle in shingles:
                inverted[shingle].add(index)
        self.inverted = dict(inverted)

    def find_match(self, text: str) -> int | None:
        candidate = word_shingles(text, n=self.n)
        if not candidate:
            return None

        possible: set[int] = set()
        for shingle in candidate:
            possible.update(self.inverted.get(shingle, ()))

        for index in sorted(possible):
            reference = self.reference_shingles[index]
            union = candidate | reference
            if not union:
                continue
            similarity = len(candidate & reference) / len(union)
            if similarity >= self.threshold:
                return index

        return None


def stable_candidate_key(uuid: str, seed: int) -> str:
    return _sha256_text(f"{seed}:{uuid}")


def _training_messages(problem: str, completion: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": CONTROLLED_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": completion},
    ]


def _formatted_token_count(tokenizer, problem: str, completion: str) -> int:
    encoded = tokenizer.apply_chat_template(
        _training_messages(problem, completion),
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "shape") and len(getattr(encoded, "shape")) == 2:
        return int(encoded.shape[-1])
    return len(encoded)


def _length_audit(lengths: list[int]) -> dict:
    values = np.asarray(lengths, dtype=np.int64)
    if values.size == 0:
        return {
            "pre_length_filter_count": 0,
            "formatted_token_percentiles": {
                "p50": None,
                "p75": None,
                "p90": None,
                "p95": None,
                "p99": None,
            },
            "formatted_token_tail_fractions": {
                "gt_2048": 0.0,
                "gt_4096": 0.0,
                "gt_8192": 0.0,
            },
        }

    return {
        "pre_length_filter_count": int(values.size),
        "formatted_token_percentiles": {
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        },
        "formatted_token_tail_fractions": {
            "gt_2048": float(np.mean(values > 2048)),
            "gt_4096": float(np.mean(values > 4096)),
            "gt_8192": float(np.mean(values > 8192)),
        },
    }


def build_sft_manifest(
    openr1_rows,
    gsm8k_reference_rows,
    tokenizer,
    target_size: int = 10_000,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    candidates = list(openr1_rows)
    references = [str(row["question"]) for row in gsm8k_reference_rows]

    exact_reference = {_nfkc_strip(text) for text in references}
    basic_reference = {normalize_problem(text) for text in references}
    aggressive_reference = {
        normalize_problem(text, aggressive=True) for text in references
    }
    near_index = NearDuplicateIndex(
        references,
        n=DEFAULT_SHINGLE_SIZE,
        threshold=DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    )

    seen_uuids: set[str] = set()
    for row in candidates:
        uuid = str(row.get("uuid", ""))
        if not uuid:
            raise ValueError("Every OpenR1 candidate must have a non-empty uuid")
        if uuid in seen_uuids:
            raise ValueError(f"Duplicate OpenR1 uuid: {uuid}")
        seen_uuids.add(uuid)

    ordered = sorted(
        candidates,
        key=lambda row: stable_candidate_key(str(row["uuid"]), seed),
    )

    audit = {
        "candidate_count": len(ordered),
        "removed_exact_duplicates": 0,
        "removed_normalized_duplicates": 0,
        "removed_near_duplicates": 0,
        "removed_too_long": 0,
        "removed_no_verified_trace": 0,
        "eligible_after_filters": 0,
        "excluded_by_target_size": 0,
        "final_count": 0,
        "max_formatted_tokens": MAX_SFT_TOKENS,
        "shingle_size": DEFAULT_SHINGLE_SIZE,
        "near_duplicate_threshold": DEFAULT_NEAR_DUPLICATE_THRESHOLD,
        "seed": seed,
    }

    eligible: list[dict] = []
    pre_length_filter_lengths: list[int] = []

    for row in ordered:
        selected = select_verified_trace(row)
        if selected is None:
            audit["removed_no_verified_trace"] += 1
            continue

        problem = str(row["problem"])
        exact_key = _nfkc_strip(problem)
        basic_key = normalize_problem(problem)
        aggressive_key = normalize_problem(problem, aggressive=True)

        if exact_key in exact_reference:
            audit["removed_exact_duplicates"] += 1
            continue

        if basic_key in basic_reference or aggressive_key in aggressive_reference:
            audit["removed_normalized_duplicates"] += 1
            continue

        if near_index.find_match(problem) is not None:
            audit["removed_near_duplicates"] += 1
            continue

        generation_index, completion = selected
        token_count = _formatted_token_count(tokenizer, problem, completion)
        pre_length_filter_lengths.append(token_count)
        if token_count > MAX_SFT_TOKENS:
            audit["removed_too_long"] += 1
            continue

        eligible.append(
            {
                "uuid": str(row["uuid"]),
                "generation_index": generation_index,
                "source": row.get("source"),
                "problem_sha256": _sha256_text(problem),
                "completion_sha256": _sha256_text(completion),
                "formatted_token_count": token_count,
            }
        )

    audit.update(_length_audit(pre_length_filter_lengths))
    audit["eligible_after_filters"] = len(eligible)

    if len(eligible) < target_size:
        raise ValueError(
            f"SFT subset requested {target_size} examples but only "
            f"{len(eligible)} survived deterministic filtering"
        )

    manifest = eligible[:target_size]
    audit["excluded_by_target_size"] = len(eligible) - target_size
    audit["final_count"] = len(manifest)
    return manifest, audit


def materialize_sft_records(manifest: list[dict], openr1_rows) -> list[dict]:
    by_uuid: dict[str, dict] = {}
    for row in openr1_rows:
        uuid = str(row.get("uuid", ""))
        if uuid in by_uuid:
            raise ValueError(f"Duplicate OpenR1 uuid while materializing: {uuid}")
        by_uuid[uuid] = row

    records: list[dict] = []
    for item in manifest:
        uuid = str(item["uuid"])
        if uuid not in by_uuid:
            raise ValueError(f"Manifest uuid not found in OpenR1 rows: {uuid}")

        row = by_uuid[uuid]
        selected = select_verified_trace(row)
        if selected is None:
            raise ValueError(f"Manifest uuid no longer has a verified trace: {uuid}")
        generation_index, completion = selected
        if generation_index != int(item["generation_index"]):
            raise ValueError(
                f"Manifest generation index mismatch for {uuid}: "
                f"expected {item['generation_index']}, got {generation_index}"
            )

        problem = str(row["problem"])
        if _sha256_text(problem) != item["problem_sha256"]:
            raise ValueError(f"Manifest problem hash mismatch for {uuid}")
        if _sha256_text(completion) != item["completion_sha256"]:
            raise ValueError(f"Manifest completion hash mismatch for {uuid}")

        records.append(
            {
                "uuid": uuid,
                "prompt": [
                    {"role": "system", "content": CONTROLLED_SYSTEM_PROMPT},
                    {"role": "user", "content": problem},
                ],
                "completion": [
                    {"role": "assistant", "content": completion},
                ],
            }
        )

    return records


def build_gsm8k_rl_rows(dataset_rows) -> list[dict]:
    rows: list[dict] = []
    for index, item in enumerate(dataset_rows):
        if "question" not in item or "answer" not in item:
            raise ValueError(
                f"GSM8K row {index} must contain question and answer fields"
            )
        question = str(item["question"])
        gold = extract_gsm8k_gold(str(item["answer"]))
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": CONTROLLED_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "answer": gold,
            }
        )
    return rows


def _token_count(encoded) -> int:
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "shape"):
        shape = getattr(encoded, "shape")
        if len(shape) >= 1:
            return int(shape[-1])
    return len(encoded)


def assert_prompt_token_limit(
    rows: list[dict],
    tokenizer,
    max_tokens: int = 512,
) -> dict:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not rows:
        raise ValueError("Cannot run prompt-token preflight on an empty dataset")

    lengths: list[int] = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"RL row {index} must contain a non-empty prompt list")

        encoded = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
        )
        length = _token_count(encoded)
        lengths.append(length)
        if length > max_tokens:
            raise ValueError(
                f"RL prompt row {index} has {length} tokens, exceeding hard "
                f"limit {max_tokens}; prompts are not truncated"
            )

    return {
        "count": len(lengths),
        "max_tokens": max(lengths),
        "p95_tokens": float(np.percentile(lengths, 95)),
        "limit": max_tokens,
    }
