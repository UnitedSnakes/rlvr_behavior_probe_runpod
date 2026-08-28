from __future__ import annotations

import numpy as np


def extended_length_audit(lengths: list[int]) -> dict:
    values = np.asarray(lengths, dtype=np.int64)
    if values.size == 0:
        return {
            "p99_5": None,
            "p99_9": None,
            "max": None,
            "gt_12288": 0.0,
            "gt_16384": 0.0,
            "gt_32768": 0.0,
        }

    return {
        "p99_5": float(np.percentile(values, 99.5)),
        "p99_9": float(np.percentile(values, 99.9)),
        "max": int(values.max()),
        "gt_12288": float(np.mean(values > 12_288)),
        "gt_16384": float(np.mean(values > 16_384)),
        "gt_32768": float(np.mean(values > 32_768)),
    }


class RecordingTokenizer:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.formatted_lengths: list[int] = []

    def apply_chat_template(self, *args, **kwargs):
        encoded = self._tokenizer.apply_chat_template(*args, **kwargs)
        if kwargs.get("tokenize") is True:
            from controlled_run.data import _token_count

            self.formatted_lengths.append(_token_count(encoded))
        return encoded

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)
