from __future__ import annotations

from collections import UserDict

from controlled_run.data import _formatted_token_count, _token_count


class MappingTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        assert tokenize is True
        return UserDict(
            {
                "input_ids": list(range(137)),
                "attention_mask": [1] * 137,
            }
        )


def test_formatted_token_count_handles_batch_encoding_mapping():
    assert _formatted_token_count(MappingTokenizer(), "problem", "completion") == 137


def test_token_count_handles_batch_encoding_mapping():
    encoded = UserDict(
        {
            "input_ids": list(range(211)),
            "attention_mask": [1] * 211,
        }
    )
    assert _token_count(encoded) == 211
