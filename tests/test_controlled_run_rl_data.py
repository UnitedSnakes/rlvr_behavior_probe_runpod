from __future__ import annotations

import pytest

from controlled_run.constants import CONTROLLED_SYSTEM_PROMPT
from controlled_run.data import assert_prompt_token_limit, build_gsm8k_rl_rows


class LengthTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        assert messages[0] == {
            "role": "system",
            "content": CONTROLLED_SYSTEM_PROMPT,
        }
        assert messages[1]["role"] == "user"
        length = int(messages[1]["content"].split()[-1])
        return list(range(length))


def test_build_gsm8k_rl_rows_uses_train_question_and_numeric_gold():
    rows = build_gsm8k_rl_rows(
        [
            {
                "question": "If Alice has 2 apples and gets 3 more, how many?",
                "answer": "Alice gets 2+3 = 5 apples.\n#### 5",
            }
        ]
    )

    assert rows == [
        {
            "prompt": [
                {"role": "system", "content": CONTROLLED_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "If Alice has 2 apples and gets 3 more, how many?",
                },
            ],
            "answer": "5",
        }
    ]


def test_build_gsm8k_rl_rows_rejects_unparseable_gold():
    with pytest.raises(ValueError, match="GSM8K gold"):
        build_gsm8k_rl_rows([{"question": "q", "answer": "no marker"}])


def test_prompt_preflight_uses_generation_chat_template_and_returns_summary():
    rows = [
        {
            "prompt": [
                {"role": "system", "content": CONTROLLED_SYSTEM_PROMPT},
                {"role": "user", "content": "length 100"},
            ],
            "answer": "1",
        },
        {
            "prompt": [
                {"role": "system", "content": CONTROLLED_SYSTEM_PROMPT},
                {"role": "user", "content": "length 200"},
            ],
            "answer": "2",
        },
        {
            "prompt": [
                {"role": "system", "content": CONTROLLED_SYSTEM_PROMPT},
                {"role": "user", "content": "length 300"},
            ],
            "answer": "3",
        },
    ]

    summary = assert_prompt_token_limit(rows, LengthTokenizer(), max_tokens=512)

    assert summary == {
        "count": 3,
        "max_tokens": 300,
        "p95_tokens": 290.0,
        "limit": 512,
    }


def test_prompt_preflight_fails_loudly_instead_of_truncating():
    rows = [
        {
            "prompt": [
                {"role": "system", "content": CONTROLLED_SYSTEM_PROMPT},
                {"role": "user", "content": "length 513"},
            ],
            "answer": "1",
        }
    ]

    with pytest.raises(ValueError, match="row 0.*513.*512"):
        assert_prompt_token_limit(rows, LengthTokenizer(), max_tokens=512)


def test_prompt_preflight_rejects_empty_dataset():
    with pytest.raises(ValueError, match="empty"):
        assert_prompt_token_limit([], LengthTokenizer(), max_tokens=512)
