from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from controlled_run.config import load_config
from controlled_run.train_sft import (
    DEFAULT_CONFIG,
    DEFAULT_RECORDS,
    DEFAULT_SOURCE_REVISIONS,
    _load_sft_classes,
    build_sft_arguments,
    load_prompt_completion_jsonl,
)

IM_END = 151645
EOT = 151643


def patch_assistant_terminal_to_native_eos(tokenizer):
    old = (
        "        {{- '<|im_end|>\\n' }}\n"
        '    {%- elif message.role == "tool" %}'
    )

    new = (
        '        {%- if message.role == "assistant" %}\n'
        "            {{- '<|endoftext|>\\n' }}\n"
        "        {%- else %}\n"
        "            {{- '<|im_end|>\\n' }}\n"
        "        {%- endif %}\n"
        '    {%- elif message.role == "tool" %}'
    )

    template = tokenizer.chat_template

    if template.count(old) != 1:
        raise RuntimeError(
            f"Expected exactly one assistant/tool boundary to patch; "
            f"found {template.count(old)}"
        )

    tokenizer.chat_template = template.replace(old, new)


def ids_from_text(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def verify_single_variable(current_tok, native_tok, row):
    # Prompt used at generation time MUST remain byte/token identical.
    prompt_a = current_tok.apply_chat_template(
        row["prompt"],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_b = native_tok.apply_chat_template(
        row["prompt"],
        tokenize=False,
        add_generation_prompt=True,
    )

    if prompt_a != prompt_b:
        raise RuntimeError("Patched template changed generation prompt")

    pa = ids_from_text(current_tok, prompt_a)
    pb = ids_from_text(native_tok, prompt_b)

    if pa != pb:
        raise RuntimeError("Patched template changed prompt token IDs")

    # Full SFT sequence should differ in exactly ONE token:
    # assistant <|im_end|> -> <|endoftext|>
    messages = row["prompt"] + row["completion"]

    full_a = current_tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_b = native_tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    ia = ids_from_text(current_tok, full_a)
    ib = ids_from_text(native_tok, full_b)

    if len(ia) != len(ib):
        raise RuntimeError(
            f"Sequence lengths differ: current={len(ia)}, native={len(ib)}"
        )

    diffs = [
        (i, a, b)
        for i, (a, b) in enumerate(zip(ia, ib))
        if a != b
    ]

    print("template verification:")
    print("  prompt tokens identical:", len(pa))
    print("  full tokens:", len(ia))
    print("  differing positions:", diffs)

    if len(diffs) != 1:
        raise RuntimeError(
            f"Expected exactly one token difference, got {diffs}"
        )

    pos, a, b = diffs[0]
    if a != IM_END or b != EOT:
        raise RuntimeError(
            f"Wrong terminal replacement at {pos}: {a} -> {b}"
        )

    print(
        "  PASS: only assistant terminal changed "
        f"{IM_END}(<|im_end|>) -> {EOT}(<|endoftext|>)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        required=True,
        choices=["current", "native_eos"],
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(DEFAULT_CONFIG)

    revisions = json.loads(
        Path(DEFAULT_SOURCE_REVISIONS).read_text()
    )
    base_sha = revisions["base_model"]["sha"]
    base_name = revisions["base_model"]["repo_id"]

    # Build both tokenizers first so we can assert that B really is
    # a one-token intervention.
    current_tok = AutoTokenizer.from_pretrained(
        base_name,
        revision=base_sha,
    )
    native_tok = AutoTokenizer.from_pretrained(
        base_name,
        revision=base_sha,
    )
    patch_assistant_terminal_to_native_eos(native_tok)

    dataset = load_prompt_completion_jsonl(DEFAULT_RECORDS)

    verify_single_variable(
        current_tok,
        native_tok,
        dataset[0],
    )

    tokenizer = (
        current_tok
        if args.variant == "current"
        else native_tok
    )

    print("\nvariant:", args.variant)
    print("runtime tokenizer eos:",
          tokenizer.eos_token,
          tokenizer.eos_token_id)

    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        revision=base_sha,
        dtype=torch.bfloat16,
        attn_implementation=config["attn_implementation"],
    )

    train_args = build_sft_arguments(
        config,
        args.output_dir / "trainer",
        max_steps=args.steps,
    )

    _, SFTTrainer = _load_sft_classes()

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        final = args.output_dir / "final"
        trainer.save_model(str(final))
        tokenizer.save_pretrained(str(final))

        print("\nsaved:", final)

    if accelerator is not None:
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
