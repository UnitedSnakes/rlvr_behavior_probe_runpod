from __future__ import annotations

import re
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


# Exact system prompt stated on the SFT model card.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Think step by step before responding to the user's query. "
    "Your thought process should be enclosed between <think> and </think> tags. "
    "Once your thought process is complete, write a response which should end in the final "
    "answer enclosed in \\boxed{}."
)

# Use one canonical tokenizer/chat template for BOTH checkpoints.
TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def resolve_checkpoint_revision(repo_id: str, requested_revision: str | None) -> str:
    """
    The SFT repository's main branch contains metadata/training config, while actual
    checkpoints live on branches. If revision='auto', inspect branches and select
    the branch with the largest trailing step number.

    For normal repos (e.g. the RL final checkpoint), use 'main'.
    """
    if requested_revision not in (None, "auto"):
        return requested_revision

    refs = HfApi().list_repo_refs(repo_id)
    branch_names = [b.name for b in refs.branches]

    # If main has actual weights, caller can simply request "main".
    # For the SFT metadata repo, prefer numbered checkpoint branches.
    numbered = []
    for name in branch_names:
        m = re.search(r"(\d+)(?!.*\d)", name)
        if m and name != "main":
            numbered.append((int(m.group(1)), name))

    if numbered:
        numbered.sort()
        chosen = numbered[-1][1]
        print(f"[revision:auto] {repo_id}")
        print(f"  available branches: {branch_names}")
        print(f"  selected latest numbered checkpoint: {chosen}")
        return chosen

    print(f"[revision:auto] No numbered checkpoint branch found for {repo_id}.")
    print(f"Available branches: {branch_names}")
    raise RuntimeError(
        "Could not auto-select an SFT checkpoint revision. "
        "Pass --sft-revision <branch-name> using one of the branches printed above."
    )


class Sampler:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype,
        revision: str = "main",
        max_input_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.max_input_tokens = max_input_tokens

        print(f"Loading tokenizer from {TOKENIZER_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print(f"Loading model {model_name} @ revision={revision}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()

    def format_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def sample(
        self,
        question: str,
        n: int,
        batch_rollouts: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        prompt = self.format_prompt(question)
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
            add_special_tokens=False,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        texts = []
        remaining = n
        batch_index = 0
        total_batches = (n + batch_rollouts - 1) // batch_rollouts

        while remaining > 0:
            batch_index += 1
            batch_size = min(batch_rollouts, remaining)

            input_ids = enc["input_ids"].repeat(batch_size, 1)
            attention_mask = enc["attention_mask"].repeat(batch_size, 1)

            start_time = time.perf_counter()

            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

            elapsed = time.perf_counter() - start_time

            print(
                f"  batch {batch_index}/{total_batches} "
                f"({batch_size} rollouts) "
                f"{elapsed:.1f}s"
            )

            for seq in out:
                texts.append(
                    self.tokenizer.decode(
                        seq[prompt_len:],
                        skip_special_tokens=True,
                    )
                )

            remaining -= batch_size

        return texts
