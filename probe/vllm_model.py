from __future__ import annotations

import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from probe.model import SYSTEM_PROMPT, TOKENIZER_NAME


def normalize_vllm_dtype(dtype) -> str:
    name = str(dtype)
    if name.startswith("torch."):
        name = name.removeprefix("torch.")

    aliases = {
        "float": "float32",
        "half": "float16",
    }
    return aliases.get(name, name)


class VLLMSampler:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype,
        revision: str = "main",
        max_input_tokens: int = 1024,
        gpu_memory_utilization: float = 0.90,
    ):
        if not str(device).startswith("cuda"):
            raise ValueError(
                "The vLLM backend currently requires a CUDA device. "
                f"Resolved device was {device!r}."
            )

        self.model_name = model_name
        self.device = device
        self.dtype = normalize_vllm_dtype(dtype)
        self.revision = revision
        self.max_input_tokens = max_input_tokens
        self.gpu_memory_utilization = gpu_memory_utilization

        print(f"Loading tokenizer from {TOKENIZER_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading vLLM model {model_name} @ revision={revision}")
        self.model = LLM(
            model=model_name,
            tokenizer=TOKENIZER_NAME,
            revision=revision,
            dtype=self.dtype,
            gpu_memory_utilization=gpu_memory_utilization,
        )

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

    def tokenize_prompt(self, question: str) -> list[int]:
        prompt = self.format_prompt(question)
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_input_tokens,
            add_special_tokens=False,
        )
        return encoded["input_ids"]

    def sample(
        self,
        question: str,
        n: int,
        batch_rollouts: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[str]:
        del batch_rollouts

        prompt_token_ids = self.tokenize_prompt(question)
        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
        )

        start_time = time.perf_counter()
        request_outputs = self.model.generate(
            [prompt],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - start_time

        if len(request_outputs) != 1:
            raise RuntimeError(
                "Expected one vLLM request output for one question, "
                f"received {len(request_outputs)}"
            )

        texts = [output.text for output in request_outputs[0].outputs]
        if len(texts) != n:
            raise RuntimeError(
                f"Requested {n} vLLM completions but received {len(texts)}"
            )

        print(f"  vLLM generated {n} rollouts in {elapsed:.1f}s")
        return texts
