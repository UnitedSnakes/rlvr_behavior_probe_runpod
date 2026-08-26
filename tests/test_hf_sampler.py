import importlib
import sys
from types import ModuleType

import torch


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "formatted prompt"

    def __call__(
        self,
        prompt,
        return_tensors,
        truncation,
        max_length,
        add_special_tokens,
    ):
        assert prompt == "formatted prompt"
        assert return_tensors == "pt"
        assert truncation is True
        assert max_length == 1024
        assert add_special_tokens is False
        return {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, sequence, skip_special_tokens):
        assert skip_special_tokens is True
        return "decoded"


class FakeAutoTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        cls.calls.append((name, kwargs))
        return FakeTokenizer()


class FakeModel:
    generate_calls = []

    def to(self, device):
        assert device == "cpu"
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        type(self).generate_calls.append(kwargs)
        return torch.tensor([[10, 11, 12, 13]])


class FakeAutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return FakeModel()


def import_hf_model(monkeypatch):
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    sys.modules.pop("probe.model", None)
    return importlib.import_module("probe.model")


def test_hf_sampler_passes_canonical_sampling_policy_explicitly(monkeypatch):
    module = import_hf_model(monkeypatch)
    FakeModel.generate_calls.clear()

    sampler = module.Sampler(
        model_name="example/model",
        device="cpu",
        dtype=torch.float32,
        revision="checkpoint-8-of-10",
    )
    texts = sampler.sample(
        question="What is 1+1?",
        n=1,
        batch_rollouts=1,
        max_new_tokens=128,
        temperature=1.0,
        top_p=0.95,
        top_k=0,
        repetition_penalty=1.0,
        seed=42,
    )

    assert texts == ["decoded"]
    tokenizer_name, tokenizer_kwargs = FakeAutoTokenizer.calls[-1]
    assert tokenizer_name == module.TOKENIZER_NAME
    assert tokenizer_kwargs["revision"] == module.TOKENIZER_REVISION

    generate_kwargs = FakeModel.generate_calls[-1]
    assert generate_kwargs["temperature"] == 1.0
    assert generate_kwargs["top_p"] == 0.95
    assert generate_kwargs["top_k"] == 0
    assert generate_kwargs["repetition_penalty"] == 1.0


def test_hf_sampler_allows_hf_like_sampling_override(monkeypatch):
    module = import_hf_model(monkeypatch)
    FakeModel.generate_calls.clear()

    sampler = module.Sampler(
        model_name="example/model",
        device="cpu",
        dtype=torch.float32,
    )
    sampler.sample(
        question="What is 1+1?",
        n=1,
        batch_rollouts=1,
        max_new_tokens=128,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.1,
        seed=42,
    )

    generate_kwargs = FakeModel.generate_calls[-1]
    assert generate_kwargs["top_k"] == 20
    assert generate_kwargs["repetition_penalty"] == 1.1
