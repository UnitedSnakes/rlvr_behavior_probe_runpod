import importlib
import sys
from types import ModuleType, SimpleNamespace


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "formatted prompt"

    def __call__(
        self,
        prompt,
        truncation,
        max_length,
        add_special_tokens,
    ):
        assert prompt == "formatted prompt"
        assert truncation is True
        assert max_length == 1024
        assert add_special_tokens is False
        return {"input_ids": [10, 11, 12]}


class FakeAutoTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        cls.calls.append((name, kwargs))
        return FakeTokenizer()


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLLM:
    init_kwargs = None
    generate_calls = []

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def generate(self, prompts, sampling_params, use_tqdm):
        type(self).generate_calls.append(
            {
                "prompts": prompts,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
            }
        )
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(text="first"),
                    SimpleNamespace(text="second"),
                    SimpleNamespace(text="third"),
                ]
            )
        ]


def import_vllm_model(monkeypatch):
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    fake_vllm_inputs = ModuleType("vllm.inputs")
    fake_vllm_inputs.TokensPrompt = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm.inputs", fake_vllm_inputs)

    sys.modules.pop("probe.vllm_model", None)
    return importlib.import_module("probe.vllm_model")


def test_vllm_sampler_preserves_model_revision_and_dtype(monkeypatch):
    module = import_vllm_model(monkeypatch)

    module.VLLMSampler(
        model_name="example/model",
        device="cuda",
        dtype="bfloat16",
        revision="checkpoint-8-of-10",
        gpu_memory_utilization=0.85,
    )

    tokenizer_name, tokenizer_kwargs = FakeAutoTokenizer.calls[-1]
    assert tokenizer_name == module.TOKENIZER_NAME
    assert tokenizer_kwargs["revision"] == module.TOKENIZER_REVISION
    assert FakeLLM.init_kwargs["model"] == "example/model"
    assert FakeLLM.init_kwargs["tokenizer"] == module.TOKENIZER_NAME
    assert FakeLLM.init_kwargs["revision"] == "checkpoint-8-of-10"
    assert FakeLLM.init_kwargs["tokenizer_revision"] == module.TOKENIZER_REVISION
    assert FakeLLM.init_kwargs["dtype"] == "bfloat16"
    assert FakeLLM.init_kwargs["gpu_memory_utilization"] == 0.85


def test_vllm_sampler_uses_one_request_with_n_completions(monkeypatch):
    module = import_vllm_model(monkeypatch)
    FakeLLM.generate_calls.clear()

    sampler = module.VLLMSampler(
        model_name="example/model",
        device="cuda",
        dtype="bfloat16",
        revision="main",
    )

    texts = sampler.sample(
        question="What is 1+1?",
        n=3,
        batch_rollouts=1,
        max_new_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.1,
        seed=4200000,
    )

    assert texts == ["first", "second", "third"]
    assert len(FakeLLM.generate_calls) == 1

    call = FakeLLM.generate_calls[0]
    assert call["prompts"] == [{"prompt_token_ids": [10, 11, 12]}]
    assert call["use_tqdm"] is False
    assert call["sampling_params"].kwargs == {
        "n": 3,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.1,
        "max_tokens": 2048,
        "seed": 4200000,
    }
