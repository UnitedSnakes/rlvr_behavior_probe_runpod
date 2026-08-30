# M5 Development / A40 Production Infrastructure Separation

## Goal

Make Apple Silicon the default environment for development, tests, deterministic CPU data preparation, and analysis, while reserving A40 RunPod instances for CUDA-specific integration, training, large-scale sampling, and canonical CUDA evaluation.

## Execution lanes

### M5 Pro development lane

The M5 lane must support:

- Python unit tests for controlled-run logic;
- deterministic `controlled_run.prepare_data` against pinned Hugging Face revisions;
- contamination and token-length audits;
- generation of the canonical 10,000-train / 512-validation SFT data bundle;
- analysis and plotting;
- optional vLLM-Metal inference smoke in a separate environment.

The ordinary controlled-run development environment must not depend on vLLM, CUDA, TRL, FlashAttention, or the vLLM-Metal plugin. The existing `~/.venv-vllm-metal` remains separate and optional.

### A40 production lane

The A40 lane begins when CUDA-specific behavior matters:

- image/runtime acceptance;
- SFT engineering smoke and canonical SFT;
- large `p_0` sampling;
- GRPO engineering pilot and canonical GRPO;
- canonical CUDA evaluation.

A40 should consume a previously materialized data bundle when available rather than rematerializing it only because the execution host changed.

## Data bundle handoff

`prepare_data` produces the six canonical data artifacts:

- `manifests/sft_10k_manifest.jsonl`
- `manifests/sft_val_512_manifest.jsonl`
- `manifests/contamination_audit.json`
- `manifests/source_revisions.json`
- `generated/sft_10k_records.jsonl`
- `generated/sft_val_512_records.jsonl`

It also writes `manifests/data_bundle_manifest.json` containing:

- schema version;
- source identity scheme;
- train/validation counts;
- pinned source revisions;
- SHA256 for every artifact above.

A platform-independent verifier must reject missing files, hash mismatches, wrong counts, or source-revision mismatch before training consumes a transferred bundle.

Scientific identity is defined by pinned source revisions, deterministic source indices/seed, and file hashes, not by the machine that materialized the data.

## Dependency boundaries

### Generic dev requirements

Create `controlled_run/requirements-dev.txt` for Python 3.12 development/data work. It should include the packages required by tests, data preparation, provenance, and analysis, but not CUDA-only packages or vLLM-Metal.

### A40 runtime

The Docker image is the canonical A40 runtime source. `controlled_run/requirements-a40.in` must not independently pin conflicting versions of vLLM, Transformers, datasets, or Accelerate. It should represent A40 training extras that are layered onto the image, or be explicitly documented as image-build input.

The built image must verify at build time that the intended Python, Torch/CUDA, Transformers, vLLM, datasets, Accelerate, TRL, and observability packages import successfully and print their versions. `gpustat` is an observability convenience, not a scientific dependency.

FlashAttention must not be silently assumed present: the image/runtime acceptance path must explicitly test the configured SFT attention backend before canonical SFT. If FlashAttention cannot be made compatible with the pinned CUDA/Torch runtime, changing the scientific SFT attention backend requires a separate documented decision rather than an implicit infra fallback.

## Documentation and workflow

The controlled checkpoint and README should describe M5-first development/data preparation and A40-only CUDA work. RunPod image workflows remain responsible only for the A40 lane.

## Constraints

- Do not change canonical SFT or GRPO scientific hyperparameters in this infra change.
- Do not change pinned dataset/model revisions or deterministic selection semantics.
- Do not merge vLLM-Metal rollouts into canonical CUDA measurements.
- Current live A40 work may continue; this refactor applies to subsequent pulls/runs.