# GRPO 2×A40 Batch Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remap the post-2048 controlled GRPO run onto exactly 2×A40 while preserving the original global optimizer batch, generation batch, prompt-group count, and steps-per-generation semantics.

**Architecture:** Keep scientific batch semantics in the frozen GRPO YAML as explicit provenance fields (`canonical_world_size`, `global_optimizer_batch_size`). Add a pure runtime-batch derivation/validation helper in `controlled_run.config` and call it in `train_grpo.run_grpo` before model construction/training. Forward only TRL-supported fields to `GRPOConfig`.

**Tech Stack:** Python 3.12, PyTorch distributed environment variables, Transformers 5.15, TRL 1.12.0, pytest, YAML.

**Spec:** `docs/superpowers/specs/2026-08-29-grpo-2xa40-batch-semantics-amendment.md`

## Global Constraints

- canonical world size = 2
- per-device train batch size = 4
- gradient accumulation steps = 4
- global optimizer batch size = 32
- generation batch size = 32
- num generations = 16
- steps per generation = 4
- unique prompt groups per generation batch = 2
- max prompt tokens = 512
- max completion length = 2048
- vLLM max model length = 2560
- pilot/canonical GRPO must fail closed when runtime world size differs from 2
- do not change reward, LR, scheduler, clipping, loss, reward scaling, decoding, or IS settings

---

### Task 1: Freeze the 2×A40 batch recipe

**Files:**
- Modify: `tests/test_controlled_run_config.py`
- Modify: `tests/test_controlled_run_grpo.py`
- Modify: `controlled_run/configs/grpo_qwen3_0_6b.yaml`
- Modify: `controlled_run/config.py`

**Interfaces:**
- Produces: `validate_grpo_runtime_batch(config: dict, *, world_size: int) -> dict`

- [ ] **Step 1: Write failing config/runtime-batch tests**

Add assertions that the frozen YAML contains:

```python
assert cfg["canonical_world_size"] == 2
assert cfg["global_optimizer_batch_size"] == 32
assert cfg["per_device_train_batch_size"] == 4
```

Add a runtime mapping test:

```python
mapping = validate_grpo_runtime_batch(cfg, world_size=2)
assert mapping == {
    "world_size": 2,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "global_optimizer_batch_size": 32,
    "generation_batch_size": 32,
    "steps_per_generation": 4,
    "num_generations": 16,
    "unique_prompts_per_generation_batch": 2,
}
```

Also assert world sizes 1 and 3 fail loudly.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python -m pytest tests/test_controlled_run_config.py tests/test_controlled_run_grpo.py -q
```

Expected: failure because the YAML still has per-device batch 8 and `validate_grpo_runtime_batch` does not exist.

- [ ] **Step 3: Implement the exact frozen mapping**

Update YAML:

```yaml
canonical_world_size: 2
global_optimizer_batch_size: 32
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
generation_batch_size: 32
```

Update `GRPO_INVARIANTS` with the two new provenance fields and batch 4.

Implement `validate_grpo_runtime_batch` to:

```python
expected_world_size = int(config["canonical_world_size"])
if world_size != expected_world_size:
    raise ValueError(...)

global_optimizer_batch = (
    int(config["per_device_train_batch_size"])
    * world_size
    * int(config["gradient_accumulation_steps"])
)
if global_optimizer_batch != int(config["global_optimizer_batch_size"]):
    raise ValueError(...)

per_step_global_batch = int(config["per_device_train_batch_size"]) * world_size
if int(config["generation_batch_size"]) % per_step_global_batch != 0:
    raise ValueError(...)
steps_per_generation = int(config["generation_batch_size"]) // per_step_global_batch
unique_prompts = int(config["generation_batch_size"]) // int(config["num_generations"])
```

Require exact values 4 and 2 for the final two derived semantics through the frozen config/invariants.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same focused pytest command and require zero failures.

- [ ] **Step 5: Commit**

Commit with message:

```text
fix: preserve GRPO batch semantics on 2xA40
```

### Task 2: Fail closed on wrong GRPO launch topology

**Files:**
- Modify: `tests/test_controlled_run_grpo.py`
- Modify: `controlled_run/train_grpo.py`

**Interfaces:**
- Consumes: `validate_grpo_runtime_batch(config, world_size=...)`
- Produces: `resolve_runtime_world_size() -> int`

- [ ] **Step 1: Write failing runner tests**

Test that `resolve_runtime_world_size()` reads `WORLD_SIZE` and defaults to 1 when absent. Test that `run_grpo` invokes batch validation before model loading/trainer construction by monkeypatching downstream model/trainer entry points and forcing `WORLD_SIZE=1`; expect a `ValueError` mentioning 2 GPUs.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest tests/test_controlled_run_grpo.py -q
```

Expected: failure because runtime world-size validation is not yet wired into `run_grpo`.

- [ ] **Step 3: Implement minimal runtime guard**

Add:

```python
def resolve_runtime_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))
```

After loading/validating the GRPO config and before tokenizer/model/trainer construction:

```python
runtime_batch = validate_grpo_runtime_batch(
    config,
    world_size=resolve_runtime_world_size(),
)
```

Record `runtime_batch` in `grpo_run_manifest.json` and in the returned result.

- [ ] **Step 4: Verify GREEN**

Run focused GRPO tests and require zero failures.

- [ ] **Step 5: Commit**

Commit with message:

```text
fix: gate GRPO on canonical 2xA40 topology
```

### Task 3: Full verification and live pilot handoff

**Files:**
- No production changes unless tests expose a concrete defect.

- [ ] **Step 1: Run the full controlled suite**

```bash
python -m pytest -q
```

Require zero failures.

- [ ] **Step 2: Verify GitHub Actions for the final commit**

Require the `Controlled Qwen3 tests` workflow to complete with `conclusion=success` for the exact final head SHA.

- [ ] **Step 3: Live runtime preflight on the A40 pod**

After pulling the exact head:

```bash
WORLD_SIZE=2 python - <<'PY'
from pathlib import Path
from controlled_run.config import load_config, validate_grpo_runtime_batch
cfg = load_config(Path("controlled_run/configs/grpo_qwen3_0_6b.yaml"))
print(validate_grpo_runtime_batch(cfg, world_size=2))
PY
```

Expected exact mapping:

```text
world_size=2
per_device_train_batch_size=4
gradient_accumulation_steps=4
global_optimizer_batch_size=32
generation_batch_size=32
steps_per_generation=4
num_generations=16
unique_prompts_per_generation_batch=2
```

- [ ] **Step 4: Run the canonical-semantics disposable pilot**

```bash
rm -rf controlled_run_outputs/grpo_pilot_20_2048_2xa40
set -o pipefail

torchrun --nproc_per_node=2 \
  -m controlled_run.train_grpo \
  --pi0-dir controlled_run_outputs/sft/pi_0 \
  --output-dir controlled_run_outputs/grpo_pilot_20_2048_2xa40 \
  --mode pilot \
  --pilot-steps 20 \
  2>&1 | tee controlled_run_outputs/grpo_pilot_20_2048_2xa40.log
```

- [ ] **Step 5: Acceptance review**

Require no OOM, runtime manifest batch mapping exactly as frozen, final epoch approximately 0.00535, and inspect clipped ratio/reward/zero-std/entropy/grad norm/clip/logp/IS diagnostics before allowing canonical GRPO.
