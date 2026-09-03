# M5 / 2×A40 execution-infrastructure amendment

Date: 2026-09-03

## Status

Operational infrastructure amendment only.

This amendment does not change the frozen GRPO or practical-MaxRL scientific
objective, sampling policy, reward, completion cap, batch geometry, optimizer,
or evaluation semantics.

## Motivation

Two operational gaps were observed while preparing the practical-MaxRL GPU
pilot.

First, a fresh 2×A40 RunPod instance successfully loaded the model and captured
vLLM CUDA graphs but then hung at the first NCCL collective. An isolated
two-rank `torch.distributed` smoke reproduced the failure:

```text
rank 0 BEFORE ALLREDUCE x=1
rank 1 BEFORE ALLREDUCE x=2
no rank reached AFTER ALLREDUCE
```

The failure occurred below the MaxRL/TRL/vLLM training logic and showed that
single-GPU/static runtime acceptance was insufficient to qualify a host for
the canonical two-GPU lane.

Second, M5-local analysis work had accumulated raw rollout, signal-ledger, and
snapshot-evaluation banks in the repository working tree. A broad
`git add .` temporarily staged those raw artifacts even though project policy
assigns them to local/private-Hugging-Face storage.

## Execution lanes

### M5 Pro

The ordinary local Python 3.12 environment owns:

- tests;
- deterministic data preparation;
- analysis;
- figures and lightweight tables;
- provenance verification;
- synthetic objective/estimator tests.

Apple-Silicon vLLM-Metal remains a separate optional smoke/development
environment.

### Controlled 2×A40

The CUDA lane owns:

- CUDA/FlashAttention runtime acceptance;
- real distributed NCCL preflight;
- canonical CUDA evaluation;
- GRPO/MaxRL pilots, shakedowns, and canonical training.

## Artifact boundary

Git stores source code, configs, scientific records, manifests, hashes,
lightweight derived tables, and paper-facing figures.

Raw rollouts, raw signal ledgers, raw snapshot banks, model checkpoints, and
large logs/intermediates remain local or in private Hugging Face repositories.

Raw computational outputs should preferentially live below the Git-ignored
`controlled_run_outputs/` tree.

## Exact repository gate

RunPod bootstrap may receive:

```text
RLVR_EXPECT_COMMIT=<approved exact execution commit SHA>
```

When supplied, bootstrap fails closed if the checked-out repository HEAD does
not match the approved commit.

The gate is operational provenance and does not redefine scientific lineage.

## Distributed host gate

The active controlled RunPod template may enable:

```text
RLVR_RUN_2XA40_PREFLIGHT=1
```

Bootstrap then launches a two-rank invocation of:

```text
controlled_run.distributed_preflight
```

The preflight requires:

- `WORLD_SIZE = 2`;
- at least two visible CUDA devices;
- an NVIDIA A40 on each rank;
- NCCL process-group initialization;
- a real two-rank GPU all-reduce;
- rank-local values 1 and 2 to produce sum 3.

The bootstrap wraps the preflight in a shell-level timeout so a host that
deadlocks below the Python training stack cannot hang bootstrap indefinitely.

Successful rank-0 preflight writes:

```text
/workspace/rlvr-2xa40-preflight.json
```

## Failure policy

A pod that fails the ordinary/default NCCL path is rejected for controlled
training.

`NCCL_P2P_DISABLE=1`, `NCCL_CUMEM_ENABLE=0`, or comparable transport
changes may be used diagnostically, but must not be silently promoted into the
canonical runtime merely to salvage a bad host.

Changing the canonical distributed transport policy requires a separate
documented decision.

## Canonical pi0 operational path

The corrected canonical pi0 Hugging Face download currently materializes the
actual model one level below its download root.

The training path is therefore the directory directly containing
`pi0_manifest.json`, currently:

```text
controlled_run_outputs/sft/pi_0/pi_0
```

The required lineage remains:

```text
f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
```

Training code remains responsible for fail-closed pi0 manifest verification.

## Image and startup behavior

The existing RunPod startup wrapper remains in force: `/start.sh` starts
first so SSH/Jupyter remain reachable, then `rlvr-bootstrap` runs and logs to
`/workspace/rlvr-bootstrap.log`. Bootstrap failure does not terminate the pod.

The image workflow now includes `codex/signal-ledger` for Docker-path changes.
Normal pushes produce a traceable `sha-*` image; the stable `0.27.1` tag
still requires explicit promotion.

## Scientific boundary

This amendment changes only host qualification, repository provenance, local
artifact hygiene, and operational documentation.

It does not authorize reinterpretation of MaxRL or GRPO outcomes and does not
modify the frozen objective intervention.
