# 2026-09-05 8xA40 host NCCL preflight rejection

## Status

The newly provisioned 8xA40 host was rejected before any replication training
or pilot outcome.

This is an infrastructure failure record, not a scientific result.

## Host and runtime

Observed host/container identifier:

```text
9936c8bc7399
```

Runtime facts:

```text
8 x NVIDIA A40, 46068 MiB each
driver 570.195.03
CUDA 13.0
NCCL 2.27.7-1
/dev/shm 187G, empty
torch.cuda.can_device_access_peer(0,1) = True both directions
```

The host topology included PIX-adjacent pairs (1,2), (3,4), and (6,7), plus
PXB and cross-NUMA SYS links.

ECC inspection showed no volatile errors, no uncorrectable DRAM/SRAM errors,
no pending row remaps, and no remapping failures. Some devices had historical
correctable aggregate counts / one correctable row remap, but those were not
used as the rejection criterion.

## Failure

The default fail-closed NCCL preflight on GPUs 0,1 initialized both ranks,
connected the communicator/rings, and both ranks entered the first one-float
ALLREDUCE. The collective made no progress and timed out after 30 seconds.

A subsequent remap test using the PIX-adjacent GPU pair 1,2 also failed to
complete the same tiny all-reduce and was terminated by the external timeout.

Therefore the failure is not explained by insufficient /dev/shm, a missing
rank, or merely choosing a PXB pair.

## Decision

Reject this physical host.

Do not run GRPO/MaxRL pilots or scientific training on it.

Do not make any of the following changes to rescue the host:

```text
NCCL_P2P_DISABLE=1
forced SHM transport
alternate NCCL transport overrides
changed distributed topology
changed scientific batch semantics
```

The active seed43/44 matched 2xA40 replication design remains unchanged.
Provision a fresh A40 host and rerun the default runtime acceptance and
2xA40 NCCL preflight before training.

No replication outcomes were produced or inspected on this rejected host.
