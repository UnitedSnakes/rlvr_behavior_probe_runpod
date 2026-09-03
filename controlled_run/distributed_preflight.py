from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


EXPECTED_WORLD_SIZE = 2
EXPECTED_ALL_REDUCE_SUM = 3.0


def validate_static_contract(
    *,
    world_size: int,
    local_rank: int,
    device_count: int,
    gpu_name: str,
) -> None:
    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"2xA40 preflight requires WORLD_SIZE={EXPECTED_WORLD_SIZE}; "
            f"found {world_size}"
        )

    if local_rank < 0 or local_rank >= EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"unexpected LOCAL_RANK={local_rank}")

    if device_count < EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"2xA40 preflight requires at least {EXPECTED_WORLD_SIZE} "
            f"visible CUDA devices; found {device_count}"
        )

    if "A40" not in gpu_name:
        raise RuntimeError(
            f"2xA40 preflight requires an NVIDIA A40 on each rank; "
            f"found {gpu_name!r}"
        )


def validate_all_reduce_result(value: float) -> None:
    if abs(float(value) - EXPECTED_ALL_REDUCE_SUM) > 1e-6:
        raise RuntimeError(
            f"NCCL all_reduce returned {value}; "
            f"expected {EXPECTED_ALL_REDUCE_SUM}"
        )


def _nccl_version() -> str | None:
    try:
        value = torch.cuda.nccl.version()
    except Exception:
        return None

    if isinstance(value, tuple):
        return ".".join(str(part) for part in value)

    return str(value)


def _nvidia_smi_summary() -> list[str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,pci.bus_id",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except Exception:
        return None

    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def run_preflight(
    *,
    collective_timeout_seconds: int,
    output_json: Path | None,
) -> dict[str, Any] | None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch.cuda.is_available():
        raise RuntimeError("2xA40 preflight requires CUDA")

    device_count = int(torch.cuda.device_count())

    torch.cuda.set_device(local_rank)
    gpu_name = str(torch.cuda.get_device_name(local_rank))

    validate_static_contract(
        world_size=world_size,
        local_rank=local_rank,
        device_count=device_count,
        gpu_name=gpu_name,
    )

    print(
        f"[2XA40-PREFLIGHT rank={rank}] "
        f"device={local_rank} name={gpu_name} INIT",
        flush=True,
    )

    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=int(collective_timeout_seconds)),
    )

    try:
        value = torch.tensor(
            [float(rank + 1)],
            device=f"cuda:{local_rank}",
        )

        torch.cuda.synchronize()

        print(
            f"[2XA40-PREFLIGHT rank={rank}] "
            f"BEFORE_ALLREDUCE x={value.item()}",
            flush=True,
        )

        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        observed = float(value.item())
        validate_all_reduce_result(observed)

        print(
            f"[2XA40-PREFLIGHT rank={rank}] "
            f"AFTER_ALLREDUCE x={observed}",
            flush=True,
        )

        dist.barrier()

        if rank != 0:
            return None

        payload: dict[str, Any] = {
            "status": "PASS",
            "backend": "nccl",
            "world_size": world_size,
            "visible_cuda_devices": device_count,
            "gpu_names": [
                str(torch.cuda.get_device_name(index))
                for index in range(device_count)
            ],
            "torch_version": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "nccl_version": _nccl_version(),
            "all_reduce_expected": EXPECTED_ALL_REDUCE_SUM,
            "all_reduce_observed": observed,
            "nvidia_smi": _nvidia_smi_summary(),
        }

        if output_json is not None:
            destination = Path(output_json)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        print(
            json.dumps(payload, indent=2, sort_keys=True),
            flush=True,
        )

        return payload

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed 2xA40 NCCL preflight "
            "for controlled CUDA runs."
        )
    )

    parser.add_argument(
        "--collective-timeout-seconds",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )

    args = parser.parse_args(argv)

    if args.collective_timeout_seconds <= 0:
        raise ValueError(
            "--collective-timeout-seconds must be positive"
        )

    run_preflight(
        collective_timeout_seconds=args.collective_timeout_seconds,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
