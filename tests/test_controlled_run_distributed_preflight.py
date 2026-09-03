import pytest

from controlled_run.distributed_preflight import (
    validate_all_reduce_result,
    validate_static_contract,
)


def test_valid_2xa40_static_contract_passes():
    validate_static_contract(
        world_size=2,
        local_rank=0,
        device_count=2,
        gpu_name="NVIDIA A40",
    )

    validate_static_contract(
        world_size=2,
        local_rank=1,
        device_count=2,
        gpu_name="NVIDIA A40",
    )


def test_wrong_world_size_fails_closed():
    with pytest.raises(RuntimeError, match="WORLD_SIZE=2"):
        validate_static_contract(
            world_size=1,
            local_rank=0,
            device_count=2,
            gpu_name="NVIDIA A40",
        )


def test_missing_second_gpu_fails_closed():
    with pytest.raises(RuntimeError, match="visible CUDA devices"):
        validate_static_contract(
            world_size=2,
            local_rank=0,
            device_count=1,
            gpu_name="NVIDIA A40",
        )


def test_wrong_gpu_class_fails_closed():
    with pytest.raises(RuntimeError, match="NVIDIA A40"):
        validate_static_contract(
            world_size=2,
            local_rank=0,
            device_count=2,
            gpu_name="NVIDIA RTX 4090",
        )


def test_expected_all_reduce_result_passes():
    validate_all_reduce_result(3.0)


def test_wrong_all_reduce_result_fails_closed():
    with pytest.raises(RuntimeError, match="expected 3.0"):
        validate_all_reduce_result(2.0)
