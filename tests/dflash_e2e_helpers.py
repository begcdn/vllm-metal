# SPDX-License-Identifier: Apache-2.0
"""Shared engine setup for the spawned DFlash e2e tests."""

import os


def spawn_env(verify_window: bool) -> None:
    """Env a spawned serving process must set before importing vllm."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_METAL_SPEC_VERIFY_WINDOW"] = "1" if verify_window else "0"


def dflash_llm(**overrides):
    """LLM with the shared DFlash test configuration, tuned by *overrides*."""
    from vllm import LLM

    return LLM(
        **{
            "model": "mlx-community/Qwen3-4B-4bit",
            "max_model_len": 128,
            "max_num_seqs": 2,
            "max_num_batched_tokens": 32,
            "block_size": 16,
            "num_gpu_blocks_override": 10,
            "gpu_memory_utilization": 0.25,
            "enable_prefix_caching": False,
            "async_scheduling": False,
            **overrides,
        }
    )
