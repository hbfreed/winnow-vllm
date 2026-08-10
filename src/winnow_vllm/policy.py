"""Validation for the first serving release."""

from __future__ import annotations

import torch


def validate_quantization(quant_config) -> None:
    """Accept unquantized BF16 or FP16 weights only."""
    if quant_config is None:
        return
    raise NotImplementedError(
        f"Winnow-vLLM 0.1 does not support quantization={quant_config.get_name()!r}"
    )


def validate_dtype(vllm_config) -> None:
    """Accept BF16 or FP16 model weights only."""
    dtype = vllm_config.model_config.dtype
    if dtype not in {torch.bfloat16, torch.float16}:
        raise ValueError("Winnow-vLLM 0.1 requires BF16 or FP16 model weights")


def validate_eager(vllm_config) -> None:
    """Require the tested eager execution path."""
    if not vllm_config.model_config.enforce_eager:
        raise ValueError("Winnow-vLLM 0.1 requires --enforce-eager")
