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


def use_int8_w8a16(quant_config) -> bool:
    """Whether vLLM requested expert-only online INT8 quantization.

    ``--quantization experts_int8`` gets stable Winnow semantics regardless of
    what the stock uniform-MoE implementation does with it: symmetric
    per-output-channel INT8 expert weights with BF16/FP16 activations, applied
    as each MoE block finishes loading.  Attention and dense weights stay in
    BF16/FP16.
    """
    return quant_config is not None and quant_config.get_name() == "experts_int8"


def validate_expert_quantization(quant_config) -> None:
    """Accept unquantized weights or expert-only INT8 W8A16."""
    if quant_config is None or use_int8_w8a16(quant_config):
        return
    raise NotImplementedError(
        "Winnow ragged experts support only unquantized BF16/FP16 or "
        f"experts_int8 W8A16, not quantization={quant_config.get_name()!r}"
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
