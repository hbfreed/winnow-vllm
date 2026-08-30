"""Register Winnow checkpoint architectures with vLLM."""

from __future__ import annotations

from importlib.metadata import version

from packaging.version import Version

__version__ = "0.1.0"


def register() -> None:
    """Register Winnow models in each vLLM process."""
    installed = Version(version("vllm"))
    if installed != Version("0.25.0"):
        raise RuntimeError(f"moe-winnow-vllm 0.1 requires vLLM 0.25.0; found {installed}")

    from transformers import AutoConfig
    from vllm import ModelRegistry
    from winnow.runtime import WinnowAfmoeConfig, WinnowOlmoeConfig, WinnowQwen3_5MoeConfig

    for model_type, config_type in (
        ("winnow_olmoe", WinnowOlmoeConfig),
        ("winnow_qwen3_5_moe", WinnowQwen3_5MoeConfig),
        ("winnow_afmoe", WinnowAfmoeConfig),
    ):
        try:
            AutoConfig.register(model_type, config_type)
        except ValueError:
            pass

    ModelRegistry.register_model(
        "WinnowOlmoeForCausalLM",
        "winnow_vllm.olmoe:WinnowOlmoeForCausalLM",
    )
    ModelRegistry.register_model(
        "WinnowQwen3_5MoeForCausalLM",
        "winnow_vllm.qwen:WinnowQwen3_5MoeForCausalLM",
    )
    ModelRegistry.register_model(
        "WinnowAfmoeForCausalLM",
        "winnow_vllm.afmoe:WinnowAfmoeForCausalLM",
    )


__all__ = ["register"]
