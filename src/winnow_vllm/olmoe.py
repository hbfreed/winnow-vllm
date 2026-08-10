"""vLLM model adapter for Winnow OLMoE checkpoints."""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.olmoe import (
    OlmoeAttention,
    OlmoeDecoderLayer,
    OlmoeForCausalLM,
)
from vllm.model_executor.models.utils import extract_layer_index
from winnow.runtime.fast import FastOlmoeMoE

from .policy import validate_dtype, validate_eager, validate_quantization

_EXPERT_WEIGHT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_up_projs|down_projs)\.(\d+)$"
)


class WinnowOlmoeDecoderLayer(OlmoeDecoderLayer):
    """Use Winnow fused experts in the stock vLLM decoder layer."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.self_attn = OlmoeAttention(vllm_config=vllm_config, prefix=f"{prefix}.self_attn")
        widths = config.expert_widths[extract_layer_index(prefix)]
        self.mlp = FastOlmoeMoE(config.hidden_size, list(widths), config.num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class WinnowOlmoeForCausalLM(OlmoeForCausalLM):
    """Serve a Winnow OLMoE checkpoint on one GPU."""

    _STACKED = (
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        validate_quantization(vllm_config.quant_config)
        validate_dtype(vllm_config)
        validate_eager(vllm_config)
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("Winnow OLMoE supports tensor parallel size 1")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("Winnow OLMoE supports pipeline parallel size 1")
        if len(config.expert_widths) != config.num_hidden_layers:
            raise ValueError("expert_widths must contain one row for each decoder layer")
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=WinnowOlmoeDecoderLayer,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load and pack every checkpoint tensor exactly once."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for original_name, weight in weights:
            match = _EXPERT_WEIGHT.match(original_name)
            if match is not None:
                layer, kind, expert = int(match.group(1)), match.group(2), int(match.group(3))
                mlp = self.model.layers[layer].mlp
                width = mlp.expert_widths[expert]
                if kind == "gate_up_projs":
                    if tuple(weight.shape) != (2 * width, mlp.hidden_size):
                        raise ValueError(f"{original_name} has the wrong shape")
                    gate = mlp.load_expert_weight_(expert, "gate", weight[:width])
                    up = mlp.load_expert_weight_(expert, "up", weight[width:])
                    loaded.add(f"model.layers.{layer}.mlp.{gate}")
                    loaded.add(f"model.layers.{layer}.mlp.{up}")
                else:
                    down = mlp.load_expert_weight_(expert, "down", weight)
                    loaded.add(f"model.layers.{layer}.mlp.{down}")
                continue

            name = original_name
            for parameter_name, checkpoint_name, shard in self._STACKED:
                if checkpoint_name in name and ".mlp." not in name:
                    name = name.replace(checkpoint_name, parameter_name)
                    parameter = params[name]
                    parameter.weight_loader(parameter, weight, shard)
                    break
            else:
                if name not in params:
                    raise ValueError(f"checkpoint weight {name} has no destination parameter")
                parameter = params[name]
                getattr(parameter, "weight_loader", default_weight_loader)(parameter, weight)
            loaded.add(name)
        return loaded
