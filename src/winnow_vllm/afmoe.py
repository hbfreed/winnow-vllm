"""vLLM model adapter for Winnow Afmoe (Arcee Trinity) checkpoints."""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.afmoe import (
    AfmoeAttention,
    AfmoeDecoderLayer,
    AfmoeForCausalLM,
    AfmoeModel,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from winnow.runtime.fast import FastSigmoidMoE

from .policy import use_int8_w8a16, validate_dtype, validate_eager, validate_expert_quantization

_EXPERT_WEIGHT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_up_projs|down_projs)\.(\d+)$"
)
_RENAMES = (
    (".mlp.router.gate.weight", ".mlp.gate.weight"),
    (".mlp.expert_bias", ".mlp.e_score_correction_bias"),
)


class _LocalMLP(nn.Module):
    """Keep the checkpoint names for the dense and shared-expert SwiGLU."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class WinnowAfmoeDecoderLayer(AfmoeDecoderLayer):
    """Use Winnow fused experts in the stock Afmoe decoder layer."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.layer_idx = extract_layer_index(prefix)
        self.self_attn = AfmoeAttention(
            config=config,
            layer_idx=self.layer_idx,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=getattr(config, "max_position_embeddings", 131072),
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.moe_enabled = self.layer_idx >= config.num_dense_layers
        if self.moe_enabled:
            widths = config.expert_widths[self.layer_idx - config.num_dense_layers]
            mlp = FastSigmoidMoE(
                config.hidden_size,
                list(widths),
                config.num_experts_per_tok,
                float(config.route_scale),
                quantize_w8a16=use_int8_w8a16(vllm_config.quant_config),
            )
            mlp.shared_experts = _LocalMLP(
                config.hidden_size,
                config.moe_intermediate_size * config.num_shared_experts,
            )
            self.mlp = mlp
        else:
            self.mlp = _LocalMLP(config.hidden_size, config.intermediate_size)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class WinnowAfmoeModel(AfmoeModel):
    """Build the Afmoe skeleton without a uniform fused MoE allocation."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.mup_enabled = config.mup_enabled

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: WinnowAfmoeDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )


class WinnowAfmoeForCausalLM(AfmoeForCausalLM):
    """Serve a Winnow Afmoe checkpoint."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        validate_expert_quantization(vllm_config.quant_config)
        validate_dtype(vllm_config)
        validate_eager(vllm_config)
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("Winnow Afmoe supports tensor parallel size 1")
        expected_rows = config.num_hidden_layers - config.num_dense_layers
        if len(config.expert_widths) != expected_rows:
            raise ValueError("expert_widths must contain one row for each sparse decoder layer")

        nn.Module.__init__(self)
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = WinnowAfmoeModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, quant_config=self.quant_config
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    _STACKED = (
        ("qkv_proj", ".q_proj", "q"),
        ("qkv_proj", ".k_proj", "k"),
        ("qkv_proj", ".v_proj", "v"),
    )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load and pack every checkpoint tensor exactly once, PP-aware."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        start_layer = self.model.start_layer
        end_layer = self.model.end_layer

        def layer_of(name: str) -> int | None:
            parts = name.split(".")
            if len(parts) > 2 and parts[:2] == ["model", "layers"]:
                return int(parts[2])
            return None

        for original_name, weight in weights:
            layer = layer_of(original_name)
            if layer is not None and not start_layer <= layer < end_layer:
                continue

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
            for source, target in _RENAMES:
                name = name.replace(source, target)
            if name.endswith(".e_score_correction_bias"):
                # The sigmoid selection bias stays float32 like the reference.
                parameter = params[name]
                parameter.data = weight.to(device=parameter.device, dtype=torch.float32)
                loaded.add(name)
                continue
            for parameter_name, checkpoint_name, shard in self._STACKED:
                if checkpoint_name in name and ".mlp." not in name:
                    name = name.replace(checkpoint_name, f".{parameter_name}")
                    parameter = params[name]
                    parameter.weight_loader(parameter, weight, shard)
                    break
            else:
                if name not in params:
                    if layer is None and (
                        isinstance(self.lm_head, PPMissingLayer)
                        or isinstance(self.model.embed_tokens, PPMissingLayer)
                        or isinstance(self.model.norm, PPMissingLayer)
                    ):
                        continue  # embed/norm/head belong to another pipeline stage
                    raise ValueError(f"checkpoint weight {name} has no destination parameter")
                parameter = params[name]
                getattr(parameter, "weight_loader", default_weight_loader)(parameter, weight)
            loaded.add(name)
        return loaded
