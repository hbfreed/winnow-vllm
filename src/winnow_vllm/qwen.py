"""vLLM model adapter for Winnow Qwen3.5 MoE checkpoints."""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLMBase,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextModel,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from winnow.runtime.fast import FastQwenMoE

from .policy import validate_dtype, validate_eager, validate_quantization

_EXPERT_WEIGHT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_up_projs|down_projs)\.(\d+)$"
)


class _SharedExpert(nn.Module):
    """Keep the checkpoint names for the stock shared expert."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class WinnowQwenDecoderLayer(Qwen3_5DecoderLayer):
    """Use Winnow fused experts in the stock hybrid decoder layer."""

    def __init__(self, vllm_config: VllmConfig, layer_type: str, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
            )
        elif layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"invalid layer_type {layer_type!r}")

        widths = config.expert_widths[self.layer_idx]
        mlp = FastQwenMoE(config.hidden_size, list(widths), config.num_experts_per_tok)
        mlp.shared_expert = _SharedExpert(
            config.hidden_size, config.shared_expert_intermediate_size
        )
        mlp.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
        self.mlp = mlp
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.ffn_layer_scale = nn.Parameter(torch.zeros(1, 1, config.hidden_size))


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class WinnowQwenModel(Qwen3_5Model):
    """Build the Qwen skeleton without a uniform fused MoE allocation."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super(Qwen3NextModel, self).__init__()
        config = vllm_config.model_config.hf_text_config
        parallel = vllm_config.parallel_config
        self.num_redundant_experts = parallel.eplb_config.num_redundant_experts
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)

        def get_layer(layer_prefix: str):
            return WinnowQwenDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(layer_prefix)],
                prefix=layer_prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.norm = (
            Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        self.aux_hidden_state_layers: tuple[int, ...] = ()


class WinnowQwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, IsHybrid, SupportsMRoPE):
    """Serve a Winnow Qwen3.5 MoE checkpoint."""

    get_mamba_state_dtype_from_config = (
        Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config
    )
    get_mamba_state_shape_from_config = (
        Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config
    )
    get_mamba_state_copy_func = Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func

    def get_mrope_input_positions(self, input_tokens, mm_features):
        """Return text-only M-RoPE positions."""
        if mm_features:
            raise ValueError("a Winnow text checkpoint cannot accept multimodal features")
        positions = torch.arange(len(input_tokens), dtype=torch.long).unsqueeze(0).expand(3, -1)
        return positions, 0

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config
        validate_quantization(vllm_config.quant_config)
        validate_dtype(vllm_config)
        validate_eager(vllm_config)
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("Winnow Qwen supports tensor parallel size 1")
        if len(config.expert_widths) != config.num_hidden_layers:
            raise ValueError("expert_widths must contain one row for each decoder layer")

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError("Qwen3.5 requires --mamba-cache-mode=align")
        self.quant_config = vllm_config.quant_config
        nn.Module.__init__(self)
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = WinnowQwenModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load expert tensors into the fused layout and pass through other weights."""
        loaded: set[str] = set()
        passthrough: list[tuple[str, torch.Tensor]] = []
        parameters: dict[str, nn.Parameter] | None = None
        start_layer = self.model.start_layer
        end_layer = self.model.end_layer

        def layer_of(name: str) -> int | None:
            parts = name.split(".")
            if len(parts) > 2 and parts[:2] == ["model", "layers"]:
                return int(parts[2])
            return None

        for name, weight in weights:
            match = _EXPERT_WEIGHT.match(name)
            if match is None:
                if ".mlp." in name:
                    layer = layer_of(name)
                    if layer is not None and not start_layer <= layer < end_layer:
                        continue
                    if parameters is None:
                        parameters = dict(self.named_parameters())
                    if name not in parameters:
                        raise ValueError(f"checkpoint weight {name} has no destination parameter")
                    with torch.no_grad():
                        parameters[name].copy_(weight)
                    loaded.add(name)
                else:
                    passthrough.append((name, weight))
                continue

            layer, kind, expert = (
                int(match.group(1)),
                match.group(2),
                int(match.group(3)),
            )
            if not start_layer <= layer < end_layer:
                continue
            mlp = self.model.layers[layer].mlp
            width = mlp.expert_widths[expert]
            if kind == "gate_up_projs":
                if tuple(weight.shape) != (2 * width, mlp.hidden_size):
                    raise ValueError(f"{name} has the wrong shape")
                gate = mlp.load_expert_weight_(expert, "gate", weight[:width])
                up = mlp.load_expert_weight_(expert, "up", weight[width:])
                loaded.add(f"model.layers.{layer}.mlp.{gate}")
                loaded.add(f"model.layers.{layer}.mlp.{up}")
            else:
                down = mlp.load_expert_weight_(expert, "down", weight)
                loaded.add(f"model.layers.{layer}.mlp.{down}")
        loaded |= super().load_weights(iter(passthrough))
        return loaded
