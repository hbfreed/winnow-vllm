import torch
from torch import nn
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase
from winnow.runtime.fast import FastOlmoeMoE, FastQwenMoE

from winnow_vllm.olmoe import WinnowOlmoeForCausalLM
from winnow_vllm.qwen import WinnowQwen3_5MoeForCausalLM

HIDDEN = 128
WIDTHS = [128, 256]


class Layer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


class Model(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.layers = nn.ModuleList([Layer(mlp)])
        self.start_layer = 0
        self.end_layer = 1


def expert_weights(seed: int):
    generator = torch.Generator().manual_seed(seed)
    weights = []
    for expert, width in enumerate(WIDTHS):
        prefix = "model.layers.0.mlp.experts"
        weights.append(
            (
                f"{prefix}.gate_up_projs.{expert}",
                torch.randn(2 * width, HIDDEN, generator=generator),
            )
        )
        weights.append(
            (
                f"{prefix}.down_projs.{expert}",
                torch.randn(HIDDEN, width, generator=generator),
            )
        )
    return weights


def test_olmoe_loader_packs_combined_expert_weights():
    model = WinnowOlmoeForCausalLM.__new__(WinnowOlmoeForCausalLM)
    nn.Module.__init__(model)
    model.model = Model(FastOlmoeMoE(HIDDEN, WIDTHS, 1))
    loaded = model.load_weights(iter(expert_weights(1)))
    assert loaded == {
        "model.layers.0.mlp.w_gate",
        "model.layers.0.mlp.w_up",
        "model.layers.0.mlp.w_down",
    }


def test_qwen_loader_packs_combined_expert_weights(monkeypatch):
    monkeypatch.setattr(Qwen3_5ForCausalLMBase, "load_weights", lambda self, weights: set())
    model = WinnowQwen3_5MoeForCausalLM.__new__(WinnowQwen3_5MoeForCausalLM)
    nn.Module.__init__(model)
    model.model = Model(FastQwenMoE(HIDDEN, WIDTHS, 1))
    loaded = model.load_weights(iter(expert_weights(2)))
    assert loaded == {
        "model.layers.0.mlp.w_gate",
        "model.layers.0.mlp.w_up",
        "model.layers.0.mlp.w_down",
    }
