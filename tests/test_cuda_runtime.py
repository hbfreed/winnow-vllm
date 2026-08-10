import pytest
import torch
import torch.nn.functional as F
from winnow.runtime.fast import FastOlmoeMoE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_runtime_matches_reference_math():
    torch.manual_seed(11)
    device = torch.device("cuda")
    module = FastOlmoeMoE(128, [128, 256, 128], 2).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        module.gate.weight.normal_(std=0.02)
        module.w_gate.normal_(std=0.02)
        module.w_up.normal_(std=0.02)
        module.w_down.normal_(std=0.02)
        inputs = torch.randn(13, 128, device=device, dtype=torch.bfloat16)
        actual = module(inputs)

        probabilities = F.softmax(module.gate(inputs), dim=-1, dtype=torch.float32)
        router_weights, experts = torch.topk(probabilities, 2, dim=-1)
        expected = torch.zeros_like(inputs, dtype=torch.float32)
        for expert in range(module.num_experts):
            tokens, slots = torch.where(experts == expert)
            if tokens.numel() == 0:
                continue
            current = inputs[tokens].float()
            gate = F.linear(current, module.expert_weight(expert, "gate").float())
            up = F.linear(current, module.expert_weight(expert, "up").float())
            current = F.silu(gate) * up
            current = F.linear(current, module.expert_weight(expert, "down").float())
            current *= router_weights[tokens, slots, None]
            expected.index_add_(0, tokens, current)

    relative_error = (actual.float() - expected).norm() / expected.norm()
    assert relative_error < 5e-3
