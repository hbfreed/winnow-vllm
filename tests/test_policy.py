import pytest
import torch

from winnow_vllm.policy import validate_dtype, validate_quantization


class Quantization:
    def get_name(self):
        return "fp8"


def test_only_plain_weights_are_supported():
    validate_quantization(None)
    with pytest.raises(NotImplementedError, match="fp8"):
        validate_quantization(Quantization())


def test_only_bfloat16_and_float16_are_supported():
    class ModelConfig:
        dtype = torch.bfloat16

    class VllmConfig:
        model_config = ModelConfig()

    validate_dtype(VllmConfig())
    ModelConfig.dtype = torch.float16
    validate_dtype(VllmConfig())
    ModelConfig.dtype = torch.float32
    with pytest.raises(ValueError, match="BF16 or FP16"):
        validate_dtype(VllmConfig())
