# Winnow-vLLM

Winnow-vLLM serves the aligned MoE checkpoints that Winnow creates.

Create an environment and install the two GitHub releases:

```bash
uv venv --python 3.12
UV_TORCH_BACKEND=cu130 uv pip install \
  "moe-winnow @ git+https://github.com/hbfreed/winnow.git@v0.1.0" \
  "moe-winnow-vllm @ git+https://github.com/hbfreed/winnow-vllm.git@v0.1.0"

.venv/bin/vllm serve MODEL --enforce-eager
```

Version 0.1 supports vLLM 0.25.0, OLMoE, and Qwen3.5 or Qwen3.6 MoE text
checkpoints. It supports BF16 or FP16 weights. Tensor parallelism is not
supported. OLMoE uses one pipeline stage. Qwen supports pipeline parallelism.
Quantization is not supported.
