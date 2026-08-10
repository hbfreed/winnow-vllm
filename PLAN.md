# Release plan

Winnow-vLLM 0.1 has these parts:

- Register Winnow OLMoE and Qwen3.5 or Qwen3.6 MoE checkpoints with vLLM.
- Load each pruned expert weight exactly once.
- Use the fused runtime from Winnow.
- Support vLLM 0.25.0 with BF16 or FP16 weights.
- Fail early for unsupported modes.
- Do not include pruning, training, or healing.

Before each release:

1. Run the loader and registration tests.
2. Run a CUDA parity test through Winnow.
3. Build the wheel and source archive.
4. Serve one model from each supported family.
