#!/usr/bin/env python
"""Verify the isolated LLaMA-Factory FlashAttention-2 environment."""

import os
from importlib.metadata import version

# FLA imports decorate some optional attention helpers with torch.compile.
# The FA2 container pins Triton for FLA, which can be incompatible with the
# PyTorch-bundled Inductor path, so keep torch.compile disabled for this env.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import flash_attn
import torch
import triton
from fla.modules.convolution import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


def main() -> None:
    imported_kernels = (causal_conv1d, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
    assert all(callable(kernel) for kernel in imported_kernels)

    print(f"[fa2] torch: {torch.__version__}")
    print(f"[fa2] torch CUDA: {torch.version.cuda}")
    print(f"[fa2] triton: {triton.__version__}")
    print(f"[fa2] flash-attn: {flash_attn.__version__}")
    print(f"[fa2] fla-core: {version('fla-core')}")
    print(f"[fa2] flash-linear-attention: {version('flash-linear-attention')}")
    print("[fa2] Qwen3.5 FLA imports: OK")


if __name__ == "__main__":
    main()
