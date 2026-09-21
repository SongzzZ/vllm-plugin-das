# SPDX-License-Identifier: Apache-2.0
# Ported from vllm-hcu-main (0.18) modelopt.py w8a16_gemm as a standalone
# module: deepgemm W8A16 GEMM with a PyTorch dequant fallback, registered as
# torch.ops.vllm.w8a16_gemm (the registration point the 0.18 overlay owned).

import torch

from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer

import vllm_hcu.platforms.envs as henvs


def _w8a16_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """W8A16 GEMM with runtime backend selection."""
    use_deepgemm = henvs.VLLM_HCU_USE_DEEPGEMM_W8A16_GEMM
    if use_deepgemm:
        try:
            import deepgemm

            out = deepgemm.w8a16_linear(x, w, scale)
            if bias is not None:
                out = out + bias
            return out
        except (ImportError, AttributeError):
            pass
    # Fallback to PyTorch native implementation
    weight_scale = scale.unsqueeze(1)
    weights = w.view(torch.int8).to(x.dtype) * weight_scale.to(x.dtype)
    return torch.nn.functional.linear(x, weights, bias)


def _w8a16_gemm_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty(
        (*x.shape[:-1], w.shape[0]), dtype=x.dtype, device=x.device
    )


try:
    direct_register_custom_op(
        op_name="w8a16_gemm",
        op_func=_w8a16_gemm,
        fake_impl=_w8a16_gemm_fake,
        tags=() if is_torch_equal_or_newer("2.7.0") else (
            torch.Tag.needs_fixed_stride_order,
        ),
    )
except RuntimeError:
    # Already registered by upstream or a previous load attempt
    pass


def w8a16_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Public entry: keep signature torch-compatible."""
    return torch.ops.vllm.w8a16_gemm(x, w, scale, bias)
