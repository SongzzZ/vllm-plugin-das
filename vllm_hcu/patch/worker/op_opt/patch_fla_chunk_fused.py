# SPDX-License-Identifier: Apache-2.0
# Ported from vllm-hcu-main d095594: gate chunk_gated_delta_rule_fwd between
# the custom fused kernels (fused_kkt_solve_wy + fused_h_o, no intermediate
# A/h tensors in DRAM) and the wheel's non-fused algorithm, controlled by
# VLLM_HCU_USE_CUSTOM_FUSED_GDN. Also registers the
# chunk_gated_delta_rule_static custom op used by the CUDA-graph static path.

from __future__ import annotations

import functools
from types import ModuleType
from typing import Optional

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
    require_unpatched,
)

TARGET_MODULE = "vllm.model_executor.layers.fla.ops.chunk"
PATCH_ID = "worker.op_opt.fla.chunk_fused_dispatch"
TARGETS = (
    f"{TARGET_MODULE}.chunk_gated_delta_rule_fwd",
    f"{TARGET_MODULE}.chunk_gated_delta_rule_static",
)
_MARKER = "_vllm_hcu_fla_chunk_fused_applied"
_WRAPPER = "_vllm_hcu_fla_chunk_fused_wrapper"

_FWD_PARAMETERS = (
    "q", "k", "v", "g", "beta", "scale",
    "initial_state", "output_final_state", "cu_seqlens",
    "chunk_indices", "chunk_offsets", "core_attn_out",
)


def _register_static_op() -> bool:
    """Register torch.ops.vllm.chunk_gated_delta_rule_static (idempotent)."""
    import torch

    try:
        torch.ops.vllm.chunk_gated_delta_rule_static
        return True
    except (AttributeError, RuntimeError):
        pass

    try:
        from vllm_hcu.model_executor.layers.fla.ops.chunk_fused import (
            chunk_gated_delta_rule_fwd_static,
        )
    except ImportError:
        return False

    from vllm.utils.torch_utils import direct_register_custom_op

    def _impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              g: torch.Tensor, beta: torch.Tensor, scale: float,
              initial_state: torch.Tensor, output_final_state: bool,
              cu_seqlens: Optional[torch.Tensor],
              use_qk_l2norm_in_kernel: bool,
              buf_w: torch.Tensor, buf_u: torch.Tensor, buf_o: torch.Tensor,
              buf_v_new: torch.Tensor,
              buf_final_state: torch.Tensor) -> torch.Tensor:
        g_out, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd_static(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=scale, initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            buf_w=buf_w, buf_u=buf_u,
            buf_o=buf_o, buf_v_new=buf_v_new,
            buf_final_state=buf_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        return o.to(q.dtype)

    def _fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              g: torch.Tensor, beta: torch.Tensor, scale: float,
              initial_state: torch.Tensor, output_final_state: bool,
              cu_seqlens: Optional[torch.Tensor],
              use_qk_l2norm_in_kernel: bool,
              buf_w: torch.Tensor, buf_u: torch.Tensor, buf_o: torch.Tensor,
              buf_v_new: torch.Tensor,
              buf_final_state: torch.Tensor) -> torch.Tensor:
        return buf_o[: q.shape[0], : q.shape[1]].to(q.dtype)

    direct_register_custom_op(
        op_name="chunk_gated_delta_rule_static",
        op_func=_impl,
        mutates_args=["buf_w", "buf_u", "buf_o", "buf_v_new", "buf_final_state"],
        fake_impl=_fake,
    )
    return True


def apply_to_module(module: ModuleType) -> bool:
    chunk = load_exact_module(TARGET_MODULE, module)
    if getattr(chunk, _MARKER, False):
        if not getattr(chunk.chunk_gated_delta_rule_fwd, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale; "
                "restart the process"
            )
        return False
    original = require_unpatched(
        chunk, "chunk_gated_delta_rule_fwd", TARGETS[0], _WRAPPER
    )
    require_exact_signature(
        original,
        TARGETS[0],
        positional=_FWD_PARAMETERS,
        defaults={
            "cu_seqlens": None,
            "chunk_indices": None,
            "chunk_offsets": None,
            "core_attn_out": None,
        },
    )

    try:
        from vllm_hcu.model_executor.layers.fla.ops.chunk_fused import (
            chunk_gated_delta_rule_fwd_fused,
        )
    except ImportError:
        chunk_gated_delta_rule_fwd_fused = None  # noqa: F811

    @functools.wraps(original)
    def chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state,
                                   output_final_state, cu_seqlens=None,
                                   chunk_indices=None, chunk_offsets=None,
                                   core_attn_out=None):
        import vllm_hcu.platforms.envs as henvs

        # The 0.18 fused kernel predates v0.25's chunk-stitching metadata
        # (chunk_indices/chunk_offsets) and the optional core_attn_out sink;
        # only divert the plain prefill form.
        if (
            henvs.VLLM_HCU_USE_CUSTOM_FUSED_GDN
            and chunk_gated_delta_rule_fwd_fused is not None
            and chunk_indices is None
            and chunk_offsets is None
            and core_attn_out is None
        ):
            return chunk_gated_delta_rule_fwd_fused(
                q=q, k=k, v=v, g=g, beta=beta,
                scale=scale, initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )
        return original(q, k, v, g, beta, scale, initial_state,
                        output_final_state, cu_seqlens, chunk_indices,
                        chunk_offsets, core_attn_out)

    setattr(chunk_gated_delta_rule_fwd, _WRAPPER, True)
    chunk._vllm_hcu_original_chunk_gated_delta_rule_fwd = original
    chunk.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd
    _register_static_op()
    setattr(chunk, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
