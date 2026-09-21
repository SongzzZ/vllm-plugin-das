# SPDX-License-Identifier: Apache-2.0
# Ported from vllm-hcu-main d095594: use lightop.op.topk_softmax for the
# plain (non-grouped, non-bias) softmax routing path.
#
# Verified on sg-t1 (DCU, qwen3.5-35B-A3B w8a16):
#   M=32, experts=256, topk=8, fp16:
#     _moe_C.topk_softmax        12.15 us
#     rocm_aiter_topk_softmax    15.25 us
#     lightop.op.topk_softmax     6.91 us   (ids/weights/indices identical)
#
# Gated on VLLM_HCU_USE_CUSTOM_OPS + VLLM_HCU_USE_FUSE_MOE_GATE and a shape
# guard. The swapped module symbol only affects the plain vllm_topk_softmax
# dispatch; grouped/bias routers keep their vLLM-selected backends.

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
    require_unpatched,
)

TARGET_MODULE = (
    "vllm.model_executor.layers.fused_moe.router.fused_topk_router"
)
PATCH_ID = "worker.op_opt.moe.router.lightop_topk_softmax"
TARGETS = (
    f"{TARGET_MODULE}.vllm_topk_softmax",
    f"{TARGET_MODULE}.dispatch_topk_softmax_func",
)
_MARKER = "_vllm_hcu_lightop_topk_softmax_applied"
_WRAPPER = "_vllm_hcu_lightop_topk_softmax_wrapper"
_DISPATCH_WRAPPER = "_vllm_hcu_lightop_topk_dispatch_wrapper"

_PARAMETERS = (
    "topk_weights",
    "topk_indices",
    "token_expert_indices",
    "gating_output",
    "renormalize",
)


def _try_hcu_topk_softmax(topk_weights, topk_indices, token_expert_indices,
                          gating_output, renormalize) -> bool:
    # Run lightop.op.topk_softmax when it applies; return True if handled.
    try:
        import lightop.op as hcu_op  # noqa: F401

        import vllm_hcu.platforms.envs as hcu_envs
    except Exception:
        return False

    if not (
        hcu_envs.VLLM_HCU_USE_CUSTOM_OPS
        and hcu_envs.VLLM_HCU_USE_FUSE_MOE_GATE
    ):
        return False

    top_k = topk_weights.shape[-1]
    num_experts = gating_output.shape[-1]
    if top_k != 8 or num_experts not in (128, 160, 256, 384):
        return False

    hcu_op.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )
    return True


def apply_to_module(module: ModuleType) -> bool:
    router = load_exact_module(TARGET_MODULE, module)
    if getattr(router, _MARKER, False):
        if not getattr(router.vllm_topk_softmax, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale; "
                "restart the process"
            )
        return False
    original = require_unpatched(
        router, "vllm_topk_softmax", TARGETS[0], _WRAPPER
    )
    require_exact_signature(
        original,
        TARGETS[0],
        positional=_PARAMETERS,
        defaults={"renormalize": False},
    )

    @functools.wraps(original)
    def vllm_topk_softmax(topk_weights, topk_indices, token_expert_indices,
                          gating_output, renormalize=False):
        if _try_hcu_topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
        ):
            return topk_weights, topk_indices
        return original(topk_weights, topk_indices, token_expert_indices,
                        gating_output, renormalize)

    setattr(vllm_topk_softmax, _WRAPPER, True)
    router._vllm_hcu_original_vllm_topk_softmax = original
    router.vllm_topk_softmax = vllm_topk_softmax

    # When the caller selected the AITER backend, dispatch_topk_softmax_func
    # returns rocm_aiter_ops.topk_softmax directly and the plain
    # vllm_topk_softmax wrapper above would never fire. Prefer lightop first
    # and fall back to the caller-selected backend (never the generic
    # ops.topk_softmax) when the shape guard fails.
    dispatch = getattr(router, "dispatch_topk_softmax_func", None)
    if callable(dispatch) and not getattr(dispatch, _DISPATCH_WRAPPER, False):
        require_exact_signature(
            dispatch,
            TARGETS[1],
            positional=("use_rocm_aiter",),
            defaults={"use_rocm_aiter": False},
        )

        @functools.wraps(dispatch)
        def dispatch_topk_softmax_func(use_rocm_aiter=False):
            module_ops = router.ops
            module_aiter_ops = router.rocm_aiter_ops

            def _dispatch(topk_weights, topk_indices, token_expert_indices,
                          gating_output, renormalize=False):
                if _try_hcu_topk_softmax(
                    topk_weights,
                    topk_indices,
                    token_expert_indices,
                    gating_output,
                    renormalize,
                ):
                    return topk_weights, topk_indices
                if use_rocm_aiter:
                    module_aiter_ops.topk_softmax(
                        topk_weights,
                        topk_indices,
                        token_expert_indices,
                        gating_output,
                        renormalize,
                    )
                else:
                    module_ops.topk_softmax(
                        topk_weights,
                        topk_indices,
                        token_expert_indices,
                        gating_output,
                        renormalize,
                    )
                return topk_weights, topk_indices

            return _dispatch

        setattr(dispatch_topk_softmax_func, _DISPATCH_WRAPPER, True)
        router._vllm_hcu_original_dispatch_topk_softmax_func = dispatch
        router.dispatch_topk_softmax_func = dispatch_topk_softmax_func

    setattr(router, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
