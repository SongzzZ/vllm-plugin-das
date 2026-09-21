# SPDX-License-Identifier: Apache-2.0
# Ported from vllm-hcu-main dffd974: dispatch the GDN non-spec packed decode
# to aiter's HIP C++ kernel
# (aiter.aiter_fused_recurrent_gated_delta_rule_packed_decode).
#
# The v0.18 source patch gated the call site inside
# Qwen3NextForCausalLayer._forward_core_decode_non_spec.  The runtime
# equivalent swaps the module-level
# ``fused_recurrent_gated_delta_rule_packed_decode`` symbol (installed by the
# stock aiter-Triton import), so the call site picks the HIP kernel through
# ordinary global lookup.  The two kernels share one signature, and the call
# site passes keyword arguments only.

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_unpatched,
)

TARGET_MODULE = "vllm.model_executor.models.qwen3_next"
PATCH_ID = "worker.op_opt.models.qwen3_next_packed_decode"
TARGETS = (f"{TARGET_MODULE}.fused_recurrent_gated_delta_rule_packed_decode",)
_MARKER = "_vllm_hcu_qwen3_next_packed_decode_applied"
_WRAPPER = "_vllm_hcu_qwen3_next_packed_decode_wrapper"

_hip_fn = None
_hip_resolved = False


def _resolve_hip_kernel():
    """Resolve the aiter HIP C++ kernel once; None when unavailable."""
    global _hip_fn, _hip_resolved
    if not _hip_resolved:
        try:
            import aiter

            _hip_fn = getattr(
                aiter, "aiter_fused_recurrent_gated_delta_rule_packed_decode", None
            )
        except ImportError:
            _hip_fn = None
        _hip_resolved = True
    return _hip_fn


def apply_to_module(module: ModuleType) -> bool:
    models = load_exact_module(TARGET_MODULE, module)
    if getattr(models, _MARKER, False):
        if not getattr(
            models.fused_recurrent_gated_delta_rule_packed_decode, _WRAPPER, False
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale; "
                "restart the process"
            )
        return False
    original = require_unpatched(
        models,
        "fused_recurrent_gated_delta_rule_packed_decode",
        TARGETS[0],
        _WRAPPER,
    )
    require_callable(
        models,
        "fused_recurrent_gated_delta_rule_packed_decode",
        TARGETS[0],
    )

    @functools.wraps(original)
    def hcu_packed_decode(*args, **kwargs):
        import vllm_hcu.platforms.envs as henvs

        hip_kernel = _resolve_hip_kernel()
        if (
            henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_USE_AITER_PACKED_DECODE
            and hip_kernel is not None
        ):
            return hip_kernel(*args, **kwargs)
        return original(*args, **kwargs)

    setattr(hcu_packed_decode, _WRAPPER, True)
    models._vllm_hcu_original_packed_decode = original
    models.fused_recurrent_gated_delta_rule_packed_decode = hcu_packed_decode
    setattr(models, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
