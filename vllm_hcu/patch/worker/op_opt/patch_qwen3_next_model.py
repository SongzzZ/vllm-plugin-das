# SPDX-License-Identifier: Apache-2.0
# Ported from vllm-hcu-main d095594: model-level shared GDN scratch buffers.
# When VLLM_HCU_USE_CUSTOM_FUSED_GDN is enabled, Qwen3NextModel.__init__
# allocates ONE holder shared by every linear_attn layer; the buffer itself is
# allocated lazily at the first layer's warmup (patch_qwen_gdn_static).
#
# v0.25 note: Qwen3_5Model subclasses Qwen3NextModel without overriding
# __init__, so wrapping Qwen3NextModel.__init__ covers the Qwen3.5 models too.

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_unpatched,
)

TARGET_MODULE = "vllm.model_executor.models.qwen3_next"
PATCH_ID = "worker.op_opt.models.qwen3_next_gdn_shared"
TARGETS = (f"{TARGET_MODULE}.Qwen3NextModel.__init__",)
_MARKER = "_vllm_hcu_qwen3_next_gdn_shared_applied"
_WRAPPER = "_vllm_hcu_qwen3_next_gdn_shared_wrapper"


class _GdnSharedBuffers:
    """Model-level holder for the ONE GDN scratch buffer set shared by every
    linear_attn layer. Plain object (not an nn.Module), so it is stored in
    __dict__ and never registered as a submodule / moved by .to(). The buffer
    itself is allocated lazily on the first layer's warmup."""

    __slots__ = ("buf", "n_layers")

    def __init__(self, n_layers: int):
        self.buf = None
        self.n_layers = n_layers


def apply_to_module(module: ModuleType) -> bool:
    models = load_exact_module(TARGET_MODULE, module)
    model_cls = require_class(models, "Qwen3NextModel", TARGETS[0])
    if getattr(models, _MARKER, False):
        if not getattr(model_cls.__init__, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale; "
                "restart the process"
            )
        return False
    original = require_unpatched(model_cls, "__init__", TARGETS[0], _WRAPPER)

    @functools.wraps(original)
    def hcu_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        import vllm_hcu.platforms.envs as henvs

        if not henvs.VLLM_HCU_USE_CUSTOM_FUSED_GDN:
            return
        self._gdn_shared = _GdnSharedBuffers(
            n_layers=sum(
                1
                for _l in self.layers
                if getattr(_l, "linear_attn", None) is not None
            )
        )
        for _l in self.layers:
            _g = getattr(_l, "linear_attn", None)
            if _g is not None:
                _g._gdn_shared = self._gdn_shared

    setattr(hcu_init, _WRAPPER, True)
    models._vllm_hcu_original_qwen3_next_model_init = original
    model_cls.__init__ = hcu_init
    setattr(models, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
