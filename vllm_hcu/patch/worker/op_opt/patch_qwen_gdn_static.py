# SPDX-License-Identifier: Apache-2.0
# Ported from vllm-hcu-main d095594: CUDA-graph static path for the GDN
# prefill chunk kernel.
#
# * _warmup_prefill_kernels is extended: after the original autotune warmup,
#   the shared ChunkGDRBuffers are allocated (once, on the model-level holder
#   installed by patch_qwen3_next_model) and attached to this layer's
#   ``chunk_gated_delta_rule`` op instance.
# * ChunkGatedDeltaRule.forward_cuda is wrapped: when the static buffers are
#   present, the environment switch is on and the call is the plain prefill
#   form (single batch, within buffer capacities, no chunk stitching
#   metadata), the call is diverted to
#   torch.ops.vllm.chunk_gated_delta_rule_static which writes into the fixed
#   pre-allocated buffers (CUDA-graph friendly). Everything else falls back
#   to the original backend dispatch.
#
# Must be registered after patch_qwen3_next_model (buffer holder) and
# patch_fla_chunk_fused (registers the chunk_gated_delta_rule_static op).

from __future__ import annotations

import functools
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_unpatched,
)

TARGET_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
PATCH_ID = "worker.op_opt.mamba.gdn.static_buffers"
TARGETS = (
    f"{TARGET_MODULE}.ChunkGatedDeltaRule.forward_cuda",
    f"{TARGET_MODULE}.QwenGatedDeltaNetAttention._warmup_prefill_kernels",
)
_MARKER = "_vllm_hcu_qwen_gdn_static_applied"
_FWD_WRAPPER = "_vllm_hcu_gdn_static_fwd_wrapper"
_WARMUP_WRAPPER = "_vllm_hcu_gdn_static_warmup_wrapper"


def _allocate_static_buffers(self):
    """Allocate the shared ChunkGDRBuffers once; attach (buf, scale) to the
    layer's chunk op instance. Returns the attached tuple or None."""
    import vllm_hcu.platforms.envs as henvs

    if not henvs.VLLM_HCU_USE_CUSTOM_FUSED_GDN:
        return None
    op = getattr(self, "chunk_gated_delta_rule", None)
    if op is None:
        return None
    existing = getattr(op, "_hcu_static", None)
    if existing is not None:
        return existing

    shared = getattr(self, "_gdn_shared", None)
    if shared is None:
        shared = type("_GdnShared", (), {"buf": None, "n_layers": 1})()
        self._gdn_shared = shared

    try:
        from vllm.logger import init_logger

        logger = init_logger(__name__)
        from vllm_hcu.model_executor.layers.fla.ops.chunk_fused import (
            ChunkGDRBuffers,
        )

        num_v_heads = self.num_v_heads // self.tp_size
        max_tokens = henvs.VLLM_HCU_GDN_MAX_TOKENS
        max_seqs = henvs.VLLM_HCU_GDN_MAX_SEQS
        max_batch = henvs.VLLM_HCU_GDN_MAX_BATCH
        kwargs = {
            "max_tokens": max_tokens,
            "max_seqs": max_seqs,
            "num_heads": num_v_heads,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
        }
        try:
            buf = ChunkGDRBuffers(max_batch=max_batch, **kwargs)
        except TypeError:
            buf = ChunkGDRBuffers(**kwargs)
            max_batch = getattr(buf, "max_batch", 1)
        shared.buf = buf
        entry = (buf, self.head_k_dim ** -0.5, max_batch)
        op._hcu_static = entry
        mem_mb = sum(
            t.numel() * t.element_size()
            for t in (buf.w, buf.u, buf.o, buf.v_new, buf.final_state)
        ) / 1e6
        logger.info(
            "GDN static buffers allocated (shared across %d linear_attn "
            "layers): %.1f MB (max_batch=%s max_tokens=%d max_seqs=%d "
            "num_v_heads=%d)",
            shared.n_layers, mem_mb, max_batch, max_tokens, max_seqs,
            num_v_heads,
        )
        return entry
    except Exception:
        from vllm.logger import init_logger

        init_logger(__name__).warning(
            "GDN static buffer allocation failed for layer %s; "
            "falling back to the non-static chunk kernel",
            getattr(self, "prefix", "?"),
            exc_info=True,
        )
        return None


def apply_to_module(module: ModuleType) -> bool:
    qwen = load_exact_module(TARGET_MODULE, module)
    if getattr(qwen, _MARKER, False):
        return False

    chunk_cls = getattr(qwen, "ChunkGatedDeltaRule", None)
    if chunk_cls is None or not hasattr(chunk_cls, "forward_cuda"):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} is missing"
        )
    original_fwd = require_unpatched(
        chunk_cls, "forward_cuda", TARGETS[0], _FWD_WRAPPER
    )

    @functools.wraps(original_fwd)
    def hcu_forward_cuda(self, *args, **kwargs):
        entry = getattr(self, "_hcu_static", None)
        if (
            entry is not None
            and not kwargs.get("chunk_indices")
            and not kwargs.get("chunk_offsets")
            and kwargs.get("output_final_state", True)
        ):
            buf, scale, max_batch = entry
            q = args[0] if len(args) > 0 else kwargs.get("q")
            initial_state = (
                args[6] if len(args) > 6 else kwargs.get("initial_state")
            )
            if (
                q is not None
                and initial_state is not None
                and q.shape[0] == 1
                and max_batch == 1
                and q.shape[1] <= buf.w.shape[1]
                and initial_state.shape[0] <= buf.final_state.shape[0]
            ):
                T = q.shape[1]
                n_seqs = initial_state.shape[0]
                try:
                    o = torch.ops.vllm.chunk_gated_delta_rule_static(
                        q=q, k=args[1], v=args[2], g=args[3], beta=args[4],
                        scale=scale,
                        initial_state=initial_state,
                        output_final_state=True,
                        cu_seqlens=args[5] if len(args) > 5 else kwargs.get("cu_seqlens"),
                        use_qk_l2norm_in_kernel=kwargs.get(
                            "use_qk_l2norm_in_kernel", True
                        ),
                        buf_w=buf.w[:, :T].contiguous(),
                        buf_u=buf.u[:, :T].contiguous(),
                        buf_o=buf.o[:, :T].contiguous(),
                        buf_v_new=buf.v_new[:, :T].contiguous(),
                        buf_final_state=buf.final_state[
                            :n_seqs
                        ].contiguous(),
                    )
                    return o, buf.final_state[:n_seqs]
                except (RuntimeError, AttributeError):
                    pass
        return original_fwd(self, *args, **kwargs)

    setattr(hcu_forward_cuda, _FWD_WRAPPER, True)
    chunk_cls._vllm_hcu_original_forward_cuda = original_fwd
    chunk_cls.forward_cuda = hcu_forward_cuda

    # --- warmup extension -------------------------------------------------
    attn_cls = getattr(qwen, "QwenGatedDeltaNetAttention", None)
    if attn_cls is None or not callable(
        getattr(attn_cls, "_warmup_prefill_kernels", None)
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[1]} is missing"
        )
    original_warmup = require_unpatched(
        attn_cls, "_warmup_prefill_kernels", TARGETS[1], _WARMUP_WRAPPER
    )

    @functools.wraps(original_warmup)
    def hcu_warmup(self, *args, **kwargs):
        original_warmup(self, *args, **kwargs)
        _allocate_static_buffers(self)

    setattr(hcu_warmup, _WARMUP_WRAPPER, True)
    attn_cls._vllm_hcu_original_warmup = original_warmup
    attn_cls._warmup_prefill_kernels = hcu_warmup

    setattr(qwen, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
