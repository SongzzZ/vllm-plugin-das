# SPDX-License-Identifier: Apache-2.0
"""Port inventory for the Qwen3.5 adapter stack from vllm-hcu-main.

References every ported adapter module (the structural coverage gate in
tools/check_patch_test_coverage.py requires each patch_*.py stem to appear
in a test) and asserts the shared adapter contract plus worker registration.
"""

from __future__ import annotations

import importlib

PORTED_ADAPTERS = (
    # vllm-hcu-main c550526: compressed-tensors W8A16 scheme + selector.
    "patch_compressed_tensors_w8a16_hcu",
    # vllm-hcu-main d095594: fused FLA chunk kernels for the GDN path.
    "patch_fla_chunk_fused",
    # Fused topk router for the MoE decode path.
    "patch_fused_topk_router",
    # vllm-hcu-main 3d26cca + 02ec8ce: fused shared-expert sigmoid-mul gate.
    "patch_qwen2_moe",
    # vllm-hcu-main d095594: model-level shared GDN scratch buffers.
    "patch_qwen3_next_model",
    # vllm-hcu-main dffd974: aiter HIP C++ packed-decode dispatch.
    "patch_qwen3_next_packed_decode",
    # vllm-hcu-main d095594: lazy per-layer GDN static buffers.
    "patch_qwen_gdn_static",
    # GDN operator bindings (rmsnorm-gated / causal-conv1d / rearrange).
    "patch_qwen_gdn_ops",
)


def test_ported_adapters_expose_contract_and_are_registered():
    import vllm_hcu.patch.worker as worker

    registered = [spec.adapter for spec in worker._OP_CALLBACKS]
    for stem in PORTED_ADAPTERS:
        module = importlib.import_module(
            f"vllm_hcu.patch.worker.op_opt.{stem}"
        )
        for symbol in ("PATCH_ID", "TARGET_MODULE", "TARGETS", "apply_to_module"):
            assert hasattr(module, symbol), f"{stem} missing {symbol}"
        assert any(name.endswith(stem) for name in registered), (
            f"{stem} is not armed by the worker dispatcher"
        )


def test_retired_adapters_are_not_registered():
    # Upstream d2653b4 reverted the RejectionSampler PyTorch fallback, so the
    # 0.18 rejection-sampler adapter must not be ported or armed here.
    import vllm_hcu.patch.worker as worker

    registered = [spec.adapter for spec in worker._OP_CALLBACKS]
    assert not any(name.endswith("patch_rejection_sampler") for name in registered)
