# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the GDN packed-decode aiter HIP dispatch (port of
vllm-hcu-main dffd974 + 02ec8ce).

CPU tests: the env flag is registered and the worker arms the packed-decode
adapter. Device tests: HIP-vs-Triton numeric parity, in-place state update,
and -1 padding index behavior; skipped without an HCU device or aiter.
"""

import importlib.util
import inspect
import os

import pytest
import torch


def _pkg_dir(name):
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin is not None, name
    return os.path.dirname(os.path.abspath(spec.origin))


def test_env_flag_registered():
    envs_spec = importlib.util.find_spec("vllm_hcu.platforms.envs")
    mod = importlib.util.module_from_spec(envs_spec)
    envs_spec.loader.exec_module(mod)
    flag = "VLLM_HCU_USE_AITER_PACKED_DECODE"
    assert flag in mod.hcu_vllm_environment_variables
    old = os.environ.pop(flag, None)
    try:
        assert mod.hcu_vllm_environment_variables[flag]() is True
        os.environ[flag] = "0"
        assert mod.hcu_vllm_environment_variables[flag]() is False
    finally:
        if old is None:
            os.environ.pop(flag, None)
        else:
            os.environ[flag] = old


def test_packed_decode_adapter_wiring():
    # The worker must arm the packed-decode adapter, and the retired
    # rejection-sampler adapter (upstream d2653b4 reverted it) must be gone.
    import vllm_hcu.patch.worker as worker

    specs = [spec.adapter for spec in worker._OP_CALLBACKS]
    assert any(s.endswith("patch_qwen3_next_packed_decode") for s in specs)
    assert not any(s.endswith("patch_rejection_sampler") for s in specs)

    from vllm_hcu.patch.worker.op_opt import (
        patch_qwen3_next_packed_decode as adapter,
    )

    src = inspect.getsource(adapter)
    assert "VLLM_HCU_USE_AITER_PACKED_DECODE" in src
    assert "VLLM_HCU_USE_CUSTOM_OPS" in src
    assert "hip_kernel is not None" in src


def _hip_and_triton_fns():
    aiter = pytest.importorskip("aiter")
    if not torch.cuda.is_available():
        pytest.skip("HCU device is unavailable")
    if importlib.util.find_spec("aiter.ops.triton.fla.fused_recurrent") is None:
        pytest.skip("aiter triton fused_recurrent unavailable")
    from aiter.ops.triton.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode as triton_fn,
    )

    hip_fn = getattr(
        aiter, "aiter_fused_recurrent_gated_delta_rule_packed_decode", None
    )
    if hip_fn is None:
        pytest.skip("aiter HIP packed decode unavailable")
    return triton_fn, hip_fn


@pytest.mark.parametrize("batch", [1, 8, 64])
def test_hip_packed_decode_matches_triton(batch):
    triton_fn, hip_fn = _hip_and_triton_fns()
    device = "cuda"
    torch.manual_seed(0)
    num_k_heads, num_v_heads, head_dim = 16, 32, 128
    state_pool = max(batch, 64)
    qkv_dim = 2 * num_k_heads * head_dim + num_v_heads * head_dim
    scale = head_dim**-0.5

    mixed_qkv = torch.randn(batch, qkv_dim, dtype=torch.float16, device=device) * 0.5
    a = torch.randn(batch, num_v_heads, dtype=torch.float32, device=device)
    b = torch.randn(batch, num_v_heads, dtype=torch.float32, device=device)
    a_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1 - 3.0
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
    indices = torch.randperm(state_pool, device=device)[:batch].to(torch.int32)
    state0 = torch.randn(
        state_pool, num_v_heads, head_dim, head_dim,
        dtype=torch.float32, device=device,
    ) * 0.1

    out_triton = torch.zeros(
        batch, 1, num_v_heads, head_dim, dtype=torch.float16, device=device
    )
    out_hip = torch.zeros_like(out_triton)
    state_triton = state0.clone()
    state_hip = state0.clone()

    kwargs = dict(
        A_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )
    triton_fn(
        mixed_qkv=mixed_qkv, a=a, b=b, initial_state=state_triton,
        out=out_triton, ssm_state_indices=indices, **kwargs
    )
    hip_fn(
        mixed_qkv=mixed_qkv, a=a, b=b, initial_state=state_hip,
        out=out_hip, ssm_state_indices=indices, **kwargs
    )
    torch.cuda.synchronize()

    assert torch.allclose(out_hip.float(), out_triton.float(), atol=1e-3, rtol=1e-3)
    sel = indices.long()
    assert torch.allclose(state_hip[sel], state_triton[sel], atol=1e-4, rtol=1e-4)
    # Both kernels must update the initial state in place.
    assert (state_hip[sel] - state0[sel]).abs().max().item() > 0


def test_hip_packed_decode_padding_index_zeroes_output():
    triton_fn, hip_fn = _hip_and_triton_fns()
    device = "cuda"
    torch.manual_seed(0)
    num_k_heads, num_v_heads, head_dim = 16, 32, 128
    state_pool = 64
    qkv_dim = 2 * num_k_heads * head_dim + num_v_heads * head_dim
    batch = 8
    mixed_qkv = torch.randn(batch, qkv_dim, dtype=torch.float16, device=device)
    a = torch.randn(batch, num_v_heads, dtype=torch.float32, device=device)
    b = torch.randn(batch, num_v_heads, dtype=torch.float32, device=device)
    a_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    indices = torch.randperm(state_pool, device=device)[:batch].to(torch.int32)
    indices[2] = -1
    indices[5] = -1
    state = torch.randn(
        state_pool, num_v_heads, head_dim, head_dim,
        dtype=torch.float32, device=device,
    )
    out_triton = torch.zeros(
        batch, 1, num_v_heads, head_dim, dtype=torch.float16, device=device
    )
    out_hip = torch.zeros_like(out_triton)
    state_triton = state.clone()
    state_hip = state.clone()
    kwargs = dict(
        A_log=a_log,
        dt_bias=dt_bias,
        scale=head_dim**-0.5,
        use_qk_l2norm_in_kernel=True,
    )
    triton_fn(
        mixed_qkv=mixed_qkv, a=a, b=b, initial_state=state_triton,
        out=out_triton, ssm_state_indices=indices, **kwargs
    )
    hip_fn(
        mixed_qkv=mixed_qkv, a=a, b=b, initial_state=state_hip,
        out=out_hip, ssm_state_indices=indices, **kwargs
    )
    torch.cuda.synchronize()
    assert out_hip[2].abs().max().item() == 0.0
    assert out_hip[5].abs().max().item() == 0.0
    assert out_triton[2].abs().max().item() == 0.0
