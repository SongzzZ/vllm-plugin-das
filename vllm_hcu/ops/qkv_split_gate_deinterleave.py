# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _qkv_split_gate_deinterleave_kernel(
    qkv_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    gate_ptr,
    stride_qkv_row,
    stride_q_row,
    stride_k_row,
    stride_v_row,
    stride_gate_row,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    hb = tl.program_id(1)

    h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < head_dim

    # q / gate: each head occupies 2*head_dim in qkv (first half q, second
    # half gate).
    mask_q = (h < num_heads)[:, None] & d_mask[None, :]
    qgate_base = row * stride_qkv_row + h[:, None] * (2 * head_dim) + d[None, :]
    q_vals = tl.load(qkv_ptr + qgate_base, mask=mask_q, other=0.0)
    gate_vals = tl.load(qkv_ptr + qgate_base + head_dim, mask=mask_q, other=0.0)

    q_off = row * stride_q_row + h[:, None] * head_dim + d[None, :]
    gate_off = row * stride_gate_row + h[:, None] * head_dim + d[None, :]
    tl.store(q_ptr + q_off, q_vals, mask=mask_q)
    tl.store(gate_ptr + gate_off, gate_vals, mask=mask_q)

    # k / v: packed right after the q_gate block.
    q_gate_width = 2 * num_heads * head_dim
    mask_kv = (h < num_kv_heads)[:, None] & d_mask[None, :]
    k_src = row * stride_qkv_row + q_gate_width + h[:, None] * head_dim + d[None, :]
    v_src = (
        row * stride_qkv_row
        + q_gate_width
        + num_kv_heads * head_dim
        + h[:, None] * head_dim
        + d[None, :]
    )
    k_vals = tl.load(qkv_ptr + k_src, mask=mask_kv, other=0.0)
    v_vals = tl.load(qkv_ptr + v_src, mask=mask_kv, other=0.0)

    k_off = row * stride_k_row + h[:, None] * head_dim + d[None, :]
    v_off = row * stride_v_row + h[:, None] * head_dim + d[None, :]
    tl.store(k_ptr + k_off, k_vals, mask=mask_kv)
    tl.store(v_ptr + v_off, v_vals, mask=mask_kv)


def qkv_split_gate_deinterleave(
    qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split qkv ([q|gate] per head + k + v) into q, k, v, gate in one kernel.

    qkv layout: [num_heads * (q | gate) | num_kv_heads * k | num_kv_heads * v]
    where q and gate are interleaved per head (each head occupies 2*head_dim).
    """
    *lead, width = qkv.shape
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    expected = 2 * q_size + 2 * kv_size
    if width != expected:
        raise ValueError(
            f"qkv_split_gate_deinterleave: qkv last dim {width} != expected "
            f"2*q_size + 2*kv_size = {expected} (num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim})"
        )

    qkv2d = qkv.reshape(-1, width)
    if not qkv2d.is_contiguous():
        qkv2d = qkv2d.contiguous()
    N = qkv2d.shape[0]
    device = qkv.device
    dtype = qkv.dtype

    q = torch.empty((N, q_size), device=device, dtype=dtype)
    k = torch.empty((N, kv_size), device=device, dtype=dtype)
    v = torch.empty((N, kv_size), device=device, dtype=dtype)
    gate = torch.empty((N, q_size), device=device, dtype=dtype)

    BLOCK_H = min(triton.next_power_of_2(max(num_heads, num_kv_heads)), 32)
    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = (N, triton.cdiv(max(num_heads, num_kv_heads), BLOCK_H))
    _qkv_split_gate_deinterleave_kernel[grid](
        qkv2d,
        q,
        k,
        v,
        gate,
        qkv2d.stride(0),
        q.stride(0),
        k.stride(0),
        v.stride(0),
        gate.stride(0),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
    )

    q = q.reshape(*lead, q_size)
    k = k.reshape(*lead, kv_size)
    v = v.reshape(*lead, kv_size)
    gate = gate.reshape(*lead, q_size)
    return q, k, v, gate


def _qkv_split_gate_deinterleave_fake(
    qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    *lead, _ = qkv.shape
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    return (
        qkv.new_empty((*lead, q_size)),
        qkv.new_empty((*lead, kv_size)),
        qkv.new_empty((*lead, kv_size)),
        qkv.new_empty((*lead, q_size)),
    )


direct_register_custom_op(
    op_name="qkv_split_gate_deinterleave",
    op_func=qkv_split_gate_deinterleave,
    mutates_args=[],
    fake_impl=_qkv_split_gate_deinterleave_fake,
)
