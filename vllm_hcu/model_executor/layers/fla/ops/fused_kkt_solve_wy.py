# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Fused kernel 1: chunk_scaled_dot_kkt + solve_tril + recompute_w_u
#
# Strategy: Compute A as 10 independent [16,16] sub-blocks (4 diagonal + 6 off-diagonal),
# perform hierarchical forward substitution entirely in registers, then apply A_inv to
# produce w and u. Eliminates the intermediate A tensor from global memory.
#
# Memory savings: 4 passes over [B, T, H, BT] float32 -> 0 passes (A never hits DRAM)
#
# Assumes T is a multiple of BT (64). Padding is handled by the caller (chunk_fused.py).
# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from vllm.model_executor.layers.fla.ops.index import prepare_chunk_indices
from vllm.model_executor.layers.fla.ops.op import exp


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [64]
        for BV in [64]
        for num_warps in [4, 8]
        for num_stages in [1]
    ],
    key=["H", "K", "V", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def fused_kkt_solve_wy_kernel(
    k,
    v,
    beta,
    g,
    w_out,
    u_out,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kkt + solve_tril + recompute_w_u.

    Per chunk, computes:
      1. A[i,j] = strict_lower_tril(beta_i * k_i @ k_j^T * exp(g_i - g_j))  [in 16x16 blocks]
      2. A_inv = (I - A)^{-1} via hierarchical 16x16 forward substitution
      3. w = A_inv @ diag(beta * exp(g)) @ k
         u = A_inv @ diag(beta) @ v

    The [BT,BT] A matrix never touches global memory.
    Caller must ensure T is a multiple of BT (padding in chunk_fused.py).
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_16 = tl.arange(0, 16)

    # ======== Load beta and g for this chunk ========
    chunk_start = i_t * BT
    k_base = k + (bos * Hg + i_h // (H // Hg)) * K
    beta_base = beta + bos * H + i_h

    off_0 = chunk_start
    off_1 = chunk_start + 16
    off_2 = chunk_start + 32
    off_3 = chunk_start + 48

    p_beta0 = tl.make_block_ptr(beta_base, (T,), (H,), (off_0,), (16,), (0,))
    p_beta1 = tl.make_block_ptr(beta_base, (T,), (H,), (off_1,), (16,), (0,))
    p_beta2 = tl.make_block_ptr(beta_base, (T,), (H,), (off_2,), (16,), (0,))
    p_beta3 = tl.make_block_ptr(beta_base, (T,), (H,), (off_3,), (16,), (0,))
    b_beta0 = tl.load(p_beta0, boundary_check=(0,))
    b_beta1 = tl.load(p_beta1, boundary_check=(0,))
    b_beta2 = tl.load(p_beta2, boundary_check=(0,))
    b_beta3 = tl.load(p_beta3, boundary_check=(0,))

    if USE_G:
        g_base = g + bos * H + i_h
        p_g0 = tl.make_block_ptr(g_base, (T,), (H,), (off_0,), (16,), (0,))
        p_g1 = tl.make_block_ptr(g_base, (T,), (H,), (off_1,), (16,), (0,))
        p_g2 = tl.make_block_ptr(g_base, (T,), (H,), (off_2,), (16,), (0,))
        p_g3 = tl.make_block_ptr(g_base, (T,), (H,), (off_3,), (16,), (0,))
        b_g0 = tl.load(p_g0, boundary_check=(0,)).to(tl.float32)
        b_g1 = tl.load(p_g1, boundary_check=(0,)).to(tl.float32)
        b_g2 = tl.load(p_g2, boundary_check=(0,)).to(tl.float32)
        b_g3 = tl.load(p_g3, boundary_check=(0,)).to(tl.float32)
        # Per-chunk local cumsum (fused, eliminates chunk_local_cumsum kernel)
        b_g0 = tl.cumsum(b_g0, axis=0)
        b_g1 = tl.cumsum(b_g1, axis=0) + tl.sum(tl.where(o_16 == 15, b_g0, 0.0))
        b_g2 = tl.cumsum(b_g2, axis=0) + tl.sum(tl.where(o_16 == 15, b_g1, 0.0))
        b_g3 = tl.cumsum(b_g3, axis=0) + tl.sum(tl.where(o_16 == 15, b_g2, 0.0))


    # ======== Step 1: Compute 10 [16,16] sub-blocks of A ========
    b_A00 = tl.zeros([16, 16], dtype=tl.float32)
    b_A10 = tl.zeros([16, 16], dtype=tl.float32)
    b_A11 = tl.zeros([16, 16], dtype=tl.float32)
    b_A20 = tl.zeros([16, 16], dtype=tl.float32)
    b_A21 = tl.zeros([16, 16], dtype=tl.float32)
    b_A22 = tl.zeros([16, 16], dtype=tl.float32)
    b_A30 = tl.zeros([16, 16], dtype=tl.float32)
    b_A31 = tl.zeros([16, 16], dtype=tl.float32)
    b_A32 = tl.zeros([16, 16], dtype=tl.float32)
    b_A33 = tl.zeros([16, 16], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k0 = tl.make_block_ptr(k_base, (T, K), (Hg * K, 1), (off_0, i_k * BK), (16, BK), (1, 0))
        p_k1 = tl.make_block_ptr(k_base, (T, K), (Hg * K, 1), (off_1, i_k * BK), (16, BK), (1, 0))
        p_k2 = tl.make_block_ptr(k_base, (T, K), (Hg * K, 1), (off_2, i_k * BK), (16, BK), (1, 0))
        p_k3 = tl.make_block_ptr(k_base, (T, K), (Hg * K, 1), (off_3, i_k * BK), (16, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_k3 = tl.load(p_k3, boundary_check=(0, 1))

        b_kb0 = (b_k0 * b_beta0[:, None]).to(b_k0.dtype)
        b_kb1 = (b_k1 * b_beta1[:, None]).to(b_k1.dtype)
        b_kb2 = (b_k2 * b_beta2[:, None]).to(b_k2.dtype)
        b_kb3 = (b_k3 * b_beta3[:, None]).to(b_k3.dtype)

        b_A00 += tl.dot(b_kb0, tl.trans(b_k0))
        b_A10 += tl.dot(b_kb1, tl.trans(b_k0))
        b_A11 += tl.dot(b_kb1, tl.trans(b_k1))
        b_A20 += tl.dot(b_kb2, tl.trans(b_k0))
        b_A21 += tl.dot(b_kb2, tl.trans(b_k1))
        b_A22 += tl.dot(b_kb2, tl.trans(b_k2))
        b_A30 += tl.dot(b_kb3, tl.trans(b_k0))
        b_A31 += tl.dot(b_kb3, tl.trans(b_k1))
        b_A32 += tl.dot(b_kb3, tl.trans(b_k2))
        b_A33 += tl.dot(b_kb3, tl.trans(b_k3))

    if USE_G:
        b_A00 *= exp(b_g0[:, None] - b_g0[None, :])
        b_A10 *= exp(b_g1[:, None] - b_g0[None, :])
        b_A11 *= exp(b_g1[:, None] - b_g1[None, :])
        b_A20 *= exp(b_g2[:, None] - b_g0[None, :])
        b_A21 *= exp(b_g2[:, None] - b_g1[None, :])
        b_A22 *= exp(b_g2[:, None] - b_g2[None, :])
        b_A30 *= exp(b_g3[:, None] - b_g0[None, :])
        b_A31 *= exp(b_g3[:, None] - b_g1[None, :])
        b_A32 *= exp(b_g3[:, None] - b_g2[None, :])
        b_A33 *= exp(b_g3[:, None] - b_g3[None, :])

    m_strict_lower = o_16[:, None] > o_16[None, :]
    b_A00 = tl.where(m_strict_lower, b_A00, 0.0)
    b_A11 = tl.where(m_strict_lower, b_A11, 0.0)
    b_A22 = tl.where(m_strict_lower, b_A22, 0.0)
    b_A33 = tl.where(m_strict_lower, b_A33, 0.0)

    # ======== Step 2: Hierarchical forward substitution ========
    m_I16 = o_16[:, None] == o_16[None, :]

    b_Ai00 = -b_A00
    for i in range(2, 16):
        b_a = tl.sum(tl.where(o_16[:, None] == i, -b_A00, 0.0), axis=0)
        b_a = b_a + tl.sum(b_a[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where(o_16[:, None] == i, b_a, b_Ai00)
    b_Ai00 = b_Ai00 + m_I16.to(tl.float32)

    b_Ai11 = -b_A11
    for i in range(2, 16):
        b_a = tl.sum(tl.where(o_16[:, None] == i, -b_A11, 0.0), axis=0)
        b_a = b_a + tl.sum(b_a[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where(o_16[:, None] == i, b_a, b_Ai11)
    b_Ai11 = b_Ai11 + m_I16.to(tl.float32)

    b_Ai22 = -b_A22
    for i in range(2, 16):
        b_a = tl.sum(tl.where(o_16[:, None] == i, -b_A22, 0.0), axis=0)
        b_a = b_a + tl.sum(b_a[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where(o_16[:, None] == i, b_a, b_Ai22)
    b_Ai22 = b_Ai22 + m_I16.to(tl.float32)

    b_Ai33 = -b_A33
    for i in range(2, 16):
        b_a = tl.sum(tl.where(o_16[:, None] == i, -b_A33, 0.0), axis=0)
        b_a = b_a + tl.sum(b_a[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where(o_16[:, None] == i, b_a, b_Ai33)
    b_Ai33 = b_Ai33 + m_I16.to(tl.float32)

    b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_A10), b_Ai00)
    b_Ai21 = -tl.dot(tl.dot(b_Ai22, b_A21), b_Ai11)
    b_Ai32 = -tl.dot(tl.dot(b_Ai33, b_A32), b_Ai22)

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00) + tl.dot(b_A21, b_Ai10),
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11) + tl.dot(b_A32, b_Ai21),
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00) + tl.dot(b_A31, b_Ai10) + tl.dot(b_A32, b_Ai20),
    )

    # ======== Step 3: Apply A_inv to produce u and w ========
    v_base = v + (bos * H + i_h) * V
    u_base = u_out + (bos * H + i_h) * V

    for i_v in range(tl.cdiv(V, BV)):
        p_v0 = tl.make_block_ptr(v_base, (T, V), (H * V, 1), (off_0, i_v * BV), (16, BV), (1, 0))
        p_v1 = tl.make_block_ptr(v_base, (T, V), (H * V, 1), (off_1, i_v * BV), (16, BV), (1, 0))
        p_v2 = tl.make_block_ptr(v_base, (T, V), (H * V, 1), (off_2, i_v * BV), (16, BV), (1, 0))
        p_v3 = tl.make_block_ptr(v_base, (T, V), (H * V, 1), (off_3, i_v * BV), (16, BV), (1, 0))
        b_v0 = tl.load(p_v0, boundary_check=(0, 1))
        b_v1 = tl.load(p_v1, boundary_check=(0, 1))
        b_v2 = tl.load(p_v2, boundary_check=(0, 1))
        b_v3 = tl.load(p_v3, boundary_check=(0, 1))

        b_vb0 = (b_v0 * b_beta0[:, None]).to(tl.float16)
        b_vb1 = (b_v1 * b_beta1[:, None]).to(tl.float16)
        b_vb2 = (b_v2 * b_beta2[:, None]).to(tl.float16)
        b_vb3 = (b_v3 * b_beta3[:, None]).to(tl.float16)

        b_u0 = tl.dot(b_Ai00.to(b_vb0.dtype), b_vb0)
        b_u1 = tl.dot(b_Ai10.to(b_vb0.dtype), b_vb0) + tl.dot(b_Ai11.to(b_vb1.dtype), b_vb1)
        b_u2 = (tl.dot(b_Ai20.to(b_vb0.dtype), b_vb0)
                + tl.dot(b_Ai21.to(b_vb1.dtype), b_vb1)
                + tl.dot(b_Ai22.to(b_vb2.dtype), b_vb2))
        b_u3 = (tl.dot(b_Ai30.to(b_vb0.dtype), b_vb0)
                + tl.dot(b_Ai31.to(b_vb1.dtype), b_vb1)
                + tl.dot(b_Ai32.to(b_vb2.dtype), b_vb2)
                + tl.dot(b_Ai33.to(b_vb3.dtype), b_vb3))

        p_u0 = tl.make_block_ptr(u_base, (T, V), (H * V, 1), (off_0, i_v * BV), (16, BV), (1, 0))
        p_u1 = tl.make_block_ptr(u_base, (T, V), (H * V, 1), (off_1, i_v * BV), (16, BV), (1, 0))
        p_u2 = tl.make_block_ptr(u_base, (T, V), (H * V, 1), (off_2, i_v * BV), (16, BV), (1, 0))
        p_u3 = tl.make_block_ptr(u_base, (T, V), (H * V, 1), (off_3, i_v * BV), (16, BV), (1, 0))
        tl.store(p_u0, b_u0.to(p_u0.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_u1, b_u1.to(p_u1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_u2, b_u2.to(p_u2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_u3, b_u3.to(p_u3.dtype.element_ty), boundary_check=(0, 1))

    # w = A_inv @ diag(beta * exp(g)) @ k
    w_base = w_out + (bos * H + i_h) * K
    if USE_G:
        b_bg0 = (b_beta0 * exp(b_g0)).to(tl.float16)
        b_bg1 = (b_beta1 * exp(b_g1)).to(tl.float16)
        b_bg2 = (b_beta2 * exp(b_g2)).to(tl.float16)
        b_bg3 = (b_beta3 * exp(b_g3)).to(tl.float16)
    else:
        b_bg0 = b_beta0.to(tl.float16)
        b_bg1 = b_beta1.to(tl.float16)
        b_bg2 = b_beta2.to(tl.float16)
        b_bg3 = b_beta3.to(tl.float16)

    for i_k in range(tl.cdiv(K, BK)):
        p_k0_w = tl.make_block_ptr(k_base, (T, K), (Hg * K, 1), (off_0, i_k * BK), (16, BK), (1, 0))
        p_k1_w = tl.make_block_ptr(k_base, (T, K), (Hg * K, 1), (off_1, i_k * BK), (16, BK), (1, 0))
        p_k2_w = tl.make_block_ptr(k_base, (T, K), (Hg * K, 1), (off_2, i_k * BK), (16, BK), (1, 0))
        p_k3_w = tl.make_block_ptr(k_base, (T, K), (Hg * K, 1), (off_3, i_k * BK), (16, BK), (1, 0))
        b_k0_w = tl.load(p_k0_w, boundary_check=(0, 1))
        b_k1_w = tl.load(p_k1_w, boundary_check=(0, 1))
        b_k2_w = tl.load(p_k2_w, boundary_check=(0, 1))
        b_k3_w = tl.load(p_k3_w, boundary_check=(0, 1))

        b_wk0 = (b_k0_w * b_bg0[:, None]).to(tl.float16)
        b_wk1 = (b_k1_w * b_bg1[:, None]).to(tl.float16)
        b_wk2 = (b_k2_w * b_bg2[:, None]).to(tl.float16)
        b_wk3 = (b_k3_w * b_bg3[:, None]).to(tl.float16)

        b_w0 = tl.dot(b_Ai00.to(b_wk0.dtype), b_wk0)
        b_w1 = tl.dot(b_Ai10.to(b_wk0.dtype), b_wk0) + tl.dot(b_Ai11.to(b_wk1.dtype), b_wk1)
        b_w2 = (tl.dot(b_Ai20.to(b_wk0.dtype), b_wk0)
                + tl.dot(b_Ai21.to(b_wk1.dtype), b_wk1)
                + tl.dot(b_Ai22.to(b_wk2.dtype), b_wk2))
        b_w3 = (tl.dot(b_Ai30.to(b_wk0.dtype), b_wk0)
                + tl.dot(b_Ai31.to(b_wk1.dtype), b_wk1)
                + tl.dot(b_Ai32.to(b_wk2.dtype), b_wk2)
                + tl.dot(b_Ai33.to(b_wk3.dtype), b_wk3))

        p_w0 = tl.make_block_ptr(w_base, (T, K), (H * K, 1), (off_0, i_k * BK), (16, BK), (1, 0))
        p_w1 = tl.make_block_ptr(w_base, (T, K), (H * K, 1), (off_1, i_k * BK), (16, BK), (1, 0))
        p_w2 = tl.make_block_ptr(w_base, (T, K), (H * K, 1), (off_2, i_k * BK), (16, BK), (1, 0))
        p_w3 = tl.make_block_ptr(w_base, (T, K), (H * K, 1), (off_3, i_k * BK), (16, BK), (1, 0))
        tl.store(p_w0, b_w0.to(p_w0.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w1, b_w1.to(p_w1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w2, b_w2.to(p_w2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w3, b_w3.to(p_w3.dtype.element_ty), boundary_check=(0, 1))


def fused_kkt_solve_wy_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    out_w: torch.Tensor | None = None,
    out_u: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused kkt + solve_tril + recompute_w_u. Supports partial chunks."""
    B, T, Hg, K = k.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = chunk_size
    assert BT == 64, "Fused kernel requires chunk_size=64"

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if out_w is None:
        out_w = k.new_zeros(B, T, H, K)
    if out_u is None:
        out_u = torch.zeros_like(v)

    fused_kkt_solve_wy_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        g=g,
        w_out=out_w,
        u_out=out_u,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return out_w, out_u
