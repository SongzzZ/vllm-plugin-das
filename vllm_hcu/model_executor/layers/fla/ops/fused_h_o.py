# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Fused kernel 2: chunk_gated_delta_rule_fwd_h + chunk_fwd_o
#
# Eliminates the intermediate h tensor ([B, NT, H, V, K]) from global memory.
# Instead of storing h per-chunk then reading it back for output, we keep h in
# registers and produce the output immediately.
#
# Key insight (online-tiling analogy):
#   FlashAttention avoids materializing N×N softmax by maintaining running max/sum.
#   This kernel avoids materializing [NT, V, K] hidden states by maintaining h in registers
#   and computing each chunk's output before advancing h.
#
# Memory savings: [B, NT, H, V, K] intermediate eliminated.
# Trade-off: chunks processed sequentially (same as fwd_h), but output is produced inline
#            rather than requiring a separate parallel pass.
#
# Partial chunk support: when T is not a multiple of BT (64), the last chunk has
# cur_len < BT valid tokens. Block pointers start at valid positions (i_t * BT < T),
# so boundary_check handles OOB elements. Loaded values are additionally masked to
# zero for invalid rows as a safety measure for ROCm/HIP.
# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from vllm.model_executor.layers.fla.ops.index import prepare_chunk_offsets
from vllm.model_executor.layers.fla.ops.op import exp
from vllm.model_executor.layers.fla.ops.utils import use_cuda_graph


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [32]
        for num_warps in [4, 8]
        for num_stages in [1]
    ],
    key=["H", "K", "V", "BT"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def fused_h_o_kernel(
    # inputs
    k,
    q,
    w,
    u,
    g,
    h0,
    ht,
    o_out,
    v_new_out,
    # common
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused hidden-state recurrence + output computation.

    For each chunk t (processed sequentially):
      1. Compute v_corrected[t] = u[t] - w[t] @ h[t]^T   (uses h in registers)
      2. Compute o_inter[t] = q[t] @ h[t]^T * scale       (inter-chunk output)
      3. Compute o_intra[t] = causal(q@k^T * gate) @ v_corrected * scale  (intra-chunk)
      4. o[t] = o_inter[t] + o_intra[t]
      5. h[t+1] = decay * h[t] + k[t]^T @ (v_corrected * gate_correction)

    Grid: (cdiv(V, BV), N * H)
    Each program handles one V-tile across all chunks for one (sequence, head).
    h is kept in registers as [BV, K] split into 64-wide tiles (up to 4 tiles for K≤256).
    Supports partial last chunk via cur_len masking.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)

    # Strides
    stride_qk = Hg * K
    stride_wv = H * V
    stride_w = H * K

    # Base pointers
    q_ptr = q + (bos * Hg + i_h // (H // Hg)) * K
    k_ptr = k + (bos * Hg + i_h // (H // Hg)) * K
    w_ptr = w + (bos * H + i_h) * K
    u_ptr = u + (bos * H + i_h) * V
    o_ptr = o_out + (bos * H + i_h) * V
    v_new_ptr = v_new_out + (bos * H + i_h) * V

    # ======== Initialize hidden state h: [BV, K] in 64-wide tiles ========
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0_base = h0 + i_nh * V * K
        p_h01 = tl.make_block_ptr(p_h0_base, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 = tl.load(p_h01, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h02 = tl.make_block_ptr(p_h0_base, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            b_h2 = tl.load(p_h02, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h03 = tl.make_block_ptr(p_h0_base, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            b_h3 = tl.load(p_h03, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h04 = tl.make_block_ptr(p_h0_base, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            b_h4 = tl.load(p_h04, boundary_check=(0, 1)).to(tl.float32)

    # ======== Main loop: process chunks sequentially ========
    for i_t in range(NT):
        # Number of valid rows in this chunk (may be < BT for the last chunk)
        cur_len = tl.minimum(BT, T - i_t * BT)

        # Row-valid mask for T-dimension loads: [BT, 1] or [BT]
        m_row = (tl.arange(0, BT) < cur_len)[:, None]

        # --- Step 1: v_corrected = u - w @ h^T ---
        # w: [BT, K], h: [BV, K] → w @ h^T: [BT, BV]
        p_w1 = tl.make_block_ptr(w_ptr, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_wh = tl.dot(tl.where(m_row, b_w1, 0.0), tl.trans(b_h1).to(b_w1.dtype))
        if K > 64:
            p_w2 = tl.make_block_ptr(w_ptr, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w2 = tl.load(p_w2, boundary_check=(0, 1))
            b_wh += tl.dot(tl.where(m_row, b_w2, 0.0), tl.trans(b_h2).to(b_w2.dtype))
        if K > 128:
            p_w3 = tl.make_block_ptr(w_ptr, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w3 = tl.load(p_w3, boundary_check=(0, 1))
            b_wh += tl.dot(tl.where(m_row, b_w3, 0.0), tl.trans(b_h3).to(b_w3.dtype))
        if K > 192:
            p_w4 = tl.make_block_ptr(w_ptr, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w4 = tl.load(p_w4, boundary_check=(0, 1))
            b_wh += tl.dot(tl.where(m_row, b_w4, 0.0), tl.trans(b_h4).to(b_w4.dtype))

        # v_new = u - w @ h^T
        p_u = tl.make_block_ptr(u_ptr, (T, V), (stride_wv, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_u = tl.load(p_u, boundary_check=(0, 1))
        b_vnew = tl.where(m_row, b_u - b_wh, 0.0)

        # Store v_new
        p_vnew = tl.make_block_ptr(v_new_ptr, (T, V), (stride_wv, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_vnew, b_vnew.to(p_vnew.dtype.element_ty), boundary_check=(0, 1))

        # --- Step 2: o_inter = q @ h^T * scale (per V-tile) ---
        # q: [BT, K], h: [BV, K] → q @ h^T: [BT, BV]
        p_q1 = tl.make_block_ptr(q_ptr, (T, K), (stride_qk, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_o_inter = tl.dot(tl.where(m_row, b_q1, 0.0), tl.trans(b_h1).to(b_q1.dtype))
        if K > 64:
            p_q2 = tl.make_block_ptr(q_ptr, (T, K), (stride_qk, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_q2 = tl.load(p_q2, boundary_check=(0, 1))
            b_o_inter += tl.dot(tl.where(m_row, b_q2, 0.0), tl.trans(b_h2).to(b_q2.dtype))
        if K > 128:
            p_q3 = tl.make_block_ptr(q_ptr, (T, K), (stride_qk, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_q3 = tl.load(p_q3, boundary_check=(0, 1))
            b_o_inter += tl.dot(tl.where(m_row, b_q3, 0.0), tl.trans(b_h3).to(b_q3.dtype))
        if K > 192:
            p_q4 = tl.make_block_ptr(q_ptr, (T, K), (stride_qk, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_q4 = tl.load(p_q4, boundary_check=(0, 1))
            b_o_inter += tl.dot(tl.where(m_row, b_q4, 0.0), tl.trans(b_h4).to(b_q4.dtype))

        # --- Step 3: o_intra = causal(q @ k^T * gate) @ v_new * scale ---
        # Compute q @ k^T: [BT, BT] (accumulate over K-tiles)
        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        p_k1_t = tl.make_block_ptr(k_ptr, (K, T), (1, stride_qk), (0, i_t * BT), (64, BT), (0, 1))
        b_k1_t = tl.load(p_k1_t, boundary_check=(0, 1))
        b_A += tl.dot(tl.where(m_row, b_q1, 0.0), b_k1_t)
        if K > 64:
            p_k2_t = tl.make_block_ptr(k_ptr, (K, T), (1, stride_qk), (64, i_t * BT), (64, BT), (0, 1))
            b_k2_t = tl.load(p_k2_t, boundary_check=(0, 1))
            b_A += tl.dot(tl.where(m_row, b_q2, 0.0), b_k2_t)
        if K > 128:
            p_k3_t = tl.make_block_ptr(k_ptr, (K, T), (1, stride_qk), (128, i_t * BT), (64, BT), (0, 1))
            b_k3_t = tl.load(p_k3_t, boundary_check=(0, 1))
            b_A += tl.dot(tl.where(m_row, b_q3, 0.0), b_k3_t)
        if K > 192:
            p_k4_t = tl.make_block_ptr(k_ptr, (K, T), (1, stride_qk), (192, i_t * BT), (64, BT), (0, 1))
            b_k4_t = tl.load(p_k4_t, boundary_check=(0, 1))
            b_A += tl.dot(tl.where(m_row, b_q4, 0.0), b_k4_t)

        # Apply gating
        if USE_G:
            g_base = g + bos * H + i_h
            p_g = tl.make_block_ptr(g_base, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            b_g = tl.where(tl.arange(0, BT) < cur_len, b_g, 0.0)
            # Per-chunk local cumsum (fused, eliminates chunk_local_cumsum kernel)
            b_g = tl.cumsum(b_g, axis=0)
            b_o_inter = b_o_inter * exp(b_g)[:, None]
            b_A = b_A * exp(b_g[:, None] - b_g[None, :])

        # Apply causal mask
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_causal = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_causal, b_A, 0.0)

        # Final output: o = (o_inter + A @ v_new) * scale
        b_o = b_o_inter * scale + tl.dot(b_A.to(b_vnew.dtype), b_vnew) * scale

        p_o = tl.make_block_ptr(o_ptr, (T, V), (stride_wv, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # --- Step 4: Update h[t] → h[t+1] ---
        # h[t+1] = decay * h[t] + k[t]^T @ (v_new * per_token_decay)
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.sum(tl.where(tl.arange(0, BT) == cur_len - 1, b_g, 0.0))
            b_vnew_gated = b_vnew * tl.where(m_t, exp(b_g_last - b_g), 0.0)[:, None]
            b_g_decay = exp(b_g_last)
            b_h1 *= b_g_decay
            if K > 64:
                b_h2 *= b_g_decay
            if K > 128:
                b_h3 *= b_g_decay
            if K > 192:
                b_h4 *= b_g_decay
        else:
            b_vnew_gated = b_vnew

        b_vnew_gated = b_vnew_gated.to(k.dtype.element_ty)

        # h += k^T @ v_gated  (k: [K, BT], v_gated: [BT, BV] → [K, BV], transposed to [BV, K])
        p_kk1 = tl.make_block_ptr(k_ptr, (K, T), (1, stride_qk), (0, i_t * BT), (64, BT), (0, 1))
        b_kk1 = tl.load(p_kk1, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_kk1, b_vnew_gated))
        if K > 64:
            p_kk2 = tl.make_block_ptr(k_ptr, (K, T), (1, stride_qk), (64, i_t * BT), (64, BT), (0, 1))
            b_kk2 = tl.load(p_kk2, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_kk2, b_vnew_gated))
        if K > 128:
            p_kk3 = tl.make_block_ptr(k_ptr, (K, T), (1, stride_qk), (128, i_t * BT), (64, BT), (0, 1))
            b_kk3 = tl.load(p_kk3, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_kk3, b_vnew_gated))
        if K > 192:
            p_kk4 = tl.make_block_ptr(k_ptr, (K, T), (1, stride_qk), (192, i_t * BT), (64, BT), (0, 1))
            b_kk4 = tl.load(p_kk4, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_kk4, b_vnew_gated))

    # ======== Epilogue: store final state ========
    if STORE_FINAL_STATE:
        p_ht_base = ht + i_nh * V * K
        p_ht1 = tl.make_block_ptr(p_ht_base, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht2 = tl.make_block_ptr(p_ht_base, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht3 = tl.make_block_ptr(p_ht_base, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            tl.store(p_ht3, b_h3.to(p_ht3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht4 = tl.make_block_ptr(p_ht_base, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            tl.store(p_ht4, b_h4.to(p_ht4.dtype.element_ty), boundary_check=(0, 1))


def fused_h_o_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    out_o: torch.Tensor | None = None,
    out_v_new: torch.Tensor | None = None,
    out_final_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Fused hidden-state recurrence + output computation.

    Replaces the 2-kernel sequence:
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k, w, u, g, ...)
        o = chunk_fwd_o(q, k, v_new, h, g, scale, ...)

    Eliminates the h tensor [B, NT, H, V, K] from global memory entirely.
    For typical prefill (B=4, T=2048, H=16, V=128, K=128):
      h = 4 × 32 × 16 × 128 × 128 × 2 bytes ≈ 2 GB saved.

    The kernel processes chunks sequentially (same serial dependency as fwd_h),
    but produces output inline rather than requiring a separate parallel pass.
    Net effect: same latency for the sequential part, one fewer kernel launch,
    and dramatically reduced memory.

    Args:
        q: [B, T, Hg, K]
        k: [B, T, Hg, K]
        w: [B, T, H, K] - from fused_kkt_solve_wy or recompute_w_u
        u: [B, T, H, V] - from fused_kkt_solve_wy or recompute_w_u
        g: [B, T, H] - cumulative log-gate
        scale: attention scale (default: K^-0.5)
        initial_state: [N, H, V, K]
        output_final_state: whether to return final h
        cu_seqlens: variable-length boundaries

    Returns:
        o: [B, T, H, V] - final output
        v_new: [B, T, H, V] - corrected values (for potential backward use)
        final_state: [N, H, V, K] or None
    """
    B, T, Hg, K = k.shape
    H, V = u.shape[-2], u.shape[-1]
    BT = chunk_size

    if scale is None:
        scale = K ** -0.5

    if cu_seqlens is None:
        N = B
        chunk_offsets = None
    else:
        N = len(cu_seqlens) - 1
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    assert K <= 256, "Kernel does not support K > 256"

    if out_final_state is None and output_final_state:
        out_final_state = k.new_empty(N, H, V, K, dtype=torch.float32)
    elif not output_final_state:
        out_final_state = None
    if out_o is None:
        out_o = torch.empty_like(u)
    if out_v_new is None:
        out_v_new = torch.empty_like(u)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    fused_h_o_kernel[grid](
        k=k,
        q=q,
        w=w,
        u=u,
        g=g,
        h0=initial_state,
        ht=out_final_state,
        o_out=out_o,
        v_new_out=out_v_new,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return out_o, out_v_new, out_final_state
