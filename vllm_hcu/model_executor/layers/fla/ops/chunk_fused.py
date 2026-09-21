# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Fused chunk_gated_delta_rule forward pass.
# No padding: kernels handle partial chunks internally.
# ruff: noqa: E501

import torch

from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd

from vllm_hcu.model_executor.layers.fla.ops.fused_h_o import fused_h_o_fwd
from vllm_hcu.model_executor.layers.fla.ops.fused_kkt_solve_wy import (
    fused_kkt_solve_wy_fwd,
)


def chunk_gated_delta_rule_fwd_fused(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens=None,
):
    w, u = fused_kkt_solve_wy_fwd(
        k=k, v=v, beta=beta, g=g, cu_seqlens=cu_seqlens, chunk_size=64,
    )
    o, v_new, final_state = fused_h_o_fwd(
        q=q, k=k, w=w, u=u, g=g,
        scale=scale, initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_size=64,
    )
    return g, o, None, final_state, w, None, v_new


# ---------------------------------------------------------------------------
# Static buffer path: uses pre-allocated tensors for CUDA graph compatibility.
# During graph replay, no internal torch.empty/zeros calls occur, so GPU
# memory addresses stay fixed across replays.
# ---------------------------------------------------------------------------

class ChunkGDRBuffers:
    """Pre-allocated buffers for chunk_gated_delta_rule static path.

    Allocate once at model init with max capture shapes, then reuse across
    forward calls. This avoids on-the-fly torch.empty/zeros which would
    invalidate CUDA graph captured addresses.
    """

    def __init__(
        self,
        max_tokens: int,
        max_seqs: int,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        max_batch: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.max_batch = max_batch

        # Intermediate buffers between the two fused kernels
        self.w = torch.empty(max_batch, max_tokens, num_heads, head_k_dim, dtype=dtype, device=device)
        self.u = torch.empty(max_batch, max_tokens, num_heads, head_v_dim, dtype=dtype, device=device)

        # Output buffers
        self.o = torch.empty(max_batch, max_tokens, num_heads, head_v_dim, dtype=dtype, device=device)
        self.v_new = torch.empty(max_batch, max_tokens, num_heads, head_v_dim, dtype=dtype, device=device)

        # Final state buffer (float32 as required by kernel)
        self.final_state = torch.empty(
            max_seqs, num_heads, head_v_dim, head_k_dim, dtype=torch.float32, device=device
        )


def chunk_gated_delta_rule_fwd_static(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
    buf_w: torch.Tensor,
    buf_u: torch.Tensor,
    buf_o: torch.Tensor,
    buf_v_new: torch.Tensor,
    buf_final_state: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Static-buffer version of chunk_gated_delta_rule_fwd_fused.

    All writable intermediate/output tensors are passed in via buf_* parameters
    with fixed memory addresses. No internal torch.empty/zeros calls — safe
    for CUDA graph capture/replay.

    The caller is responsible for providing correctly-sized buffers (e.g. via
    ChunkGDRBuffers). The kernel writes to these buffers in-place.
    """
    B, T, _, _ = q.shape
    if buf_w.shape[0] < B or buf_w.shape[1] < T:
        raise ValueError(
            f"chunk_gated_delta_rule_fwd_static: buf_w {tuple(buf_w.shape)} "
            f"smaller than q B/T ({B},{T})"
        )
    if output_final_state and buf_final_state.shape[0] < initial_state.shape[0]:
        raise ValueError(
            f"chunk_gated_delta_rule_fwd_static: buf_final_state rows "
            f"{buf_final_state.shape[0]} < N_seqs {initial_state.shape[0]}"
        )

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    w, u = fused_kkt_solve_wy_fwd(
        k=k, v=v, beta=beta, g=g,
        cu_seqlens=cu_seqlens, chunk_size=64,
        out_w=buf_w[:B, :T], out_u=buf_u[:B, :T],
    )
    o, v_new, final_state = fused_h_o_fwd(
        q=q, k=k, w=w, u=u, g=g,
        scale=scale, initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_size=64,
        out_o=buf_o[:B, :T], out_v_new=buf_v_new[:B, :T],
        out_final_state=buf_final_state[:initial_state.shape[0]] if output_final_state else None,
    )
    return g, o, None, final_state, w, None, v_new
