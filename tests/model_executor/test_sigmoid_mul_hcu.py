# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_hcu.ops.sigmoid_mul import sigmoid_mul_hcu


def _device():
    if not torch.cuda.is_available():
        pytest.skip("HCU device is unavailable")
    return "cuda"


@pytest.mark.parametrize("num_tokens", [1, 2, 8])
def test_sigmoid_mul_matches_native_fp16(num_tokens):
    device = _device()
    torch.manual_seed(0)
    gate = torch.randn(num_tokens, 1, dtype=torch.float16, device=device)
    value = torch.randn(num_tokens, 512, dtype=torch.float16, device=device)
    expected = torch.sigmoid(gate) * value
    actual = sigmoid_mul_hcu(gate, value)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("num_tokens", [1, 2, 8])
def test_sigmoid_mul_matches_native_bf16(num_tokens):
    # The fused kernel supports bf16 (sigmoid computed in fp32 with the same
    # double rounding as torch.sigmoid(gate) * value).
    device = _device()
    torch.manual_seed(0)
    gate = torch.randn(num_tokens, 1, dtype=torch.bfloat16, device=device)
    value = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device=device)
    expected = torch.sigmoid(gate) * value
    actual = sigmoid_mul_hcu(gate, value)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


def test_sigmoid_mul_fp32_falls_back_to_native():
    # fp32 is not a fused-kernel dtype; the wrapper must take the PyTorch
    # path instead of failing in the kernel's TORCH_CHECK.
    device = _device()
    torch.manual_seed(0)
    gate = torch.randn(4, 1, dtype=torch.float32, device=device)
    value = torch.randn(4, 512, dtype=torch.float32, device=device)
    actual = sigmoid_mul_hcu(gate, value)
    assert torch.equal(torch.sigmoid(gate) * value, actual)


def test_sigmoid_mul_dtype_mismatch_falls_back_to_native():
    device = _device()
    torch.manual_seed(0)
    gate = torch.randn(4, 1, dtype=torch.float16, device=device)
    value = torch.randn(4, 512, dtype=torch.bfloat16, device=device)
    expected = torch.sigmoid(gate) * value  # torch promotes fp16*bf16 -> fp32
    actual = sigmoid_mul_hcu(gate, value)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sigmoid_mul_fake_tensor_metadata(dtype):
    # torch.compile / FakeTensor relies on the Meta kernel registered by
    # vllm_hcu.ops.sigmoid_mul to propagate output metadata.
    from torch._subclasses.fake_tensor import FakeTensorMode

    device = _device()
    with FakeTensorMode():
        gate = torch.randn(4, 1, dtype=dtype, device=device)
        value = torch.randn(4, 512, dtype=dtype, device=device)
        out = torch.ops.hcu_ops.sigmoid_mul(gate, value)
    assert out.shape == value.shape
    assert out.dtype == dtype
    assert out.device.type == "cuda"


def test_sigmoid_mul_torch_compile():
    # The fused call site sits inside the model forward; Dynamo must trace
    # through the wrapper (dtype guard + custom op) in a full graph.
    device = _device()
    torch.manual_seed(0)
    gate = torch.randn(4, 1, dtype=torch.float16, device=device)
    value = torch.randn(4, 512, dtype=torch.float16, device=device)
    compiled = torch.compile(sigmoid_mul_hcu, backend="eager", fullgraph=True)
    out = compiled(gate, value)
    assert out.shape == value.shape
    assert torch.allclose(out, torch.sigmoid(gate) * value)


@pytest.mark.parametrize("batch", [1, 9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sigmoid_mul_opcheck(batch, dtype):
    device = _device()
    torch.manual_seed(0)
    gate = torch.randn(batch, 1, dtype=dtype, device=device)
    value = torch.randn(batch, 512, dtype=dtype, device=device)
    torch.library.opcheck(
        torch.ops.hcu_ops.sigmoid_mul,
        (gate, value),
        test_utils=("test_schema", "test_faketensor"),
    )
