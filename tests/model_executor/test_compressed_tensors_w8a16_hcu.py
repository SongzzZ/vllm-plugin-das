# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the compressed-tensors W8A16 HCU scheme (MR !283).

CPU-only. Every byte-level case drives the production conversion in
process_weights_after_loading directly (with hand-built layer parameters,
because create_weights requires an initialized TP group), so a future
byte-order or sign-flip regression in the scheme itself fails here.
"""

import types

import pytest
import torch

pytest.importorskip("compressed_tensors")

from compressed_tensors.config import CompressionFormat

from vllm_hcu.model_executor.layers.quantization.compressed_tensors_w8a16_hcu import (
    CompressedTensorsW8A16Hcu,
    is_hcu_w8a16_int8_channel,
)


def _wq(num_bits=8, dynamic=False, strategy="channel", symmetric=True):
    return types.SimpleNamespace(
        num_bits=num_bits, dynamic=dynamic, strategy=strategy, symmetric=symmetric
    )


def test_selector_positive():
    assert is_hcu_w8a16_int8_channel(
        _wq(), None, CompressionFormat.pack_quantized.value
    )


@pytest.mark.parametrize(
    "wq_kwargs,fmt,input_quant",
    [
        ({"num_bits": 4}, CompressionFormat.pack_quantized.value, None),
        ({"dynamic": True}, CompressionFormat.pack_quantized.value, None),
        ({"strategy": "group"}, CompressionFormat.pack_quantized.value, None),
        ({"symmetric": False}, CompressionFormat.pack_quantized.value, None),
        ({}, "dense", None),
        ({}, CompressionFormat.pack_quantized.value, types.SimpleNamespace()),
    ],
)
def test_selector_negative(wq_kwargs, fmt, input_quant):
    assert not is_hcu_w8a16_int8_channel(_wq(**wq_kwargs), input_quant, fmt)


def _make_layer(out_features, in_features):
    layer = torch.nn.Module()
    layer.weight_packed = torch.nn.Parameter(
        torch.zeros(out_features, in_features // 4, dtype=torch.int32),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.full((out_features, 1), 0.5, dtype=torch.float16),
        requires_grad=False,
    )
    layer.weight_shape = torch.nn.Parameter(
        torch.tensor([out_features, in_features], dtype=torch.int64),
        requires_grad=False,
    )
    return layer


def test_process_weights_known_pattern():
    # Non-neutral bytes through the production conversion. Word 0x80808081
    # has little-endian bytes 81 80 80 80; the uint8b128 flip ^0x80 yields
    # 01 00 00 00 -> int8 [1, 0, 0, 0]. Word 0xFF7F0180 has bytes
    # 80 01 7F FF -> 00 81 FF 7F -> [0, -127, -1, 127].
    scheme = CompressedTensorsW8A16Hcu()
    layer = _make_layer(1, 8)
    layer.weight_packed.data.view(torch.uint8)[0] = torch.tensor(
        [0x81, 0x80, 0x80, 0x80, 0x80, 0x01, 0x7F, 0xFF], dtype=torch.uint8
    )
    scheme.process_weights_after_loading(layer)
    assert layer.weight.shape == (1, 8)
    assert layer.weight.dtype == torch.int8
    assert layer.weight.tolist() == [[1, 0, 0, 0, 0, -127, -1, 127]]
    assert layer.weight_scale.shape == (1,)
    assert layer.weight_packed is None
    assert layer.weight_shape is None


def test_process_weights_random_matches_byte_reference():
    # Independent reference: uint8b128 means real int8 = byte - 128. The
    # production conversion must reproduce it for arbitrary packed data.
    torch.manual_seed(0)
    out_features, in_features = 8, 32
    scheme = CompressedTensorsW8A16Hcu()
    layer = _make_layer(out_features, in_features)
    packed = torch.randint(
        -(2**31), 2**31 - 1, (out_features, in_features // 4), dtype=torch.int32
    )
    layer.weight_packed.data.copy_(packed)
    scheme.process_weights_after_loading(layer)
    reference = (
        packed.view(torch.uint8).to(torch.int16) - 128
    ).clamp(-128, 127).to(torch.int8).view_as(layer.weight)
    assert layer.weight.shape == (out_features, in_features)
    assert torch.equal(layer.weight, reference)


def test_process_weights_neutral_pattern_is_zero():
    # Every byte 0x80 is the uint8b128 zero: the unpacked weight must be 0.
    scheme = CompressedTensorsW8A16Hcu()
    layer = _make_layer(16, 32)
    layer.weight_packed.data.view(torch.uint8).fill_(0x80)
    scheme.process_weights_after_loading(layer)
    assert layer.weight.shape == (16, 32)
    assert torch.all(layer.weight == 0)
