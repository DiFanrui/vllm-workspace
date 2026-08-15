#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Ascend W4A16-AWQ linear scheme (dense layers).

Loads weights produced by the hand-written AWQ quantizer
(``vllm-workspace/test_0811/awq/awq_core.py``):

- ``qweight``:   ``torch.int32``, ``[output_size, input_size // 8]``.
  Each int32 packs 8 signed int4 values of ``Q(W * s)`` (low nibble first).
- ``qscales``:   ``params_dtype``, ``[output_size, input_size // group_size]``.
  Per-(row, group) symmetric scale of ``W * s``, i.e. ``max|W*s| / 7``.
- ``awq_scale``: ``params_dtype``, ``[input_size]``. Per-input-channel AWQ
  activation-aware scale ``s``.

Inference (activation-side folding of ``1 / s``):
    dequant = unpack(qweight) * qscales.repeat_interleave(group_size, -1)  # ~ W*s
    out     = F.linear(x / awq_scale, dequant, bias)                       # ~ x @ W^T

This keeps the AWQ scale self-contained per layer (no cross-layer folding
bookkeeping as in AutoAWQ's ``scale_fc_fc``).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import AscendLinearScheme
from .registry import register_scheme


def unpack_int4_packed_int32(packed: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """Unpack int4 values packed into int32 back to signed int8.

    Args:
        packed: ``[out, in // (32 // num_bits)]`` int32, each element stores
            ``32 // num_bits`` values (low ``num_bits`` first).
        num_bits: Number of bits per value (must divide 32).

    Returns:
        ``[out, in]`` int8 with signed values in ``[-2^(num_bits-1),
        2^(num_bits-1) - 1]``.
    """
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    out = torch.zeros(
        packed.shape[0],
        packed.shape[1] * pack_factor,
        device=packed.device,
        dtype=torch.int32,
    )
    for i in range(pack_factor):
        out[:, i::pack_factor] = (packed >> (num_bits * i)) & mask

    # Convert unsigned nibbles back to signed values.
    out = torch.where(out >= (1 << (num_bits - 1)), out - (1 << num_bits), out)
    return out.to(torch.int8)


@register_scheme("W4A16_AWQ", "linear")
class AscendW4A16AWQLinearMethod(AscendLinearScheme):
    """Dense linear method for Ascend W4A16-AWQ (weight-only INT4)."""

    def __init__(self, group_size: int = 128, num_bits: int = 4) -> None:
        self.group_size = group_size
        self.num_bits = num_bits
        self.pack_factor = 32 // num_bits

    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        assert input_size % self.group_size == 0, (
            f"input_size {input_size} must be divisible by group_size {self.group_size}"
        )
        assert input_size % self.pack_factor == 0, (
            f"input_size {input_size} must be divisible by pack_factor {self.pack_factor}"
        )
        return {
            "qweight": torch.empty(
                output_size, input_size // self.pack_factor, dtype=torch.int32
            ),
            "qscales": torch.empty(
                output_size, input_size // self.group_size, dtype=params_dtype
            ),
            # Shape [1, in] (not [in]): the AscendLinearMethod adapter tags every
            # param with input_dim=1/output_dim=0.  ColumnParallelLinear.weight_loader
            # narrows along output_dim (0) and RowParallelLinear.weight_loader along
            # input_dim (1) — a 1-D tensor would IndexError on shape[1] in the row
            # path.  [1, in] narrows to itself in both paths and still broadcasts
            # against x in ``apply``.
            "awq_scale": torch.empty(1, input_size, dtype=params_dtype),
        }

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # Unpack Q(W*s) -> int8 and dequantize in one pass, so ``apply`` stays a
        # plain linear (correctness-first torch fallback).  Dequant is stored in
        # bf16 (not fp32): the full model would otherwise hold ~32GB of fp32
        # dequant weights and risk OOM at gpu_memory_utilization=0.6 on one 64GB
        # NPU.  bf16 rounding is ~0.4%, far below the int4 quantization error.
        unpacked = unpack_int4_packed_int32(layer.qweight.data, self.num_bits)
        qscales = layer.qscales.data
        dequant = unpacked.to(torch.float32) * qscales.to(torch.float32).repeat_interleave(
            self.group_size, dim=-1
        )
        layer.awq_dequant_weight = dequant.to(qscales.dtype)  # [out, in], ~ W*s

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        # (x / s) @ (W*s)^T ~ x @ W^T. Division in fp32 to avoid bf16 underflow.
        x_f = x.to(torch.float32) / layer.awq_scale.to(torch.float32)
        out = torch.nn.functional.linear(x_f, layer.awq_dequant_weight, bias)
        return out.to(x.dtype)
