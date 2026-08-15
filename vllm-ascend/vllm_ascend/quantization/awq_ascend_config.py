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
"""Quantization config for the hand-written AWQ (W4A16-AWQ) method.

A custom vLLM quantization method (``--quantization awq_ascend``) whose
checkpoint format is produced by ``vllm-workspace/test_0811/awq/save_awq_model.py``.

Per dense Linear layer the checkpoint stores three tensors:

- ``qweight``   ``[out, in//8] int32``   packed Q(W*s)
- ``qscales``   ``[out, in//group_size]``  per-(row, group) scale of W*s
- ``awq_scale`` ``[1, in]``              per-input-channel activation scale s (2D so both
                                          Column- and RowParallel weight loaders accept it)

Unquantized modules (embeddings, norms, vision tower, ...) keep their
original ``weight`` tensors and are loaded as-is.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

from vllm_ascend.utils import AWQ_ASCEND_METHOD

from .method_adapters import AscendFusedMoEMethod, AscendLinearMethod
from .methods.w4a16_awq import AscendW4A16AWQLinearMethod
from .methods.w4a16_awq_moe import AscendAWQFusedMoEMethod


@register_quantization_config(AWQ_ASCEND_METHOD)
class AscendAWQConfig(QuantizationConfig):
    """Config class for the Ascend W4A16-AWQ quantization method."""

    def __init__(
        self,
        group_size: int = 128,
        num_bits: int = 4,
        modules_to_not_convert: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.num_bits = num_bits
        self.modules_to_not_convert = modules_to_not_convert or []
        # AscendRMSNorm (vllm_ascend/ops/layernorm.py) 在 quant_config 非空时迭代
        # quant_description，检查是否有 "norm.bias" 量化；空 dict 即不生成 norm bias。
        self.quant_description: dict = {}

    def get_name(self) -> str:
        return AWQ_ASCEND_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware does not support "get_min_capability" feature.')

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        # Detection is handled by vllm_ascend.quantization.utils (reads
        # ``quantization_config.quant_method`` from config.json), so no
        # separate config file is needed here.
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendAWQConfig":
        group_size = cls.get_from_keys_or(config, ["group_size", "q_group_size"], 128)
        num_bits = cls.get_from_keys_or(config, ["bits", "w_bit"], 4)
        modules_to_not_convert = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(group_size=group_size, num_bits=num_bits, modules_to_not_convert=modules_to_not_convert)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
        if isinstance(layer, LinearBase):
            # Skip layers listed in modules_to_not_convert (e.g. the visual
            # tower).  Uses substring matching so that "visual" also covers
            # "visual.patch_embedding.linear_fc1", mirroring vLLM AWQ's
            # is_layer_skipped(..., skip_with_substr=True).  Must return an
            # actual method — LinearBase raises if get_quant_method() is None.
            if any(m in prefix for m in self.modules_to_not_convert):
                from vllm.model_executor.layers.linear import UnquantizedLinearMethod

                return UnquantizedLinearMethod()
            return AscendLinearMethod(
                AscendW4A16AWQLinearMethod(
                    group_size=self.group_size, num_bits=self.num_bits
                )
            )
        if isinstance(layer, FusedMoE):
            # MoE experts quantized by the hand-written AWQ tool (fused 3D
            # int4 tensors + torch-fallback apply).  The 30B-A3B checkpoint
            # stores w13/w2 qweight/qscales/awq_scale on the FusedMoE layer.
            return AscendFusedMoEMethod(
                AscendAWQFusedMoEMethod(
                    group_size=self.group_size, num_bits=self.num_bits
                ),
                layer.moe_config,
                tid2eid=tid2eid,
            )
        # attention / KV cache: keep unquantized.
        return None
