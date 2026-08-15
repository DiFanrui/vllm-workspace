#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Ascend W4A16-AWQ MoE scheme (hand-written AWQ, fused 3D tensors).

Checkpoint format produced by ``test_0811/awq/save_awq_model.py`` for
Qwen3-VL-30B-A3B (MoE).  Each FusedMoE layer stores six fused 3D tensors
(the ``experts.`` prefix is exactly the vLLM FusedMoE param name):

- ``w13_qweight``   ``[num_experts, 2*intermediate, hidden//8]``  int32
  packed Q(W13*s13), low-nibble-first (``pack_int4_int32`` in the save script)
- ``w13_qscales``   ``[num_experts, 2*intermediate, hidden//group_size]`` bf16
- ``w13_awq_scale`` ``[num_experts, 1, hidden]`` bf16  per-input-channel s13
- ``w2_qweight``    ``[num_experts, hidden, intermediate//8]``  int32
- ``w2_qscales``    ``[num_experts, hidden, intermediate//group_size]`` bf16
- ``w2_awq_scale``  ``[num_experts, 1, intermediate]`` bf16

Loading: none of these names contain ``experts.gate_up_proj`` /
``experts.down_proj`` / any stacked-shard suffix, so they fall through to the
generic 2-arg branch in ``Qwen3VLMoe.load_weights``.  That branch calls
``weight_loader(param, loaded_weight)``, so every param must carry
``default_weight_loader`` (set via ``load_whole_tensor = True`` in the adapter
gate) instead of the 6-arg ``FusedMoE.weight_loader``.

Dequant math (per expert, matching the save-side ``w_hat``):
    w13 = unpack(w13_qweight[e]) * w13_qscales[e].repeat_interleave(group_size)
    w2  = unpack(w2_qweight[e])  * w2_qscales[e].repeat_interleave(group_size)
    out = silu(linear(x/s13[e], w13[:intm])) * linear(x/s13[e], w13[intm:])
    out = linear(out / s2[e], w2)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult

from .base import AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme
from .w4a16_awq import unpack_int4_packed_int32


@register_scheme("W4A16_AWQ", "moe")
class AscendAWQFusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for the hand-written W4A16-AWQ (fused 3D tensors).

    ``apply`` is a plain-torch fallback (per-expert dequant + matmul) that
    produces identical numerics to the dense AWQ path, trading speed for
    portability.  It is correct for tensor-parallel size 1 (logical expert id
    == physical expert id); EPLB / TP>1 need an expert-id remap added here.
    """

    quant_type: QuantType = QuantType.W4A16

    # Whole fused tensors are copied verbatim by default_weight_loader; the
    # adapter gates on this to avoid the 6-arg FusedMoE.weight_loader.
    load_whole_tensor: bool = True

    def __init__(self, group_size: int = 128, num_bits: int = 4) -> None:
        self.group_size = group_size
        self.num_bits = num_bits
        self.pack_factor = 32 // num_bits  # 8 int4 per int32

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        assert hidden_sizes % self.pack_factor == 0
        assert intermediate_size_per_partition % self.pack_factor == 0
        assert hidden_sizes % self.group_size == 0
        assert intermediate_size_per_partition % self.group_size == 0

        # gate_up: 2 * intermediate rows (silu(gate) * up), hidden input cols
        return {
            "w13_qweight": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_sizes // self.pack_factor,
                dtype=torch.int32,
            ),
            "w13_qscales": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_sizes // self.group_size,
                dtype=torch.bfloat16,
            ),
            "w13_awq_scale": torch.empty(
                num_experts, 1, hidden_sizes, dtype=torch.bfloat16
            ),
            "w2_qweight": torch.empty(
                num_experts,
                hidden_sizes,
                intermediate_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            "w2_qscales": torch.empty(
                num_experts,
                hidden_sizes,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.bfloat16,
            ),
            "w2_awq_scale": torch.empty(
                num_experts, 1, intermediate_size_per_partition, dtype=torch.bfloat16
            ),
        }

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        # No dynamic (activation) quantization tensors.
        return {}

    def _dequant(self, qweight: torch.Tensor, qscales: torch.Tensor,
                 dtype: torch.dtype) -> torch.Tensor:
        """Per-expert: unpack Q(W*s) * per-group scale -> [out, in] (bf16)."""
        group = self.group_size
        return (
            unpack_int4_packed_int32(qweight, self.num_bits).to(dtype)
            * qscales.repeat_interleave(group, dim=-1)
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: Any | None = None,
    ) -> torch.Tensor:
        num_shared_experts = getattr(layer, "n_shared_experts", 0) or 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
        assert router_logits.shape[1] == num_logical_experts, (
            "Number of global experts mismatch (excluding redundancy): "
            f"router_logits.shape[1]={router_logits.shape[1]}, "
            f"num_logical_experts={num_logical_experts}"
        )

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_logical_experts,
            tid2eid=tid2eid,
        )
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)

        num_phys_experts = layer.w13_qweight.shape[0]
        hidden = layer.w13_awq_scale.shape[-1]
        intm = layer.w2_awq_scale.shape[-1]
        out = torch.zeros(
            x.shape[0], hidden, dtype=x.dtype, device=x.device
        )

        # per-expert: token rows -> dequant -> gate*up -> down -> scatter
        for e in range(num_phys_experts):
            routed = torch.nonzero(topk_ids == e)  # [n_e, 2] (row, col)
            if routed.shape[0] == 0:
                continue
            rows = routed[:, 0]
            cols = routed[:, 1]
            w = topk_weights[rows, cols].view(-1, 1)  # [n_e, 1]

            x_e = x[rows]  # [n_e, hidden]
            s13 = layer.w13_awq_scale[e]  # [1, hidden]
            x_scaled = x_e / s13
            w13 = self._dequant(layer.w13_qweight[e], layer.w13_qscales[e], x.dtype)
            gate = F.linear(x_scaled, w13[:intm])  # [n_e, intm]
            up = F.linear(x_scaled, w13[intm:])
            intm_val = F.silu(gate) * up  # [n_e, intm]

            s2 = layer.w2_awq_scale[e]  # [1, intm]
            w2 = self._dequant(layer.w2_qweight[e], layer.w2_qscales[e], x.dtype)
            d = F.linear(intm_val / s2, w2)  # [n_e, hidden]
            out[rows] += w * d

        # vllm-ascend 契约：AscendFusedMoE.forward_impl (fused_moe.py:723) 在 apply 后
        # 访问 fused_experts_results.routed_out，因此必须返回 FusedExpertsResult 而非裸 Tensor。
        return FusedExpertsResult(routed_out=out)
