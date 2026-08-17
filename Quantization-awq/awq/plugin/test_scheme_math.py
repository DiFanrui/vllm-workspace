#!/usr/bin/env python3
"""数值验证：vllm-ascend W4A16_AWQ scheme 的解包 + apply 与 MVP 的 w_hat 数学等价。

检查 (x/s) @ unpack(q)·qscales^T ≈ x @ w_hat^T，其中 w_hat = dequant(Q(W·s))/s。
"""
import torch

import sys
sys.path.insert(0, "/vllm-workspace/vllm-ascend")

from vllm_ascend.quantization.methods.w4a16_awq import (
    AscendW4A16AWQLinearMethod,
    unpack_int4_packed_int32,
)

torch.manual_seed(0)
OUT, IN, G = 64, 512, 128

w = (torch.randn(OUT, IN) * 0.02).double()
s = torch.linspace(0.5, 2.0, IN).double()          # 逐输入通道 AWQ scale
x = torch.randn(8, IN).double()

# --- MVP 路径: w_hat = dequant(Q(W·s)) / s ---
w_s = w * s
w_s_g = w_s.reshape(OUT, -1, G)
gscale = w_s_g.abs().amax(dim=-1).clamp_min(1e-5) / 7.0
q = (w_s_g / gscale.unsqueeze(-1)).round().clamp(-8, 7)
w_hat = (q.to(torch.float32) * gscale.unsqueeze(-1)).reshape(OUT, IN) / s
out_mvp = x @ w_hat.T

# --- 插件路径: pack int32 -> scheme.process_weights_after_loading -> apply ---
q_int8 = q.reshape(OUT, IN).to(torch.int8)

# 打包: 8 个 int4 一个 int32（低位先），带符号（-8..7 -> 0..15）
q_packed = torch.zeros(OUT, IN // 8, dtype=torch.int32)
for i in range(8):
    q_packed |= ((q_int8[:, i::8].to(torch.int32) & 0x0F) << (4 * i))

# 解包回读（模拟 scheme 内部）
unpacked = unpack_int4_packed_int32(q_packed)
assert torch.equal(unpacked, q_int8), "unpack roundtrip 不一致"

# 用 scheme 的 process_weights_after_loading + apply
import torch.nn as nn

class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.qweight = nn.Parameter(q_packed, requires_grad=False)
        self.qscales = nn.Parameter(gscale.reshape(OUT, -1).float(), requires_grad=False)
        # [1, in]：与 scheme get_weight / checkpoint 形状一致
        self.awq_scale = nn.Parameter(s.float().unsqueeze(0), requires_grad=False)

layer = FakeLayer()
scheme = AscendW4A16AWQLinearMethod(group_size=G)
scheme.process_weights_after_loading(layer)
out_plugin = scheme.apply(layer, x.float(), bias=None)

rel = (out_plugin - out_mvp.float()).abs().max().item()
print(f"out 与 w_hat 路径最大绝对差: {rel:.3e}")
# dequant 存 bf16（省内存），误差由 bf16 舍入主导，放宽到 1e-2
assert rel < 1e-2, f"数学不等价: {rel}"
print("PASS: scheme apply == x @ w_hat^T (bf16 dequant)")

# 顺便验证 process_weights_after_loading 后 dequant 确实是 W·s 近似
deq = layer.awq_dequant_weight
print(f"dequant ~ W·s 相对误差: {(deq - w_s).abs().max().item():.3e}")
