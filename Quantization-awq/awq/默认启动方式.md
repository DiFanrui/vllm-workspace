# Qwen3-VL-30B-A3B 双卡部署 + MoE + 量化 配套项目

> 日期：2026-08-11
> 环境：2 × Ascend 910B2C（64GB HBM ×2 = 128GB），CANN 9.0.0，vllm 0.21.0 + vllm-ascend 0.21.0rc1

## 一句话方案

选型 **Qwen3-VL-30B-A3B-Instruct**（多模态 MoE，30B 总参 / 3B 激活，bf16 62.1GB），用 **2 卡 tensor-parallel（TP=2）** 部署，覆盖 **多模态（图像/视频）**、**MoE 特性（稀疏激活 + EPLB 专家负载均衡）**、**量化（W8A8 → AWQ/FP8，量化前后对比）** 三个完整链路。

## 为什么是 Qwen3-VL-30B-A3B

| 需求 | 满足情况 |
|---|---|
| 多模态 | ✅ 图像/视频/文本，Qwen3-VL 全家系 |
| MoE | ✅ 30B 总参 3B 激活（A3B = All-3B-active），稀疏路由 |
| 双卡部署 | ✅ 62.1GB 权重 TP=2 后每卡 31GB，余量充足 |
| 量化 | ✅ W8A8/FP8/AWQ(W4A16)/MXFP4 均支持 |
| vllm-ascend 官方支持 | ✅ CI accuracy-group-3 实测 + 专属 patch |

Qwen3-VL-235B-A22B（"max"型号，470GB bf16）**无法在 2 张卡上运行**：FP8 量化后 ~235GB、AWQ INT4 后 ~118GB 仍大于 128GB 且无 KV cache 余量，需 4 卡以上。详见 [01_调研报告.md](01_调研报告.md)。

## 项目结构

```
test_0811/
├── README.md          # 本文件：方案总览
├── 01_调研报告.md       # 模型选型 / 显存核算 / 量化 / 磁盘 调研
├── 02_实施计划.md       # 分阶段实施步骤（下载 → 部署 → MoE → 量化 → 验证）
└── scripts/           # 执行脚本（阶段推进时填充）
```

## 已确认的决策

| 决策项 | 结论 |
|---|---|
| 模型选型 | Qwen3-VL-30B-A3B-Instruct（modelscope / HF 均有，13 分片 62.1GB） |
| 235B max | 不可行，2 卡装不下（470GB bf16） |
| 磁盘 | 删除纯文本 Qwen3-30B-A3B-Instruct-2507（57GB）腾空间，用户已批准 |
| 量化路线 | W8A8（msmodelslim 自做或现成）→ AWQ（现成 QuantTrio 权重 或 msmodelslim）→ FP8（官方 Thinking-FP8） |
| 执行节奏 | 先方案文档（本文件），审阅后逐步执行 |

## 关键启动命令（预览）

```bash
# 双卡 TP=2
ASCEND_RT_VISIBLE_DEVICES=0,1 vllm serve /data/models/Qwen3-VL-30B-A3B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 1}' --trust-remote-code

# 量化版（AWQ 示例）
ASCEND_RT_VISIBLE_DEVICES=0,1 vllm serve /data/models/Qwen3-VL-30B-A3B-Instruct-AWQ \
  --quantization awq --tensor-parallel-size 2 ...
```

> 注意：`--limit-mm-per-prompt` 必须传 JSON 字符串（`'{"image": 1}'`），vLLM 0.21 用 `json.loads` 解析，写成 `image=1` 会启动即报错。
