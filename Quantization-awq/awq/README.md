# 手写 AWQ 量化工具（Stage 1：算法 MVP）

> 目标：不依赖社区 AWQ 权重，也不依赖 GPU 工具链（AutoAWQ 需要 CUDA），
> 在 Ascend NPU 上用纯 torch 从零实现 AWQ（Activation-aware Weight Quantization）。

## 为什么自己写
- vLLM 核心的 `--quantization awq` 在 Ascend 上不可用：`ops.awq_gemm`/`ops.awq_dequantize` 都是 CUDA op（vllm/awq.py:283）。
- AutoAWQ 校准需要 GPU，本机只有 NPU。
- msmodelslim 的 `anti_method="m3"`（=AWQ）支持存在但未被本机验证，且它是"黑盒"，
  自己做一遍能真正理解 AWQ 原理，面试可讲清楚。

## AWQ 算法（一句话）
**不改变激活值，只缩放权重再量化**（对齐 AutoAWQ v0.2.9）：
1. 校准：收集每一层 Linear 的输入激活 X（喂少量样本前向）。
2. 激活感知 scale 搜索：逐输入通道 `s = x_mean^ratio`（x_mean=该通道激活均值），
   在 20 个 ratio 网格上找让"量化后该层真实输出 vs fp16"误差最小的那个（AutoAWQ get_best_scale）。
3. weight clipping：在已 scale 的 W·s 上，搜索每组最优裁剪阈值，裁掉 outlier 权重
   （AutoAWQ _compute_best_clip）。
4. INT4 group 量化：`Q(clip(W·s))` 按 group_size=128 对称量化（范围 [-8,7]）。
5. 推理时 dequant 后除以 s：`W_hat = dequant(Q(W·s))/s`，scale 折进权重重建。

## 目录
- `awq_core.py` — 算法核心（校准收集 + scale 搜索 + INT4 打包/解包）
- `run_awq_mvp.py` — MVP 驱动：加载 Qwen3-VL-8B，量化前几层，验证误差
- `RESULTS.md` — 各次运行结果存档

## 验证逻辑（MVP 标准）
对同一层同一批激活（误差 = 量化后该层真实输出的 L2）：
- 普通 INT4（不做 AWQ scale）的输出误差
- AWQ scale 的输出误差
- AWQ scale + clip 的输出误差
- 期望：**AWQ < 普通 INT4**，且整模型 logits 相似度（FP16 vs 全量化）> 0.99
- 实测：252 个 Linear 全部提升（平均 31%），36 层全量化后 logits 余弦 0.9985
