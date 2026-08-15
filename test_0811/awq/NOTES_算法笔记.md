# AWQ 算法笔记（Stage 1 实现过程中的关键认知）

> 2026-08-12。手写 AWQ 时踩的坑 + 修正依据，全部来自对 AutoAWQ v0.2.9 源码的核对
> （参考文件在 [refs/](refs/)，从 PyPI 源码包提取）。

## 一句话：AWQ 在做什么
量化前先按"输入通道的激活强度"缩放权重 W·s，再 INT4 量化，推理时反量化后逐通道除回 s。
误差大但激活强的通道（salient channels）被保护，整体输出误差下降。

## 踩过的坑（重要）

### 坑 1：整组用同一个标量 s → scale-invariant，搜索无效
第一版实现：对每个 (输出行, 输入 group) 试标量候选 `s ∈ {0.5^0.5, 0.75^0.5, 0.875^0.5, 1.0}`，
把**整个 group 的所有通道**乘同一个 s。
结果：所有候选的重建误差完全相同（提升 0.0%）。

原因：对称 group 量化下
`round(w·s / (s·max|w|/7)) = round(w·7/max|w|)` —— s 在分子分母抵消。
只要缩放是"组内统一的标量"，量化决策就与 s 无关。

**正确做法**：scale 必须**逐输入通道**（每个通道的 s 不同），
这样 W·s 改变了组内的相对分布 → 组 max 变化 → 量化粒度改变 → 有区分度。

### 坑 2：AWQ 不是论文式逐通道贪心，而是"激活形状 + 单参数搜索"
AutoAWQ 生产实现（`quantizer.py get_best_scale`）：
```
x_mean = mean(|X|) 每输入通道激活均值          # [in]
for ratio in range(20):                        # 20 个候选
    ratio = ratio / 20
    s = clamp(x_mean^ratio, 1e-4)              # scale 形状 = 激活均值，指数待选
    s = s / sqrt(max(s)·min(s))                # 归一化
    w_hat = dequant(quant(W·s)) / s            # 量化 W·s 再逐通道除回
    loss  = MSE(x·W^T, x·w_hat^T)              # ★ 真实输出空间误差
    取 loss 最小的 ratio 对应的 s
```
- scale 形状完全由激活均值决定，只有一个指数 ratio 被搜索（0~1）。
- 损失是**量化后该层的真实输出 vs fp16 输出**的 L2，不是权重重建误差。
  这符合 AWQ 的最终目标：输出误差最小。
- `ratio=0` 时 s=1（即普通 INT4），`ratio→1` 时 s∝激活均值（保护激活大的通道）。

### 坑 3：weight clipping 必须在"已 scale 的权重 W·s"上搜索
第一版把 clip 搜索用在未 scale 的 W 上，再把阈值套到 W·s 上 → 误差暴涨（+430%）。
原因：AutoAWQ 的流程是 `apply_scale`（先把 W 乘 s）→ `search_best_clip`（在 W·s 上搜阈值）
→ `apply_clip`。clip 阈值相对的是缩放后的组 max，跨尺度不通用。
修正后 clip 在 `W·s` 上搜索。

### 坑 4：clip 搜索的参考输出必须用缩放后的激活 X/s
只改权重还不够：AutoAWQ 的 `apply_scale` 同时做了 `inp.div_(scales)`（激活 ÷s），
`search_best_clip` 拿到的输入特征是 `X/s`。这样 `(X/s)·(W·s)^T = X·W^T`，
clip 搜索优化的正是"裁剪量化后 vs 真实 fp16 输出"的误差。
如果像我第一版那样传原始 X，参考输出变成 `X·(W·s)^T`，搜索目标和最终度量不一致，
导致个别层（实测 layer 35 down_proj）clip 反而略差。
修正后（传 X/s），clip 搜索的每 (行, 组) 独立代理误差与 AutoAWQ 一致。

### 已知近似（AutoAWQ 自身也如此，非 bug）
clip 搜索按 (输出行, 组) 独立最小化"该组输出贡献"误差 `||x·Δw_g||²`，
但真实总误差 `||Σ_g x·Δw_g||²` 含组间交叉项 `2Σ_{g<g'}⟨x·Δw_g, x·Δw_g'⟩`。
对大多数层交叉项可忽略 → clip 恒优于 scale（实测 251/252 层）；
对个别病态层（实测 layer 35 down_proj，深层 MLP 大 outlier）代理指标与真实总误差
不完全一致，clip 从 94.4% 降到 91.6%（仍远优于普通 INT4）。
如需严格保证，可在最后加一道"用真实总误差验证，clip 更差则回退 scale"的保险。

## 与"经典 AWQ 论文"的差异（面试可讲）
- 论文（arXiv 2306.00978）描述的是逐通道 grid search + 权重裁剪（clipping）。
- AutoAWQ 实际用激活形状的指数搜索（更快更稳），并保留 weight clipping
  （`_compute_best_clip`，搜索每组的裁剪阈值，refs/autoawq_quantizer.py:496）。
- 我们的 Stage 1 先做 scale 部分；clipping 作为后续增强（Stage 1.5）。

## 参考坐标（AutoAWQ v0.2.9）
| 函数 | 位置（refs/） | 作用 |
|---|---|---|
| `pseudo_quantize_tensor` | autoawq_quantizer.py:74 | 对称 INT4：scale=max|w|/7，范围[-8,7] |
| `get_best_scale` | autoawq_quantizer.py:375 | 激活感知 scale 搜索（核心） |
| `_compute_loss` | autoawq_quantizer.py:444 | 输出空间 L2 |
| `_compute_best_clip` | autoawq_quantizer.py:496 | weight clipping |
| `apply_scale` | autoawq_scale.py:37 | 应用 scale：权重×s，激活/÷s |

## 量化格式细节（Stage 2 复用）
- 对称 INT4，per-(输出行, 输入 group) scale，group_size=128，无 zero_point（默认）。
- 量化范围 clamp 到 [-8, 7]。
- 推理时把 `/s` 折进 dequant（`dequant(Q(W·s))/s`），或把 `1/s` 折进上一层激活
  （AutoAWQ 两种都做：GEMM 版折进权重，融合核版折进激活）。
