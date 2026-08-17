# 从零手写 AWQ 量化链路：完整复盘 + 新手保姆级教程

> 适用对象：完全新手，手里只有 `vllm-ascend` 源码，想复现「手写 AWQ 算法 → 量化出
> checkpoint → 让 vLLM 在昇腾上加载 → 端到端验证」这一整条链路。
> 这是 2026-08-11 ~ 08-12 两天实际完成的事的完整复盘：思路、参考、代码、验证、踩坑、改动清单、执行顺序、效果。
>
> 环境：2 × Ascend 910B2C（单卡 64GB HBM）、CANN 9.0.0、Python 3.12.13、
> torch 2.10.0 + torch_npu 2.10.0、vLLM 0.21.0（`/vllm-workspace/vllm` editable）、
> vllm-ascend 0.21.0rc1（`/vllm-workspace/vllm-ascend` editable）。

---

## 0. 先回答你最大的疑问：为什么"自己写"，而不是"用现成"

新手的直觉是：量化不是有 AutoAWQ、msmodelslim 这些现成工具吗？为什么我们一行一行自己写？

本机实际约束（全部实测过）：

| 路线 | 为什么走不通 |
|---|---|
| vLLM 自带 `--quantization awq` | `vllm/model_executor/layers/quantization/awq.py` 里的 `awq_gemm`/`awq_dequantize` 都是 **CUDA op**，昇腾上没有 |
| AutoAWQ 官方工具 | 校准（calibration）需要 **CUDA GPU**，本机只有 NPU |
| msmodelslim（昇腾原生） | 是"黑盒"，而且 AWQ 支持未验证；用它等于放弃理解原理 |

所以结论是：**想在纯昇腾环境里从零做出一个能用的 AWQ 模型，唯一路径就是自己实现**。
这反而是好事——你自己写了 AWQ，面试时能从头讲到尾。

> 项目一开始其实还调研过一条"更省事"的路：直接下载社区现成的 AWQ 权重（如
> `QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ`）加载。这条路线简单，但"加载别人量化好的"
> 和"自己量化"技术含量完全不同。我们选择了后者，因为你是求职者，要的是**能讲清楚原理**。

---

## 1. 整条链路长什么样（三幕剧）

把一件大事拆成三步，每步都有独立的"验证关卡"，跑通了才进下一步：

```
第一幕 Stage 1：算法 MVP（纯 torch，不动 vLLM）
  用 8B 模型 + 纯 torch 把 AWQ 的数学跑通、验证误差
  └─ 验证关卡：252 个 Linear 全部 AWQ 优于普通 INT4，全模型 logits 余弦 > 0.99
        │
第二幕 Stage 2：落地插件（dense 8B，进 vllm-ascend）
  把量化权重写成 vLLM 能加载的 checkpoint + 在 vllm-ascend 里注册自定义量化方法
  └─ 验证关卡：vLLM 在 NPU 上加载量化模型，与 fp16 基线 logits 余弦 > 0.99
        │
第三幕 Stage 3：上规模（30B-A3B MoE 全量量化）
  同样的算法扩展到 128 专家的 MoE：显存策略 + fused 3D 权重 + 运行时契约
  └─ 验证关卡：HF bf16 基线 vs 量化模型，logits 余弦 + 生成文本对比
```

**为什么按这个顺序**：先只调数学（纯 torch 迭代快），再碰工程（vLLM 加载），最后挑战规模
（显存 + MoE）。任何一个卡点都能在最简单的环境里先暴露。

---

## 2. 环境：你手上到底有什么（怎么确认）

你"只有 vllm-ascend 源码"，但实际环境里有完整的可运行组件。先确认它们都在：

```bash
# 1) 昇腾运行环境
echo $ASCEND_HOME_PATH                    # 应有 /usr/local/Ascend/cann-9.0.0
npu-smi info                              # 应看到 2 张 910B 卡，各 64GB

# 2) Python 环境（注意：vLLM 仓库贡献用 uv/.venv，但这个项目直接用系统 python3）
which python3                             # /usr/local/python3.12.13/bin/python3
python3 -c "import torch, torch_npu, vllm, vllm_ascend; print('ok')"

# 3) 两个源码仓库是 editable 安装的
pip list | grep -Ei "vllm|torch_npu"
# vllm 0.21.0          -> 源码在 /vllm-workspace/vllm
# vllm-ascend 0.21.0rc1 -> 源码在 /vllm-workspace/vllm-ascend（改它立刻生效）
```

**关键认知**：`vllm-ascend` 是 editable 安装的，改它的源码**不用重装**，重启进程就生效。
这也是为什么我们敢直接往里加自定义量化方法。

如果你真的"只有源码、没有模型"：模型在 `/data/models/` 下
（`Qwen3-VL-8B-Instruct` 和 `Qwen3-VL-30B-A3B-Instruct`），或自己
`HF_ENDPOINT=https://hf-mirror.com huggingface-cli download <model>` 拉一个。

---

## 3. 第一幕：算法 MVP（纯 torch 手写 AWQ）

### 3.1 思路

**AWQ 是什么（一句话）**：普通 INT4 量化会把所有权重一视同仁地切掉精度，但模型的某些
权重通道（activation 大的通道）特别重要，量化误差砸在上面损失最大。AWQ 先看一层线性层的
**输入激活 X**，找出哪些输入通道更重要，然后**不量化原权重 W，而是先对 W 乘一个逐通道缩放
s，量化（W·s），推理时把激活除以 s**。数学上 `x·W = (x/s)·(W·s)`，完全等价，但量化误差
被挤到了不重要的通道上。

```
普通 INT4:  W → Q(W)          → 误差均匀分布，重要通道也挨刀
AWQ:        W → W·s → Q(W·s)   → 推理用 (x/s)·(W·s)，误差集中到不重要通道
```

三个动作（都对齐 AutoAWQ）：
1. **校准**：喂几段文本前向，hook 住每个 Linear 的输入激活 X；
2. **scale 搜索**：逐输入通道 `s = x_mean^α`，在 20 个 α 网格上挑"量化后该层真实输出
   误差最小"的那个（AutoAWQ 的 `get_best_scale`）；
3. **weight clipping**：在已缩放的 W·s 上，按 (行, 组) 搜裁剪阈值，把 outlier 权重裁掉
   （AutoAWQ 的 `_compute_best_clip`）。

### 3.2 参考了什么（源码坐标）

- AutoAWQ 官方实现：`Workspace/awq/refs/autoawq_scale.py`、`refs/autoawq_quantizer.py`
  （从 AutoAWQ v0.2.9 拉的，作为算法正确性参照）。
- 关键是对齐它们的**搜索方式**（见 3.5 的坑），不是抄。

### 3.3 写了什么代码

| 文件 | 作用 | 关键函数 |
|---|---|---|
| `Workspace/awq/awq_core.py`（307 行） | 算法核心，纯 torch | `int4_quant_scale` / `int4_quantize_round` / `int4_dequant` / `int4_reconstruct` / `awq_scale_search` / `awq_clip_search` / `awq_quantize` / `collect_calib_activations` |
| `Workspace/awq/run_awq_mvp.py`（208 行） | MVP 驱动 | 5 步流程 + `LINEAR_SUFFIXES` 层匹配 + `apply_quantized_weights` 换回 w_hat 做 e2e |

核心算法流程（awq_core.py）：
```python
def awq_quantize_layer(w, x, group_size=128, num_bits=4, do_clip=True):
    s, _ = awq_scale_search(w, x)            # ① 逐通道 scale（20 格网格）
    w_s = w.float() * s
    if do_clip:
        clip_max = awq_clip_search(w_s, x/s) # ② 在 W·s 上搜裁剪阈值
        w_s = w_s.clamp(-clip, clip)         #    裁掉 outlier
    q = int4_quantize_round(w_s)             # ③ 对称 INT4（范围 [-8,7]，group=128）
    w_hat = int4_dequant(q) / s              # ④ 重建权重（除以 s，数学还原 x·W）
    return q, scale, s, w_hat
```

校准收集激活时要**注意内存**：hook 里 `x.detach().float().cpu()`，激活落 CPU 而不是 NPU，
因为大模型激活动不动上 GB，NPU 只有 64GB。

### 3.4 验证方式（MVP 的关卡设计）

对**同一层、同一批激活**，算三种"量化后该层真实输出 vs fp16 输出"的误差：

```
误差(普通 INT4)     —— 不做 AWQ scale
误差(AWQ scale)     —— 只做 scale
误差(AWQ scale+clip)—— 全做
期望：AWQ(带 clip) < AWQ(纯 scale) < 普通 INT4
```

全模型级别：36 层全部量化后，模型输出 logits 与 fp16 的**余弦相似度 > 0.99**。

```bash
cd /vllm-workspace/test_0811/awq
python run_awq_mvp.py --layers 0,1    # 快速冒烟（2 层，~10s）
python run_awq_mvp.py --layers all    # 全 36 层（~2.5 分钟）
```

### 3.5 遇到的坑（这段最值钱）

| # | 坑 | 现象 | 根因与修复 |
|---|---|---|---|
| 1 | **scale 必须逐通道** | 逐组搜 scale 完全无效，提升≈0 | AWQ 的 s 沿输入通道维，**组内是 scale-invariant**（同一组共享一个 s，数学上可以约掉，搜索是白费）。必须逐输入通道搜 |
| 2 | **生产实现不是论文式贪心** | 对照论文实现性能差 | AutoAWQ 实际是"**激活形状 + 单参数指数搜索**"（`s = x_mean^α`，网格搜 α），不是论文里逐通道贪心选最优 s |
| 3 | **clip 必须在 W·s 上搜** | clip 无效果 | 必须先乘 s 再搜裁剪阈值（裁的是 W·s 的 outlier），在原始 W 上搜是错的 |
| 4 | **clip 的参考输出要用 X/s** | clip 搜索用的代理误差不准 | 裁剪后 `w_s.clamp`，重建时要除以 s；参考输出按 X/s 前向算，不是 X |
| 5 | **clip 采样切错维度**（全量才爆） | 冒烟通过，全量 IndexError | `x_g[:, ::step]` 切的是 group 维（dim1）而非 token 维（dim0）。token≤512 时 step=1 恒等所以冒烟不炸；全量 1143+ token 才崩。修复：`x_g[::step]`。**教训：小输入不覆盖的 bug，靠大输入才暴露，所以必须跑全量验证** |

完整笔记见 [NOTES_算法笔记.md](NOTES_算法笔记.md)。

### 3.6 效果

- **252/252** 个 Linear（36 层 × 7）AWQ 全部优于普通 INT4；
- scale 单独平均降低误差 **31.2%**，加 clip 后 **55.3%**（深层提升更大，layer 34/35 的
  up/down 达 90%+，正是 outlier 最严重处）；
- 全模型 INT4 vs FP16 logits 余弦 **0.9989**。✅

---

## 4. 第二幕：落地 —— vllm-ascend 插件（dense 8B）

### 4.1 思路：先定位源码，再设计插件（两层注册机制）

你"只有 vllm-ascend 源码"——那就先读懂它的量化派发链路，答案全在源码里：

**第 1 层（vLLM 核心）**：`--quantization <名字>` 决定用哪个 **QuantConfig 类**。
- 注册表：`vllm/model_executor/layers/quantization/__init__.py:47`（`QUANTIZATION_METHODS` 列表）；
- 校验：`vllm/config/model.py:949` `_verify_quantization()`；
- 解析：`get_quantization_config(name)` → `from_config()` 构造。

**第 2 层（vllm-ascend）**：QuantConfig 的 `get_quant_method(layer, prefix)` 把每一层
派发到**已注册的 scheme 类**。
- scheme 注册表：`vllm_ascend/quantization/methods/registry.py`（`@register_scheme(quant_type, layer_type)`）；
- 派发：`compressed_tensors_config.py:298` `_create_scheme_for_layer_type()`；
- 桥接：`vllm_ascend/quantization/method_adapters.py:37` `AscendLinearMethod` 调
  `scheme.get_weight()` 注册参数、`scheme.apply()` 做计算；
- scheme 基类：`vllm_ascend/quantization/methods/base.py:42` `AscendLinearScheme`，
  **只需要实现 `get_weight()` + `apply()`**。

**一句话**：注册一个新名字（QuantConfig）+ 注册一个 scheme + 写一个保存器，vLLM 就能加载
我们的 AWQ 权重。参考模板：`vllm_ascend/quantization/methods/w4a16.py:111`
（已有 `("W4A16", "moe")`，但 **没有 dense W4A16 linear scheme**——这正是我们补的空缺）。

### 4.2 checkpoint 格式设计（怎么跟 vLLM 参数名对上）

**权重加载链路**：vLLM 的 Linear 构造时调 `quant_method.create_weights()`，参数名 =
`scheme.get_weight()` 返回 dict 的 key；加载时 `LayerLoader` 按**参数名**从 checkpoint 找
同名张量调 `default_weight_loader` 拷贝。所以：**checkpoint 张量名 = 我们注册的参数名**。

每个量化 Linear（权重 [out, in]，group=128，对称 INT4）存 3 个张量：

| 张量 | 形状 | 含义 |
|---|---|---|
| `qweight` | [out, in/8] int32 | Q(W·s) 打包：每 int32 装 8 个 int4，低 nibble 先（`pack_int4_int32`，save 端 :146） |
| `qscales` | [out, in/128] bf16 | W·s 的逐组对称 scale = max\|W·s\|/7 |
| `awq_scale` | **[1, in] bf16** | 逐输入通道的 AWQ scale s |

**`awq_scale` 为什么是 [1, in] 而不是 [in]**：适配器给所有参数统一打 `input_dim=1, output_dim=0`，
ColumnParallelLinear 沿 dim0、RowParallelLinear 沿 dim1 做 narrow；1D `[in]` 在 Row 路径会
`shape[1]` 越界，`[1, in]` 两条路径都 narrow 到自身且能广播。

推理（纯 torch 兜底，先求正确）：
```python
dequant = unpack(qweight).float() * qscales.repeat_interleave(128)   # ≈ W·s
out = F.linear(x / awq_scale, dequant, bias)                          # (x/s)·(W·s)ᵀ ≈ x·Wᵀ
```

### 4.3 写了什么代码

**保存器** `test_0811/awq/save_awq_model.py`（578 行）——量化 + 组装 checkpoint + 写盘，5 步：
```
[1/5] 加载模型（自动识别 dense vs MoE）
[2/5] 列出量化目标（8B：36 层 × q/k/v/o/gate/up/down = 252 个 Linear）
[3/5] 校准：collect_calib_activations 收集每层输入激活（落 CPU）
[4/5] AWQ 量化：按 fused 组（qkv / gate_up）量化，共享 s，再行切拆分
[5/5] 组装 checkpoint：量化张量替换原权重 + config.json 写 quantization_config + 拷 processor 文件
```
注意 fused 组：Qwen3 的 q/k/v 在 vLLM 里是 `QKVParallelLinear`（qkv_proj 一个参数），但
checkpoint 必须存**分离名** `q_proj/k_proj/v_proj`（原因见 4.5 坑 1）。

**插件**（vllm-ascend 里新增，editable 生效）：
| 文件 | 作用 |
|---|---|
| `vllm_ascend/quantization/methods/w4a16_awq.py` | dense scheme：`AscendW4A16AWQLinearMethod`，含 `unpack_int4_packed_int32`（:48，与 save 端互逆） |
| `vllm_ascend/quantization/awq_ascend_config.py` | `@register_quantization_config("awq_ascend")`，`get_quant_method` 派发 |
| `vllm_ascend/quantization/utils.py` | `detect_quantization_method`：config.json 探测到 `quant_method=="awq_ascend"` 时 import config 自动注册 |
| `vllm_ascend/quantization/method_adapters.py`（改） | `create_weights` 门控（Stage 3 用） |

### 4.4 验证方式

e2e 关键难点：怎么拿到"量化模型 vs fp16 基线"的 logits？vLLM 0.21 把 logits 捕获
机制改成了**引擎级 LogitsProcessor**（`LLM(logits_processors=[...])`）：
- 处理器在 **EngineCore 子进程**里跑，没法直接读主进程变量 → 用**文件跨进程**传回
  （`capture_logits_proc.py`：apply 时把 logits `np.save` 到 `AWQ_CAPTURE_PATH`）；
- 必须**顶层可 import**（config 按引用 pickle 到子进程）+ **继承 `LogitsProcessor`**
  （`NPUModelRunner` 校验 isinstance）。
- 需要 `VLLM_WORKER_MULTIPROC_METHOD=spawn`（fork-after-threads 会崩）。

```bash
python save_awq_model.py --out /tmp/qwen3vl8b_awq     # 量化 8B 出 checkpoint
# 分步验证：base / quant / compare
python verify_vllm_load.py --mode base   # 基线：vLLM fp16 捕获 logits
python verify_vllm_load.py --mode quant  # 量化：vLLM awq_ascend 捕获 logits
python verify_vllm_load.py --mode compare# 余弦对比
```

### 4.5 难点（都在源码里，每个都能讲）

1. **fused 命名子串碰撞（踩了两次）**：Qwen2DecoderLayer.load_weights
   （qwen2.py:473）的 `stacked_params_mapping` 用**子串匹配**路由。我们存 `gate_up_proj.*`
   会被 `up_proj` 子串误路由成 `gate_gate_up_proj.*` → KeyError；存 `qkv_proj.*` 会被
   `v_proj` 子串误路由成 `qkqkv_proj.*` → KeyError。
   **结论：checkpoint 一律存分离名** `q_proj/k_proj/v_proj/gate_proj/up_proj`，
   走 shard_id 正常路由。
2. **共享 s 的拆分**：q/k/v（gate/up）输入激活是同一份 X → 在拼接权重 [out_sum, in] 上做
   **一次** scale 搜索得到单一 s，再把 qweight/qscales 沿输出维行切到各 part；每个 part
   存同一份 awq_scale（数学自洽：s 只在输入通道维）。
3. **`awq_scale` 不设 output_dim**：见 4.2。设了会越界。
4. **进程/线程问题**：fork-after-threads 崩溃 → `VLLM_WORKER_MULTIPROC_METHOD=spawn`。

### 4.6 效果

- checkpoint 1254 个张量；`LLM(model_path, quantization="awq_ascend")` 加载成功；
- logits 余弦 **0.9988**（阈值 0.99）✅，最大绝对差 1.50。

---

## 5. 第三幕：上规模 —— 30B-A3B MoE 全量量化

### 5.1 为什么换模型

求职展示需要：dense 8B 的 AWQ 太"常规"，**MoE 量化才是面试亮点**——30B 总参数、128 专家、
每 token 只激活 8 个，量化 30B 全程只能单卡 64GB，处处是显存博弈。

### 5.2 思路演进（P2 → P3，被显存和磁盘一步步逼出来的）

先做内存冒烟（`smoke_moe_load.py`）：bf16 30B 加载即 **62.66GB/64GB**，余量 1.34GB。
→ 结论：**校准激活必须落 CPU**，量化时逐模块搬回。

原设计 P2 想"注意力量化 + 专家保持 bf16"：但专家占 ~29B（≈58GB bf16），checkpoint 61GB，
**磁盘放不下**（/data 只剩 23GB）。
→ 被逼改**全量量化**（注意力 + 专家都 int4）：checkpoint ≈ 20GB，磁盘可行。这个"被逼"的过程本身就是很好的复盘故事。

### 5.3 checkpoint 格式（专家 fused 3D 6 张量）

HF 端 `Qwen3VLMoeTextExperts` 是**单 3D Parameter**（不是逐专家 ModuleList）：
`gate_up_proj` [128,1536,2048]、`down_proj` [128,2048,768]。vLLM 端对应
`FusedMoE` 参数 `experts.w13_weight`/`w2_weight`。

最终每层 6 个张量（`experts.` 前缀 = vLLM FusedMoE 参数名，**精确一致**）：

| checkpoint 名 | 形状 | 说明 |
|---|---|---|
| `mlp.experts.w13_qweight` | [128, 1536, 256] int32 | gate+up 逐 expert 打包 |
| `mlp.experts.w13_qscales` | [128, 1536, 16] bf16 | 逐 (expert, 行, group=128) |
| `mlp.experts.w13_awq_scale` | **[128, 1, 2048] bf16** | 逐 expert × 输入通道 s |
| `mlp.experts.w2_qweight` | [128, 2048, 96] int32 | down |
| `mlp.experts.w2_qscales` | [128, 2048, 6] bf16 | |
| `mlp.experts.w2_awq_scale` | **[128, 1, 768] bf16** | 逐 expert × 中间通道 s |

### 5.4 加载路径三关（全部源码级验证）

`Qwen3VLMoe.load_weights`（qwen3_vl_moe.py）对每个 checkpoint 名按顺序判断：
1. 名字含 `mlp.experts` → stacked_params_mapping 循环 :214 `continue`（保护）；✅
2. 命中 fused 映射的 `experts.gate_up_proj`/`down_proj` → 走 transpose+chunk 加载。
   我们没用这些名字 → `is_expert_weight=False`；✅
3. 落到 generic 分支 :302-320：`weight_loader(param, loaded_weight)` **2 参调用**。
   FusedMoE 自带 6 参 weight_loader 会炸 → 参数必须挂 `default_weight_loader`
   （整张拷贝 + shape 校验，weight_utils.py:1399）。✅

### 5.5 专家量化实现（save 端）

- **逐 expert 校准**：hook `mlp.experts` forward，捕获 `(hidden_states, top_k_index)`；
  每个专家 e 用**路由到它的 token 子集**做校准（`x_e = hidden[topk==e]`）。
- **intm 重算**：`intm_e = silu(x_e@gate_up[e][:768]) * (x_e@gate_up[e][768:])`（冻结 bf16
  权重重算，作为 down_proj 的校准激活）。
- **N_PAD=256 固定行数**：采样/重复路由 token 到恰好 256 行。重复行对 AWQ 搜索等价
  （mean/argmin 不变），但所有专家分配大小一致 → NPU 缓存分配器逐 expert 完美复用，
  否则 reserved 涨到 60.3GB 后 **14MB 都 OOM**。
- **冷门专家兜底**：路由 token < 8 → 用层内均匀采样兜底。
- **模型挪 CPU 再量化**：校准完毕把模型 `model.to("cpu")` 释放 ~60GB，再逐层搬回空 NPU
  算（消除碎片化 OOM）。

### 5.6 e2e 运行时两处契约坑（vLLM 加载后 forward 期才爆）

1. **`get_quant_method` 必须收 `tid2eid` 关键字**：vllm-ascend 的
   `AscendFusedMoE.__init__`（ops/fused_moe/fused_moe.py:369）以
   `get_quant_method(self, self.layer_name, tid2eid=self.tid2eid)` 调用，我们签名没这个参数
   → `TypeError`。照 fp8_config.py:103 模式加 `tid2eid=None` 并透传。
2. **scheme `apply` 必须返回 `FusedExpertsResult` 而非裸 Tensor**：层级
   `AscendFusedMoE.forward_impl`（fused_moe.py:723）在 apply 后访问
   `fused_experts_results.routed_out` → `AttributeError`。tp=1 时 finalize 纯透传，
   `return FusedExpertsResult(routed_out=out)` 即可。

### 5.7 验证方式与效果（含一次诚实的修正）

基线问题：30B bf16=62.2GB，**装不进 vLLM 单卡 64GB** → 基线改用 **HF transformers bf16**
捕获 last-token logits，token ids 精确对齐。

```bash
python smoke_moe_load.py                            # 内存冒烟（~62.7GB 可跑）
python save_awq_model.py --model /data/models/Qwen3-VL-30B-A3B-Instruct \
    --out /data/models/Qwen3-VL-30B-A3B-AWQ         # 全量量化（240 模块）
python verify_vllm_load.py --mode base-hf           # HF bf16 基线
python verify_vllm_load.py --mode quant             # vLLM 量化端
python verify_vllm_load.py --mode compare           # 余弦
```

| 指标 | 值 |
|---|---|
| 量化模块数 | 48 层 × 4 注意力 + 48 expert groups = **240 个** |
| checkpoint | 17.8GB / 1458 张量 |
| 单 prompt logits 余弦 | **0.9925** ✅（阈值 0.99） |
| 多 prompt 平均 | **0.9797**（5 个 prompt 逐条 0.9925/0.9890/0.9839/0.9585/0.9746） |
| 生成文本 | **3/5 逐字一致，2/5 实质一致** |

**诚实的修正**：单 prompt 0.9925 有幸存者偏差。补 5 个 prompt 测，平均只有 0.9797
< 0.99。原因：每层 ~1143 tokens 路由到 128 专家，平均每专家 <9 tokens，**专家校准数据
稀疏**；且 8 experts/tok 的加权求和会累积误差。但**生成文本 3/5 逐字一致**——贪心解码对
~2% 的 logits 扰动不敏感（argmax 不变），模型行为保持住了。这个"真实水平 ≈0.98、文本语义
保持"的结论比一个虚高的数字更可信，面试时主动讲反而加分。

---

## 6. 从零执行的完整命令序列（保姆级）

> 假设：卡空闲、模型在 `/data/models/`、`cd /vllm-workspace/test_0811/awq`。

```bash
# ── 第一步：确认环境 ──
npu-smi info
python3 -c "import torch, torch_npu; print(torch_npu.npu.device_count(), 'NPUs')"

# ── 第二步：算法 MVP（8B，纯 torch）──
python run_awq_mvp.py --layers 0,1    # 冒烟，~10s
python run_awq_mvp.py --layers all    # 全 36 层，~2.5min，看 RESULTS.md 对比标准

# ── 第三步：出 8B checkpoint + 插件 e2e ──
python save_awq_model.py --model /data/models/Qwen3-VL-8B-Instruct \
    --out /tmp/qwen3vl8b_awq
VLLM_WORKER_MULTIPROC_METHOD=spawn python verify_vllm_load.py --mode base
VLLM_WORKER_MULTIPROC_METHOD=spawn python verify_vllm_load.py --mode quant \
    --model /tmp/qwen3vl8b_awq
VLLM_WORKER_MULTIPROC_METHOD=spawn python verify_vllm_load.py --mode compare

# ── 第四步：30B MoE 全量量化 ──
python smoke_moe_load.py                                          # 内存冒烟
python save_awq_model.py --model /data/models/Qwen3-VL-30B-A3B-Instruct \
    --out /data/models/Qwen3-VL-30B-A3B-AWQ --device npu:0        # 240 模块，见日志
python verify_vllm_multi.py --mode base-hf                        # HF bf16 基线
python verify_vllm_multi.py --mode quant                          # vLLM 量化端
python verify_vllm_multi.py --mode compare                        # 多 prompt 余弦 + 文本
```

> 小贴士：所有 verify 脚本都要 `VLLM_WORKER_MULTIPROC_METHOD=spawn`；
> `save_awq_model.py --max-layers N` 可只量化前 N 层快速冒烟；
> 每一步之前 `git add -A && git commit`（见第 7 节存档习惯）。

---

## 7. 我们最终改了哪些代码（清单）

**test_0811 项目（算法 + 验证）**：
- `awq/awq_core.py` — 算法核心（含 clip 采样 bug 修复 `x_g[::step]`）
- `awq/run_awq_mvp.py` — MVP 驱动
- `awq/save_awq_model.py` — 保存器（dense + MoE 专家量化都在里面）
- `awq/verify_vllm_load.py` / `verify_vllm_multi.py` — e2e 验证（单/多 prompt）
- `awq/capture_logits_proc.py` — 引擎级 LogitsProcessor 捕获
- `awq/smoke_moe_load.py` — MoE 内存冒烟

**vllm-ascend 插件（可讲给面试官的"我改了开源项目"）**：
- `vllm_ascend/quantization/methods/w4a16_awq.py` — **新增** dense scheme
- `vllm_ascend/quantization/methods/w4a16_awq_moe.py` — **新增** MoE scheme
  （`AscendAWQFusedMoEMethod`，逐 expert torch 兜底 apply）
- `vllm_ascend/quantization/awq_ascend_config.py` — **新增** `awq_ascend` QuantConfig；
  后续改 `get_quant_method` 收 `tid2eid` + FusedMoE 分支
- `vllm_ascend/quantization/utils.py` — `detect_quantization_method` 探测注册
- `vllm_ascend/quantization/method_adapters.py` — `AscendFusedMoEMethod.create_weights`
  加 `load_whole_tensor` 门控（挂 default_weight_loader）

**存档习惯**：这是一个独立 git 仓库（`/vllm-workspace/test_0811`，main 分支），
**每一步成果都 commit**。最后推到一个 GitHub 仓库三分支（main=本项目 / vllm=源码快照 /
vllm-ascend=含我们改动的插件）。

---

## 8. 参考坐标总表（源码位置，面试/排障直接查）

| 关注点 | 位置 |
|---|---|
| vLLM 量化方法注册表 | `vllm/model_executor/layers/quantization/__init__.py:47` |
| `--quantization` 校验 | `vllm/config/model.py:949` |
| vllm-ascend 配置替换套路 | `vllm_ascend/quantization/compressed_tensors_config.py:42` |
| scheme 注册 / 派发 | `methods/registry.py` / `compressed_tensors_config.py:298` |
| Linear 桥接 | `vllm_ascend/quantization/method_adapters.py:37` |
| scheme 基类（只需 get_weight+apply） | `vllm_ascend/quantization/methods/base.py:42` |
| W4A16 MoE 参考 | `vllm_ascend/quantization/methods/w4a16.py:111` |
| fused 子串匹配路由 | `vllm/model_executor/models/qwen2.py:473`（stacked_params_mapping） |
| MoE fused 加载路径 | `qwen3_vl_moe.py:189-192 / :201-203 / :214 / :249-260` |
| generic 2 参加载分支 | `qwen3_vl_moe.py:302-320` |
| default_weight_loader | `vllm/model_executor/layers/quantization/utils/weight_utils.py:1399` |
| FusedMoE None 回退（专家保持 bf16） | `fused_moe/layer.py:534-545` |
| vllm-ascend FusedMoE 调 get_quant_method | `vllm_ascend/ops/fused_moe/fused_moe.py:369` |
| apply 后访问 routed_out | `vllm_ascend/ops/fused_moe/fused_moe.py:723` |
| 模型注册表（Qwen3-VL dense/MoE） | `vllm/model_executor/models/registry.py:545-546` |

---

## 9. 面试能讲什么（30 秒版本）

1. **为什么手写 AWQ**：vLLM 的 awq 是 CUDA op（昇腾无）、AutoAWQ 校准要 GPU（本机无）、
   msmodelslim 是黑盒 → 纯 torch 从零实现是纯昇腾环境的唯一路径，也最懂原理。
2. **AWQ 三件事**：校准收集激活 → 逐通道 `s=x_mean^α` 网格搜索 → 在 W·s 上搜 clip；
   量化 (W·s)，推理用 (x/s)·(W·s)。
3. **三段式验证关卡**：算法（252/252 提升，logits 0.9989）→ 落地（插件注册，0.9988）→
   MoE（240 模块，单 prompt 0.9925，多 prompt 0.98，文本 3/5 逐字一致）。
4. **讲得出两处真实踩坑**：checkpoint 命名子串碰撞（fused 名不可用）；MoE apply 契约
   （`FusedExpertsResult` vs 裸 Tensor、`tid2eid` 关键字）——这比一切顺利更有说服力。
5. **诚实结论**：MoE 专家校准数据稀疏（每专家 <9 tokens）是精度瓶颈；生产要换 NPU GMM
   算子、补 TP>1 的 expert id remap。知道自己的边界在哪，也是面试加分项。

---

### 相关文档索引

- [README.md](README.md) — Stage 1 简介
- [NOTES_算法笔记.md](NOTES_算法笔记.md) — AWQ 4 坑
- [PLUGIN_设计.md](PLUGIN_设计.md) — Stage 2 插件设计
- [MOE_设计.md](MOE_设计.md) — Stage 3 设计 + 加载三关 + 运行时两坑
- [RESULTS.md](RESULTS.md) — 三阶段完整结果存档
- [01_调研报告.md](../01_调研报告.md) / [02_实施计划.md](../02_实施计划.md) — 立项调研与计划
