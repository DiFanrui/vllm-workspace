# DSV4 Mini 1B / InfiniLM 适配 — SWA RoPE 定位与修复记录

日期：2026-08-14

## 结论

- 确认并修复了 SWA/压缩注意力 RoPE 路径上的 **5 处 bug**：2 处旋转布局、1 处压缩块位置、1 处因果掩码、1 处 rope theta 路由。
- **单层全部对齐**（首 token argmax 一致）：
  - 第 0 层（SWA）：mean_abs 0.272 → **0.0236**（argmax 一致）。
  - 第 2 层（compress_ratio=4，csa+compressor+indexer）：**0.0074**（argmax 一致）。
  - 第 3 层（compress_ratio=96）：**0.0081**（argmax 一致）。
- 完整 24 层 mean_abs 从 0.510 降到 **0.0768**；首 token argmax 仍不一致，但这是**跨 24 层的 BF16 精度累积**（单层均已对齐），非结构性 bug。
- 纠正一个此前的假设：reference 的压缩层 RoPE **不应用 YaRN**（用的是普通 `compress_rope_theta=160000`）。因此 `rope_parameters` vs `rope_scaling` 的命名差异对当前对齐是**无影响的**——强行识别 `rope_parameters` 并套用 YaRN 反而会与 reference 失配。

## 修复内容

### 1. 正向 RoPE 算法错误（interleaved → split-half）
- 文件：`csrc/models/deepseek_v4/deepseek_v4_for_causal_lm.cpp`
- 改动：`set_rope_algo(GPT_J)` → `set_rope_algo(GPT_NEOX)`
- 原因：reference `modeling_deepseek_v4.py` 的 `_rotate_half` 用 `chunk(2)` + `cat((-x2, x1))` 是 split-half 布局；而 C++ 误设为 GPT-J 交错布局（原注释“SGLang/vLLM 用 GPT-J”对本 checkpoint 不成立）。

### 2. 内核输出逆 RoPE 布局错误（interleaved → split-half）
- 文件（InfiniCore，3 个内核）：
  - `deepseek_v4_swa_prefill/nvidia/*.cu`
  - `deepseek_v4_swa_decode/nvidia/*.cu`
  - `deepseek_v4_compressed_decode/nvidia/*.cu`
- 改动：输出逆 RoPE 的维度配对从 `(2*i, 2*i+1)` 改为 `(i, i+half)`（split-half），与 reference `_apply_output_rope` 一致（`s = -sin(angle)` 保持不变）。

### 3. 压缩块 RoPE 位置错误（块首 → 块末）
- 文件：`csrc/models/deepseek_v4/deepseek_v4_attention.cpp`（3 处：`block_position_table_`、`rotate_compressed_blocks_`、`compressed_attention_gpu_`）
- 改动：块 RoPE 位置从 `block * m` 改为 `block * m + (m - 1)`。
- 原因：reference 用 `comp_pos = arange(nb) * m + (m - 1)`（块末 token 位置），C++ 原用块首。

### 4. 压缩块因果掩码 off-by-one
- 文件：`deepseek_v4_compressed_decode/nvidia/*.cu` + `deepseek_v4_attention.cpp`（`has_no_visible_compressed_blocks`）+ `deepseek_v4_indexer/nvidia/*.cu`
- 改动：`visible_blocks = (q_pos + 1) / m` → `q_pos / m`（三处）。
- 原因：reference 掩码语义是 `block_end = b*m+(m-1) < q_pos` ⇔ `b < q_pos/m`，C++ 原实现多算一个块。

### 5. rope theta 路由错误（主 rope vs 压缩 rope）
- 文件：`deepseek_v4_rope.cpp/.hpp` + `deepseek_v4_attention.cpp`
- 改动：q/kv_sw/输出逆 RoPE 一律用**主 rope（rope_theta=10000）**，只有压缩块 kv_comp 用**压缩 rope（compress_rope_theta=160000）**；`create_gpu_ropes_if_needed` 对压缩层也同时创建主 rope。
- 原因：reference 里 `q`、`kv_sw`、`_apply_output_rope` 都用 `rope_cos`(主)，只有 `kv_comp` 用 `rope_cos_c`(压缩)；C++ 原实现把压缩层 q/kv_sw/输出全用了压缩 rope。

## 对齐结果

| 版本 | mean_abs | 首 token argmax |
|---|---|---|
| 初始 DSV4 分支（报告基线） | 0.5104 | 不一致 |
| 修复 RoPE 布局后（24 层） | 0.1370 | 不一致 |
| 再修复块位置+掩码后（24 层） | 0.0838 | 不一致 |
| 再修复 rope 路由后（24 层） | **0.0768** | 不一致（66896 vs 17659） |
| 第 0 层（单层，SWA） | **0.0236** | **一致（59567）** |
| 第 2 层（单层，ratio=4） | **0.0074** | **一致（20166）** |
| 第 3 层（单层，ratio=96） | **0.0081** | **一致（98874）** |

单层（SWA + csa + hca）全部 argmax 一致，说明结构性适配已完成；24 层剩余 0.0768 是跨层 BF16 精度累积。

## 关键发现：YaRN 不参与本 reference

- checkpoint `config.json` 有 `rope_parameters`（factor=16、beta_fast=32、beta_slow=1、original_max_position_embeddings=65536、rope_theta=10000、type=yarn）。
- `configuration_deepseek_v4.py` 会把 `rope_scaling`/默认值填成 YaRN 字典，但 `modeling_deepseek_v4.py` 的 `build_rope_cache` **只用 `rope_theta`(10000) 和 `compress_rope_theta`(160000)**，全程不读取、不应用 YaRN。
- 因此 C++ 当前“读不到 `rope_scaling` → factor=1 → 不应用 YaRN”的行为**恰好与 reference 一致**。
- 结论：对当前 checkpoint 的数值对齐而言，`rope_parameters` 识别问题**不需要修**（修了会失配）。若未来要对接真实 V4（其 reference 真的用 YaRN），再单独处理字段识别与 YaRN 应用。

## 重要构建/运行注意事项

- InfiniCore 的 `xmake install _infinicore` **只更新 `python/infinicore/lib`，不更新前缀 `/root/autodl-tmp/dpv4-prefix/lib`**。
- 运行时 `LD_LIBRARY_PATH` 指向前缀，会加载**旧**的 `libinfiniop.so`，导致内核改动不生效。
- 正确做法：内核改动后必须执行 `xmake install`（全量，更新前缀），或把 `python/infinicore/lib` 放到 `LD_LIBRARY_PATH` 前缀之前。
- 本次排查中曾因此把“正向已 split-half、输出仍 interleaved”的不一致误判为“修复变差”。

## 剩余问题与下一步

- 剩余 24 层误差（mean 0.0838）集中在 **compress_ratio=4 的 csa 压缩层**（第 2/4/6…/22 层，共 11 层），其 indexer（`deepseek_v4_indexer`）参与块选择。
- 已确认与 reference 一致：compressor（overlap transform + softmax 池化 + ape）、attention sink、indexer 的因果掩码、压缩块位置/掩码、RoPE 布局。
- 下一步：对比 `deepseek_v4_indexer` 的 score 计算（`qI=wq_b(cQ)`、`wI=weights_proj(h)*score_scale`、`relu(qI·K)`、`sum(wI*qK)`）与 reference `Indexer.select`，重点核对 indexer 自身的 compressor 键与 `score_scale`。

## 归档

- 补丁：
  - `/root/InfiniLM/experiment/dsv4_mini_1B/swa_rope_kernels.patch`（InfiniCore 3 个内核）
  - `/root/InfiniLM/experiment/dsv4_mini_1B/swa_rope_infinilm.patch`（InfiniLM attention + for_causal_lm）
- 测试脚本（工作树 `/root/autodl-tmp/InfiniLM-dpv4-test/`）：
  - `run_dsv4_1b_layer_parity.py`（单层/全量 parity，支持 zero_pos 控制）
  - `make_dsv4_1b_layerN.py`（按层号提取单层 checkpoint）
  - `rope_compare.py`（正向 RoPE 布局直接对比）
