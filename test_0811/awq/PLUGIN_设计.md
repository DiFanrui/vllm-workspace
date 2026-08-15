# Stage 2 路 B：vllm-ascend 自定义量化插件 —— 机制定位 + 设计

> 2026-08-12。目标：让 vllm-ascend 能加载我们手写 AWQ 量化出的权重。
> 先定位源码链路（vLLM 核心 + vllm-ascend 两层注册机制），再设计插件。

## 一句话
vLLM 用 `--quantization <名字>` 找 QuantConfig；vllm-ascend 在 QuantConfig 的
`get_quant_method()` 里，把每一层派发到「已注册的 scheme」。我们只要
**注册一个新名字 + 注册一个 scheme + 写一个保存器**，就能让 vllm 加载我们的 AWQ 权重。

## 两层注册机制（源码定位）

### 第 1 层：vLLM 核心 `--quantization` → QuantConfig 类
- 注册表：`vllm/model_executor/layers/quantization/__init__.py:47`
  `QUANTIZATION_METHODS: list[str]`，`register_quantization_config()` 把新名字 append 进去。
- 校验：`vllm/config/model.py:949` `_verify_quantization()`
  - 显式 `--quantization xxx` 只要在 `QUANTIZATION_METHODS` 里即可。
  - 若模型 config.json 里有 `quantization_config.quant_method`，会逐个调已注册类的
    `override_quantization_method()` 探测。我们的方法名不进「overrides 列表」，
    因此会排在前面被探测。
- 解析：`get_quantization_config(name)` → 返回 `from_config()` 构造的 QuantConfig。

### 第 2 层：vllm-ascend `get_quant_method` → scheme 类
- 配置替换：`vllm_ascend/quantization/compressed_tensors_config.py:42-52` 演示了
  「删掉 vLLM 原方法 → `@register_quantization_config` 重注册」的套路
  （fp8_config.py、modelslim_config.py 同款）。
- scheme 注册表：`vllm_ascend/quantization/methods/registry.py`
  `@register_scheme(quant_type, layer_type)` → `_SCHEME_REGISTRY[(quant_type, layer_type)]`。
- 派发：`compressed_tensors_config.py:298-329` `_create_scheme_for_layer_type()`：
  `scheme_cls = get_scheme_class(quant_type, layer_type); return scheme_cls()`。
- 桥接：`vllm_ascend/quantization/method_adapters.py:37` `AscendLinearMethod(LinearMethodBase)`
  - `create_weights()` 调 `scheme.get_weight(...)`，把返回的 dict 注册成 layer 的参数；
  - `apply()` 调 `scheme.apply(layer, x, bias, tp_rank)`。
- scheme 基类：`vllm_ascend/quantization/methods/base.py:42` `AscendLinearScheme`，
  只需实现 `get_weight()` + `apply()`，可选 `process_weights_after_loading()`。
- 现有 W4A16 参考：`methods/w4a16.py:111` 只有 `("W4A16", "moe")`（MoE 专用），
  **没有 dense W4A16 linear scheme** —— 这正是我们要补的空缺。

### 权重加载链路（为什么注册好参数就能加载）
- vLLM 的 Linear（ColumnParallelLinear 等）构造时调 `quant_method.create_weights()`，
  参数名 = scheme.get_weight() 返回 dict 的 key。
- 模型加载时 `LayerLoader` 按**参数名**从 checkpoint 找同名张量，调 `default_weight_loader`
  （tp_size=1 时就是整张 copy 进 `param.data`）。
- 所以：**checkpoint 里的张量名 = 我们注册的参数名**，形状一致即可加载。

## 设计决定

### 方法名：自定义 `awq_ascend`（不覆盖 vLLM 核心的 `awq`）
- 避免破坏社区 AWQ 路径；我们控制整个格式，最稳。
- `--quantization awq_ascend` 显式指定即可（无需 config.json 探测）。

### 权重格式（每个量化模块存 3 个张量）
对每个量化 Linear（权重 [out, in]，group_size=128，对称 INT4，范围 [-8,7]）：

| 张量 | 形状 | 含义 |
|---|---|---|
| `qweight` | [out, in/8] int32 | Q(W·s) 打包：每 int32 装 8 个 int4（低4位起，-8~7 用 &0xF 存，解包时 ≥8 减 16） |
| `qscales` | [out, in/128] fp16 | W·s 的逐组量化 scale = max\|W·s\|/7 |
| `awq_scale` | **[1, in] fp16** | 逐输入通道的 AWQ scale s（激活感知缩放） |

> **awq_scale 为什么是 [1, in] 而不是 [in]**：AscendLinearMethod 适配器会给所有参数统一打上
> `input_dim=1, output_dim=0`。ColumnParallelLinear.weight_loader 沿 output_dim(0)、
> RowParallelLinear.weight_loader 沿 input_dim(1) 做 narrow；1D `[in]` 在 Row 路径会
> `shape[1]` 越界。`[1, in]` 两条路径都 narrow 到自身（no-op），且与 x 广播兼容。

推理（纯 torch 兜底，先求正确）：
```
dequant = unpack(qweight).float() * qscales.repeat_interleave(group_size)   # ≈ W·s, [out,in]
out = F.linear(x / awq_scale, dequant, bias)                                 # (x/s)·(W·s)^T ≈ x·W^T
```

### ⭐ 关键发现：Qwen3 的 fused 线性 + stacked_params_mapping 子串碰撞
- vLLM `Qwen3Attention` 用 **`QKVParallelLinear`（qkv_proj）**、`Qwen3MLP` 用
  **`MergedColumnParallelLinear`（gate_up_proj）**，权重在磁盘上是**合成一个** [q+k+v, in] 参数；
  o_proj / down_proj 是 RowParallelLinear（不融合）。
- 加载机制（源码定位）：`Qwen2DecoderLayer.load_weights`（qwen2.py:473）用
  `stacked_params_mapping = [("qkv_proj","q_proj","q"), ..., ("gate_up_proj","gate_proj",0),
  ("gate_up_proj","up_proj",1)]`，把 HF 分离的 `q_proj.*` 按 shard_id 路由进 fused 参数。
- **⚠ 子串碰撞（e2e 实测踩到两次）**：stacked 匹配是**子串判断** `if weight_name not in name`。
  我们连续踩了两个 fused 名：
  - `gate_up_proj.*` 含 `up_proj` 子串 → 误路由成 `gate_gate_up_proj.*` → KeyError；
  - `qkv_proj.*` 含 `v_proj` 子串（index 2-7）→ 误路由成 `qkqkv_proj.*` → KeyError。
  即：**fused 命名在 Qwen2DecoderLayer.load_weights 下全部不可用**。
- **结论（最终格式）**：checkpoint 必须存**分离的 HF 名**
  `q_proj/k_proj/v_proj/gate_proj/up_proj`（+ `o_proj/down_proj` 本来就是单层），
  走 stacked mapping 的 shard_id 正常路由（`QKVParallelLinear.weight_loader` 处理
  "q"/"k"/"v"，`MergedColumnParallelLinear.weight_loader` 处理 0/1，都沿 output_dim narrow）。
  每层 7 个子模块 → **7 个量化模块**。
- 共享 s：fused 组（qkv、gate+up）在拼接权重 [out_sum, in] 上做一次 scale 搜索得到
  **单一逐通道 s**（q/k/v、gate/up 输入激活相同，数学自洽）。gate/up 拆分时各自存同一份
  awq_scale（拆分前先算好共享 s，再行切 qweight/qscales）。
- `awq_scale` 参数属性：`input_dim=1`，**不设 output_dim**。这样 q/k/v、gate/up 各 shard
  加载时走 QKVParallel/Merged weight_loader 的 else 分支**整张拷贝**（幂等写入同一个
  [1, in]）；若设 output_dim=0，shard narrow(0, offset, size) 会在 [1, in] 上越界。

### 为什么存 `awq_scale` 而不是像 AutoAWQ 那样折进前一层权重
- AutoAWQ 社区格式把 `/s` 折进**前一层**权重（refs/autoawq_scale.py `scale_fc_fc`），
  checkpoint 里不存 s，省内存但要做跨层簿记、首层还要 ScaledActivation。
- 我们**同时控制保存器和插件**，逐层自包含地存 `s` 更简单、更不易错；
  数学等价（都是 `x·(dequant(Q(W·s))/s)^T`）。性能优化（真 int4 GEMM）留作后续。

### 部署形态（已定：加到 vllm-ascend 仓库本体）
- 代码已并入 vllm-ascend（editable 安装自 /vllm-workspace/vllm-ascend）：
  - `vllm_ascend/quantization/methods/w4a16_awq.py`：dense W4A16 scheme（解包 + dequant + apply）
  - `vllm_ascend/quantization/awq_ascend_config.py`：`@register_quantization_config("awq_ascend")`
  - `vllm_ascend/quantization/utils.py` `detect_quantization_method`：config.json 探测到
    `quant_method=="awq_ascend"` 时 import config 完成注册（自动生效）
- 卸载路径：模型卸载时 config.json 里写 `quantization_config.quant_method="awq_ascend"`，
  加载时 detect 到即用；显式 `--quantization awq_ascend` 也行。
- visual 塔保持 bf16：config.json `modules_to_not_convert=["visual"]`，
  `get_quant_method` 用 substr 匹配排除（返回 `UnquantizedLinearMethod()`，
  LinearBase 里返回 None 会 raise）。

## 实施步骤
1. `save_awq_model.py`：加载 Qwen3-VL-8B，对 252 个 Linear 跑 scale+clip（复用 awq_core），
   输出完整 checkpoint（未量化部分存 bf16/fp16 原权重），附带 `config.json`。
   注意：按 fused 模块（qkv/gate_up）输出，36 层 → 144 个量化模块。
2. 插件包：`awq_ascend_config.py`（QuantConfig）+ `scheme`（AscendLinearScheme）+ entry point。
3. 端到端验证：vLLM Python API（enforce_eager，tp=1）加载量化模型，
   与 FP16 基线对比 logits 余弦（沿用 MVP 的度量）。
4. 存档 + git commit。

## 参考坐标汇总
| 关注点 | 位置 |
|---|---|
| vLLM 量化方法注册表 | vllm/.../quantization/__init__.py:47 |
| `--quantization` 校验 | vllm/config/model.py:949 |
| vllm-ascend 配置替换套路 | compressed_tensors_config.py:42 |
| scheme 注册 | vllm_ascend/.../methods/registry.py |
| scheme 派发 | compressed_tensors_config.py:298 |
| Linear 桥接 | vllm_ascend/.../method_adapters.py:37 |
| scheme 基类 | vllm_ascend/.../methods/base.py:42 |
| W4A16 MoE 参考 | vllm_ascend/.../methods/w4a16.py:111 |
| 插件加载 | vllm/plugins/__init__.py:28 |
