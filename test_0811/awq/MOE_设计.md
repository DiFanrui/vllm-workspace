# Stage 3：Qwen3-VL-30B-A3B（MoE）量化 —— 源码定位 + P2 设计

> 2026-08-12。目标模型 `/data/models/Qwen3-VL-30B-A3B-Instruct`（58GB bf16）。
> 定位 vLLM MoE 加载链路 + HF transformers 5.5.4 模块结构，为 P2（注意力量化 + MoE bf16）
> 定 checkpoint 格式。P3（专家量化）在文末给源码坐标。

## 模型结构（config + 磁盘实测）

| 项 | 值 |
|---|---|
| 架构 | Qwen3VLMoeForConditionalGeneration（text 骨干 48 层） |
| hidden / heads / vocab | 2048 / 32 / 151936 |
| MoE | 128 experts，8 experts/tok，moe_intermediate=768 |
| shared_expert | **无**（shared_expert_intermediate_size=None） |
| 磁盘权重名 | `model.language_model.layers.{i}.mlp.experts.gate_up_proj` [128,1536,2048] 与 `...down_proj` [128,2048,768]，**fused 3D**（48×2=96 个）；注意力 q/k/v/o 全分离 |

## vLLM 加载链路（源码定位）

### 注意力：与 dense 8B 完全同构
`Qwen3VLMoe`（qwen3_vl_moe.py:165-172）`stacked_params_mapping` 与 Qwen3 相同：
q_proj/k_proj/v_proj → qkv_proj（QKVParallelLinear），gate_proj/up_proj → gate_up_proj。
→ checkpoint 存**分离名** q/k/v + o_proj，复用现有 awq_ascend dense scheme。✅

### MoE 专家：fused 3D 专用路径（关键区别）
- HF 端 `Qwen3VLMoeTextExperts`（modeling_qwen3_vl_moe.py:75）：`gate_up_proj` [128,1536,2048]、
  `down_proj` [128,2048,768] 是**单 3D Parameter**，不是逐专家 ModuleList。
- vLLM 端 `Qwen3VLMoe.load_weights`（qwen3_vl_moe.py:189-192）：
  ```
  fused_expert_params_mapping = [
      ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
      ("experts.w2_weight",  "experts.down_proj",   0, "w2"),
  ]
  ```
  名字里含 `experts.gate_up_proj`/`experts.down_proj` 即切到 fused 映射（:201-203）。
  加载时 transpose(-1,-2) 后 chunk(2, dim=-2) 拆 w1/w3，逐 expert 写进 w13_weight（:249-260）。
- `UnquantizedFusedMoEMethod.create_weights`（unquantized_fused_moe_method.py:83-126）
  注册 `w13_weight` [128,1536,2048]、`w2_weight` [128,2048,768]。与 mapping 一一对应。

### FusedMoE 的 None 回退（P2 能用的关键）
`FusedMoE.__init__._get_quant_method()`（fused_moe/layer.py:534-545）：
quant_config 的 get_quant_method 返回 **None 时自动用 UnquantizedFusedMoEMethod**。
我们的 awq_ascend_config.get_quant_method 对非 LinearBase（FusedMoE 是 PluggableLayer）返回 None
（awq_ascend_config.py:114-115）→ **专家自动保持 bf16，加载端零改动**。✅

## P2 设计（注意力量化 + MoE bf16）

### checkpoint 格式（与磁盘同名）
| 权重 | 名 | 形状 | 处理 |
|---|---|---|---|
| q/k/v_proj | `self_attn.{q,k,v}_proj.qweight/qscales/awq_scale` | 分离 | **量化**（复用 awq_core） |
| o_proj | `self_attn.o_proj.{qweight,qscales,awq_scale}` | — | **量化** |
| experts | `mlp.experts.gate_up_proj` / `down_proj` | [128,1536,2048] / [128,2048,768] | **原样 bf16** |
| gate（router） | `mlp.gate.weight` | [128,2048] | **原样 bf16** |
| visual / embed / norm / lm_head | 原样 | — | bf16 |

量化模块数：48 层 × 4 = **192 个**。

### ⚠ 必须把 `mlp.gate` 加进 modules_to_not_convert
vLLM 的 router 是 `ReplicatedLinear(quant_config=quant_config)`（qwen3_moe.py:179）——
是 **LinearBase**，我们的 scheme 会套到它头上；但 checkpoint 里 gate 是 bf16。
→ config.json 写 `"modules_to_not_convert": ["visual", "mlp.gate"]`
（config.get_quant_method 用子串匹配，`mlp.gate` 命中 `...mlp.gate`）。

### 专家 fused 名不会被子串碰撞误伤
`experts.gate_up_proj` 里其实含 `up_proj` 子串，但 stacked 循环先命中
`if "mlp.experts" in name: continue`（qwen3_vl_moe.py:214）保护，再落到 fused 映射。✅

## P2 实测：内存冒烟 + 磁盘约束（2026-08-12）
- 单卡加载 30B-A3B bf16：**max_memory_allocated=62.18GB，memory_reserved=62.66GB / 64GB**，
  余量仅 1.34GB → 校准激活必须落 CPU（awq_core.py hook 已改 `.cpu()`），量化时逐组搬回。
- **磁盘硬约束**：模型 31.1B 参数中专家占 ~29B（≈58GB bf16）。P2 原设计（专家 bf16）
  checkpoint ≈ 61GB > /data 剩 38GB、/vllm-workspace 剩 22GB → **装不下**。
  → **策略改为全量量化**（注意力 + 专家都 int4）：专家 int4 后 checkpoint ≈ 20GB，落盘可行。
- 方向：`experts.gate_up_proj`（fused 3D）与 `experts.down_proj`，量化后 checkpoint 名见下节。

## P3 源码坐标（专家量化）
- vLLM 专家参数：`experts.w13_weight`/`w2_weight`，weight_loader 见
  fused_moe/layer.py:1395 `load_weights` + :1412 的 3D fused 分支。
- vllm-ascend MoE scheme 模板：`vllm_ascend/quantization/methods/w4a16.py:111`
  `("W4A16", "moe")`；桥接 `AscendFusedMoEMethod`（method_adapters.py:208）。
- 逐 expert 校准激活 = 路由到该 expert 的 token 子集（HF 端 experts.forward 里
  `token_idx = torch.where(expert_mask[expert_idx])`，modeling_qwen3_vl_moe.py:103）。
- MoE 共享 s 策略待定：所有 expert 共享 vs 逐 expert 独立（默认逐 expert）。
- ⚠ qwen3_vl_moe.py:249-260 fused 加载路径对含 `experts.gate_up_proj` 的**任意**张量做
  transpose+chunk → 量化张量须避开该子串命名（见 save 脚本实现）。

## 专家量化加载设计（源码级验证，2026-08-12 定稿）
### 关键约束（踩坑推演）
1. **`register_parameter` 禁止带点参数名**（`KeyError: parameter name can't contain "."`）
   → per-expert mapping 生成的 `experts.w13_.qweight`（qwen2/qwen3_vl_moe 的
   make_expert_params_mapping，param_name=`experts.w13_`）物理不可行。
2. fused 名 `experts.gate_up_proj.*` 会触发 qwen3_vl_moe.py:249 的 transpose+chunk
   （为未量化 3D 权重设计），打包布局会被 transpose 打乱 → 也不能用。
3. generic 路径（qwen3_vl_moe.py:302-320）对未命中任何 mapping 的名做
   `weight_loader(param, loaded_weight)` **2 参调用**，而 FusedMoE.weight_loader 是 6 参
   → 参数必须挂 `default_weight_loader`（整张拷贝，tp=1 正确）。

### ✅ 最终 checkpoint 格式（fused 3D 直接参数名，2026-08-12 实测定稿）
每层 6 个张量（不是 128×3 分离！），`experts.` 前缀即 vLLM FusedMoE 参数名（已在
qwen3_moe.py:211 确认 `Qwen3MoeSparseMoeBlock.experts = FusedMoE`，方案注册在 FusedMoE
上 → 参数名与 checkpoint 键**精确一致**）：
| checkpoint 名 | 形状 | 说明 |
|---|---|---|
| `mlp.experts.w13_qweight` | [128, 1536, 256] int32 | gate+up 逐 expert 打包（[1536,2048]→in/8） |
| `mlp.experts.w13_qscales` | [128, 1536, 16] bf16 | 逐 (expert, 行, group=128) scale |
| `mlp.experts.w13_awq_scale` | **[128, 1, 2048] bf16** | 逐 **expert×** 输入通道 s（gate/up 共享） |
| `mlp.experts.w2_qweight` | [128, 2048, 96] int32 | down [2048,768]→in/8 |
| `mlp.experts.w2_qscales` | [128, 2048, 6] bf16 | |
| `mlp.experts.w2_awq_scale` | **[128, 1, 768] bf16** | 逐 expert× 中间通道 s |

> ⚠ 早期文档写 `[1, 2048]/[1, 768]`（全局共享 s）；实现改为**逐 expert 独立 s**
> （save 端每个 expert 独立 AWQ 搜索出各自的 s）。加载路径不变。

加载路径三关（全部源码级验证）：
1. `mlp.experts` 子串 → stacked_params_mapping 循环 qwen3_vl_moe.py:214 `continue`。
2. fused 映射的 `experts.gate_up_proj`/`experts.down_proj` 不命中 → is_expert_weight=False。
3. generic 分支 :302-320：`weight_loader(param, loaded_weight)` **2 参调用**；
   `maybe_remap_kv_scale_name`（weight_utils.py:1542）只拦 `.kv_scale`，qscales/awq_scale 透传。
   → 参数必须挂 `default_weight_loader`（default_weight_loader 整张拷贝 + shape 校验，
   weight_utils.py:1399）。✅

### MoE scheme（vllm-ascend，2026-08-12 实现）
- 新文件 `vllm_ascend/quantization/methods/w4a16_awq_moe.py`：
  `AscendAWQFusedMoEMethod(AscendMoEScheme)`，`@register_scheme("W4A16_AWQ", "moe")`。
  - `get_weight` 返回上面 6 参数（与 checkpoint 形状逐一对上，已单测验证）。
  - `get_dynamic_quant_param` 返回 {}。
  - `apply`：**torch 兜底** —— `select_experts` 算 top-k →
    逐 expert `unpack_int4_packed_int32(qweight[e]) * qscales[e].repeat_interleave(128)` 反量化
    → `gate/up = F.linear(x/s13[e], w13[:768])`、`intm = silu(gate)*up`、
    `down = F.linear(intm/s2[e], w2)` → `out[rows] += topk_weights * down`。
    已用独立 fp32 参考实现合成验证（逐 token 误差均匀 = bf16 精度，无 scatter 离群）。
    ⚠ tp=1 前提（logical==physical expert）；EPLB/TP>1 需补 expert id 映射。
- adapter 门控 `method_adapters.py` `AscendFusedMoEMethod.create_weights`：
  `if getattr(scheme, "load_whole_tensor", False)` → weight_loader 替换为
  `default_weight_loader`（否则 2 参 generic 调用打 6 参 FusedMoE.weight_loader 会炸）。
  scheme 定义类属性 `load_whole_tensor = True`。
- config 接线 `awq_ascend_config.py` `get_quant_method`：新增
  `if isinstance(layer, FusedMoE)` 分支 → `AscendFusedMoEMethod(AscendAWQFusedMoEMethod(...), layer.moe_config)`。

### 专家校准（save 端，实测定稿）
- hook `experts` 模块 forward：捕获 `hidden_states` [n,2048] 与 `top_k_index` [n,8]。
- 逐 expert：`x_e = hidden_states[topk==e]`（路由 token 子集，<8 用均匀采样兜底）；
  N_PAD=256 固定行数（行重复对 AWQ 搜索等价但 NPU 分配器逐 expert 完美复用 → 不 OOM）。
- `intm_e = silu(linear(x_e, gate_up[e][:768])) * linear(x_e, gate_up[e][768:])` 用冻结权重重算，
  作为 down_proj 校准激活。
- 逐 expert AWQ（gate/up 共享 s、down 独立 s）→ 填进 fused 数组。
- ⚠ **模型校准完毕挪 CPU 再量化**（save 端 [3/5] 后）：专家驻留 NPU 时逐 expert 变长分配
  碎片化 OOM（reserved 60.3GB 后 14MB 都失败）；挪走后在空 NPU 量化，彻底消除。

### 2026-08-12 修的 bug：awq_clip_search 采样切错维度
`awq_core.py:193` 原为 `x_g = x_g[:, ::step]` —— 本意采样 token（dim0），实际切的是
group 维（dim1）。tokens≤512 时 step=1 是恒等（冒烟不炸）；全量每层 1143+ tokens →
step≥2 → dim1 16→8，循环 range(ng)=16 越界 `IndexError: index 8 out of bounds...`。
已修为 `x_g = x_g[::step]`。⚠ 影响：仅全量（>512 tokens/层）会触发；8B dense 前期跑
tokens 小未触发，结果仍有效。

### 2026-08-12 e2e 运行时踩坑两处（vLLM 加载后 forward 期）
1. **`get_quant_method` 必须收 `tid2eid` 关键字**：vllm-ascend 的
   `AscendFusedMoE.__init__`（ops/fused_moe/fused_moe.py:369）以
   `get_quant_method(self, self.layer_name, tid2eid=self.tid2eid)` 调用；
   `AscendAWQConfig.get_quant_method(self, layer, prefix)` 签名缺此参数 →
   `TypeError: ... unexpected keyword argument 'tid2eid'`。照 fp8_config.py 模式
   加 `tid2eid=None` 并透传给 `AscendFusedMoEMethod(..., tid2eid=tid2eid)`。
2. **scheme `apply` 必须返回 `FusedExpertsResult` 而非裸 Tensor**：层级
   `AscendFusedMoE.forward_impl`（fused_moe.py:723）在 apply 后访问
   `fused_experts_results.routed_out`；裸 Tensor 报
   `AttributeError: 'Tensor' object has no attribute 'routed_out'`。
   tp=1 时 `finalize`（PrepareAndFinalizeWithAllGather，dp/pcp=1）纯透传，
   `routed_out=out` 即最终结果。修为 `return FusedExpertsResult(routed_out=out)`。

### 2026-08-12 e2e 结果（PASS）
- 基线：HF transformers bf16 `Qwen3VLMoeForConditionalGeneration` 捕获 last-token
  logits，norm=1251.323，tokens=19（30B bf16 62.2GB 装不进 vLLM 单卡 64GB）。
- 量化：vLLM 0.21 `quantization=awq_ascend` 加载 17.8GB checkpoint，同一 token ids
  生成，norm=1282.432。
- **余弦相似度 0.992530（≥0.99 PASS），最大绝对差 2.078**。
- 逐 expert torch 兜底 apply 慢（19 tokens 约 11.7s/生成，128 层 × 48 层循环），
  仅验证用途；上线需换 NPU GMM 算子（w4a16.py fused_experts 路径）。
