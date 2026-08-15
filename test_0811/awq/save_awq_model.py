#!/usr/bin/env python3
"""保存器：把 Qwen3-VL-8B 用手写 AWQ（scale + clip）量化，产出 vllm-ascend 可加载的 checkpoint。

输出格式（vllm-ascend `awq_ascend` 方法，见 vllm_ascend/quantization/awq_ascend_config.py）：

vLLM 的 Qwen3 骨干里注意力与 MLP 的 Linear 是 **fused** 的：
  - self_attn.qkv_proj   = QKVParallelLinear（q/k/v 合成一个，out = q+k+v）
  - self_attn.mlp.gate_up_proj = MergedColumnParallelLinear（gate/up 合成一个，out = 2*）
而 o_proj / down_proj 是 RowParallelLinear，不融合。

输出命名（⚠ 见 PLUGIN_设计.md "stacked_params_mapping 子串碰撞"）：
  - vLLM 的 Qwen3 里 qkv_proj / gate_up_proj 是 fused 参数，但 checkpoint 必须存
    **分离的 HF 名** q_proj/k_proj/v_proj/gate_proj/up_proj：fused 名
    qkv_proj/gate_up_proj 分别含 "v_proj"/"up_proj" 子串，会被
    Qwen2DecoderLayer.load_weights 的 stacked_params_mapping 子串匹配误路由成
    qkqkv_proj.* / gate_gate_up_proj.* 而 KeyError。分离名按 shard_id 走正常路由。
  - 每层 7 个 Linear → 7 个量化模块：q/k/v/o/gate/up/down。
  对每个量化 Linear（分离后即单层）：
    - `<name>.qweight`  [out, in/8] int32    Q(W·s) 打包（每 int32 装 8 个 int4，低 nibble 先）
    - `<name>.qscales`  [out, in/128] bf16   W·s 的逐组对称 scale = max|W·s|/7
    - `<name>.awq_scale`[1, in] bf16         逐输入通道 AWQ scale s（2D，见 w4a16_awq.py 注释）
  共享 s：q/k/v（或 gate/up）输入是同一份激活 X，在拼接后的 [out_sum, in] 权重上做一次
  scale 搜索得到单一 s，再把 qweight/qscales 沿输出维行切到各 part；每个 part 存同一份
  awq_scale（apply 里 x/s @ dequant(Q(W·s))，s 只出现在输入通道维，数学自洽）。

  其余模块（embedding/norm/vision/lm_head 等）原样保存 bf16 weight。
  vision 塔在 config.json 里用 modules_to_not_convert=["visual"] 排除（保持 bf16）。

用法:
  python save_awq_model.py --out /path/to/qwen3vl8b_awq [--max-layers N]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from awq_core import (
    awq_clip_search,
    awq_scale_search,
    collect_calib_activations,
)

# 校准文本（与 run_awq_mvp 相同）
CALIB_TEXTS = [
    "人工智能是研究如何让机器具有智能的学科，主要包括机器学习、深度学习等方向。",
    "深度学习通过多层神经网络学习数据的层次化表示，在图像识别与自然语言处理中取得了突破。",
    "The transformer architecture relies on self-attention to model long-range dependencies in sequences.",
    "Large language models are trained on massive text corpora using next-token prediction objectives.",
    "分布式训练通常涉及数据并行、模型并行以及流水线并行等不同的并行策略。",
    "Quantization reduces the numerical precision of weights to accelerate inference and save memory.",
]

# MoE（30B-A3B）专用校准文本：128 experts × 8 experts/tok，每个专家平均分到
# total_tokens × 8/128 个 token。短文本只有 ~250 token → 每专家 ~15 token 太少。
# 用较长段落凑 ~3500 token（每专家 ~220 token），AWQ 搜索更稳。
MOE_CALIB_TEXTS = [
    "人工智能是研究如何让机器具有智能的学科，主要包括机器学习、深度学习、自然语言处理与计算机视觉等方向。"
    "深度学习通过多层神经网络学习数据的层次化表示，在图像识别、语音识别与机器翻译等任务中取得了突破性进展。"
    "大语言模型基于 Transformer 架构，通过自注意力机制建模序列中的长距离依赖关系，并在海量文本上预训练。",

    "Transformer 由编码器与解码器两部分组成，核心是自注意力与逐位置前馈网络。"
    "自注意力计算查询、键与值之间的相关性，允许每个位置关注序列中的任意其他位置。"
    "多头注意力将输入投影到多个子空间并行计算，从而捕捉不同粒度的语义关系。"
    "位置编码为序列注入顺序信息，使模型能够区分不同位置的 token。",

    "Mixture of Experts 将网络中的前馈层替换为多个并行的专家网络，并通过路由器为每个输入动态选择少量专家。"
    "稀疏激活使得模型总参数大幅增加，而每次前向计算的算力开销保持不变，从而在同等推理成本下获得更强能力。"
    "专家的负载均衡与路由稳定性训练是 MoE 模型训练中的关键难点。",

    "Model quantization converts high-precision weights into low-precision representations to reduce memory and "
    "accelerate inference. Activation-aware weight quantization, or AWQ, searches per-channel scales based on the "
    "statistics of representative calibration activations, protecting the most salient weight channels from clipping "
    "errors. AWQ does not require backpropagation and can be applied to a pretrained model in a data-free manner.",

    "Large language models are usually trained with a next-token prediction objective on massive text corpora. "
    "During inference, the model generates one token at a time in an autoregressive fashion, caching key and value "
    "tensors to avoid recomputation. Decoding strategies such as greedy search, beam search, and sampling control "
    "the trade-off between diversity and determinism of the generated text.",

    "Distributed training of large models relies on data, tensor, pipeline, and sequence parallelism. "
    "In tensor parallelism the weight matrices are sharded across devices and collective communication such as "
    "all-reduce is used to aggregate partial results. Sequence parallelism further shards the sequence dimension "
    "of activations to reduce memory footprint in attention and normalization layers.",

    "推理引擎的量化链路通常包括校准集的选择、逐层激活统计的收集与按模块量化三个环节。"
    "校准集需要覆盖目标任务的输入分布，激活统计以每个输入通道的均值与绝对值均值为主。"
    "逐层量化时，需要把激活按需搬回计算设备，避免大模型的中间结果长期驻留显存导致溢出。",

    "vLLM 是一个面向大语言模型推理的高吞吐引擎，采用 PagedAttention 管理 KV 缓存，以近乎零碎片的方式分配显存。"
    "其模型加载层根据 config 中的 quantization_config 分发到不同的量化方法，方法内实现参数的创建、权重的加载与算子的执行。"
    "在昇腾硬件上，vllm-ascend 插件通过桥接层把 vLLM 的量化方法与 Ascend 的自研算子对接起来。",

    "The routing mechanism of a mixture of experts model computes a probability distribution over all experts and "
    "selects the top-k experts with the largest scores for each token. The outputs of the selected experts are "
    "weighted by the corresponding routing scores and summed together. Load balancing losses encourage a uniform "
    "distribution of tokens across experts during training, improving both efficiency and stability.",

    "Group quantization packs a small number of consecutive input channels into one group sharing a single scale "
    "factor. A common choice is a group size of 128 with 4-bit symmetric quantization, which stores eight quantized "
    "values in one 32-bit integer. The scale factors are stored separately in a lower precision format to save "
    "additional memory, and dequantization multiplies each group by its scale during inference.",

    "推理引擎在加载量化模型时需要把 checkpoint 中的量化张量与运行时算子要求的布局对齐，例如打包顺序、通道顺序与 "
    "scale 的形状。若层内存在融合参数，命名子串的匹配顺序会直接影响加载路由，任何一次误匹配都会导致 KeyError 或 "
    "权重错位，因此 checkpoint 的命名必须与加载端的映射规则逐条核对。",

    "Memory management is one of the key challenges when serving large models on a single accelerator. Both the "
    "weight tensors and the activation tensors occupy high bandwidth memory, and the KV cache grows with the batch "
    "size and sequence length. Quantizing the weights reduces the static footprint, while efficient paged attention "
    "reduces the dynamic footprint, together enabling larger batches and longer sequences on a fixed device.",

    "在昇腾 910B 上加载 300 亿参数的混合专家模型时，模型本身的 bf16 权重约占用 62GB 显存，余量非常有限。"
    "因此校准阶段的激活统计必须全部落到 CPU 内存，量化阶段再按模块逐组搬回 NPU 计算，避免任何中间张量长期驻留。"
    "这种逐层逐组的内存策略是单卡完成大模型量化改造的前提。",

    "A clean architecture separates the algorithm of quantization from the deployment format. The calibration, "
    "scale search, and clipping steps only depend on the linear algebra of the weight matrices and their input "
    "activations. The serialization step then maps the quantized tensors into the exact parameter names and shapes "
    "that the inference engine expects, decoupling the two concerns cleanly.",
]

# 多模态校准样本（--vision-calib 时叠加到纯文本校准集）：手写几何图形 + 一句叙述性描述。
# 走 chat template 编码（Qwen3-VL processor 只有 apply_chat_template 才把 image 替换成
# <|image_pad|> tokens），让 image tokens 与 text tokens 一起进入文本骨干，使 AWQ 的
# scale/clip 覆盖视觉输入下的激活分布——纯文本校准集覆盖不到（30B 多模态 logits 掉 0.85 的根因）。
# 每条描述用陈述句描述图像内容，256x256 统一尺寸 → 每图 image tokens 数一致（~84），token 比例可控。
VISION_CALIB = [
    ("img_red_ring.png", "这张图片里有一个红色的禁止标志，圆形轮廓，中间一条红色斜杠。"),
    ("img_blue_tri.png", "这张图片里有一个蓝色的三角形，三个顶点分别指向左、上、右。"),
    ("img_green_sq.png", "这张图片里有一个绿色的正方形，四条边长度相等。"),
    ("img_ring_tri.png", "这张图片里有两个图形，左边一个红色圆环，右边一个蓝色三角形。"),
    ("img_yellow_star.png", "这张图片里有一个黄色的五角星，五个角均匀分布。"),
    ("img_red_sq.png", "这张图片里有一个红色的正方形，中心有一个蓝色小圆点。"),
    ("img_blue_ring.png", "这张图片里有一个蓝色的圆环，内部是白色背景。"),
    ("img_green_tri.png", "这张图片里有一个绿色的三角形，倒置放置。"),
]


def _make_vision_calib_images(out_dir: str) -> list[str]:
    """PIL 现场生成 8 张手写几何图形（256x256 白底），返回路径列表（与 VISION_CALIB 一一对齐）。"""
    import math

    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)
    paths = []

    def save(name: str, draw_fn) -> None:
        img = Image.new("RGB", (256, 256), "white")
        draw_fn(ImageDraw.Draw(img))
        p = os.path.join(out_dir, name)
        img.save(p)
        paths.append(p)

    def red_ring(d):   # 红色禁止标志
        d.ellipse([48, 48, 208, 208], outline="red", width=14)
        d.line([76, 180, 180, 76], fill="red", width=14)

    def blue_tri(d):   # 蓝色三角形
        d.polygon([(128, 52), (32, 200), (224, 200)], fill="blue")

    def green_sq(d):   # 绿色正方形
        d.rectangle([52, 52, 204, 204], fill="green")

    def ring_tri(d):   # 左红环 + 右蓝三角（两个图形）
        d.ellipse([16, 76, 118, 178], outline="red", width=12)
        d.polygon([(156, 200), (204, 56), (244, 200)], fill="blue")

    def yellow_star(d):  # 黄五角星
        cx, cy, R, r = 128, 128, 96, 38
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rad = R if i % 2 == 0 else r
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
        d.polygon(pts, fill="gold")

    def red_sq(d):     # 红方块 + 中心蓝圆点
        d.rectangle([52, 52, 204, 204], outline="red", width=12)
        d.ellipse([114, 114, 142, 142], fill="blue")

    def blue_ring(d):  # 蓝色圆环
        d.ellipse([48, 48, 208, 208], outline="blue", width=14)

    def green_tri(d):  # 倒置绿三角
        d.polygon([(128, 204), (40, 56), (216, 56)], fill="green")

    for fn, draw_fn in zip([v[0] for v in VISION_CALIB],
                           (red_ring, blue_tri, green_sq, ring_tri, yellow_star,
                            red_sq, blue_ring, green_tri)):
        save(fn, draw_fn)
    return paths


GROUP_SIZE = 128
NUM_BITS = 4
MAX_Q = 2 ** (NUM_BITS - 1) - 1  # 7
MIN_Q = -(2 ** (NUM_BITS - 1))  # -8

# 层内 7 个可量化模块。fused 定义参考 vLLM Qwen3ForCausalLM.packed_modules_mapping：
#   qkv_proj = [q_proj, k_proj, v_proj];  gate_up_proj = [gate_proj, up_proj]
QKV_PARTS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
GATE_UP_PARTS = ("mlp.gate_proj", "mlp.up_proj")
# 输出名 -> (组成模块, 校准激活取哪个模块的输入)
FUSED_GROUPS = {
    "self_attn.qkv_proj": (QKV_PARTS, "self_attn.q_proj"),
    "mlp.gate_up_proj": (GATE_UP_PARTS, "mlp.gate_proj"),
}
SINGLE_SUFFIXES = ("self_attn.o_proj", "mlp.down_proj")
ALL_CONSTITUENT = QKV_PARTS + GATE_UP_PARTS + SINGLE_SUFFIXES  # 7 个子模块


def pack_int4_int32(q: torch.Tensor) -> torch.Tensor:
    """把 [out, in] int8（值域 -8..7）打包成 [out, in/8] int32。

    每 int32 装 8 个 int4，低 nibble 先；带符号（-8..7 -> 0..15，解包时 ≥8 减 16）。
    与 vllm_ascend.quantization.methods.w4a16_awq.unpack_int4_packed_int32 互逆。
    """
    out, in_dim = q.shape
    packed = torch.zeros(out, in_dim // 8, dtype=torch.int32, device=q.device)
    for i in range(8):
        packed |= ((q[:, i::8].to(torch.int32) & 0x0F) << (4 * i))
    return packed


def awq_quantize_layer(w: torch.Tensor, x: torch.Tensor,
                       group_size: int = GROUP_SIZE, num_bits: int = NUM_BITS,
                       do_clip: bool = True):
    """对一层（或拼接后的 fused 层）做 AWQ 量化，返回 (qweight_int32, qscales, awq_scale, w_hat)。

    w: [out, in]; x: [n, in] 校准激活。
    流程：
      s = awq_scale_search(w, x)                    # 逐通道 scale
      clip_max = awq_clip_search(W·s, X/s)          # 在已 scale 权重上搜裁剪阈值
      q = Q(clip(W·s))                              # 对称 INT4
    """
    s, _ = awq_scale_search(w, x, group_size=group_size, num_bits=num_bits)
    w_s = w.to(torch.float32) * s
    if do_clip:
        clip_max = awq_clip_search(w_s, x.to(torch.float32) / s,
                                   group_size=group_size, num_bits=num_bits)
        clip = clip_max.repeat_interleave(group_size, dim=-1)
        w_s = w_s.clamp(-clip, clip)

    out, in_dim = w_s.shape
    ng = in_dim // group_size
    w_g = w_s.reshape(out, ng, group_size)
    gscale = w_g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / MAX_Q
    q = (w_g / gscale).round().clamp(MIN_Q, MAX_Q).to(torch.int8)
    w_hat = (q.to(torch.float32) * gscale / s.reshape(1, ng, group_size)).reshape(out, in_dim)

    return pack_int4_int32(q.reshape(out, in_dim)), gscale.reshape(out, -1), s, w_hat


def is_quant_target(module_name: str, constituents: tuple[str, ...]) -> bool:
    """命中 text 骨干的量化目标 Linear（.layers. 限定文本层，排除 vision）。

    dense：q/k/v/o/gate/up/down 7 类；MoE：q/k/v/o 4 类（专家 fused 3D 不量化）。
    """
    return any(module_name.endswith(sfx) for sfx in constituents) and ".layers." in module_name


# ---- 专家量化（MoE 专用）----
# Qwen3VLMoeTextExperts 单 3D Parameter：gate_up_proj [128,1536,2048]、down_proj [128,2048,768]
NUM_EXPERTS = 128
INTM_DIM = 768          # moe_intermediate_size
GATE_UP_HIDDEN = 2048   # hidden_size
# 专家校准固定 token 数：采样/重复路由 token 到恰好 N_PAD 个。
# 重复行对 AWQ 搜索完全等价（mean/argmin 不变），但所有专家分配大小一致 → NPU 缓存
# 分配器逐 expert 完美复用，reserved 有界（模型占 ~60GB，之前逐 expert 变长分配
# 使 reserved 涨到 60.3GB 后 14MB 都 OOM）。路由 token < 该值时是"校准数据不足"，改全局兜底。
N_PAD = 256
EXPERT_FALLBACK_MIN = 8  # 路由 token < 该值 → 用层内均匀采样兜底


def collect_expert_routing(model, batches, layer_nums, base, max_tokens=None) -> dict:
    """hook 每层 `mlp.experts` 模块，捕获 (hidden_states, top_k_index) 存 CPU。

    forward 签名（modeling_qwen3_vl_moe.py:87-92）：(hidden_states, top_k_index, top_k_weights)。
    hidden_states 是 post_attention_layernorm 后的真实 gate_up 输入（decoder layer :348-349），
    top_k_index [n, 8] 给出每个 token 路由到的 8 个专家。
    返回 {layer_i: (hidden[n,2048] fp32, topk[n,8] int64)}，均 CPU。
    """
    captures: dict[int, list] = {li: [] for li in layer_nums}

    def make_hook(li):
        def hook(module, args, kwargs):
            hs, tki, _tkw = args
            captures[li].append((hs.detach().float().cpu(), tki.detach().cpu()))
        return hook

    handles = []
    for li in layer_nums:
        mod = model.get_submodule(f"{base}{li}.mlp.experts")
        handles.append(mod.register_forward_hook(make_hook(li)))

    model.eval()
    with torch.no_grad():
        for batch in batches:
            model(**batch)
    for h in handles:
        h.remove()

    out = {}
    for li, lst in captures.items():
        hs = torch.cat([h for h, _ in lst], dim=0)
        tki = torch.cat([t for _, t in lst], dim=0)
        if max_tokens is not None and hs.shape[0] > max_tokens:
            hs, tki = hs[:max_tokens], tki[:max_tokens]
        out[li] = (hs, tki)
    return out


def quantize_experts(sd, routing, layer_nums, base, device, do_clip=True):
    """逐 expert AWQ，输出 6 个 fused 张量/层（MOE_设计.md 最终格式）。

    sd: 模型 state_dict（模型已挪 CPU，见 main）。routing: {li: (hidden, topk)}（CPU）。
    模型挪 CPU 后 NPU 近乎空置 → 每层把 gate_up/down 搬到 NPU，逐 expert 量化无 OOM。
    每 expert e：
      x_e = hidden[topk==e]                          # 路由 token 子集
      intm_e = silu(x_e@gate_up[e][:768]ᵀ) * (x_e@gate_up[e][768:]ᵀ)  # 冻结 bf16 重算
      gate_up[e] [1536,2048] 与 x_e 做 AWQ（gate/up 共享同一份 x → 同一 s）
      down[e]   [2048,768]   与 intm_e 做 AWQ
    输出（均 CPU）：
      w13_qweight[128,1536,256] int32, w13_qscales[128,1536,16] bf16, w13_awq_scale[128,1,2048] bf16
      w2_qweight[128,2048,96]   int32, w2_qscales[128,2048,6]   bf16, w2_awq_scale[128,1,768]  bf16
    返回 (quant_tensors, sources)：quant_tensors["...mlp.experts"] = 6 个张量；
    sources[...] = [gate_up_proj 全名, down_proj 全名]（state_dict 删除键，fused 3D 无 .weight 后缀）。
    """
    quant_tensors: dict[str, dict] = {}
    sources: dict[str, list[str]] = {}

    torch_npu.npu.empty_cache()

    for li in layer_nums:
        # 模型在 CPU，当前层专家权重搬到（空置的）NPU
        gate_up = sd[f"{base}{li}.mlp.experts.gate_up_proj"].to(device)  # [128,1536,2048] bf16
        down = sd[f"{base}{li}.mlp.experts.down_proj"].to(device)        # [128,2048,768]
        hidden, topk = routing[li]      # [n,2048] fp32 / [n,8] int64（CPU）
        n_tok = hidden.shape[0]
        # ⚠ 不能 topk.view(-1) 再 nonzero：那是拍平位置（0..n*8-1），hidden 只有 n 行。
        # 用 (topk==e).any(dim=-1) 直接得到逐 token 布尔掩码。

        # fused 结果预分配（CPU，逐 expert 填充）
        W13_Q = torch.zeros(NUM_EXPERTS, 2 * INTM_DIM, GATE_UP_HIDDEN // 8, dtype=torch.int32)
        W13_S = torch.zeros(NUM_EXPERTS, 2 * INTM_DIM, GATE_UP_HIDDEN // GROUP_SIZE, dtype=torch.float32)
        W13_A = torch.zeros(NUM_EXPERTS, 1, GATE_UP_HIDDEN, dtype=torch.float32)
        W2_Q = torch.zeros(NUM_EXPERTS, GATE_UP_HIDDEN, INTM_DIM // 8, dtype=torch.int32)
        W2_S = torch.zeros(NUM_EXPERTS, GATE_UP_HIDDEN, INTM_DIM // GROUP_SIZE, dtype=torch.float32)
        W2_A = torch.zeros(NUM_EXPERTS, 1, INTM_DIM, dtype=torch.float32)

        # 全局兜底索引：层内均匀采样（给 0-token 专家）
        fb_idx = torch.linspace(0, n_tok - 1, N_PAD).round().long()

        def pad_indices(idx: torch.Tensor) -> torch.Tensor:
            """把 token 索引采样/重复到恰好 N_PAD 个（重复行对 AWQ 搜索等价）。"""
            n_real = idx.numel()
            if n_real >= N_PAD:
                return torch.linspace(0, n_real - 1, N_PAD).round().long()  # 均匀降采样
            base = torch.arange(n_real)
            rep = (N_PAD + n_real - 1) // n_real
            return base.repeat(rep)[:N_PAD]

        for e in range(NUM_EXPERTS):
            token_idx = torch.nonzero((topk == e).any(dim=-1)).view(-1)
            if token_idx.numel() < EXPERT_FALLBACK_MIN:
                token_idx = fb_idx
            else:
                token_idx = token_idx[pad_indices(token_idx)]
            x_e = hidden[token_idx].to(gate_up.device)  # [N_PAD,2048] fp32 NPU，大小固定

            # intm 重算：模型内实际 bf16（experts.forward:105-106 silu(gate)*up）
            x_bf = x_e.to(torch.bfloat16)
            g = F.linear(x_bf, gate_up[e, :INTM_DIM])
            u = F.linear(x_bf, gate_up[e, INTM_DIM:])
            intm = (F.silu(g) * u).to(torch.float32)
            del x_bf, g, u

            qw13, qs13, s13, _ = awq_quantize_layer(gate_up[e].float(), x_e, do_clip=do_clip)
            qw2, qs2, s2, _ = awq_quantize_layer(down[e].float(), intm, do_clip=do_clip)
            del x_e, intm

            W13_Q[e] = qw13.cpu(); W13_S[e] = qs13.cpu(); W13_A[e, 0] = s13.cpu()
            W2_Q[e] = qw2.cpu();   W2_S[e] = qs2.cpu();   W2_A[e, 0] = s2.cpu()
            del qw13, qs13, s13, qw2, qs2, s2
            torch_npu.npu.empty_cache()  # 逐专家归还碎片块（~0.01ms，模型占 60GB 时必须）

        out_name = f"{base}{li}.mlp.experts"
        quant_tensors[out_name] = {
            "w13_qweight": W13_Q,
            "w13_qscales": W13_S.to(torch.bfloat16),
            "w13_awq_scale": W13_A.to(torch.bfloat16),
            "w2_qweight": W2_Q,
            "w2_qscales": W2_S.to(torch.bfloat16),
            "w2_awq_scale": W2_A.to(torch.bfloat16),
        }
        sources[out_name] = [f"{base}{li}.mlp.experts.gate_up_proj",
                             f"{base}{li}.mlp.experts.down_proj"]
        torch_npu.npu.empty_cache()
        print(f"      层 {li}: 128 专家量化完成")

        # 自检：解包专家 0 的 gate_up，对比原始 bf16（验证打包顺序 + scale 形状自洽）
        q0 = torch.zeros(2 * INTM_DIM, GATE_UP_HIDDEN, dtype=torch.int32, device=device)
        for i in range(8):
            q0[:, i::8] = ((W13_Q[0] >> (4 * i)) & 0x0F).to(torch.int32)
        q0 = torch.where(q0 >= 8, q0 - 16, q0).to(torch.float32)
        w_hat0 = (q0.reshape(2 * INTM_DIM, -1, GROUP_SIZE)
                  * W13_S[0].unsqueeze(-1).to(device)).reshape(2 * INTM_DIM, GATE_UP_HIDDEN)
        w_hat0 = w_hat0 / W13_A[0].to(device)
        rel0 = (w_hat0 - gate_up[0].float()).abs().max().item()
        del q0, w_hat0
        print(f"        专家0 gate_up max|w_hat-w|={rel0:.3e}")

    return quant_tensors, sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data/models/Qwen3-VL-8B-Instruct")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--out", default="/vllm-workspace/test_0811/awq/out/qwen3vl8b_awq")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-layers", type=int, default=None,
                   help="只量化前 N 层（快速冒烟测试）")
    p.add_argument("--do-clip", action="store_true", default=True)
    p.add_argument("--vision-calib", action="store_true",
                   help="叠加多模态校准样本（手写几何图形+描述）到纯文本校准集，"
                        "让 scale/clip 覆盖视觉输入下的激活分布")
    p.add_argument("--vision-calib-only", action="store_true",
                   help="仅多模态校准：去掉纯文本段，8 图×5 份≈3600 tokens（image ~89%），"
                        "验证 image/text 占比对 scale 的主导")
    args = p.parse_args()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

    os.makedirs(args.out, exist_ok=True)
    print(f"[1/5] 加载模型 {args.model} -> {args.device} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    # 自动识别 dense（Qwen3-VL-8B）vs MoE（Qwen3-VL-30B-A3B）
    import json as _json
    _cfg = _json.load(open(os.path.join(args.model, "config.json")))
    _arch = _cfg.get("architectures", [""])[0]
    model_cls = Qwen3VLMoeForConditionalGeneration if "Moe" in _arch else Qwen3VLForConditionalGeneration
    model = model_cls.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(args.device)
    model.eval()
    print(f"      完成 ({time.time()-t0:.0f}s), 架构 {_arch}")

    # dense：q/k/v/o/gate/up/down 7 类；MoE：q/k/v/o 4 类（专家 fused 3D 不量化）
    is_moe = any("mlp.experts" in n for n, _ in model.named_modules())
    if is_moe:
        constituents = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                        "self_attn.o_proj")
        modules_not_conv = ["visual", "mlp.gate"]  # gate 是 vLLM 的 ReplicatedLinear，排除
    else:
        constituents = ALL_CONSTITUENT
        modules_not_conv = ["visual"]

    targets = [n for n, m in model.named_modules() if is_quant_target(n, constituents)]
    if args.max_layers is not None:
        targets = [t for t in targets if int(t.split(".layers.")[1].split(".")[0]) < args.max_layers]
    kind = "q/k/v/o" if is_moe else "q/k/v/o/gate/up/down"
    nlayers = len(set(int(t.split(".layers.")[1].split(".")[0]) for t in targets))
    print(f"[2/5] 量化目标: {len(targets)} 个 Linear（每层 {kind}，共 {nlayers} 层 → {len(targets)//nlayers} 个量化模块/层）")

    print("[3/5] 校准: 收集输入激活 ...")
    t0 = time.time()
    if args.vision_calib_only:
        print("      [vision-calib-only] 仅多模态校准：无纯文本段，8 图×5 份≈3600 tokens（image ~89%）")
        calib_texts = []
    else:
        calib_texts = MOE_CALIB_TEXTS if is_moe else CALIB_TEXTS
    batches = []
    for t in calib_texts:
        enc = processor(text=t, return_tensors="pt")
        batches.append({k: v.to(args.device) for k, v in enc.items() if hasattr(v, "to")})
    if args.vision_calib or args.vision_calib_only:
        # 叠加多模态样本：手写几何图形 + 描述，chat template 编码（同 verify_vllm_vision.encode）。
        # 逐图前向（Qwen3-VL 多图 batch 的 grid_thw 对齐复杂），统一 256x256 → image tokens 一致。
        # only 模式复制 5 份：分布不变只增样本量，保住专家路由的 token 数（~3500/每专家 ~220）。
        from PIL import Image
        img_paths = _make_vision_calib_images(os.path.join(args.out, "calib_imgs"))
        v_batches = []
        for img_path, (_, desc) in zip(img_paths, VISION_CALIB):
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": desc}]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            enc = processor(text=text, images=[Image.open(img_path).convert("RGB")],
                            return_tensors="pt")
            v_batches.append({k: v.to(args.device) for k, v in enc.items() if hasattr(v, "to")})
        n_rep = 5 if args.vision_calib_only else 1
        batches.extend(v_batches * n_rep)
        print(f"      已叠加 {len(VISION_CALIB) * n_rep} 条多模态校准样本"
              f"（image tokens 占比 ~{89 if args.vision_calib_only else 17}%）")
    # 只收集目标层：把完整模块名作为 matcher（精确命中）
    calib = collect_calib_activations(model, batches, targets, max_tokens=args.max_tokens)
    print(f"      收集到 {len(calib)} 层激活 ({time.time()-t0:.0f}s), {sum(x.shape[0] for x in calib.values())} tokens")

    # MoE 额外：专家路由 token 子集采集（hook mlp.experts，单独一次前向）
    routing = None
    if is_moe:
        layer_nums0 = sorted({int(t.split(".layers.")[1].split(".")[0]) for t in targets})
        base0 = targets[0].split(".layers.", 1)[0] + ".layers."
        routing = collect_expert_routing(model, batches, layer_nums0, base0,
                                         max_tokens=args.max_tokens)
        topk0 = routing[layer_nums0[0]][1].view(-1)
        cnt0 = torch.bincount(topk0, minlength=NUM_EXPERTS)
        cold = int((cnt0 < EXPERT_FALLBACK_MIN).sum())
        print(f"      专家路由采集完成: 层{layer_nums0[0]} 每专家 tokens "
              f"min={cnt0.min().item()} median={cnt0.median().item()} max={cnt0.max().item()}, "
              f"冷门(<{EXPERT_FALLBACK_MIN}) {cold}/{NUM_EXPERTS}")

    # ---- 校准完毕，模型挪 CPU、释放 NPU ----
    # 模型驻留时逐 expert 量化会因分配器碎片化 OOM（reserved 涨到 60.3GB 后 14MB 都失败）。
    # 挪到 CPU 后 NPU 近乎空置，注意力/专家量化都搬到空 NPU 算，彻底消除 OOM。
    print("      校准完毕: 模型挪到 CPU（释放 ~60GB NPU）...")
    t0 = time.time()
    model.to("cpu")
    sd = {k: v for k, v in model.state_dict().items()}  # 引用共享（模型已是 bf16），不复制
    del model
    torch_npu.npu.empty_cache()
    print(f"      NPU 已释放 ({time.time()-t0:.0f}s)")

    # ---- 按层分组成 fused 模块，逐组量化 ----
    print("[4/5] AWQ 量化（scale + clip）...")
    t0 = time.time()
    quant_tensors: dict[str, dict] = {}      # 输出模块名 -> {qweight, qscales, awq_scale}
    sources: dict[str, list[str]] = {}       # 输出模块名 -> [源模块名]（删除其 .weight）

    layer_nums = sorted({int(t.split(".layers.")[1].split(".")[0]) for t in targets})
    base = targets[0].split(".layers.", 1)[0] + ".layers."  # 如 model.language_model.layers.
    target_set = set(targets)

    def emit_part(out_name: str, part_names: list[str],
                  qweight: torch.Tensor, qscales: torch.Tensor, s: torch.Tensor,
                  w_hat: torch.Tensor, ws: list[torch.Tensor]) -> None:
        """把一个（或一段）量化结果写入 quant_tensors / sources，并报告误差。"""
        quant_tensors[out_name] = {
            "qweight": qweight.to("cpu"),
            "qscales": qscales.to(torch.bfloat16).to("cpu"),
            # [1, in]：与 scheme 的 get_weight 形状一致（避免 1D 在 RowParallel 的 shape[1] 越界）
            "awq_scale": s.unsqueeze(0).to(torch.bfloat16).to("cpu"),
        }
        sources[out_name] = part_names
        if len(part_names) == 1:
            rel = (w_hat - ws[0]).abs().max().item()
            print(f"      {out_name:52s} max|w_hat-w|={rel:.3e}")
        else:
            off = 0
            for i, p in enumerate(part_names):
                seg_w_hat = w_hat[off:off + ws[i].shape[0]]
                rel = (seg_w_hat - ws[i]).abs().max().item()
                print(f"      {out_name:40s} [{p.rsplit('.',1)[-1]:6s}] max|w_hat-w|={rel:.3e}")
                off += ws[i].shape[0]

    def quantize_group(out_names: list[str], part_names: list[str], x: torch.Tensor):
        """量化一组（W = cat(各 part)，共享 s）。

        len(out_names)==1：fused 输出（如 qkv_proj）——整张写一个模块；
        len(out_names)>1 ：按 part 拆分输出（如 gate_proj/up_proj）——每个 part 独立张量，
        但共用同一逐通道 s（q/k/v、gate/up 输入激活相同，数学自洽）。
        """
        ws = []
        for p in part_names:
            if p not in target_set or p not in calib:
                return None
            # 模型已挪 CPU（见 main [3/5] 后），权重从 sd 读、搬到空置 NPU 算
            ws.append(sd[p + ".weight"].detach().float().to(args.device))
        W = torch.cat(ws, dim=0) if len(ws) > 1 else ws[0]
        assert W.shape[-1] % GROUP_SIZE == 0, f"{out_names}: in={W.shape[-1]} 非 group 整倍"
        x = x.to(W.device)  # 校准激活在 CPU，量化时搬回设备

        qweight, qscales, s, w_hat = awq_quantize_layer(W, x, do_clip=args.do_clip)

        if len(out_names) == 1:
            emit_part(out_names[0], part_names, qweight, qscales, s, w_hat, ws)
            return
        # 拆分：qweight/qscales 沿输出维(0)按 part 行切；awq_scale 共享，各自存同一份
        off = 0
        for i, p in enumerate(part_names):
            n = ws[i].shape[0]
            part_w_hat = w_hat[off:off + n]
            emit_part(out_names[i], [p], qweight[off:off + n], qscales[off:off + n],
                      s, part_w_hat, [ws[i]])
            off += n

    for li in layer_nums:
        # 注意：注意力输入是同一份 hidden_states，q/k/v 输入相同 → 用 q_proj 的激活
        qkv = [f"{base}{li}.{p}" for p in QKV_PARTS]
        qkv_out = [f"{base}{li}.{p}" for p in QKV_PARTS]
        # ⚠ q/k/v、gate/up 都必须分离命名：fused 名 qkv_proj/gate_up_proj 分别含
        # "v_proj"/"up_proj" 子串，会被 Qwen2DecoderLayer.load_weights 的
        # stacked_params_mapping 子串匹配误路由（qkv_proj→qkqkv_proj、gate_up_proj→
        # gate_gate_up_proj）而 KeyError。分离名走 shard_id 的正常路由
        # （QKVParallel/MergedColumnParallel weight_loader 沿 output_dim narrow）。
        quantize_group(qkv_out, qkv, calib.get(qkv[0]))
        oproj = f"{base}{li}.self_attn.o_proj"
        quantize_group([oproj], [oproj], calib.get(oproj))
        if not is_moe:
            # dense 额外量化 gate_proj/up_proj（共享同一份 MLP 输入激活）
            gup = [f"{base}{li}.{p}" for p in GATE_UP_PARTS]
            quantize_group([f"{base}{li}.mlp.gate_proj", f"{base}{li}.mlp.up_proj"],
                           gup, calib.get(gup[0]))

    if is_moe:
        # 专家：逐 expert AWQ（路由 token 子集校准）
        print(f"      专家 AWQ 量化（{len(layer_nums)} 层 × {NUM_EXPERTS} experts）...")
        exp_t0 = time.time()
        exp_tensors, exp_sources = quantize_experts(sd, routing, layer_nums, base,
                                                    args.device, do_clip=args.do_clip)
        quant_tensors.update(exp_tensors)
        sources.update(exp_sources)
        print(f"      专家量化完成 ({time.time()-exp_t0:.0f}s)")

    print(f"      量化完成 ({time.time()-t0:.0f}s), 共 {len(quant_tensors)} 个量化模块")

    print("[5/5] 组装 checkpoint + 写盘 ...")
    t0 = time.time()
    # sd 已在 [3/5] 后建立（引用 model 的 CPU bf16 张量），这里直接换量化张量
    for out_name, tens in quant_tensors.items():
        for src in sources[out_name]:
            wkey = f"{src}.weight"
            if wkey not in sd:
                wkey = src  # 专家 fused 3D Parameter（experts.gate_up_proj）键名无 .weight 后缀
            assert wkey in sd, f"state_dict 里没有 {wkey}"
            del sd[wkey]  # 用量化张量替换原始权重
        for k, v in tens.items():
            # dense 输出 {qweight, qscales, awq_scale}; 专家输出 {w13_qweight, w13_qscales, ...}
            sd[f"{out_name}.{k}"] = v

    # config.json：保留原配置 + 加 quantization_config
    import safetensors.torch as st
    cfg_path = os.path.join(args.model, "config.json")
    with open(cfg_path) as f:
        model_cfg = json.load(f)
    model_cfg["quantization_config"] = {
        "quant_method": "awq_ascend",
        "bits": NUM_BITS,
        "group_size": GROUP_SIZE,
        "zero_point": False,
        # vision 塔保持 bf16；MoE 的 router gate 也是 LinearBase，须一并排除
        # （vLLM 端 get_quant_method 用 substr 匹配）。
        "modules_to_not_convert": modules_not_conv,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(model_cfg, f, indent=2, ensure_ascii=False)

    st.save_file(sd, os.path.join(args.out, "model.safetensors"))
    print(f"      写盘完成 ({time.time()-t0:.0f}s), 共 {len(sd)} 个张量")

    # 拷贝 processor/tokenizer 文件：vLLM 构造多模态模型时还要 preprocessor_config.json
    # 等（Qwen3-VL 的 image processor 初始化会失败）。仅拷贝缺失的。
    import shutil
    for fname in (
        "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
        "chat_template.json", "generation_config.json", "video_preprocessor_config.json",
    ):
        src = os.path.join(args.model, fname)
        dst = os.path.join(args.out, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
            print(f"      拷贝 {fname}")

    nq = len(quant_tensors)
    print(f"\n完成: 量化 {nq} 个 Linear, checkpoint 在 {args.out}")
    print(f"  config.json quantization_config.quant_method = awq_ascend")
    print(f"  加载方式: vllm --quantization awq_ascend --load-format safetensors ...")


if __name__ == "__main__":
    main()
