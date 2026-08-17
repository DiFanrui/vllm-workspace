#!/usr/bin/env python3
"""AWQ MVP 驱动：在 Ascend NPU 上用纯 torch 手写 AWQ，量化 Qwen3-VL-8B 的若干层并验证。

流程：
  1. 加载 Qwen3-VL-8B（bf16, npu:0）
  2. 用少量纯文本校准样本前向，hook 收集目标 Linear 层的输入激活 X
  3. 对每个目标层做 AWQ scale 搜索 + INT4 group 量化（group_size=128）
  4. 对比报告：普通 INT4 vs AWQ 的重建误差（激活加权 + 相对 F-norm）
  5. 端到端验证：把量化后的 W_hat 换回模型，跑一个"非校准"的 prompt，
     与 FP16 基线对比 last-token logits 的余弦相似度

用法:
  python run_awq_mvp.py                          # 量化 layer 0-1（快速 MVP）
  python run_awq_mvp.py --layers all             # 量化全部 36 层（更完整的信号）
  python run_awq_mvp.py --layers 0,1,2,3
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch_npu  # noqa: F401  (注册 npu 后端)

from awq_core import (
    awq_quantize,
    awq_reconstruct,
    awq_clip_search,
    awq_layer_error,
    awq_scale_search,
    collect_calib_activations,
    int4_quant_scale,
    int4_quantize_round,
    int4_dequant,
    _rel_error,
)

MODEL_ID = "/data/models/Qwen3-VL-8B-Instruct"
HIDDEN = 4096

# 校准文本（纯文本即可，AWQ 只看激活的通道分布）
CALIB_TEXTS = [
    "人工智能是研究如何让机器具有智能的学科，主要包括机器学习、深度学习等方向。",
    "深度学习通过多层神经网络学习数据的层次化表示，在图像识别与自然语言处理中取得了突破。",
    "The transformer architecture relies on self-attention to model long-range dependencies in sequences.",
    "Large language models are trained on massive text corpora using next-token prediction objectives.",
    "分布式训练通常涉及数据并行、模型并行以及流水线并行等不同的并行策略。",
    "Quantization reduces the numerical precision of weights to accelerate inference and save memory.",
]

# 非校准、用于端到端验证的 prompt
EVAL_TEXT = (
    "自然语言处理的核心任务包括文本分类、机器翻译、问答系统和文本生成等。"
)

# Qwen3 text 骨干结构: model.model.language_model.layers.N.{self_attn|mlp}.xxx
LINEAR_SUFFIXES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                   "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--layers", default="0,1",
                   help="要量化的层，逗号分隔；或 'all' 量化全部 36 层")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="每层校准激活最多保留的 token 数")
    p.add_argument("--num-bits", type=int, default=4)
    p.add_argument("--results-dir", default="/vllm-workspace/test_0811/awq/results")
    return p.parse_args()


def build_batches(processor, texts, device):
    batches = []
    for t in texts:
        enc = processor(text=t, return_tensors="pt")
        batch = {k: v.to(device) for k, v in enc.items() if hasattr(v, "to")}
        batches.append(batch)
    return batches


def layer_names(layer_idxs):
    """返回该层所有 Linear 子层的完整模块名（含 self_attn./mlp. 中间段）。"""
    return [f"language_model.layers.{i}.{sfx}" for i in layer_idxs for sfx in LINEAR_SUFFIXES]


def apply_quantized_weights(model, module_name, w_hat, group_size, num_bits):
    """把 AWQ 反量化重建的 W_hat 换回对应 Linear，模拟量化推理。"""
    mod = model
    for part in module_name.split("."):
        mod = getattr(mod, part)
    mod.weight.data = w_hat.to(mod.weight.dtype).to(mod.weight.device)
    return mod


def main():
    args = parse_args()
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report = {"model": args.model, "device": args.device, "group_size": args.group_size,
              "num_bits": args.num_bits, "timestamp": stamp}

    print(f"[1/5] 加载模型 {args.model} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(args.device)
    model.eval()
    print(f"      模型加载完成 ({time.time()-t0:.0f}s)")

    if args.layers == "all":
        layer_idxs = list(range(36))
    else:
        layer_idxs = [int(i) for i in args.layers.split(",")]
    targets = layer_names(layer_idxs)
    print(f"[2/5] 目标层: {len(layer_idxs)} 层 x {len(LINEAR_SUFFIXES)} 个 Linear = {len(targets)} 个")

    print("[3/5] 校准: 收集输入激活 X ...")
    t0 = time.time()
    batches = build_batches(processor, CALIB_TEXTS, args.device)
    calib = collect_calib_activations(model, batches, targets, max_tokens=args.max_tokens)
    print(f"      收集到 {len(calib)} 层激活, 耗时 {time.time()-t0:.0f}s")
    for name, x in calib.items():
        print(f"        {name:70s} X={tuple(x.shape)}")

    print("[4/5] AWQ 量化 + 逐层误差对比 ...")
    t0 = time.time()
    per_layer = {}
    for name, x in calib.items():
        mod = model
        for part in name.split("."):
            mod = getattr(mod, part)
        w = mod.weight.detach().float()

        if w.shape[-1] % args.group_size != 0:
            # 非整倍数（如 k_proj/v_proj 的 4096 是整的；这里兜底跳过）
            print(f"        {name}: in_dim={w.shape[-1]} 不是 group_size 整数倍，跳过")
            continue

        err = awq_layer_error(w, x, group_size=args.group_size, num_bits=args.num_bits)
        err["out"] = w.shape[0]
        err["in"] = w.shape[1]
        per_layer[name] = err
        print(f"        {name:60s} plain={err['plain_out_l2']:.3e}  "
              f"scale={err['awq_out_l2']:.3e}(+{err['improve']*100:.1f}%)  "
              f"scale+clip={err['awq_clip_out_l2']:.3e}(+{err['improve_clip']*100:.1f}%)")
    print(f"      逐层量化耗时 {time.time()-t0:.0f}s")

    # 统计
    imps = [e["improve"] for e in per_layer.values()]
    imps_c = [e["improve_clip"] for e in per_layer.values()]
    report["per_layer"] = per_layer
    report["summary"] = {
        "n_layers_quantized": len(per_layer),
        "improve_mean_pct": sum(imps) / len(imps) * 100 if imps else None,
        "improve_min_pct": min(imps) * 100 if imps else None,
        "improve_clip_mean_pct": sum(imps_c) / len(imps_c) * 100 if imps_c else None,
        "improve_clip_min_pct": min(imps_c) * 100 if imps_c else None,
    }

    print("[5/5] 端到端验证: 量化权重换回模型, 对比 logits ...")
    t0 = time.time()
    with torch.no_grad():
        batch = build_batches(processor, [EVAL_TEXT], args.device)[0]
        logits_base = model(**batch).logits

    for name, x in calib.items():
        mod = model
        for part in name.split("."):
            mod = getattr(mod, part)
        w = mod.weight.detach().float()
        s, _ = awq_scale_search(w, x, group_size=args.group_size, num_bits=args.num_bits)
        w_s = w * s
        clip_max = awq_clip_search(w_s, x / s, group_size=args.group_size,
                                   num_bits=args.num_bits)
        w_hat = awq_reconstruct(w, s, group_size=args.group_size, num_bits=args.num_bits,
                                clip_max=clip_max)
        apply_quantized_weights(model, name, w_hat, args.group_size, args.num_bits)

    with torch.no_grad():
        batch = build_batches(processor, [EVAL_TEXT], args.device)[0]
        logits_q = model(**batch).logits

    # 度量: last-token logits 余弦相似度 + 相对误差
    cos = torch.nn.functional.cosine_similarity(
        logits_base[:, -1].float(), logits_q[:, -1].float(), dim=-1).item()
    rel = _rel_error(logits_q, logits_base).item()
    report["e2e"] = {"logits_cosine": cos, "logits_rel_fro": rel}
    print(f"      端到端耗时 {time.time()-t0:.0f}s")
    print(f"      last-token logits 余弦相似度: {cos:.6f}")
    print(f"      last-token logits 相对误差:   {rel:.6f}")

    # 存档
    json_path = os.path.join(args.results_dir, f"mvp_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n结果已存档: {json_path}")


if __name__ == "__main__":
    main()
