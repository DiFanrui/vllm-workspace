#!/usr/bin/env python3
"""多 prompt 证据补强：30B-A3B 量化模型 vs bf16 基线的 last-token logits 余弦 + 生成文本对比。

相比 verify_vllm_load.py（单 prompt）补两件事：
  1. 多 prompt（5 个）last-token logits 逐条余弦 + 平均
  2. 一次实际生成文本对比（量化模型 vs bf16 基线"说出的话"）

三步（每步独立进程，避免 fork-after-threads）：
  python verify_vllm_multi.py --mode base-hf   # HF bf16 基线：捕获 logits + 生成文本
  python verify_vllm_multi.py --mode quant     # vLLM awq_ascend 量化端：捕获 logits + 生成文本
  python verify_vllm_multi.py --mode compare   # 逐 prompt 余弦 + 文本对比表

多 prompt 实现要点：
  - HF 端一次 batch forward（padding=True），按 attention_mask 取每行 last-token logits。
  - vLLM 端一次 batch generate（max_tokens=1）：引擎级 processor 收到 [B, vocab] 整批 logits
    （注意：单请求版 CaptureLogitsProcessor 只存 logits[0]，多请求必须存整批）。
  - 文本生成：HF 原生 generate vs vLLM generate，均为贪心、短输出。
"""
from __future__ import annotations

import argparse
import glob
import json
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import torch

from vllm.v1.sample.logits_processor.interface import LogitsProcessor

import vllm_ascend.quantization.awq_ascend_config  # noqa: F401

PROMPTS = [
    "自然语言处理的核心任务包括文本分类、机器翻译、问答系统和文本生成等。",
    "中国的首都是哪个城市？",
    "计算 17 乘以 8 等于多少？",
    "把'你好，世界'翻译成英文。",
    "请用一句话描述春天的景色。",
]
MAX_NEW_TOKENS = 8


# vLLM 0.21 引擎级 LogitsProcessor：必须顶层可 import（config 按引用 pickle 到 EngineCore 子进程），
# 且必须继承 LogitsProcessor（NPUModelRunner init 校验 isinstance）。
class BatchCaptureLogitsProcessor(LogitsProcessor):
    """把每次采样步的 last-token logits 写到序号文件。

    vLLM 0.21 sampler 按序列调用 apply（每次 logits=[1, vocab]），且同一引擎的每次
    generate 都会触发 —— 因此主进程逐个 prompt generate（max_tokens=1），
    每次恰好 1 次 apply，文件序号即 prompt 序号；堆叠 0..B-1 得到 [B, vocab]。
    """

    def __init__(self, vllm_config, device, is_pin_memory):
        self.count = 0

    def apply(self, logits):
        path = os.environ.get("AWQ_CAPTURE_PATH")
        if path:
            np.save(f"{path}.{self.count}", logits.float().cpu().numpy())
            self.count += 1
        return logits

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        pass


def run_base_hf(base_model: str, capture_dir: str, device: str) -> None:
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

    print(f"[base-hf] 加载 bf16 基线 {base_model} ...")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(device).eval()

    # batch encode（padding），保存 HF 版 ids/mask 供 vLLM 端 unpad 精确对齐
    enc = processor(text=PROMPTS, return_tensors="pt", padding=True,
                    truncation=True, max_length=256)
    np.save(os.path.join(capture_dir, "hf_input_ids.npy"), enc["input_ids"].cpu().numpy())
    np.save(os.path.join(capture_dir, "hf_attention_mask.npy"), enc["attention_mask"].cpu().numpy())

    inputs = {k: v.to(device) for k, v in enc.items() if hasattr(v, "to")}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits  # [B, seq, vocab]
    last_idx = enc["attention_mask"].sum(-1) - 1  # 每行最后一个非 pad token
    base_logits = torch.stack(
        [logits[b, last_idx[b]].float().cpu() for b in range(len(PROMPTS))])
    np.save(os.path.join(capture_dir, "base_logits.npy"), base_logits.numpy())
    print(f"  base_logits: {tuple(base_logits.shape)}")

    # 生成文本（逐个短生成，防 62GB bf16 + KV cache OOM）
    texts = []
    for pr in PROMPTS:
        in_ids = processor(text=pr, return_tensors="pt")["input_ids"].to(device)
        gen = model.generate(input_ids=in_ids, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False)
        texts.append(processor.batch_decode(
            gen[:, in_ids.shape[-1]:], skip_special_tokens=True)[0])
    with open(os.path.join(capture_dir, "base_texts.json"), "w") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(f"  base 生成文本已存: {texts}")


def run_quant(model_path: str, capture_dir: str, device: str, gpu_mem: float) -> None:
    from vllm import LLM, SamplingParams

    ids = np.load(os.path.join(capture_dir, "hf_input_ids.npy"))
    mask = np.load(os.path.join(capture_dir, "hf_attention_mask.npy"))
    # 用 HF 的 attention_mask 恢复每 prompt 原始（未 padding）token ids，保证与基线精确对齐
    raw_ids = [ids[i, : int(mask[i].sum())].tolist() for i in range(len(PROMPTS))]
    print(f"[quant] 加载量化模型 {model_path} (quantization=awq_ascend) ...")

    q_path = os.path.join(capture_dir, "q_logits.npy")
    for f in glob.glob(q_path + ".*"):
        os.remove(f)
    if os.path.exists(q_path):
        os.remove(q_path)
    os.environ["AWQ_CAPTURE_PATH"] = q_path
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=2048,
        trust_remote_code=True,
        max_num_seqs=8,
        gpu_memory_utilization=gpu_mem,
        logits_processors=[BatchCaptureLogitsProcessor],
    )
    # 逐个 prompt 捕获（sampler 按序列调用 apply，序号文件即 prompt 序号）
    for rids in raw_ids:
        llm.generate(rids, SamplingParams(max_tokens=1, temperature=0.0))
    files = sorted(glob.glob(q_path + ".*"))
    assert len(files) == len(PROMPTS), f"捕获到 {len(files)} 行，期望 {len(PROMPTS)}"
    q_logits = np.concatenate([np.load(f) for f in files], axis=0)  # [B, vocab]
    np.save(q_path, q_logits)  # 供 compare 直接读
    print(f"  q_logits: {tuple(q_logits.shape)}")

    # 文本生成
    outs = llm.generate(PROMPTS, SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0))
    texts = [o.outputs[0].text for o in outs]
    with open(os.path.join(capture_dir, "q_texts.json"), "w") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(f"  quant 生成文本已存: {texts}")


def run_compare(capture_dir: str) -> int:
    base = torch.from_numpy(np.load(os.path.join(capture_dir, "base_logits.npy")))
    q = torch.from_numpy(np.load(os.path.join(capture_dir, "q_logits.npy")))
    assert base.shape == q.shape, f"形状不一致: {base.shape} vs {q.shape}"
    base_texts = json.load(open(os.path.join(capture_dir, "base_texts.json")))
    q_texts = json.load(open(os.path.join(capture_dir, "q_texts.json")))

    print(f"{'#':>2}  {'cosine':>8}  {'max_abs':>10}  prompt")
    cosines = []
    for b, pr in enumerate(PROMPTS):
        cos = torch.nn.functional.cosine_similarity(base[b], q[b], dim=-1).item()
        mabs = (q[b] - base[b]).abs().max().item()
        cosines.append(cos)
        print(f"{b:>2}  {cos:>8.4f}  {mabs:>10.3e}  {pr[:24]}")
    mean = float(np.mean(cosines))
    print(f"\n平均 cosine: {mean:.6f}   {'✅ PASS (≥0.99)' if mean >= 0.99 else '❌ FAIL (<0.99)'}")

    print("\n生成文本对比（量化 vs 基线）:")
    for b, (bt, qt) in enumerate(zip(base_texts, q_texts)):
        mark = "✓" if bt == qt else "~"
        print(f"[{b}] {PROMPTS[b][:16]}...")
        print(f"  {mark} base : {bt!r}")
        print(f"  {mark} quant: {qt!r}")
    return 0 if mean >= 0.99 else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["base-hf", "quant", "compare"], required=True)
    p.add_argument("--model", default="/data/models/Qwen3-VL-30B-A3B-AWQ")
    p.add_argument("--base-model", default="/data/models/Qwen3-VL-30B-A3B-Instruct")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--capture-dir", default="/tmp/awq_capture_30b")
    p.add_argument("--gpu-mem", type=float, default=0.7)
    args = p.parse_args()

    os.makedirs(args.capture_dir, exist_ok=True)
    if args.mode == "base-hf":
        run_base_hf(args.base_model, args.capture_dir, args.device)
    elif args.mode == "quant":
        run_quant(args.model, args.capture_dir, args.device, args.gpu_mem)
    else:
        return run_compare(args.capture_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
