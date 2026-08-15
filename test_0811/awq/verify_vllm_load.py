#!/usr/bin/env python3
"""端到端验证：vLLM 在 Ascend NPU 上加载我们的 AWQ checkpoint，与 FP16 基线对比 logits。

前提：
  - vllm-ascend 已安装且包含 awq_ascend 插件（editable）
  - save_awq_model.py 已产出量化 checkpoint

用法（每步独立进程，避免 fork-after-threads 崩溃）:
  python verify_vllm_load.py --mode base     [--base-model <原始模型目录>]   # 捕获 FP16 基线 logits
  python verify_vllm_load.py --mode quant    [--model <量化模型目录>]         # 捕获 AWQ logits
  python verify_vllm_load.py --mode compare  [--capture-dir <目录>]           # 计算余弦对比

度量：同一 prompt 的 last-token logits 余弦相似度（沿用 MVP 标准，目标 > 0.99）。
"""
from __future__ import annotations

import argparse
import os

# 本机 fork-after-threads 竞态（父进程 import torch 后 fork EngineCore 会撞
# "Invalid thread pool!" abort）；用 spawn 全新解释器启动 EngineCore 子进程规避。
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import torch

# 先注册 awq_ascend 方法（@register_quantization_config）再构造 LLM
import vllm_ascend.quantization.awq_ascend_config  # noqa: F401

# 引擎级 LogitsProcessor：在 EngineCore 子进程里抓 last-token logits 并写文件传回
from capture_logits_proc import CaptureLogitsProcessor


def make_prompt() -> str:
    return "自然语言处理的核心任务包括文本分类、机器翻译、问答系统和文本生成等。"


def capture_logits(llm, prompt: str, out_path: str) -> torch.Tensor:
    """生成 1 个 token，把第一步采样的 last-token logits 从子进程文件读回。"""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    llm.generate([prompt], sp)
    assert os.path.exists(out_path), "没有捕获到 logits（子进程未写文件）"
    return torch.from_numpy(np.load(out_path))


def run_one(model_path: str, prompt: str, quantized: bool, device: str,
            out_path: str, prompt_tokens: list[int] | None = None,
            gpu_mem: float = 0.6) -> torch.Tensor:
    from vllm import LLM
    # 必须在 LLM() 构造（EngineCore 子进程 fork）之前设置，worker 才会继承
    os.environ["AWQ_CAPTURE_PATH"] = out_path
    if os.path.exists(out_path):
        os.remove(out_path)
    kwargs = dict(
        model=model_path,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=2048,
        trust_remote_code=True,
        max_num_seqs=8,
        gpu_memory_utilization=gpu_mem,
        # vLLM 0.21: logits_processors 从 SamplingParams 移到引擎级，传 processor 类
        logits_processors=[CaptureLogitsProcessor],
    )
    if quantized:
        kwargs["quantization"] = "awq_ascend"
    llm = LLM(**kwargs)
    # 传入 token ids（list[int]）而非字符串，保证与 HF 基线 token 完全对齐
    logits = capture_logits(llm, prompt_tokens if prompt_tokens is not None else prompt, out_path)
    print(f"  [{model_path}] 捕获 last-token logits: {tuple(logits.shape)}, "
          f"norm={logits.norm().item():.3f}")
    return logits


def run_base_hf(model_path: str, prompt: str, device: str, out_path: str,
                tokens_path: str) -> torch.Tensor:
    """bf16 基线经 HF transformers 捕获（30B 权重 62.2GB，vLLM 单卡 64GB 装不下）。

    返回 last-token logits，并把输入 token ids 存到 tokens_path 供 vLLM 端精确对齐。
    """
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(device).eval()

    enc = processor(text=prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    np.save(tokens_path, input_ids.cpu().numpy())
    inputs = {k: v.to(device) for k, v in enc.items() if hasattr(v, "to")}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :].float().cpu()  # [vocab]
    np.save(out_path, logits.numpy())
    print(f"  [{model_path}] HF 基线捕获: {tuple(logits.shape)}, "
          f"norm={logits.norm().item():.3f}, tokens={input_ids.shape[-1]}")
    return logits


def load_tokens(tokens_path: str) -> list[int]:
    import numpy as np
    return np.load(tokens_path).reshape(-1).tolist()


def compare(capture_dir: str) -> int:
    """对比两个 npy：计算 last-token logits 余弦相似度。"""
    base_path = os.path.join(capture_dir, "base_logits.npy")
    q_path = os.path.join(capture_dir, "q_logits.npy")
    assert os.path.exists(base_path), f"缺少基线 logits: {base_path}"
    assert os.path.exists(q_path), f"缺少量化 logits: {q_path}"
    base = torch.from_numpy(np.load(base_path))
    q = torch.from_numpy(np.load(q_path))
    assert base.shape == q.shape, f"形状不一致: {base.shape} vs {q.shape}"
    cos = torch.nn.functional.cosine_similarity(base, q, dim=-1).item()
    rel = (q - base).abs().max().item()
    print(f"  last-token logits 余弦相似度: {cos:.6f}")
    print(f"  last-token logits 最大绝对差: {rel:.3e}")
    ok = cos >= 0.99
    print(f"  结论: {'✅ PASS (≥0.99)' if ok else '❌ FAIL (<0.99)'}")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["base", "base-hf", "quant", "compare"],
                   required=True)
    p.add_argument("--model", default="/data/models/Qwen3-VL-30B-A3B-AWQ",
                   help="量化 checkpoint 目录")
    p.add_argument("--base-model", default="/data/models/Qwen3-VL-30B-A3B-Instruct")
    p.add_argument("--prompt", default=make_prompt())
    p.add_argument("--device", default="npu:0")
    p.add_argument("--capture-dir", default="/tmp/awq_capture_30b",
                   help="跨进程 logits 捕获目录")
    p.add_argument("--gpu-mem", type=float, default=0.6,
                   help="vLLM gpu_memory_utilization")
    args = p.parse_args()

    os.makedirs(args.capture_dir, exist_ok=True)
    base_path = os.path.join(args.capture_dir, "base_logits.npy")
    q_path = os.path.join(args.capture_dir, "q_logits.npy")
    tokens_path = os.path.join(args.capture_dir, "prompt_tokens.npy")

    if args.mode == "base":
        print("[base] 加载并运行 bf16 基线 (vLLM) ...")
        run_one(args.base_model, args.prompt, quantized=False, device=args.device,
                out_path=base_path, gpu_mem=args.gpu_mem)
    elif args.mode == "base-hf":
        print("[base-hf] 加载并运行 bf16 基线 (HF transformers, 30B 专用) ...")
        run_base_hf(args.base_model, args.prompt, args.device, base_path,
                    tokens_path)
    elif args.mode == "quant":
        print("[quant] 加载并运行 AWQ 量化模型 (quantization=awq_ascend) ...")
        toks = load_tokens(tokens_path) if os.path.exists(tokens_path) else None
        run_one(args.model, args.prompt, quantized=True, device=args.device,
                out_path=q_path, prompt_tokens=toks, gpu_mem=args.gpu_mem)
    else:
        print("[compare] 对比 ...")
        return compare(args.capture_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
