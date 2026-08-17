#!/usr/bin/env python3
"""P1 内存冒烟：Qwen3-VL-30B-A3B 单卡加载 + 一次 forward，测峰值 HBM。

跑通即证明 P2 的校准阶段在 64GB 单卡上可行。
"""
import time

import torch
import torch_npu  # noqa: F401


def main():
    model_path = "/data/models/Qwen3-VL-30B-A3B-Instruct"
    device = "npu:0"
    torch.manual_seed(0)

    print("[1/3] 加载模型（bf16）...")
    t0 = time.time()
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    print(f"      完成 ({time.time()-t0:.0f}s)")
    print(f"      模型参数: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

    torch_npu.npu.reset_peak_memory_stats(device)
    print(f"[2/3] forward（text-only，~32 token）...")
    t0 = time.time()
    with torch.no_grad():
        enc = processor(text="介绍一下深度学习。", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out = model.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=torch.arange(input_ids.shape[-1], device=device).unsqueeze(0),
        )
        print(f"      out shape: {out.last_hidden_state.shape}")
        print(f"      完成 ({time.time()-t0:.0f}s)")
    torch.npu.synchronize(device)

    print("[3/3] 内存统计")
    peak = torch_npu.npu.max_memory_allocated(device) / 1e9
    total = torch_npu.npu.memory_reserved(device) / 1e9
    print(f"      max_memory_allocated = {peak:.2f} GB")
    print(f"      memory_reserved      = {total:.2f} GB")
    print(f"      HBM 余量 ≈ {64 - total:.2f} GB")
    print("结果: " + ("✅ 可行" if total < 60 else "⚠ 太挤，需要逐层收集/分卡"))


if __name__ == "__main__":
    main()
