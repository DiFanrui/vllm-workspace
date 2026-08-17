#!/usr/bin/env python3
"""多模态（图像）prompt 证据补强：30B-A3B 量化模型 vs bf16 基线。

验证"视觉塔 bf16 + 文本骨干 INT4"的真实使用链路在图像输入下误差是否保持：
  1. 5 张简单几何图形图片（PIL 现场生成，沿用"红色圆形禁止标志"的简单度）
  2. 每张配一个简单问题，逐条对比 last-token logits 余弦
  3. 一次实际生成文本对比（量化 vs 基线"说出的话"）

与 verify_vllm_multi.py（纯文本）的关键区别：
  - HF 端用 processor(images=..., text=...) 编码，input_ids 含 image tokens
  - vLLM 端传 HF 编出的 prompt_token_ids + multi_modal_data（图像）——
    见 vllm/inputs/preprocess.py:143 `_process_multimodal`，TokensPrompt 支持
    {"prompt_token_ids", "multi_modal_data"}，token 精确对齐基线
  - 每张图逐个前向（避免多图 batch padding 的 grid_thw 复杂度）

三步（每步独立进程，避免 fork-after-threads）：
  python verify_vllm_vision.py --mode base-hf   # HF bf16 基线：捕获 logits + 生成文本
  python verify_vllm_vision.py --mode quant     # vLLM awq_ascend 量化端：捕获 + 生成
  python verify_vllm_vision.py --mode compare   # 逐图余弦 + 文本对比表

⚠ 对齐前提：HF 和 vLLM 用同一个模型的 image processor（同一 preprocessor_config.json，
默认 mm_processor_kwargs），image tokens 数量才会一致；若 vLLM 替换报错，需两端
传一致的 mm_processor_kwargs 再跑。
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

# 5 张图 + 问题。Qwen3-VL processor 只有走 chat template 才把 image 替换成
# <|vision_start|><|image_pad|>...<|vision_end|> tokens（裸 text 带 `<image>`
# 不会被替换 → "tokens: 0" 报错）。编码统一用 apply_chat_template。
PROMPTS = [
    "这张图片里有什么？请用一句话回答。",
    "图片中的蓝色图形有几条边？",
    "图片中的图形是什么形状？",
    "图片中一共有几个图形？",
    "图片中是什么图形？",
]
MAX_NEW_TOKENS = 8


def make_images(out_dir: str) -> list[str]:
    """用 PIL 现场生成 5 张简单几何图形（256x256 白底），返回路径列表。

    与 HF/vLLM 复用同一批文件，保证两端输入图片完全一致。
    """
    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)
    paths = []

    def save(name: str, draw_fn) -> str:
        img = Image.new("RGB", (256, 256), "white")
        draw_fn(ImageDraw.Draw(img))
        p = os.path.join(out_dir, name)
        img.save(p)
        paths.append(p)
        return p

    def red_ring(d):  # 红色圆形"禁止标志"
        d.ellipse([48, 48, 208, 208], outline="red", width=14)
        d.line([76, 180, 180, 76], fill="red", width=14)

    def blue_tri(d):  # 蓝色三角形
        d.polygon([(128, 52), (32, 200), (224, 200)], fill="blue")

    def green_sq(d):  # 绿色正方形
        d.rectangle([52, 52, 204, 204], fill="green")

    def ring_tri(d):  # 左红圈 + 右蓝三角（两个图形）
        d.ellipse([16, 76, 118, 178], outline="red", width=12)
        d.polygon([(156, 200), (204, 56), (244, 200)], fill="blue")

    def yellow_star(d):  # 黄色五角星
        import math
        cx, cy, R, r = 128, 128, 96, 38
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rad = R if i % 2 == 0 else r
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
        d.polygon(pts, fill="gold")

    save("img_0.png", red_ring)
    save("img_1.png", blue_tri)
    save("img_2.png", green_sq)
    save("img_3.png", ring_tri)
    save("img_4.png", yellow_star)
    return paths


def _hf_model_class(model_path: str):
    """按 config 架构选 HF 类：dense（8B）用 Qwen3VLForConditionalGeneration，
    MoE（30B-A3B）用 Qwen3VLMoeForConditionalGeneration。"""
    import json as _json
    with open(os.path.join(model_path, "config.json")) as f:
        arch = _json.load(f).get("architectures", [""])[0]
    from transformers import (Qwen3VLForConditionalGeneration,
                              Qwen3VLMoeForConditionalGeneration)
    return Qwen3VLMoeForConditionalGeneration if "Moe" in arch \
        else Qwen3VLForConditionalGeneration


class BatchCaptureLogitsProcessor(LogitsProcessor):
    """vLLM 0.21 引擎级 LogitsProcessor：每次 apply 写序号文件。

    sampler 按序列调用 apply，主进程逐个 generate（max_tokens=1）→ 每图恰好 1 次
    apply，序号文件即图片序号；堆叠 0..B-1 得到 [B, vocab]。
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
    from transformers import AutoProcessor

    imgs = [Image_open(p) for p in _image_paths(capture_dir)]
    print(f"[base-hf] 加载 bf16 基线 {base_model} ...")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = _hf_model_class(base_model).from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(device).eval()

    def encode(img, pr):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": pr}]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return processor(text=text, images=[img], return_tensors="pt")

    # 逐个编码 + 前向（避免多图 batch padding 复杂度），取 last-token logits
    base_logits, lens = [], []
    ids_padded = np.zeros((len(PROMPTS), 512), dtype=np.int64)  # chat template，512 足够
    for i, (img, pr) in enumerate(zip(imgs, PROMPTS)):
        enc = encode(img, pr)
        ids = enc["input_ids"]
        lens.append(ids.shape[-1])
        ids_padded[i, : ids.shape[-1]] = ids.cpu().numpy()
        inputs = {k: v.to(device) for k, v in enc.items() if hasattr(v, "to")}
        with torch.no_grad():
            out = model(**inputs)
        base_logits.append(out.logits[0, -1, :].float().cpu())
        print(f"  [{i}] tokens={ids.shape[-1]} norm={out.logits[0,-1].norm().item():.1f}")

    np.save(os.path.join(capture_dir, "hf_input_ids.npy"), ids_padded)
    np.save(os.path.join(capture_dir, "hf_lens.npy"), np.array(lens))
    base_logits = torch.stack(base_logits)
    np.save(os.path.join(capture_dir, "base_logits.npy"), base_logits.numpy())
    print(f"  base_logits: {tuple(base_logits.shape)}")

    # 生成文本（逐个短生成，防 62GB bf16 + KV cache OOM）
    texts = []
    for i, (img, pr) in enumerate(zip(imgs, PROMPTS)):
        enc = encode(img, pr)
        gen = model.generate(
            **{k: v.to(device) for k, v in enc.items() if hasattr(v, "to")},
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        texts.append(processor.batch_decode(
            gen[:, enc["input_ids"].shape[-1]:], skip_special_tokens=True)[0])
    with open(os.path.join(capture_dir, "base_texts.json"), "w") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(f"  base 生成文本已存: {texts}")


def run_quant(model_path: str, capture_dir: str, device: str, gpu_mem: float,
              quantized: bool = True) -> None:
    from vllm import LLM, SamplingParams

    ids = np.load(os.path.join(capture_dir, "hf_input_ids.npy"))
    lens = np.load(os.path.join(capture_dir, "hf_lens.npy"))
    raw_ids = [ids[i, : int(lens[i])].tolist() for i in range(len(PROMPTS))]
    imgs = [Image_open(p) for p in _image_paths(capture_dir)]
    tag = "quantization=awq_ascend" if quantized else "bf16 未量化"
    print(f"[quant] 加载模型 {model_path} ({tag}) ...")

    q_path = os.path.join(capture_dir, "q_logits.npy")
    for f in glob.glob(q_path + ".*"):
        os.remove(f)
    if os.path.exists(q_path):
        os.remove(q_path)
    os.environ["AWQ_CAPTURE_PATH"] = q_path
    kwargs = dict(
        model=model_path,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=2048,
        trust_remote_code=True,
        max_num_seqs=8,
        gpu_memory_utilization=gpu_mem,
        logits_processors=[BatchCaptureLogitsProcessor],
    )
    if quantized:
        kwargs["quantization"] = "awq_ascend"
    llm = LLM(**kwargs)
    # 逐个 prompt 捕获（HF ids + 图片，token 精确对齐基线）
    for rids, img in zip(raw_ids, imgs):
        llm.generate(
            {"prompt_token_ids": rids, "multi_modal_data": {"image": img}},
            SamplingParams(max_tokens=1, temperature=0.0))
    files = sorted(glob.glob(q_path + ".*"))
    assert len(files) == len(PROMPTS), f"捕获到 {len(files)} 行，期望 {len(PROMPTS)}"
    q_logits = np.concatenate([np.load(f) for f in files], axis=0)  # [B, vocab]
    np.save(q_path, q_logits)
    print(f"  q_logits: {tuple(q_logits.shape)}")

    # 生成文本
    prompts = [{"prompt_token_ids": r, "multi_modal_data": {"image": img}}
               for r, img in zip(raw_ids, imgs)]
    outs = llm.generate(prompts, SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0))
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
        print(f"{b:>2}  {cos:>8.4f}  {mabs:>10.3e}  {pr[:22]}")
    mean = float(np.mean(cosines))
    print(f"\n平均 cosine: {mean:.6f}   {'✅ PASS (≥0.99)' if mean >= 0.99 else '❌ FAIL (<0.99)'}")

    print("\n生成文本对比（量化 vs 基线）:")
    for b, (bt, qt) in enumerate(zip(base_texts, q_texts)):
        mark = "✓" if bt == qt else "~"
        print(f"[{b}] {PROMPTS[b][:16]}...")
        print(f"  {mark} base : {bt!r}")
        print(f"  {mark} quant: {qt!r}")
    return 0 if mean >= 0.99 else 1


def _image_paths(capture_dir: str) -> list[str]:
    """返回 {capture_dir}/img_{0..4}.png；不存在则先生成。"""
    paths = [os.path.join(capture_dir, f"img_{i}.png") for i in range(len(PROMPTS))]
    if not all(os.path.exists(p) for p in paths):
        make_images(capture_dir)
    return paths


def Image_open(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["base-hf", "quant", "compare"], required=True)
    p.add_argument("--model", default="/data/models/Qwen3-VL-30B-A3B-AWQ")
    p.add_argument("--base-model", default="/data/models/Qwen3-VL-30B-A3B-Instruct")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--capture-dir", default="/tmp/awq_capture_30b_vision")
    p.add_argument("--gpu-mem", type=float, default=0.7)
    p.add_argument("--no-quant", action="store_true",
                   help="quant 模式加载 bf16 未量化模型（8B 对照实验用）")
    args = p.parse_args()

    os.makedirs(args.capture_dir, exist_ok=True)
    if args.mode == "base-hf":
        run_base_hf(args.base_model, args.capture_dir, args.device)
    elif args.mode == "quant":
        run_quant(args.model, args.capture_dir, args.device, args.gpu_mem,
                  quantized=not args.no_quant)
    else:
        return run_compare(args.capture_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
