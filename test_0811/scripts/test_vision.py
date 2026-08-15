#!/usr/bin/env python3
"""视觉问答测试：向 vLLM 服务发送一张图 + 一个问题。

用法:
    python test_vision.py --url http://127.0.0.1:8001 --image /path/to/img.png [--prompt "问什么"]

已实测: Qwen3-VL-8B 单卡跑通；Qwen3-VL-30B-A3B 双卡应同样可用。
"""
import argparse
import base64
import json
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8001")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="图片里有什么？用一句话回答")
    p.add_argument("--model", default="/data/models/Qwen3-VL-30B-A3B-Instruct")
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args()

    with open(args.image, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": args.prompt},
            ],
        }],
        "max_tokens": args.max_tokens,
        "temperature": 0.2,
    }

    req = urllib.request.Request(
        f"{args.url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    print("RESPONSE:", resp["choices"][0]["message"]["content"])
    print("usage:", resp.get("usage"))


if __name__ == "__main__":
    main()
