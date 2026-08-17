#!/usr/bin/env python3
"""并发压测：N 个并发请求，每个请求都带一张图片。

用法:
    python load_test_concurrent.py --url http://127.0.0.1:8001 \
        --image /tmp/test_red_circle.png \
        --concurrency 8 --requests 40 --max-tokens 64

指标:
    - 成功率 / 总耗时 / 吞吐(requests/s)
    - 延迟分布 p50 / p90 / p95 / p99
    - 生成吞吐 tokens/s（累计 completion_tokens）
    - 首请求单独报告（vLLM 首请求会触发图编译，延迟极高，属正常现象）
"""
import argparse
import asyncio
import base64
import json
import time

import aiohttp

PERCENTILES = [50, 90, 95, 99]


def parse_args():
    p = argparse.ArgumentParser(description="带图片的并发压测")
    p.add_argument("--url", default="http://127.0.0.1:8001")
    p.add_argument("--model", default="/data/models/Qwen3-VL-30B-A3B-Instruct")
    p.add_argument("--image", required=True, help="每张请求都带这张图片")
    p.add_argument("--prompt", default="图片里有什么？用一句话回答")
    p.add_argument("--concurrency", type=int, default=8, help="并发数")
    p.add_argument("--requests", type=int, default=32, help="总请求数")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--timeout", type=float, default=180, help="单请求超时(秒)")
    p.add_argument("--warmup", action="store_true", help="先发 1 个请求热图编译，不计入统计")
    return p.parse_args()


def build_payload(args, image_b64):
    return {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": args.prompt},
            ],
        }],
        "max_tokens": args.max_tokens,
        "temperature": 0.2,
    }


async def one_request(session, args, payload):
    """发送单个请求，返回 (ok, latency_s, completion_tokens)。"""
    start = time.perf_counter()
    try:
        async with session.post(f"{args.url}/v1/chat/completions",
                                json=payload, timeout=aiohttp.ClientTimeout(total=args.timeout)) as resp:
            body = await resp.json()
        latency = time.perf_counter() - start
        if resp.status != 200:
            return False, latency, 0, f"HTTP {resp.status}: {str(body)[:200]}"
        if "choices" not in body:
            return False, latency, 0, f"异常响应: {str(body)[:200]}"
        n_tok = body.get("usage", {}).get("completion_tokens", 0)
        return True, latency, n_tok, None
    except Exception as e:
        return False, time.perf_counter() - start, 0, str(e)[:200]


async def worker(session, args, payload, q, results):
    while True:
        try:
            idx = q.get_nowait()
        except asyncio.QueueEmpty:
            return
        ok, lat, tok, err = await one_request(session, args, payload)
        results[idx] = (ok, lat, tok, err)


async def main():
    args = parse_args()

    with open(args.image, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    payload = build_payload(args, image_b64)

    connector = aiohttp.TCPConnector(limit=args.concurrency, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        # 热图编译：vLLM 首次请求会触发 decode 图编译，单独跑并报告
        if args.warmup:
            print(">>> warmup 1 个请求（触发图编译，可能较慢）...")
            t0 = time.perf_counter()
            ok, lat, tok, err = await one_request(session, args, payload)
            print(f"    warmup 完成: ok={ok} latency={lat:.1f}s {'' if ok else ('ERR: '+str(err))}")

        q = asyncio.Queue()
        for i in range(args.requests):
            q.put_nowait(i)
        results = [None] * args.requests

        print(f">>> 开始压测: concurrency={args.concurrency} requests={args.requests}")
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(session, args, payload, q, results)
                               for _ in range(args.concurrency)])
        wall = time.perf_counter() - t0

    oks = [r for r in results if r[0]]
    fails = [r for r in results if not r[0]]
    latencies = sorted(r[1] for r in oks)
    tokens = sum(r[2] for r in oks)
    n_ok = len(oks)

    print("\n===== 压测结果 =====")
    print(f"总请求:      {args.requests}")
    print(f"成功/失败:   {n_ok} / {len(fails)}  (成功率 {100*n_ok/max(args.requests,1):.1f}%)")
    print(f"总耗时:      {wall:.1f}s")
    print(f"吞吐:        {n_ok/wall:.2f} requests/s")
    print(f"生成吞吐:    {tokens/wall:.1f} tokens/s")
    if latencies:
        print(f"延迟分布:    p50={latencies[max(0,int(len(latencies)*0.50))-1]*1000:.0f}ms"
              f"  p90={latencies[max(0,int(len(latencies)*0.90))-1]*1000:.0f}ms"
              f"  p95={latencies[max(0,int(len(latencies)*0.95))-1]*1000:.0f}ms"
              f"  p99={latencies[max(0,int(len(latencies)*0.99))-1]*1000:.0f}ms")
        print(f"平均延迟:    {sum(latencies)/len(latencies)*1000:.0f}ms")
        print(f"最长延迟:    {latencies[-1]*1000:.0f}ms")
    if fails:
        print("\n失败样例（前3条）:")
        for f in fails[:3]:
            print("   ", f[3])


if __name__ == "__main__":
    asyncio.run(main())
