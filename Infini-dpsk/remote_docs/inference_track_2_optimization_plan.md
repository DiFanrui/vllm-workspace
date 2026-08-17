# 推理赛道 2 服务能力优化思路

记录时间：2026-07-11

当前背景：
- 目标赛题：启元实验室推理服务能力优化，推理赛道 2-1-2。
- 当前模型：Llama-3.1-8B-Instruct。
- 当前平台：RTX 4090 24GB。
- 当前服务端：InfiniLM commit `7c1efb5`。
- 当前 baseline 会话：`tmux attach -t infini_server` 和 `tmux attach -t infini_bench`。
- 当前启动参数核心：`--enable-graph --enable-paged-attn --attn flash-attn --num-blocks 64 --max-batch-size 64 --ignore-eos`。

## 1. 先建立判断标准

比赛目标不是单点低延迟，而是多并发、长文本服务场景下的输出吞吐和总吞吐提升，同时小规模性能不能明显下降。

所以优化时建议始终用这几类指标分组判断：

1. 短输入长输出：`input=32, output=1024/4096`
   - 主要看 decode 阶段吞吐。
   - 当前 `con=1,in=32,out=256` 的 TPOT 约 17.5ms，单流约 57 tok/s。

2. 中输入长输出：`input=256, output=1024/4096`
   - 兼顾 prefill 和 decode。

3. 长输入：`input=4096`
   - 主要暴露 prefill、KV cache 容量、token budget 和调度策略问题。

4. 高并发：`con=16/64`
   - 主要暴露 batch 调度、KV block 预留策略、CPU 侧请求处理和 decode 合批效率。

优化前后必须按同一套矩阵比较：
- `Successful requests`
- `Output token throughput`
- `Total Token throughput`
- `Mean/P99 TTFT`
- `Mean/P99 TPOT`
- 是否出现 timeout、OOM、No available cache blocks、服务 unhealthy。

## 2. 第一优先级：参数和容量扫描

这部分最适合先做，因为不改代码、风险低、收益可能很直接。

### 2.1 `max-batch-size`

之前启动命令里同时出现过：

```bash
--max-batch-size 64 ... --max-batch-size 4
```

后面的 `4` 很可能覆盖前面的 `64`，导致高并发测试结果偏低。现在已改成只保留：

```bash
--max-batch-size 64
```

建议后续对比：
- `--max-batch-size 4`
- `--max-batch-size 16`
- `--max-batch-size 64`

观察点：
- `con=1/4` 是否下降。
- `con=16/64` 输出吞吐是否提升。
- 高并发是否出现显著 TTFT 变差。

### 2.2 `num-blocks`

当前是：

```bash
--num-blocks 64
--block-size 256
```

总 KV block token 容量大约是：

```text
64 * 256 = 16384 token slots
```

这对 `con=64, output=4096` 理论上远远不够，因为调度器会为运行中请求的剩余输出预留 block。官方参考里 `num-blocks=6144`，但 4090 24GB 很可能放不下。

建议做容量扫描，先不要一步到 6144：

```bash
--num-blocks 64
--num-blocks 96
--num-blocks 128
--num-blocks 192
--num-blocks 256
```

每次重启后先跑小集合：

```text
con=4,  input=256, output=256
con=16, input=256, output=256
con=64, input=256, output=256
con=4,  input=4096, output=256
```

如果启动 OOM 或显存逼近上限，就回退。重点找“能启动且高并发不失败”的最大 `num-blocks`。

### 2.3 `block-size`

当前默认：

```bash
--block-size 256
```

大 block 的优点是 block table 较短、元数据少；缺点是短请求和高并发时内部碎片大。官方矩阵有很多 `input=32` 和 `output=256` 的短/中请求，高并发下 block 内部浪费可能很明显。

建议对比：

```bash
--block-size 128
--block-size 256
```

可能收益：
- `con=16/64, input=32/256` 更容易容纳更多请求。
- KV block 预留更细，减少因为单个请求预留过粗导致的拒绝/等待。

风险：
- block table 变长，paged attention 元数据开销增加。
- 某些底层 kernel 可能对 256 更友好。

### 2.4 `INFINILM_MAX_NUM_BATCHED_TOKENS`

代码位置：`python/infinilm/llm/llm.py`

调度器会读取：

```python
max_num_batched_tokens = int(
    os.getenv("INFINILM_MAX_NUM_BATCHED_TOKENS", max_position_embeddings)
)
assert 1024 <= max_num_batched_tokens <= max_position_embeddings
```

这个值限制一次 prefill schedule 能合批多少 token。

建议扫描：

```bash
export INFINILM_MAX_NUM_BATCHED_TOKENS=1024
export INFINILM_MAX_NUM_BATCHED_TOKENS=2048
export INFINILM_MAX_NUM_BATCHED_TOKENS=4096
export INFINILM_MAX_NUM_BATCHED_TOKENS=8192
```

判断：
- 长输入 `input=4096` 的 TTFT 是否改善。
- 短输入高并发是否被大 prefill 阻塞，导致 TTFT 变差。

可能策略：
- 若比赛只按吞吐，较大 token budget 可能更好。
- 若 P99 TTFT 也被看重，可能需要限制 prefill 一次吃太多。

## 3. 第二优先级：调度器优化

代码位置：`python/infinilm/llm/scheduler.py`

当前调度策略大致是：

1. 只要 waiting queue 能调度出 prefill batch，就先返回 prefill。
2. 如果没有 prefill，再处理 running queue 的 decode。

这意味着新请求的 prefill 可能打断 decode，尤其高并发和长输入时，decode TPOT/ITL 可能被 prefill 干扰。

### 3.1 Decode 优先或 decode/prefill 配额

可尝试策略：
- 如果 running queue 非空，优先调度 decode。
- 或每轮最多调度一定量 prefill token，避免长 prefill 阻塞 decode。
- 或按 `running_queue` 大小动态决定 prefill token budget。

预期收益：
- 长输出场景 TPOT 更稳。
- `con=16/64, output=1024/4096` 输出吞吐可能提升。

风险：
- 新请求 TTFT 可能上升。
- `input=4096` 的长输入总吞吐可能下降。

建议先做实验分支，不直接改主线：
- `scheduler_decode_first`
- `scheduler_prefill_budget`

### 3.2 高并发下 KV 预留策略

代码位置：`Scheduler.can_accept_request()`

当前逻辑会为 running requests 的所有剩余输出 token 预留 block：

```python
remaining_tokens = req.sampling_params.max_tokens - req.get_num_generated_tokens()
num_blocks_needed = (remaining_tokens + block_size - 1) // block_size
```

这很保守。比如 `output=4096` 时，每个请求都会提前预留最多 16 个 block。`con=64` 时会要求非常多 block，即使实际 decode 是逐 token 进行。

可探索：
- 改成只预留短期 decode headroom，例如每个 running request 预留 1-2 个 block。
- 或增加参数控制保守程度，例如 `INFINILM_DECODE_BLOCK_RESERVE`.

预期收益：
- 更容易接收高并发请求。
- 减少 waiting queue 因 KV block 预估过大而停滞。

主要风险：
- 如果低估预留，decode 中途可能无 block，当前代码会抛 `No available cache blocks`。
- 需要配合 admission control，不能让服务在中途失败。

建议先只做实验：
- 对高并发短输出 `con=64,out=256` 看是否提升。
- 再逐步试 `out=1024/4096`。

## 4. 第三优先级：服务端 CPU/JSON/解码路径

代码位置：
- `python/infinilm/server/inference_server.py`
- `python/infinilm/llm/llm.py`

当前非流式 `_chat()` 仍然通过 `stream_request()` 收 token，然后逐 token 拼接：

```python
output_text += token_output.token_text
```

而 `LLMEngine._update_requests()` 每个 token 都会：

```python
pending_tokens = req.generated_token_ids[req._token_decode_offset:]
delta = self.tokenizer.decode(pending_tokens)
```

在 GPU 利用率不满、并发很高、输出很长时，这条 CPU 路径可能成为瓶颈。

可优化方向：

1. benchmark 专用 fast path
   - 因为 vLLM bench 主要依赖 token 数和完成状态，不一定需要完整高质量文本。
   - 可以增加可选参数或环境变量，跳过逐 token decode，只在结束时 decode 一次，甚至返回占位文本。

2. 非流式请求批量输出
   - 对 `_chat()`，内部不用真的走 SSE 语义。
   - 让 request 完成后一次性生成 response。

风险：
- OpenAI API 兼容性可能下降。
- 如果 benchmark 会校验返回文本结构，不能破坏 JSON schema。

建议仅作为比赛实验开关：

```bash
export INFINILM_BENCH_FAST_TEXT=1
```

在开关关闭时保持原行为。

## 5. 第四优先级：投机解码

代码里已有 draft model 相关入口：
- `--draft-model`
- `--num-draft-tokens`
- `SpeculativeRunner`

如果能找到合适的 draft model，并且底层支持稳定，长输出吞吐可能明显提升。

但当前比赛时间紧，环境脆弱，优先级不如参数扫描和调度器改动。建议作为后备大招，不要一开始就碰。

## 6. 推荐执行顺序

### 阶段 A：不改代码，只跑参数表

目标：找到 RTX 4090 上稳定的最佳启动参数。

建议先扫：

```text
max_batch_size: 64
num_blocks: 64, 96, 128, 192
block_size: 128, 256
INFINILM_MAX_NUM_BATCHED_TOKENS: 2048, 4096, 8192
```

每组先跑小矩阵：

```text
con=1,  in=32,   out=256
con=4,  in=256,  out=256
con=16, in=256,  out=256
con=64, in=256,  out=256
con=4,  in=4096, out=256
```

找到稳定参数后，再跑官方 30 点。

### 阶段 B：低风险代码优化

候选：
1. 增加 decode-first 或 prefill budget 调度策略开关。
2. 增加 decode block reserve 策略开关。
3. 增加 benchmark fast text 开关。

原则：
- 全部用环境变量或 CLI 参数控制。
- 默认行为保持不变。
- 每次只改一个变量，跑对照。

### 阶段 C：高风险优化

候选：
1. 投机解码。
2. 更激进的 KV block 超卖/动态回收。
3. 修改底层 kernel 或 graph capture 形状策略。

这些可能收益大，但最容易破坏环境或正确性，建议最后再碰。

## 7. 当前最值得优先尝试的三个点

1. `num_blocks` 和 `block_size` 扫描
   - 这是高并发能否跑满的基础。

2. `INFINILM_MAX_NUM_BATCHED_TOKENS`
   - 这是长输入 prefill 的关键旋钮。

3. 调度器 decode 优先策略
   - 这是长输出、多并发下 TPOT/ITL 的关键。

如果时间只够做一个代码改动，我倾向先做“decode 优先/限制 prefill 干扰”的调度开关；如果时间只够做实验，不改代码，就先扫 `num_blocks + block_size + token budget`。
