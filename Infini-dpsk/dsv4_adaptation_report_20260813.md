# DeepSeek-V4 Mini 1B / InfiniLM 适配记录

日期：2026-08-13

## 结论

- 当前服务器空间足够开展 1B checkpoint 的适配与单卡验证。
- checkpoint 已完整下载并通过 safetensors 索引、张量数量和文件完整性检查。
- InfiniLM 已能正确装载该 checkpoint，并完成 forward 和 generate。
- 已修复 safetensors MTP 权重过滤、routed scaling 和 bounded SwiGLU 三处兼容问题。
- 当前尚未达到端到端数值一致：剩余误差已定位到 routed expert 执行/加权路径。因此当前输出只可用于适配调试，暂不应视为可信模型输出或进入服务压测。
- 原始工作区 /root/InfiniLM 未被这些源码改动覆盖；适配在隔离 worktree 中完成。

## 设备与磁盘

- GPU：NVIDIA GeForce RTX 4090，24564 MiB。
- 系统盘清理前：30 GB，总剩余约 1.7 GB，使用率 95%。
- 系统盘清理后：剩余约 3.6 GB，使用率 89%。
- 数据盘 /root/autodl-tmp：50 GB，总剩余约 26 GB，使用率约 50%。

本轮仅删除可再生成缓存：

- /root/.cache/pip：约 1.5 GB
- /root/.vscode-server/data/CachedExtensionVSIXs：约 366 MB
- /root/xmake-pkg-downloads：约 130 MB

保留的主要内容：

- /root/autodl-tmp/models/Qwen3-4B-Instruct-2507：约 7.6 GB
- /root/InfiniLM/.venv：约 11 GB
- DSV4 的 InfiniLM/InfiniCore worktree、构建前缀和实验文件

如未来确实需要继续腾数据盘，可优先复核：

- /root/autodl-tmp/vllm-bench312-conda：约 9.4 GB
- /root/autodl-tmp/InfiniCore-official-850：约 1.3 GB
- /root/autodl-tmp/InfiniLM-official-d807：约 991 MB

上述候选本轮均未删除。

## Checkpoint

- 路径：/root/autodl-tmp/models/deepseek-v4-mini-1B-from-flash
- 磁盘占用：约 2.0 GB
- 参数量：1,021,129,744
- 精度：BF16
- 结构摘要：24 层、hidden size 1024、16 attention heads、16 routed experts、top-2、hc_mult=4
- checkpoint 张量键：1889
- MTP 张量键：77
- 去除 MTP 后 checkpoint 键与 InfiniLM C++ 模型键完全一致：1812 / 1812，missing=0，unexpected=0

## 隔离适配环境

- InfiniLM：/root/autodl-tmp/InfiniLM-dpv4-test
- InfiniLM commit：9aa937fa，分支 dpv4-compress
- InfiniCore：/root/autodl-tmp/InfiniCore-dpv4-test
- InfiniCore commit：37c73a8，分支 dpv4
- 构建前缀：/root/autodl-tmp/dpv4-prefix
- CUDA 架构：sm_89

## 本轮源码改动

1. python/infinilm/modeling_utils.py
   - safetensors remap 时同样过滤 mtp.* 权重。
   - 原逻辑只在 .bin 加载路径过滤，导致 safetensors checkpoint 出现多余键。

2. csrc/models/deepseek_v4/deepseek_v4_moe.cpp
   - 在普通、fused hash 和 fallback hash routing 路径中应用 routed_scaling_factor。
   - 原实现读取了该配置，但没有作用到 top-k 权重。

3. csrc/models/deepseek_v4/deepseek_v4_mlp.cpp/.hpp
   - 保存并应用 swiglu_limit。
   - 实现该 checkpoint 所需的 bounded SwiGLU：up 分支限制在 [-limit, limit]，gate 分支只限制上界，再计算 silu(gate) * up。

隔离 worktree 已重新构建和安装，git diff --check 通过。

## 端到端对齐结果

输入为同一条 5-token 序列，对比 checkpoint 自带 Python reference 与 InfiniLM C++。

| 版本 | max abs | mean abs | RMSE | 首 token argmax |
|---|---:|---:|---:|---|
| 初始 DSV4 分支 | 4.671875 | 0.510425 | 0.762983 | 不一致 |
| 加 routed scaling | 4.671875 | 0.509156 | 0.761610 | 不一致 |
| 加 bounded SwiGLU | 4.406250 | 0.322815 | 0.542848 | 不一致 |

修复带来了明显改善，但尚未达到可接受的数值一致性。

## 定位结果

- MHC 等价测试通过。
- 0 层模型的 logits 基本一致，且 argmax 一致。
- 从第 1 个 transformer layer 开始出现显著误差。
- 将 FFN 置零时误差回落到接近 0 层水平，说明 attention 路径基本正常。
- 将 attention 置零、保留 FFN 时误差显著。
- bounded SwiGLU 修复后 shared expert 误差明显下降。
- routed expert 仍是主要误差来源。
- 禁用 fused hash top-k 后结果不变，说明问题不只在 fused router。
- 将 gate 置零形成均匀路由权重后仍有明显误差，进一步指向 routed expert 的执行或权重应用语义。

当前最窄的剩余排查范围是：单个 routed expert 的矩阵权重映射、激活计算、expert 输出聚合和 top-k 权重应用。

## 下一步

1. 建立单层、单 expert 的直接单元测试，固定输入和 expert id。
2. 分别导出 Python reference 与 C++ 的 gate/up/down 中间张量。
3. 先验证单 expert 输出，再验证 top-k 聚合与 routed_scaling_factor。
4. 数值一致后再运行真实生成质量检查和服务端部署测试。

当前不建议直接做吞吐或调度压测，因为模型数值语义仍未完全对齐。
