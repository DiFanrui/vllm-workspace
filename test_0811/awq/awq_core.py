"""手写 AWQ（Activation-aware Weight Quantization）核心算法 —— 纯 torch，Ascend NPU 可用。

实现与 AutoAWQ v0.2.9（awq/quantize/quantizer.py）对齐：
  - pseudo_quantize_tensor: 对称 INT4，per-group scale = max|w|/7，量化范围 [-8, 7]
  - get_best_scale: 逐输入通道 scale s = clamp(x_mean^ratio, 1e-4)，归一化后
    用 20 个 ratio 网格搜索，损失 = 量化权重后该层真实输出 vs fp16 输出的 L2
  - 关键认知：scale 必须逐通道（各输入通道激活强度不同），不能整组同一标量，
    否则对称量化是 scale-invariant 的（round(w·s/(s·max/7)) 与 s 无关），搜索无效果。

Stage 1 验证标准：对同一层，AWQ 的量化输出误差 < 普通 INT4 的输出误差。
Stage 2 再对齐可加载格式（msmodelslim / vllm-ascend）。
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# 1. INT4 对称 group 量化（基础算子，AWQ 与非 AWQ 共用）
# ---------------------------------------------------------------------------

def int4_quant_scale(w: torch.Tensor, group_size: int = 128, num_bits: int = 4) -> torch.Tensor:
    """每个 (输出行, 输入 group) 的对称量化 scale。

    w: [*, in]; 返回 [*, in/group_size]。
    scale = max(|w|) / (2^(num_bits-1) - 1)，与 AutoAWQ pseudo_quantize_tensor 一致。
    """
    max_q = 2 ** (num_bits - 1) - 1
    w_g = w.to(torch.float32).reshape(*w.shape[:-1], -1, group_size)
    return w_g.abs().amax(dim=-1).clamp_min(1e-5) / max_q


def int4_quantize_round(w: torch.Tensor, scale: torch.Tensor,
                        group_size: int = 128, num_bits: int = 4) -> torch.Tensor:
    """对权重做取整量化（不打包），返回 int4 数值（int8 存放）。

    w: [*, in]; scale: [*, in/group_size]。量化范围 [-2^(b-1), 2^(b-1)-1]。
    """
    max_q = 2 ** (num_bits - 1) - 1
    min_q = -(2 ** (num_bits - 1))
    w_g = w.to(torch.float32).reshape(*w.shape[:-1], -1, group_size)
    q = (w_g / scale.unsqueeze(-1)).round().clamp(min_q, max_q).to(torch.int8)
    return q.reshape_as(w)


def int4_dequant(q: torch.Tensor, scale: torch.Tensor,
                 group_size: int = 128) -> torch.Tensor:
    """int4 反量化回 fp32: w_hat = q * scale。q: [*, in]; scale: [*, in/group_size]。"""
    q_f = q.to(torch.float32).reshape(*q.shape[:-1], -1, group_size)
    return (q_f * scale.unsqueeze(-1)).reshape_as(q)


def int4_reconstruct(w: torch.Tensor, group_size: int = 128, num_bits: int = 4) -> torch.Tensor:
    """普通 INT4 重建（无 AWQ scale）。"""
    scale = int4_quant_scale(w, group_size, num_bits)
    q = int4_quantize_round(w, scale, group_size, num_bits)
    return int4_dequant(q, scale, group_size)


# ---------------------------------------------------------------------------
# 2. AWQ：激活感知 scale 搜索 + 重建（对齐 AutoAWQ get_best_scale）
# ---------------------------------------------------------------------------

def awq_scale_search(w: torch.Tensor, x: torch.Tensor, group_size: int = 128,
                     num_bits: int = 4, n_grid: int = 20) -> tuple:
    """AutoAWQ 风格的激活感知 scale 搜索。

    w: [out, in]（Linear 权重）; x: [n_tokens, in]（该层输入激活，校准集）。
    返回 (best_s, best_err):
      best_s:  [in] 逐输入通道 scale = x_mean^ratio（ratio 由网格搜索选出）
      best_err: 对应量化后该层输出的 L2 误差

    对 ratio ∈ {0/20, 1/20, ..., 19/20}：
      s = clamp(x_mean^ratio, 1e-4) / sqrt(max(s)·min(s))   # x_mean = 逐通道激活均值
      w_hat = dequant(quant(W·s)) / s                       # 量化 W·s，再逐通道除回 s
      loss  = || x·W^T - x·w_hat^T ||²                      # 真实输出空间误差
    """
    x_mean = x.to(torch.float32).abs().mean(dim=0).clamp_min(1e-4)  # [in]
    w_f = w.to(torch.float32)
    out_base = x.to(torch.float32) @ w_f.T  # [n, out]

    best_s = torch.ones_like(x_mean)
    best_err = float("inf")

    for i in range(n_grid):
        ratio = i / n_grid
        s = x_mean.pow(ratio)
        s = s / (s.max() * s.min()).sqrt()

        w_s = w_f * s                              # W·s（逐通道缩放）
        gscale = int4_quant_scale(w_s, group_size, num_bits)
        q = int4_quantize_round(w_s, gscale, group_size, num_bits)
        w_hat = int4_dequant(q, gscale, group_size) / s   # dequant 后逐通道 /s

        out_q = x.to(torch.float32) @ w_hat.T
        err = ((out_q - out_base) ** 2).mean().item()
        if err < best_err:
            best_err, best_s = err, s

    return best_s, best_err


def awq_reconstruct(w: torch.Tensor, s: torch.Tensor, group_size: int = 128,
                    num_bits: int = 4, clip_max: torch.Tensor | None = None) -> torch.Tensor:
    """用选出的 scale（可选 clip）做 AWQ 重建。

    w_hat = dequant(quant(clip(W·s, ±clip_max))) / s。
    """
    w_s = w.to(torch.float32) * s
    if clip_max is not None:
        clip = clip_max.repeat_interleave(group_size, dim=-1)
        w_s = w_s.clamp(-clip, clip)
    gscale = int4_quant_scale(w_s, group_size, num_bits)
    q = int4_quantize_round(w_s, gscale, group_size, num_bits)
    return int4_dequant(q, gscale, group_size) / s


def awq_layer_error(w: torch.Tensor, x: torch.Tensor, group_size: int = 128,
                    num_bits: int = 4, n_grid: int = 20,
                    do_clip: bool = True) -> dict:
    """一层 Linear 的三路对比：普通 INT4 vs AWQ-scale vs AWQ-scale+clip。

    误差均为该层量化后真实输出 vs fp16 输出的 L2。
    """
    x_f = x.to(torch.float32)
    out_base = x_f @ w.to(torch.float32).T

    # 普通 INT4 对照
    w_plain = int4_reconstruct(w, group_size, num_bits)
    err_plain = ((x_f @ w_plain.T - out_base) ** 2).mean().item()

    # AWQ scale
    s, _ = awq_scale_search(w, x, group_size, num_bits, n_grid)
    w_awq = awq_reconstruct(w, s, group_size, num_bits)
    err_awq = ((x_f @ w_awq.T - out_base) ** 2).mean().item()

    # AWQ scale + clip（clip 阈值在 W·s 上搜索，激活传 X/s —— 与 AutoAWQ 的 apply_scale → search_best_clip 一致）
    clip_stats = None
    if do_clip:
        w_s = w.to(torch.float32) * s
        clip_max = awq_clip_search(w_s, x_f / s, group_size, num_bits, n_grid)
        w_awqc = awq_reconstruct(w, s, group_size, num_bits, clip_max)
        err_awqc = ((x_f @ w_awqc.T - out_base) ** 2).mean().item()
        # 只存小统计量，避免把整个 clip 矩阵写进 JSON（会到 GB 级）
        org_max = w_s.reshape(w_s.shape[0], -1, group_size).abs().amax(dim=-1)
        shrink = clip_max / org_max
        clip_stats = {
            "clipped_groups_pct": ((org_max - clip_max).abs() > 1e-9).float().mean().item(),
            "shrink_min": shrink.min().item(),
            "shrink_mean": shrink.mean().item(),
        }
    else:
        err_awqc = err_awq

    return {
        "awq_out_l2": err_awq,
        "awq_clip_out_l2": err_awqc,
        "plain_out_l2": err_plain,
        "improve": (1 - err_awq / err_plain) if err_plain else float("nan"),
        "improve_clip": (1 - err_awqc / err_plain) if err_plain else float("nan"),
        "clip_stats": clip_stats,
    }


# ---------------------------------------------------------------------------
# 3. AWQ weight clipping（AutoAWQ _compute_best_clip：搜每组最优裁剪阈值）
# ---------------------------------------------------------------------------

def awq_clip_search(w: torch.Tensor, x: torch.Tensor, group_size: int = 128,
                    num_bits: int = 4, n_grid: int = 20, max_shrink: float = 0.5,
                    n_sample_token: int = 512) -> torch.Tensor:
    """搜索每组的最优权重裁剪阈值 clip_max。

    w: [out, in] 应为已 scale 的权重 W·s; x: [n, in] 应为缩放后的激活 X/s
      （AutoAWQ apply_scale 里 inp.div_(scales) 后传给 search_best_clip），
      这样 (X/s)·(W·s)^T = X·W^T，参考输出就是真实 fp16 输出。
    返回 clip_max: [out, in/group_size]。
    思路：outlier 权重会撑大组 scale、让组内其他通道量化变粗。
    对每个 (输出行, 组) 试 clip 阈值 max_val = org_max·(1 - i/20)：
      把权重裁到 ±max_val → INT4 量化 → 算该组输出误差 → 选误差最小的阈值。
    """
    out, in_dim = w.shape
    ng = in_dim // group_size
    max_q = 2 ** (num_bits - 1) - 1
    min_q = -(2 ** (num_bits - 1))

    w_g = w.to(torch.float32).reshape(out, ng, group_size)   # [out, ng, g]
    x_g = x.to(torch.float32).reshape(x.shape[0], ng, group_size)  # [n, ng, g]

    # 采样 tokens，控制计算量
    # ⚠ 必须切 dim0（token）：`[:, ::step]` 会切 dim1（group），把组数减半后
    # 循环 range(ng) 越界（IndexError: index 8 out of bounds... size 8）。
    n_tok = x_g.shape[0]
    step = max(1, n_tok // n_sample_token)
    x_g = x_g[::step]
    n_tok = x_g.shape[0]

    org_max = w_g.abs().amax(dim=-1, keepdim=True)           # [out, ng, 1]
    best_max = org_max.clone()
    min_errs = torch.full_like(org_max, float("inf"))
    org_out = torch.empty(out, n_tok, ng, device=w.device, dtype=torch.float32)
    for g in range(ng):                                      # 逐组算，避免大张量
        org_out[:, :, g] = (x_g[:, g, :] @ w_g[:, g, :].T).T  # [n, out] -> [out, n]

    for i_s in range(int(max_shrink * n_grid)):              # 10 个候选阈值
        max_val = org_max * (1 - i_s / n_grid)
        cur_w = w_g.clamp(-max_val, max_val)
        gscale = cur_w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / max_q
        q = (cur_w / gscale).round().clamp(min_q, max_q) * gscale

        cur_out = torch.empty_like(org_out)
        for g in range(ng):
            cur_out[:, :, g] = (x_g[:, g, :] @ q[:, g, :].T).T
        err = ((cur_out - org_out) ** 2).mean(dim=1).unsqueeze(-1)  # [out, 1, ng]

        better = err < min_errs
        best_max[better] = max_val[better]
        min_errs[better] = err[better]

    return best_max.squeeze(-1)                              # [out, ng]


# ---------------------------------------------------------------------------
# 4. AWQ 打包（对齐 AWQ 权重的标准存储：qweight + qscales）
# ---------------------------------------------------------------------------

def awq_quantize(w: torch.Tensor, s: torch.Tensor, group_size: int = 128,
                 num_bits: int = 4):
    """用选出的 scale 做 AWQ 量化并打包。

    返回 (qweight_packed, qscales, w_hat):
      qweight_packed: [out, in/2] int8，每字节打包 2 个 int4（低4位 + 高4位）
      qscales:        [out, ng] fp32，W·s 的每组量化 scale
      w_hat:          [out, in] fp32，dequant(Q(W·s))/s —— 用于替换权重做端到端验证
    """
    out, in_dim = w.shape
    ng = in_dim // group_size

    w_s = (w.to(torch.float32) * s).reshape(out, ng, group_size)
    gscale = w_s.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / (2 ** (num_bits - 1) - 1)
    q = (w_s / gscale).round().clamp(-(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1).to(torch.int8)

    # 打包：每字节两个 int4（低4位 + 高4位）
    q_even = q[..., 0::2].to(torch.uint8)
    q_odd = (q[..., 1::2].to(torch.uint8) & 0x0F) << 4
    packed = (q_even | q_odd).reshape(out, in_dim // 2)

    w_hat = (q.to(torch.float32) * gscale / s.reshape(out, ng, group_size)).reshape(out, in_dim)
    return packed, gscale.reshape(out, ng), w_hat


# ---------------------------------------------------------------------------
# 4. 校准数据收集
# ---------------------------------------------------------------------------

def collect_calib_activations(model, batches, target_matchers,
                              max_tokens: int | None = None) -> dict:
    """前向跑校准样本，用 hook 收集匹配层的输入激活。

    batches: list[dict]，每次是 model(**batch) 的输入。
    target_matchers: 层名字符串子串列表，命中即收集（如 "self_attn.q_proj"）。
    返回 {layer_name: X}，X = [Σ tokens, in_dim] fp32，**存 CPU**（NPU 只留瞬时
    单层激活；量化时按需搬回设备，避免 30B 级模型全层激活驻留 NPU 撑爆 64GB）。
    """
    collected: dict[str, list[torch.Tensor]] = {}

    def make_hook(name: str):
        def hook(module, args, kwargs):
            x = args[0]
            if isinstance(x, tuple):
                x = x[0]
            x = x.detach().to(torch.float32).cpu()  # 落 CPU：大模型全层激活驻留 NPU 会 OOM
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            collected.setdefault(name, []).append(x)
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(m in name for m in target_matchers):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in batches:
            model(**batch)

    for h in hooks:
        h.remove()

    out = {}
    for name, xs in collected.items():
        x = torch.cat(xs, dim=0)
        if max_tokens is not None and x.shape[0] > max_tokens:
            x = x[:max_tokens]
        out[name] = x
    return out


# ---------------------------------------------------------------------------
# 5. 工具函数
# ---------------------------------------------------------------------------

def _rel_error(w_hat: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """相对 Frobenius 误差: ||w_hat - w||_F / ||w||_F。"""
    w = w.to(torch.float32)
    return (w_hat.to(torch.float32) - w).norm() / w.norm().clamp_min(1e-8)
